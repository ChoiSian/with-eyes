// MediaPipe FaceLandmarker 기반 상/하 시선 제스처 감지기.
//
// 신호 설계 (보정 기반):
// - 1차 신호(기하): 홍채 중심에서 눈꼬리(내안각-외안각) 선까지의 부호 있는 수직거리
//   / 눈꼬리 간 거리. 머리 기울기(roll)에 불변이고 거리에 불변.
// - 2차 신호: FaceLandmarker 블렌드셰이프 (eyeLookUp/Down 평균 차).
// - 보정에서 각 신호의 판별력(d')을 재서 가중 융합한 z-점수 S를 만든다.
//   S ≈ 0 = 정면, S_up 근처 = 위, S_down 근처 = 아래.
// - 슈미트 트리거(진입/이탈 임계값 분리) + 체류(dwell) + 중립 복귀 재장전으로
//   한 번의 시선 = 정확히 한 번의 응답을 보장한다.
// - 깜빡임 동안 신호를 동결(freeze)하고, 깜빡임 직후 150ms도 무시한다
//   (벨 현상: 눈 감을 때 안구가 위로 굴러 거짓 '위' 신호가 나옴).

const TASKS_VISION_VERSION = '0.10.14';
const WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/wasm`;
const BUNDLE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/vision_bundle.mjs`;
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

const CALIB_KEY = 'aac.calib.v1';

// 랜드마크 인덱스 (478 포인트 모델)
const R_OUTER = 33, R_INNER = 133, R_IRIS = 468, R_LID_UP = 159, R_LID_DOWN = 145;
const L_INNER = 362, L_OUTER = 263, L_IRIS = 473, L_LID_UP = 386, L_LID_DOWN = 374;

export const DEFAULT_PARAMS = {
  dwellMs: 800, // 응답 확정까지 응시 유지 시간
  debounceMs: 100, // 구역 진입 확인 시간
  exitGraceMs: 150, // 구역 이탈 허용 시간 (이보다 길면 체류 취소)
  lockoutMs: 600, // 응답 후 입력 잠금
  neutralArmMs: 300, // 다음 응답 전 중립 유지 요구 시간
  retractEnabled: true,
  retractWarnMs: 2000, // 계속 응시 시 취소 경고 시점 (체류 시작 기준)
  retractMs: 2500, // 이 시점까지 계속 응시하면 방금 응답 취소
  blinkPauseMaxMs: 300, // 이하의 깜빡임은 체류 타이머만 일시정지
  blinkAbortMs: 400, // 초과하면 체류 취소
  postBlinkHoldMs: 150, // 깜빡임 직후 신호 무시 (벨 현상)
  pauseGestureMs: 3000, // 두 눈을 이 시간 이상 감으면 일시정지 제스처
  faceLossMs: 500,
  faceStableMs: 500,
  faceLossAlertMs: 10000,
  emaTauMs: 80,
  neutralBand: 1.5, // |S| < 1.5 = 중립
  driftBetaPerSec: 0.02, // 중립 기준선 적응 속도
  eyeMode: 'both', // 'both' | 'left' | 'right'
};

function median(arr) {
  if (arr.length === 0) return NaN;
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function mad(arr, med) {
  return median(arr.map((x) => Math.abs(x - med)));
}

function percentile(arr, p) {
  if (arr.length === 0) return NaN;
  const s = [...arr].sort((a, b) => a - b);
  const idx = Math.min(s.length - 1, Math.floor((p / 100) * s.length));
  return s[idx];
}

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

// 프레임별 원시 특징 계산. landmarks: 픽셀 좌표로 변환된 배열.
export function extractFeatures(landmarks, blendMap, width, height) {
  const px = (i) => ({ x: landmarks[i].x * width, y: landmarks[i].y * height });

  const eyeGeo = (outerIdx, innerIdx, irisIdx) => {
    const a = px(outerIdx);
    const b = px(innerIdx);
    const p = px(irisIdx);
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len = Math.hypot(dx, dy);
    if (len < 1e-6) return null;
    // 부호 있는 수직거리 (이미지 좌표: 외적이 양수면 홍채가 선 아래쪽).
    // 화면 위쪽 = y 감소이므로 부호를 뒤집어 '위를 보면 양수'가 되게 한다.
    const cross = dx * (p.y - a.y) - dy * (p.x - a.x);
    return -cross / (len * len); // 눈 폭으로 정규화
  };

  const earOf = (upIdx, downIdx, outerIdx, innerIdx) => {
    const up = px(upIdx);
    const down = px(downIdx);
    const a = px(outerIdx);
    const b = px(innerIdx);
    const w = Math.hypot(b.x - a.x, b.y - a.y);
    if (w < 1e-6) return 0;
    return Math.hypot(up.x - down.x, up.y - down.y) / w;
  };

  const geoR = eyeGeo(R_OUTER, R_INNER, R_IRIS);
  const geoL = eyeGeo(L_INNER, L_OUTER, L_IRIS);
  const earR = earOf(R_LID_UP, R_LID_DOWN, R_OUTER, R_INNER);
  const earL = earOf(L_LID_UP, L_LID_DOWN, L_INNER, L_OUTER);

  const blinkR = blendMap.eyeBlinkRight ?? 0;
  const blinkL = blendMap.eyeBlinkLeft ?? 0;
  const blend =
    ((blendMap.eyeLookUpLeft ?? 0) + (blendMap.eyeLookUpRight ?? 0)) / 2 -
    ((blendMap.eyeLookDownLeft ?? 0) + (blendMap.eyeLookDownRight ?? 0)) / 2;

  // 얼굴 기울기(roll)와 눈 사이 거리 (설정 화면용)
  const rMid = { x: (px(R_OUTER).x + px(R_INNER).x) / 2, y: (px(R_OUTER).y + px(R_INNER).y) / 2 };
  const lMid = { x: (px(L_INNER).x + px(L_OUTER).x) / 2, y: (px(L_INNER).y + px(L_OUTER).y) / 2 };
  const rollDeg = (Math.atan2(lMid.y - rMid.y, lMid.x - rMid.x) * 180) / Math.PI;
  const interocularPx = Math.hypot(lMid.x - rMid.x, lMid.y - rMid.y);

  return { geoR, geoL, earR, earL, blinkR, blinkL, blend, rollDeg, interocularPx };
}

// 보정 통계로부터 임계값을 계산한다. samples: {center|up|down: [{geo, blend}...]}
export function computeCalibration(samples, blinkDownSamples) {
  const featureNames = ['geo', 'blend'];
  const stats = {};
  for (const name of featureNames) {
    stats[name] = {};
    for (const target of ['center', 'up', 'down']) {
      const values = samples[target].map((s) => s[name]).filter((v) => Number.isFinite(v));
      if (values.length < 20) {
        return { ok: false, message: `유효 샘플이 부족합니다 (${target}). 조명을 밝게 하고 다시 시도하세요.` };
      }
      const med = median(values);
      // 이상치 제거 후 재계산 (|x-med| > 3*1.4826*MAD)
      const sigma0 = 1.4826 * mad(values, med) + 1e-9;
      const kept = values.filter((v) => Math.abs(v - med) <= 3 * sigma0);
      const med2 = median(kept);
      stats[name][target] = {
        med: med2,
        sigma: 1.4826 * mad(kept, med2) + 1e-9,
      };
    }
  }

  // 특징별 부호/판별력 → 융합 가중치
  const features = {};
  let wSum = 0;
  for (const name of featureNames) {
    const st = stats[name];
    const spanUp = st.up.med - st.center.med;
    const spanDown = st.down.med - st.center.med;
    if (Math.sign(spanUp) === Math.sign(spanDown) || spanUp === 0 || spanDown === 0) {
      features[name] = { weight: 0 }; // 방향을 구분하지 못하는 특징은 제외
      continue;
    }
    const sign = Math.sign(spanUp);
    const sigma = st.center.sigma;
    const dUp = Math.abs(spanUp) / sigma;
    const dDown = Math.abs(spanDown) / sigma;
    const weight = Math.min(dUp, dDown) ** 2;
    features[name] = { weight, sign, med: st.center.med, sigma, dUp, dDown };
    wSum += weight;
  }

  if (wSum <= 0) {
    return { ok: false, message: '위/아래 시선을 구분할 수 없습니다. 카메라를 눈높이에 맞추고 얼굴을 정면으로 비추세요.' };
  }

  // 융합 z-점수에서의 위/아래 평균 위치
  const fusedAt = (target) => {
    let acc = 0;
    for (const name of featureNames) {
      const f = features[name];
      if (!f.weight) continue;
      acc += f.weight * f.sign * ((stats[name][target].med - f.med) / f.sigma);
    }
    return acc / wSum;
  };
  const sUp = fusedAt('up');
  const sDown = fusedAt('down');

  if (sUp < 2.5 || sDown > -2.5) {
    return {
      ok: false,
      dPrimeUp: sUp,
      dPrimeDown: -sDown,
      message:
        '위/아래 구분이 약합니다 (조명을 밝게, 카메라를 40~60cm 눈높이로, 얼굴 정면에서). 다시 보정해 주세요.',
    };
  }

  const calib = {
    features,
    sUp,
    sDown,
    upEnter: Math.max(0.55 * sUp, 3.0),
    upExit: Math.max(0.35 * sUp, 2.0),
    downEnter: Math.min(0.55 * sDown, -3.0),
    downExit: Math.min(0.35 * sDown, -2.0),
    // 아래를 볼 때 눈꺼풀이 따라 내려와 blink 값이 커지므로, 환자 본인의
    // 아래 응시 blink 최대치와 1.0의 중간을 깜빡임 게이트로 쓴다.
    blinkGate: clamp((percentile(blinkDownSamples, 95) + 1) / 2, 0.5, 0.8),
    neutralEar: median(samples.center.map((s) => s.ear).filter(Number.isFinite)),
    weakSignal: Math.min(sUp, -sDown) < 3.0, // 약하면 dwell을 늘리도록 권고
    createdAt: Date.now(),
  };
  return { ok: true, dPrimeUp: sUp, dPrimeDown: -sDown, calib };
}

export class EyeTracker extends EventTarget {
  constructor(params = {}) {
    super();
    this.params = { ...DEFAULT_PARAMS, ...params };
    this.calib = null;
    this.landmarker = null;
    this.video = null;
    this.running = false;
    this.#resetRuntime();
    this.collecting = null; // 보정 수집 상태
  }

  #resetRuntime() {
    this.S = 0;
    this.medianBuf = [];
    this.lastTs = 0;
    this.state = 'idle'; // idle | debounce | dwell | held | lockout
    this.zone = 'neutral'; // up | down | neutral
    this.stateSince = 0;
    this.dwellStart = 0;
    this.dwellPausedAt = 0;
    this.dwellPausedTotal = 0;
    this.outOfZoneSince = 0;
    this.neutralSince = 0;
    this.armed = true;
    this.blinkSince = 0;
    this.blinkGated = false;
    this.postBlinkUntil = 0;
    this.pauseGestureFired = false;
    this.faceLostSince = 0;
    this.faceStableSince = 0;
    this.faceLost = false;
    this.faceLostAlerted = false;
    this.retractWarned = false;
    this.driftNeutralSince = 0;
  }

  setParams(params) {
    Object.assign(this.params, params);
  }

  loadStoredCalibration() {
    try {
      const raw = localStorage.getItem(CALIB_KEY);
      if (raw) {
        this.calib = JSON.parse(raw);
        return true;
      }
    } catch { /* 무시 */ }
    return false;
  }

  saveCalibration() {
    try {
      localStorage.setItem(CALIB_KEY, JSON.stringify(this.calib));
    } catch { /* 무시 */ }
  }

  async init(video) {
    this.video = video;
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    const { FaceLandmarker, FilesetResolver } = await import(BUNDLE_URL);
    const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
    this.landmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true,
    });
  }

  start() {
    if (!this.landmarker || this.running) return;
    this.running = true;
    this.#resetRuntime();
    const scheduleNext = () => {
      if (!this.running) return;
      if (this.video.requestVideoFrameCallback) {
        this.video.requestVideoFrameCallback(() => {
          this.#processFrame(performance.now());
          scheduleNext();
        });
      } else {
        let lastVideoTime = -1;
        const raf = () => {
          if (!this.running) return;
          if (this.video.currentTime !== lastVideoTime) {
            lastVideoTime = this.video.currentTime;
            this.#processFrame(performance.now());
          }
          requestAnimationFrame(raf);
        };
        requestAnimationFrame(raf);
      }
    };
    scheduleNext();
  }

  stop() {
    this.running = false;
  }

  // ===== 보정 =====
  // main이 순서대로 호출: collectTarget('center', 3000) → ... (두 번 반복)
  collectTarget(target, durationMs = 3000, discardMs = 700) {
    return new Promise((resolve) => {
      this.collecting = {
        target,
        samples: [],
        blinkSamples: [],
        startAt: performance.now(),
        discardUntil: performance.now() + discardMs,
        endAt: performance.now() + durationMs,
        resolve,
      };
    });
  }

  static emptyCalibrationSamples() {
    return { center: [], up: [], down: [], blinkDown: [] };
  }

  finishCalibration(allSamples) {
    const result = computeCalibration(
      { center: allSamples.center, up: allSamples.up, down: allSamples.down },
      allSamples.blinkDown,
    );
    if (result.ok) {
      this.calib = result.calib;
      this.saveCalibration();
      if (result.calib.weakSignal && this.params.dwellMs < 1000) {
        this.params.dwellMs = 1000;
      }
    }
    return result;
  }

  // ===== 프레임 처리 =====
  #processFrame(now) {
    let result;
    try {
      result = this.landmarker.detectForVideo(this.video, now);
    } catch {
      return;
    }
    const landmarks = result?.faceLandmarks?.[0];

    if (!landmarks || landmarks.length < 478) {
      this.#onFaceMissing(now);
      return;
    }
    this.#onFacePresent(now);

    const blendMap = {};
    for (const c of result.faceBlendshapes?.[0]?.categories ?? []) {
      blendMap[c.categoryName] = c.score;
    }
    const feat = extractFeatures(landmarks, blendMap, this.video.videoWidth, this.video.videoHeight);

    // 눈 유효성 (모드 + 깜빡임 + EAR)
    const gate = this.calib?.blinkGate ?? 0.5;
    const earFloor = this.calib?.neutralEar ? 0.6 * this.calib.neutralEar : 0;
    const rValid = this.params.eyeMode !== 'left' && feat.blinkR < gate &&
      (earFloor === 0 || feat.earR > earFloor) && Number.isFinite(feat.geoR);
    const lValid = this.params.eyeMode !== 'right' && feat.blinkL < gate &&
      (earFloor === 0 || feat.earL > earFloor) && Number.isFinite(feat.geoL);

    const geoValues = [];
    if (rValid) geoValues.push(feat.geoR);
    if (lValid) geoValues.push(feat.geoL);
    const geo = geoValues.length ? geoValues.reduce((a, b) => a + b, 0) / geoValues.length : NaN;
    const ear = (feat.earR + feat.earL) / 2;

    const blinkMax = Math.max(
      this.params.eyeMode === 'left' ? 0 : feat.blinkR,
      this.params.eyeMode === 'right' ? 0 : feat.blinkL,
    );
    const blinkNow = blinkMax > gate || geoValues.length === 0;

    // 상태 이벤트 (설정/디버그 화면용)
    this.dispatchEvent(new CustomEvent('status', {
      detail: {
        facePresent: true,
        interocularPx: feat.interocularPx,
        rollDeg: feat.rollDeg,
        blink: blinkMax,
        validEyes: geoValues.length,
      },
    }));

    // 보정 샘플 수집 중이면 여기서 끝
    if (this.collecting) {
      this.#collectFrame(now, { geo, blend: feat.blend, ear, blink: blinkMax, blinkNow });
      return;
    }

    // 일시정지 제스처: 두 눈을 오래 감음
    if (blinkNow) {
      if (this.blinkSince === 0) this.blinkSince = now;
      if (!this.pauseGestureFired && now - this.blinkSince >= this.params.pauseGestureMs) {
        this.pauseGestureFired = true;
        this.dispatchEvent(new CustomEvent('pausegesture'));
      }
    } else {
      if (this.blinkSince !== 0) {
        this.postBlinkUntil = now + this.params.postBlinkHoldMs;
      }
      this.blinkSince = 0;
      this.pauseGestureFired = false;
    }

    if (!this.calib) return;

    const gatedNow = blinkNow || now < this.postBlinkUntil;

    // 원시 융합 점수
    if (!gatedNow) {
      const rawS = this.#fusedScore({ geo, blend: feat.blend });
      this.medianBuf.push(rawS);
      if (this.medianBuf.length > 3) this.medianBuf.shift();
      const med = median(this.medianBuf);
      const dt = this.lastTs ? now - this.lastTs : 33;
      const alpha = 1 - Math.exp(-dt / this.params.emaTauMs);
      this.S = this.S + alpha * (med - this.S);

      // 중립 기준선 서서히 적응 (조명 변화/눈꺼풀 처짐 보상)
      if (Math.abs(this.S) < this.params.neutralBand) {
        if (this.driftNeutralSince === 0) this.driftNeutralSince = now;
        if (now - this.driftNeutralSince > 2000) {
          const beta = this.params.driftBetaPerSec * (dt / 1000);
          for (const name of ['geo', 'blend']) {
            const f = this.calib.features[name];
            if (f?.weight) {
              const x = name === 'geo' ? geo : feat.blend;
              if (Number.isFinite(x)) f.med += beta * (x - f.med);
            }
          }
        }
      } else {
        this.driftNeutralSince = 0;
      }
    }
    this.lastTs = now;

    this.#dwellMachine(now, gatedNow);

    this.dispatchEvent(new CustomEvent('gaze', {
      detail: {
        S: this.S,
        zone: this.zone,
        state: this.state,
        progress: this.#dwellProgress(now),
        gated: gatedNow,
        upEnter: this.calib.upEnter,
        downEnter: this.calib.downEnter,
      },
    }));
  }

  #fusedScore({ geo, blend }) {
    let acc = 0;
    let wSum = 0;
    for (const [name, x] of [['geo', geo], ['blend', blend]]) {
      const f = this.calib.features[name];
      if (!f?.weight || !Number.isFinite(x)) continue;
      acc += f.weight * f.sign * ((x - f.med) / f.sigma);
      wSum += f.weight;
    }
    return wSum > 0 ? acc / wSum : 0;
  }

  #collectFrame(now, sample) {
    const col = this.collecting;
    if (now < col.discardUntil) return;
    if (now >= col.endAt) {
      const done = this.collecting;
      this.collecting = null;
      done.resolve({ target: done.target, samples: done.samples, blinkSamples: done.blinkSamples });
      return;
    }
    // 아래 응시 타깃에서는 깜빡임 게이트로 버리지 않는다 (눈꺼풀이 원래 내려옴).
    // 대신 blink 값을 기록해 게이트 보정에 쓴다.
    if (col.target === 'down') {
      col.blinkSamples.push(sample.blink);
      if (Number.isFinite(sample.geo)) col.samples.push(sample);
    } else if (!sample.blinkNow && Number.isFinite(sample.geo)) {
      col.samples.push(sample);
    }
  }

  #zoneOf(S) {
    if (S >= this.calib.upEnter) return 'up';
    if (S <= this.calib.downEnter) return 'down';
    return 'neutral';
  }

  #inZoneSustain(S, zone) {
    // 이탈 임계값 기준 (히스테리시스)
    if (zone === 'up') return S >= this.calib.upExit;
    if (zone === 'down') return S <= this.calib.downExit;
    return false;
  }

  #dwellProgress(now) {
    if (this.state !== 'dwell') return 0;
    const elapsed = now - this.dwellStart - this.dwellPausedTotal -
      (this.dwellPausedAt ? now - this.dwellPausedAt : 0);
    return clamp(elapsed / this.params.dwellMs, 0, 1);
  }

  #dwellMachine(now, gated) {
    const p = this.params;
    const S = this.S;

    // 얼굴 소실 시 리셋은 #onFaceMissing에서 처리
    switch (this.state) {
      case 'idle': {
        if (Math.abs(S) < p.neutralBand) {
          if (this.neutralSince === 0) this.neutralSince = now;
          if (!this.armed && now - this.neutralSince >= p.neutralArmMs) this.armed = true;
        } else {
          this.neutralSince = 0;
        }
        if (!this.armed || gated) break;
        const zone = this.#zoneOf(S);
        if (zone !== 'neutral') {
          this.state = 'debounce';
          this.zone = zone;
          this.stateSince = now;
        }
        break;
      }

      case 'debounce': {
        if (gated || !this.#inZoneSustain(S, this.zone)) {
          this.state = 'idle';
          this.zone = 'neutral';
          break;
        }
        if (now - this.stateSince >= p.debounceMs) {
          this.state = 'dwell';
          this.dwellStart = now;
          this.dwellPausedAt = 0;
          this.dwellPausedTotal = 0;
          this.outOfZoneSince = 0;
          this.dispatchEvent(new CustomEvent('dwellstart', { detail: { dir: this.zone } }));
        }
        break;
      }

      case 'dwell': {
        // 깜빡임: 짧으면 일시정지, 길면 취소
        if (gated) {
          if (this.dwellPausedAt === 0) this.dwellPausedAt = now;
          if (now - this.dwellPausedAt > p.blinkAbortMs) {
            this.#abortDwell('blink');
          }
          break;
        }
        if (this.dwellPausedAt !== 0) {
          const pausedFor = now - this.dwellPausedAt;
          if (pausedFor > p.blinkPauseMaxMs) {
            this.#abortDwell('blink');
            break;
          }
          this.dwellPausedTotal += pausedFor;
          this.dwellPausedAt = 0;
        }
        // 구역 이탈 판정 (이탈 임계값 + 유예)
        if (!this.#inZoneSustain(S, this.zone)) {
          if (this.outOfZoneSince === 0) this.outOfZoneSince = now;
          if (now - this.outOfZoneSince > p.exitGraceMs) {
            this.#abortDwell('left-zone');
            break;
          }
        } else {
          this.outOfZoneSince = 0;
        }
        if (this.#dwellProgress(now) >= 1) {
          const dir = this.zone;
          this.state = 'held';
          this.stateSince = now;
          this.retractWarned = false;
          this.dispatchEvent(new CustomEvent('answer', { detail: { dir } }));
        }
        break;
      }

      case 'held': {
        // 응답 후에도 같은 방향을 계속 응시하면 취소(retract)
        const heldFor = now - this.dwellStart;
        const stillIn = !gated && this.#inZoneSustain(S, this.zone);
        if (p.retractEnabled && stillIn) {
          if (!this.retractWarned && heldFor >= p.retractWarnMs) {
            this.retractWarned = true;
            this.dispatchEvent(new CustomEvent('retractwarn', { detail: { dir: this.zone } }));
          }
          if (heldFor >= p.retractMs) {
            this.dispatchEvent(new CustomEvent('retract', { detail: { dir: this.zone } }));
            this.#toLockout(now);
          }
        } else {
          this.#toLockout(now);
        }
        break;
      }

      case 'lockout': {
        if (now - this.stateSince >= p.lockoutMs) {
          this.state = 'idle';
          this.zone = 'neutral';
          this.armed = false;
          this.neutralSince = 0;
        }
        break;
      }
    }
  }

  #abortDwell(reason) {
    this.state = 'idle';
    this.zone = 'neutral';
    this.dispatchEvent(new CustomEvent('dwellabort', { detail: { reason } }));
  }

  #toLockout(now) {
    this.state = 'lockout';
    this.zone = 'neutral';
    this.stateSince = now;
  }

  #onFaceMissing(now) {
    // 보정 수집 중 얼굴이 안 잡혀도 시간이 다 되면 수집을 끝낸다
    if (this.collecting && now >= this.collecting.endAt) {
      const done = this.collecting;
      this.collecting = null;
      done.resolve({ target: done.target, samples: done.samples, blinkSamples: done.blinkSamples });
    }
    this.faceStableSince = 0;
    if (this.faceLostSince === 0) this.faceLostSince = now;
    if (!this.faceLost && now - this.faceLostSince >= this.params.faceLossMs) {
      this.faceLost = true;
      this.faceLostAlerted = false;
      this.state = 'idle';
      this.zone = 'neutral';
      this.medianBuf = [];
      this.dispatchEvent(new CustomEvent('facelost'));
    }
    if (this.faceLost && !this.faceLostAlerted && now - this.faceLostSince >= this.params.faceLossAlertMs) {
      this.faceLostAlerted = true;
      this.dispatchEvent(new CustomEvent('facelostlong'));
    }
    this.dispatchEvent(new CustomEvent('status', { detail: { facePresent: false } }));
  }

  #onFacePresent(now) {
    this.faceLostSince = 0;
    if (this.faceLost) {
      if (this.faceStableSince === 0) this.faceStableSince = now;
      if (now - this.faceStableSince >= this.params.faceStableMs) {
        this.faceLost = false;
        this.armed = false;
        this.neutralSince = 0;
        this.dispatchEvent(new CustomEvent('facefound'));
      }
    }
  }
}

// 카메라 없이 키보드로 테스트하는 대체 입력기 (같은 이벤트 인터페이스)
export class KeyboardTracker extends EventTarget {
  constructor() {
    super();
    this.running = false;
    this.handler = (e) => {
      if (!this.running) return;
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        this.dispatchEvent(new CustomEvent('answer', { detail: { dir: 'up' } }));
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        this.dispatchEvent(new CustomEvent('answer', { detail: { dir: 'down' } }));
      } else if (e.key === 'Backspace') {
        e.preventDefault();
        this.dispatchEvent(new CustomEvent('retract', { detail: { dir: 'up' } }));
      } else if (e.key === 'p' || e.key === 'P') {
        this.dispatchEvent(new CustomEvent('pausegesture'));
      }
    };
  }

  start() {
    this.running = true;
    window.addEventListener('keydown', this.handler);
  }

  stop() {
    this.running = false;
    window.removeEventListener('keydown', this.handler);
  }
}

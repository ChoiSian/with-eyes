// MediaPipe FaceLandmarker 기반 '위 응시' 제스처 감지기.
// 환자는 아래를 볼 수 없고, 눈을 위로 올렸다 내리는 동작만 가능하다고 가정한다.
// 단일 스위치처럼 동작: 위를 일정 시간 응시하면 '선택' 이벤트 하나를 낸다.
//
// 신호 설계 (보정 기반):
// - 1차 신호(기하): 홍채 중심에서 눈꼬리(내안각-외안각) 선까지의 부호 있는 수직거리
//   / 눈꼬리 간 거리. 머리 기울기(roll)에 불변이고 거리에 불변.
// - 2차 신호: FaceLandmarker 블렌드셰이프 (eyeLookUp/Down 평균 차).
// - 보정(가운데/위 2지점)에서 각 신호의 판별력(d')을 재서 가중 융합한 z-점수 S를 만든다.
//   S ≈ 0 = 정면, S_up 근처 = 위.
// - 슈미트 트리거(진입/이탈 임계값 분리) + 체류(dwell) + 중립 복귀 재장전으로
//   한 번의 위 응시 = 정확히 한 번의 선택을 보장한다.
// - 깜빡임 동안 신호를 동결(freeze)하고, 깜빡임 직후 150ms도 무시한다
//   (벨 현상: 눈 감을 때 안구가 위로 굴러 거짓 '위' 신호가 나옴).

const TASKS_VISION_VERSION = '0.10.14';
const WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/wasm`;
const BUNDLE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/vision_bundle.mjs`;
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

// v2: 결정 신호를 눈동자(홍채 중심) 기하만으로 재구성 — 이전 보정과 호환되지 않음
const CALIB_KEY = 'aac.calib.v2';
const ROTATION_KEY = 'aac.rotation.v1';

// 랜드마크 인덱스 (478 포인트 모델)
const R_OUTER = 33, R_INNER = 133, R_IRIS = 468, R_LID_UP = 159, R_LID_DOWN = 145;
const L_INNER = 362, L_OUTER = 263, L_IRIS = 473, L_LID_UP = 386, L_LID_DOWN = 374;

export const DEFAULT_PARAMS = {
  // '위로 움직이는 이벤트' 감지: 기준선 대비 상승(R)이 임계값을 넘어
  // confirmMs 동안만 유지되면 즉시 선택. 긴 응시 유지가 필요 없다.
  confirmMs: 150, // 잡음 구분용 최소 확인 시간 (선택까지 debounce+confirm ≈ 0.2초)
  debounceMs: 60, // 구역 진입 확인 시간
  exitGraceMs: 120, // 구역 이탈 허용 시간 (이보다 길면 확인 취소)
  lockoutMs: 600, // 선택 후 입력 잠금
  neutralArmMs: 300, // 다음 선택 전 기준선 근처 복귀 요구 시간
  retractEnabled: true,
  retractWarnMs: 1200, // 선택 확정 '후' 계속 응시 시 취소 경고 시점
  retractMs: 1800, // 선택 확정 후 이 시간까지 계속 응시하면 방금 선택 취소
  // 깜빡임 게이트 구간에는 깜빡임 직후 무시 시간(postBlinkHoldMs)도 포함되므로
  // 자연 깜빡임(100~400ms) + 150ms 를 감당할 수 있어야 한다
  blinkPauseMaxMs: 450, // 이하의 게이트 구간은 확인 타이머만 일시정지
  blinkAbortMs: 550, // 초과하면 확인 취소
  postBlinkHoldMs: 150, // 깜빡임 직후 신호 무시 (벨 현상)
  pauseGestureMs: 3000, // 두 눈을 이 시간 이상 감으면 일시정지 제스처
  faceLossMs: 500,
  faceStableMs: 500,
  faceLossAlertMs: 10000,
  emaTauMs: 40, // 빠른 경로 스무딩 (가볍게 — 이벤트 감지가 무뎌지지 않게)
  baselineTauMs: 1200, // 기준선(느린 경로) 추적 시간 상수
  baselineFreezeZ: 1.0, // 이보다 크게 올라가 있는 동안은 기준선 갱신 동결
  neutralBand: 1.0, // R < 1.0 = 기준선 근처 (보정 전 기본값)
  // 이벤트 임계값 = upSensitivity × 보정된 위 응시 크기(sUp).
  // 낮을수록 민감. 보호자 설정에서 실시간 조절 가능.
  upSensitivity: 0.35,
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

  // 눈꼬리 기준선(내안각-외안각)에 대한 임의 점의 부호 있는 수직거리 / 눈 폭.
  // 이미지 좌표에서 화면 위쪽 = y 감소이므로 부호를 뒤집어 '위 = 양수'가 되게 한다.
  // 회전/거리 불변.
  const perpOf = (outerIdx, innerIdx, pointIdx) => {
    const a = px(outerIdx);
    const b = px(innerIdx);
    const p = px(pointIdx);
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len = Math.hypot(dx, dy);
    if (len < 1e-6) return null;
    const cross = dx * (p.y - a.y) - dy * (p.x - a.x);
    return -cross / (len * len);
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

  // 결정 신호는 눈동자(홍채 중심) 기하만 사용한다:
  // geo    = 홍채 중심이 눈꼬리 기준선 위로 올라간 높이
  // geoLow = 홍채 중심이 아래 눈꺼풀에서 떨어진 높이 (아래 눈꺼풀은 시선을
  //          따라 움직이지 않아 안정적인 기준이 된다)
  const geoR = perpOf(R_OUTER, R_INNER, R_IRIS);
  const geoL = perpOf(L_INNER, L_OUTER, L_IRIS);
  const lowLidR = perpOf(R_OUTER, R_INNER, R_LID_DOWN);
  const lowLidL = perpOf(L_INNER, L_OUTER, L_LID_DOWN);
  const geoLowR = geoR !== null && lowLidR !== null ? geoR - lowLidR : null;
  const geoLowL = geoL !== null && lowLidL !== null ? geoL - lowLidL : null;
  const earR = earOf(R_LID_UP, R_LID_DOWN, R_OUTER, R_INNER);
  const earL = earOf(L_LID_UP, L_LID_DOWN, L_INNER, L_OUTER);

  // 블렌드셰이프는 깜빡임 게이트에만 사용 (시선 판정에는 쓰지 않음)
  const blinkR = blendMap.eyeBlinkRight ?? 0;
  const blinkL = blendMap.eyeBlinkLeft ?? 0;

  // 얼굴 기울기(roll)와 눈 사이 거리 (설정 화면용)
  const rMid = { x: (px(R_OUTER).x + px(R_INNER).x) / 2, y: (px(R_OUTER).y + px(R_INNER).y) / 2 };
  const lMid = { x: (px(L_INNER).x + px(L_OUTER).x) / 2, y: (px(L_INNER).y + px(L_OUTER).y) / 2 };
  const rollDeg = (Math.atan2(lMid.y - rMid.y, lMid.x - rMid.x) * 180) / Math.PI;
  const interocularPx = Math.hypot(lMid.x - rMid.x, lMid.y - rMid.y);

  return { geoR, geoL, geoLowR, geoLowL, earR, earL, blinkR, blinkL, rollDeg, interocularPx };
}

// 보정 통계를 계산한다. samples: {center|up: [{geo, geoLow, ear}...]}
// blinkNeutralSamples: 가운데 응시 중의 깜빡임 블렌드셰이프 값들 (게이트 산출용)
// 임계값 자체는 저장하지 않고 런타임에 민감도 설정 × sUp 으로 계산한다.
export function computeCalibration(samples, blinkNeutralSamples) {
  const featureNames = ['geo', 'geoLow'];
  const stats = {};
  for (const name of featureNames) {
    stats[name] = {};
    for (const target of ['center', 'up']) {
      const values = samples[target].map((s) => s[name]).filter((v) => Number.isFinite(v));
      if (values.length < 20) {
        return {
          ok: false,
          message: `유효 샘플이 부족합니다 (${target}). 조명을 밝게 하고, 보정 중에는 이 화면을 계속 켜 둔 채 다시 시도하세요.`,
        };
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
    if (spanUp === 0) {
      features[name] = { weight: 0 }; // 위 응시를 구분하지 못하는 특징은 제외
      continue;
    }
    const sign = Math.sign(spanUp);
    const sigma = st.center.sigma;
    const dUp = Math.abs(spanUp) / sigma;
    const weight = dUp ** 2;
    features[name] = { weight, sign, med: st.center.med, sigma, dUp };
    wSum += weight;
  }

  if (wSum <= 0) {
    return { ok: false, message: '위 응시를 구분할 수 없습니다. 카메라를 눈높이에 맞추고 얼굴을 정면으로 비추세요.' };
  }

  // 융합 z-점수에서의 위 응시 평균 위치
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

  if (sUp < 2.0) {
    return {
      ok: false,
      dPrimeUp: sUp,
      message:
        '위 응시 구분이 약합니다 (조명을 밝게, 카메라를 40~60cm 눈높이로, 얼굴 정면에서). 다시 보정해 주세요.',
    };
  }

  const calib = {
    features,
    sUp,
    // 자연 깜빡임 수준보다 확실히 높은 값을 깜빡임 게이트로 사용
    blinkGate: clamp(percentile(blinkNeutralSamples, 80) + 0.3, 0.5, 0.8),
    neutralEar: median(samples.center.map((s) => s.ear).filter(Number.isFinite)),
    weakSignal: sUp < 3.0, // 약하면 dwell을 늘리도록 권고
    createdAt: Date.now(),
  };
  return { ok: true, dPrimeUp: sUp, calib };
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
    this.zone = 'neutral'; // up | neutral
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
    this.baseline = 0; // 느린 경로: 최근 시선 위치의 기준선
    this.baselineInit = false;
    this.riseHighSince = 0;
    this.R = 0; // 기준선 대비 상승량 (결정 신호)
  }

  setParams(params) {
    Object.assign(this.params, params);
  }

  loadStoredCalibration() {
    try {
      const raw = localStorage.getItem(CALIB_KEY);
      if (!raw) return false;
      const calib = JSON.parse(raw);
      // 손상되었거나 구버전 형식이면 버린다 (첫 프레임 TypeError 방지)
      const valid =
        calib && typeof calib === 'object' &&
        Number.isFinite(calib.sUp) && calib.sUp > 0 &&
        calib.features && typeof calib.features === 'object' &&
        Object.values(calib.features).some(
          (f) => f && f.weight > 0 &&
            Number.isFinite(f.med) && Number.isFinite(f.sigma) && f.sigma > 0 &&
            (f.sign === 1 || f.sign === -1),
        );
      if (!valid) {
        localStorage.removeItem(CALIB_KEY);
        return false;
      }
      this.calib = calib;
      return true;
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
      video: {
        facingMode: 'user', // 모바일에서 후면 카메라가 잡히지 않도록 전면 지정
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30 },
      },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    const { FaceLandmarker, FilesetResolver } = await import(BUNDLE_URL);
    const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
    const options = (delegate) => ({
      baseOptions: { modelAssetPath: MODEL_URL, delegate },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true,
      // 침대 옆 조명/각도 조건에서도 얼굴을 놓치지 않도록 기본값(0.5)보다 관대하게
      minFaceDetectionConfidence: 0.3,
      minFacePresenceConfidence: 0.3,
      minTrackingConfidence: 0.3,
    });
    try {
      this.landmarker = await FaceLandmarker.createFromOptions(fileset, options('GPU'));
    } catch {
      // 하드웨어 가속이 꺼진 환경(병원 관리 PC 등)에서는 CPU로 대체
      this.landmarker = await FaceLandmarker.createFromOptions(fileset, options('CPU'));
    }

    // 얼굴이 화면에서 옆/거꾸로 보일 때를 위한 입력 회전 (아래 #frameSource 참고)
    this.canvas = document.createElement('canvas');
    this.canvasCtx = this.canvas.getContext('2d', { willReadFrequently: false });
    this.rotation = this.#loadRotation();
    this.rotationSearchIdx = Math.max(0, [0, 90, 270, 180].indexOf(this.rotation));
    this.rotationTriedAt = 0;
    this.lastSavedRotation = this.rotation;
  }

  #loadRotation() {
    try {
      const r = Number(localStorage.getItem(ROTATION_KEY));
      if ([0, 90, 180, 270].includes(r)) return r;
    } catch { /* 무시 */ }
    return 0;
  }

  #saveRotation() {
    try {
      localStorage.setItem(ROTATION_KEY, String(this.rotation));
    } catch { /* 무시 */ }
  }

  // 현재 회전 설정에 맞는 추론 입력을 돌려준다.
  // 감지기는 대략 정자세 얼굴만 찾으므로, 카메라가 옆/거꾸로 거치된 경우
  // 프레임을 회전시켜 넣어야 인식이 된다. 시선 신호(홍채-눈꼬리선 수직거리)는
  // 프레임 회전에 불변이라 보정값은 그대로 유효하다.
  #frameSource() {
    const vw = this.video.videoWidth;
    const vh = this.video.videoHeight;
    if (this.rotation === 0 || vw === 0) {
      return { src: this.video, w: vw, h: vh };
    }
    const [cw, ch] = this.rotation === 180 ? [vw, vh] : [vh, vw];
    if (this.canvas.width !== cw || this.canvas.height !== ch) {
      this.canvas.width = cw;
      this.canvas.height = ch;
    }
    const ctx = this.canvasCtx;
    ctx.save();
    ctx.translate(cw / 2, ch / 2);
    ctx.rotate((this.rotation * Math.PI) / 180);
    ctx.drawImage(this.video, -vw / 2, -vh / 2);
    ctx.restore();
    return { src: this.canvas, w: cw, h: ch };
  }

  // 얼굴을 계속 못 찾으면 다른 회전을 순서대로 시도한다.
  // 옆으로 거치(90/270)가 침대 환경에서 가장 흔하므로 먼저 시도.
  #searchRotation(now) {
    if (now - this.rotationTriedAt < 700) return;
    this.rotationTriedAt = now;
    const candidates = [0, 90, 270, 180];
    this.rotationSearchIdx = (this.rotationSearchIdx + 1) % candidates.length;
    this.rotation = candidates[this.rotationSearchIdx];
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
    return { center: [], up: [], blinkCenter: [] };
  }

  finishCalibration(allSamples) {
    const result = computeCalibration(
      { center: allSamples.center, up: allSamples.up },
      allSamples.blinkCenter,
    );
    if (result.ok) {
      this.calib = result.calib;
      this.saveCalibration();
      if (result.calib.weakSignal && this.params.confirmMs < 250) {
        this.params.confirmMs = 250;
      }
    }
    return result;
  }

  // ===== 프레임 처리 =====
  #processFrame(now) {
    let result;
    const frame = this.#frameSource();
    try {
      result = this.landmarker.detectForVideo(frame.src, now);
      this.detectErrors = 0;
    } catch {
      // 추론 실패(GPU 컨텍스트 소실 등)가 이어지면 얼굴 소실로 취급해
      // 조용히 죽어 있는 대신 사용자에게 보이게 한다
      this.detectErrors = (this.detectErrors ?? 0) + 1;
      if (this.detectErrors > 15) this.#onFaceMissing(now);
      return;
    }
    const landmarks = result?.faceLandmarks?.[0];

    if (!landmarks || landmarks.length < 478) {
      this.#onFaceMissing(now);
      return;
    }
    this.#onFacePresent(now);

    // 프레임 공백(탭 숨김, 절전 등) 뒤에는 벽시계 시간이 점프해 있으므로
    // 진행 중이던 체류를 그대로 완성시키지 않는다
    if (this.lastTs && now - this.lastTs > 500) {
      this.medianBuf = [];
      if (this.state === 'debounce' || this.state === 'dwell' || this.state === 'held') {
        this.#abortDwell('frame-gap');
      }
      this.armed = false;
      this.neutralSince = 0;
    }

    const blendMap = {};
    for (const c of result.faceBlendshapes?.[0]?.categories ?? []) {
      blendMap[c.categoryName] = c.score;
    }
    const feat = extractFeatures(landmarks, blendMap, frame.w, frame.h);

    // 눈 유효성 (모드 + 깜빡임 + EAR)
    const gate = this.calib?.blinkGate ?? 0.5;
    const earFloor = this.calib?.neutralEar ? 0.6 * this.calib.neutralEar : 0;
    const rValid = this.params.eyeMode !== 'left' && feat.blinkR < gate &&
      (earFloor === 0 || feat.earR > earFloor) && Number.isFinite(feat.geoR);
    const lValid = this.params.eyeMode !== 'right' && feat.blinkL < gate &&
      (earFloor === 0 || feat.earL > earFloor) && Number.isFinite(feat.geoL);

    const geoValues = [];
    const geoLowValues = [];
    if (rValid) {
      geoValues.push(feat.geoR);
      if (Number.isFinite(feat.geoLowR)) geoLowValues.push(feat.geoLowR);
    }
    if (lValid) {
      geoValues.push(feat.geoL);
      if (Number.isFinite(feat.geoLowL)) geoLowValues.push(feat.geoLowL);
    }
    const geo = geoValues.length ? geoValues.reduce((a, b) => a + b, 0) / geoValues.length : NaN;
    const geoLow = geoLowValues.length
      ? geoLowValues.reduce((a, b) => a + b, 0) / geoLowValues.length : NaN;
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
        rotation: this.rotation,
      },
    }));

    // 보정 샘플 수집 중이면 여기서 끝
    if (this.collecting) {
      this.#collectFrame(now, { geo, geoLow, ear, blink: blinkMax, blinkNow });
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

    // 이벤트 감지 신호:
    // 빠른 경로(this.S) = 가볍게 스무딩한 현재 시선 위치
    // 느린 경로(this.baseline) = 최근 1~2초의 기준선 (드리프트/자세 변화 흡수)
    // R = S - baseline = "지금 눈동자가 기준선보다 얼마나 위로 움직였나"
    // 절대 위치나 보정 오차와 무관하게 작은 상승도 즉시 잡힌다.
    if (!gatedNow) {
      const rawS = this.#fusedScore({ geo, geoLow });
      this.medianBuf.push(rawS);
      if (this.medianBuf.length > 3) this.medianBuf.shift();
      const med = median(this.medianBuf);
      const dt = this.lastTs ? now - this.lastTs : 33;
      const alpha = 1 - Math.exp(-dt / this.params.emaTauMs);
      this.S = this.S + alpha * (med - this.S);

      if (!this.baselineInit) {
        this.baseline = this.S;
        this.baselineInit = true;
      } else {
        // 위로 움직이는 중(이벤트 진행)에는 기준선을 동결해
        // 이벤트 자체가 기준선에 흡수되지 않게 한다
        const rise = this.S - this.baseline;
        if (this.state === 'idle' && rise < this.params.baselineFreezeZ) {
          const beta = 1 - Math.exp(-dt / this.params.baselineTauMs);
          this.baseline = this.baseline + beta * (this.S - this.baseline);
          this.riseHighSince = 0;
        } else if (this.state === 'idle') {
          // 자세 변화 등으로 기준선보다 높은 상태가 오래 이어지면
          // 아주 느리게 따라가 교착(입력 불능)을 방지한다
          if (this.riseHighSince === 0) this.riseHighSince = now;
          if (now - this.riseHighSince > 2500) {
            const beta = 1 - Math.exp(-dt / 4000);
            this.baseline = this.baseline + beta * (this.S - this.baseline);
          }
        }
      }
      this.R = this.S - this.baseline;
    }
    this.lastTs = now;

    this.#dwellMachine(now, gatedNow);

    this.dispatchEvent(new CustomEvent('gaze', {
      detail: {
        S: this.R, // 게이지에는 기준선 대비 상승량을 표시
        zone: this.zone,
        state: this.state,
        progress: this.#dwellProgress(now),
        gated: gatedNow,
        upEnter: this.#upEnter(),
      },
    }));
  }

  #fusedScore({ geo, geoLow }) {
    let acc = 0;
    let wSum = 0;
    for (const [name, x] of [['geo', geo], ['geoLow', geoLow]]) {
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
    // 가운데 응시 중의 blink 값을 기록해 깜빡임 게이트 산출에 쓴다
    if (col.target === 'center') col.blinkSamples.push(sample.blink);
    if (!sample.blinkNow && Number.isFinite(sample.geo)) {
      col.samples.push(sample);
    }
  }

  #zoneOf(S) {
    return S >= this.#upEnter() ? 'up' : 'neutral';
  }

  #inZoneSustain(S, zone) {
    // 이탈 임계값 기준 (히스테리시스)
    return zone === 'up' && S >= this.#upExit();
  }

  #dwellProgress(now) {
    if (this.state !== 'dwell') return 0;
    const elapsed = now - this.dwellStart - this.dwellPausedTotal -
      (this.dwellPausedAt ? now - this.dwellPausedAt : 0);
    return clamp(elapsed / this.params.confirmMs, 0, 1);
  }

  // 이벤트 임계값(기준선 대비 상승량 기준)은 민감도 설정 × sUp 에서 매번 계산한다.
  // 슬라이더 조절이 재보정 없이 즉시 반영된다.
  #upEnter() {
    const s = this.calib.sUp;
    // 하한 1.5σ: 기준선 대비 측정이라 절대 위치보다 잡음이 작으므로 낮게 잡아도 안전.
    // 상한 0.8·sUp: 약한 신호도 도달 가능하게.
    return Math.min(Math.max(this.params.upSensitivity * s, 1.5), 0.8 * s);
  }

  #upExit() {
    return 0.5 * this.#upEnter(); // 슈미트 트리거 (진입/이탈 분리)
  }

  // 재장전 구역(기준선 근처)은 이탈 임계값보다 확실히 안쪽에
  #neutralBand() {
    if (!this.calib) return this.params.neutralBand;
    return Math.min(1.0, 0.7 * this.#upExit());
  }

  #dwellMachine(now, gated) {
    const p = this.params;
    const S = this.R; // 결정 신호 = 기준선 대비 상승량

    // 얼굴 소실 시 리셋은 #onFaceMissing에서 처리
    switch (this.state) {
      case 'idle': {
        if (Math.abs(S) < this.#neutralBand()) {
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
        // 선택 확정 후에도 계속 응시하면 취소(retract).
        // 확정 시점(stateSince)부터 재므로 dwell 길이/일시정지와 무관하게 일정하다.
        const heldFor = now - this.stateSince;
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
    // 얼굴이 계속 안 잡히면 카메라가 옆/거꾸로 거치됐을 수 있으므로
    // 입력 회전을 바꿔 가며 탐색한다
    if (this.faceLostSince !== 0 && now - this.faceLostSince > 700) {
      this.#searchRotation(now);
    }
    // 보정 수집 중 얼굴이 안 잡혀도 시간이 다 되면 수집을 끝낸다
    if (this.collecting && now >= this.collecting.endAt) {
      const done = this.collecting;
      this.collecting = null;
      done.resolve({ target: done.target, samples: done.samples, blinkSamples: done.blinkSamples });
    }
    this.faceStableSince = 0;
    if (this.faceLostSince === 0) this.faceLostSince = now;
    // 짧은 추적 끊김 동안 체류 시간이 그대로 쌓이지 않도록
    // 깜빡임과 같은 방식으로 일시정지/취소 처리한다
    if (this.state === 'debounce') {
      this.state = 'idle';
      this.zone = 'neutral';
    } else if (this.state === 'dwell') {
      if (this.dwellPausedAt === 0) this.dwellPausedAt = now;
      if (now - this.dwellPausedAt > this.params.blinkAbortMs) this.#abortDwell('face-gap');
    }
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
    // 회전 탐색 중이었다면 현재 회전으로 고정하고 기억한다
    if (this.rotation !== this.lastSavedRotation) {
      this.lastSavedRotation = this.rotation;
      this.rotationSearchIdx = [0, 90, 270, 180].indexOf(this.rotation);
      this.#saveRotation();
      if (this.rotation !== 0) {
        this.dispatchEvent(new CustomEvent('rotationlock', { detail: { rotation: this.rotation } }));
      }
    }
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

// 카메라 없이 테스트하는 대체 입력기 (같은 이벤트 인터페이스)
// 키보드: Space/Enter/↑ = 위 응시(선택), Backspace = 되돌리기, P = 쉬기
// 터치(모바일): 짧게 탭 = 선택, 길게 누르기(0.6초+) = 되돌리기
export class KeyboardTracker extends EventTarget {
  constructor() {
    super();
    this.running = false;
    this.pointerDownAt = 0;

    const interactive = (target) =>
      target.closest?.('#settings, button, input, select, textarea, a') ||
      ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON', 'A'].includes(target.tagName);

    this.handler = (e) => {
      if (!this.running) return;
      // 설정 패널의 슬라이더/입력을 조작할 때는 가로채지 않는다
      if (interactive(e.target)) return;
      if (e.key === 'ArrowUp' || e.key === ' ' || e.key === 'Enter') {
        e.preventDefault();
        this.dispatchEvent(new CustomEvent('answer', { detail: { dir: 'up' } }));
      } else if (e.key === 'Backspace') {
        e.preventDefault();
        this.dispatchEvent(new CustomEvent('retract', { detail: { dir: 'up' } }));
      } else if (e.key === 'p' || e.key === 'P') {
        this.dispatchEvent(new CustomEvent('pausegesture'));
      }
    };

    this.pointerDown = (e) => {
      if (!this.running || interactive(e.target)) return;
      this.pointerDownAt = performance.now();
    };
    this.pointerUp = (e) => {
      if (!this.running || interactive(e.target) || this.pointerDownAt === 0) return;
      const heldMs = performance.now() - this.pointerDownAt;
      this.pointerDownAt = 0;
      if (heldMs >= 600) {
        this.dispatchEvent(new CustomEvent('retract', { detail: { dir: 'up' } }));
      } else {
        this.dispatchEvent(new CustomEvent('answer', { detail: { dir: 'up' } }));
      }
    };
  }

  start() {
    this.running = true;
    window.addEventListener('keydown', this.handler);
    window.addEventListener('pointerdown', this.pointerDown);
    window.addEventListener('pointerup', this.pointerUp);
  }

  stop() {
    this.running = false;
    window.removeEventListener('keydown', this.handler);
    window.removeEventListener('pointerdown', this.pointerDown);
    window.removeEventListener('pointerup', this.pointerUp);
  }
}

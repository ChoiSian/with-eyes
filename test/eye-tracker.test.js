import test from 'node:test';
import assert from 'node:assert/strict';
import { computeCalibration, extractFeatures, DEFAULT_PARAMS } from '../js/eye-tracker.js';

// 가짜 보정 샘플 생성: 평균 mu, 잡음 폭 noise
function makeSamples(n, { geo, blend }, noise = 0.002) {
  const out = [];
  for (let i = 0; i < n; i++) {
    // 결정적 의사 잡음 (테스트 재현성)
    const j = Math.sin(i * 12.9898) * 43758.5453;
    const r = (j - Math.floor(j)) - 0.5;
    out.push({ geo: geo + r * noise, blend: blend + r * 0.05, ear: 0.3 });
  }
  return out;
}

test('computeCalibration: 정상 신호에서 보정 성공', () => {
  const samples = {
    center: makeSamples(60, { geo: 0.0, blend: 0.0 }),
    up: makeSamples(60, { geo: 0.05, blend: 0.5 }),
    down: makeSamples(60, { geo: -0.05, blend: -0.5 }),
  };
  const result = computeCalibration(samples, [0.3, 0.4, 0.35]);
  assert.equal(result.ok, true);
  assert.ok(result.calib.upEnter > 0);
  assert.ok(result.calib.downEnter < 0);
  assert.ok(result.calib.upExit < result.calib.upEnter);
  assert.ok(result.calib.downExit > result.calib.downEnter);
  assert.ok(result.calib.blinkGate >= 0.5 && result.calib.blinkGate <= 0.8);
});

test('computeCalibration: 분리도가 낮으면 실패와 안내 메시지', () => {
  const samples = {
    center: makeSamples(60, { geo: 0.0, blend: 0.0 }, 0.05),
    up: makeSamples(60, { geo: 0.005, blend: 0.02 }, 0.05),
    down: makeSamples(60, { geo: -0.005, blend: -0.02 }, 0.05),
  };
  const result = computeCalibration(samples, [0.3]);
  assert.equal(result.ok, false);
  assert.ok(result.message.length > 0);
});

test('computeCalibration: 샘플 부족 시 실패', () => {
  const samples = {
    center: makeSamples(5, { geo: 0, blend: 0 }),
    up: makeSamples(60, { geo: 0.05, blend: 0.5 }),
    down: makeSamples(60, { geo: -0.05, blend: -0.5 }),
  };
  const result = computeCalibration(samples, [0.3]);
  assert.equal(result.ok, false);
});

test('computeCalibration: 한 특징이 고장나도 다른 특징으로 성공', () => {
  const samples = {
    center: makeSamples(60, { geo: 0.0, blend: 0.0 }),
    // blend가 방향을 구분하지 못함 (위/아래 모두 양수)
    up: makeSamples(60, { geo: 0.05, blend: 0.3 }),
    down: makeSamples(60, { geo: -0.05, blend: 0.3 }),
  };
  const result = computeCalibration(samples, [0.3]);
  assert.equal(result.ok, true);
  assert.equal(result.calib.features.blend.weight, 0);
  assert.ok(result.calib.features.geo.weight > 0);
});

test('extractFeatures: 홍채가 위로 가면 geo 신호가 양수', () => {
  // 눈꼬리 수평선 y=0.5, 홍채가 그 위(y 작음)에 있는 간단한 얼굴 모형
  const landmarks = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5 }));
  // 오른눈: 바깥 33 (0.3), 안쪽 133 (0.4), 홍채 468
  landmarks[33] = { x: 0.3, y: 0.5 };
  landmarks[133] = { x: 0.4, y: 0.5 };
  landmarks[468] = { x: 0.35, y: 0.48 }; // 위를 봄
  landmarks[159] = { x: 0.35, y: 0.47 };
  landmarks[145] = { x: 0.35, y: 0.53 };
  // 왼눈: 안쪽 362 (0.6), 바깥 263 (0.7), 홍채 473
  landmarks[362] = { x: 0.6, y: 0.5 };
  landmarks[263] = { x: 0.7, y: 0.5 };
  landmarks[473] = { x: 0.65, y: 0.48 };
  landmarks[386] = { x: 0.65, y: 0.47 };
  landmarks[374] = { x: 0.65, y: 0.53 };

  const feat = extractFeatures(landmarks, {}, 1000, 1000);
  assert.ok(feat.geoR > 0, `geoR ${feat.geoR} > 0`);
  assert.ok(feat.geoL > 0, `geoL ${feat.geoL} > 0`);

  // 아래를 보면 음수
  landmarks[468] = { x: 0.35, y: 0.52 };
  landmarks[473] = { x: 0.65, y: 0.52 };
  const feat2 = extractFeatures(landmarks, {}, 1000, 1000);
  assert.ok(feat2.geoR < 0);
  assert.ok(feat2.geoL < 0);
});

test('extractFeatures: 머리 기울기(roll)에 불변', () => {
  // 같은 상대 위치를 15도 회전시켜도 geo 부호와 크기가 유지되어야 함
  const rot = (p, deg) => {
    const rad = (deg * Math.PI) / 180;
    const cx = 0.5, cy = 0.5;
    const dx = p.x - cx, dy = p.y - cy;
    return {
      x: cx + dx * Math.cos(rad) - dy * Math.sin(rad),
      y: cy + dx * Math.sin(rad) + dy * Math.cos(rad),
    };
  };
  const base = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5 }));
  base[33] = { x: 0.3, y: 0.5 };
  base[133] = { x: 0.4, y: 0.5 };
  base[468] = { x: 0.35, y: 0.48 };
  base[159] = { x: 0.35, y: 0.47 };
  base[145] = { x: 0.35, y: 0.53 };
  base[362] = { x: 0.6, y: 0.5 };
  base[263] = { x: 0.7, y: 0.5 };
  base[473] = { x: 0.65, y: 0.48 };
  base[386] = { x: 0.65, y: 0.47 };
  base[374] = { x: 0.65, y: 0.53 };

  const flat = extractFeatures(base, {}, 1000, 1000);
  const rotated = base.map((p) => rot(p, 15));
  const tilted = extractFeatures(rotated, {}, 1000, 1000);
  assert.ok(Math.abs(flat.geoR - tilted.geoR) < 0.005, `roll 불변: ${flat.geoR} vs ${tilted.geoR}`);
  assert.ok(Math.abs(tilted.rollDeg - 15) < 1.5);
});

test('DEFAULT_PARAMS: 안전 관련 상수 확인', () => {
  // 벨 현상 방지: 깜빡임 직후 무시 시간과 디바운스는 안전-필수 값
  assert.ok(DEFAULT_PARAMS.postBlinkHoldMs >= 150);
  assert.ok(DEFAULT_PARAMS.debounceMs >= 100);
  assert.ok(DEFAULT_PARAMS.dwellMs >= 600);
});

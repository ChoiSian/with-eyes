import test from 'node:test';
import assert from 'node:assert/strict';
import { computeCalibration, extractFeatures, DEFAULT_PARAMS } from '../js/eye-tracker.js';

// 가짜 보정 샘플 생성: 평균 mu, 잡음 폭 noise
function makeSamples(n, { geo, geoLow }, noise = 0.002) {
  const out = [];
  for (let i = 0; i < n; i++) {
    // 결정적 의사 잡음 (테스트 재현성)
    const j = Math.sin(i * 12.9898) * 43758.5453;
    const r = (j - Math.floor(j)) - 0.5;
    out.push({ geo: geo + r * noise, geoLow: geoLow + r * noise * 1.2, ear: 0.3 });
  }
  return out;
}

test('computeCalibration: 정상 신호에서 보정 성공 (가운데/위 2지점)', () => {
  const samples = {
    center: makeSamples(60, { geo: 0.0, geoLow: 0.12 }),
    up: makeSamples(60, { geo: 0.05, geoLow: 0.2 }),
  };
  const result = computeCalibration(samples, [0.05, 0.1, 0.08]);
  assert.equal(result.ok, true);
  assert.ok(result.calib.sUp >= 2.0);
  assert.ok(result.calib.blinkGate >= 0.5 && result.calib.blinkGate <= 0.8);
  assert.ok(result.calib.features.geo.weight > 0);
  assert.ok(result.calib.features.geoLow.weight > 0);
  assert.ok(Number.isFinite(result.calib.neutralEar));
});

test('computeCalibration: 분리도가 낮으면 실패와 안내 메시지', () => {
  const samples = {
    center: makeSamples(60, { geo: 0.0, geoLow: 0.12 }, 0.05),
    up: makeSamples(60, { geo: 0.005, geoLow: 0.125 }, 0.05),
  };
  const result = computeCalibration(samples, [0.05]);
  assert.equal(result.ok, false);
  assert.ok(result.message.length > 0);
});

test('computeCalibration: 샘플 부족 시 실패', () => {
  const samples = {
    center: makeSamples(5, { geo: 0, geoLow: 0.1 }),
    up: makeSamples(60, { geo: 0.05, geoLow: 0.2 }),
  };
  const result = computeCalibration(samples, [0.05]);
  assert.equal(result.ok, false);
});

test('computeCalibration: 한 특징이 고장나도 다른 특징으로 성공', () => {
  const samples = {
    center: makeSamples(60, { geo: 0.0, geoLow: 0.12 }),
    // geoLow가 위 응시를 전혀 구분하지 못함 (span 0)
    up: makeSamples(60, { geo: 0.05, geoLow: 0.12 }),
  };
  const result = computeCalibration(samples, [0.05]);
  assert.equal(result.ok, true);
  assert.ok(result.calib.features.geo.weight > 0);
  // geoLow는 가중치가 매우 작아야 함 (span이 잡음 수준)
  assert.ok(result.calib.features.geo.weight > result.calib.features.geoLow.weight * 100);
});

test('computeCalibration: 깜빡임 게이트는 자연 깜빡임 수준보다 높다', () => {
  const samples = {
    center: makeSamples(60, { geo: 0.0, geoLow: 0.12 }),
    up: makeSamples(60, { geo: 0.05, geoLow: 0.2 }),
  };
  const result = computeCalibration(samples, [0.02, 0.03, 0.1, 0.05]);
  assert.ok(result.calib.blinkGate >= 0.4);
});

function faceModel() {
  const landmarks = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5 }));
  // 오른눈: 바깥 33 (0.3), 안쪽 133 (0.4), 홍채 468, 윗꺼풀 159, 아랫꺼풀 145
  landmarks[33] = { x: 0.3, y: 0.5 };
  landmarks[133] = { x: 0.4, y: 0.5 };
  landmarks[468] = { x: 0.35, y: 0.5 };
  landmarks[159] = { x: 0.35, y: 0.47 };
  landmarks[145] = { x: 0.35, y: 0.53 };
  // 왼눈: 안쪽 362 (0.6), 바깥 263 (0.7), 홍채 473, 윗꺼풀 386, 아랫꺼풀 374
  landmarks[362] = { x: 0.6, y: 0.5 };
  landmarks[263] = { x: 0.7, y: 0.5 };
  landmarks[473] = { x: 0.65, y: 0.5 };
  landmarks[386] = { x: 0.65, y: 0.47 };
  landmarks[374] = { x: 0.65, y: 0.53 };
  return landmarks;
}

test('extractFeatures: 홍채(눈동자)가 위로 가면 geo/geoLow 모두 증가', () => {
  const landmarks = faceModel();
  const neutral = extractFeatures(landmarks, {}, 1000, 1000);

  landmarks[468] = { x: 0.35, y: 0.48 }; // 위를 봄
  landmarks[473] = { x: 0.65, y: 0.48 };
  const up = extractFeatures(landmarks, {}, 1000, 1000);

  assert.ok(up.geoR > neutral.geoR, `geoR 증가: ${neutral.geoR} -> ${up.geoR}`);
  assert.ok(up.geoL > neutral.geoL);
  assert.ok(up.geoLowR > neutral.geoLowR, `geoLowR 증가: ${neutral.geoLowR} -> ${up.geoLowR}`);
  assert.ok(up.geoLowL > neutral.geoLowL);
  // geoLow = 홍채가 아래 눈꺼풀에서 떨어진 높이이므로 항상 양수여야 정상
  assert.ok(up.geoLowR > 0 && neutral.geoLowR > 0);
});

test('extractFeatures: 머리 기울기(roll)에 불변', () => {
  const rot = (p, deg) => {
    const rad = (deg * Math.PI) / 180;
    const cx = 0.5, cy = 0.5;
    const dx = p.x - cx, dy = p.y - cy;
    return {
      x: cx + dx * Math.cos(rad) - dy * Math.sin(rad),
      y: cy + dx * Math.sin(rad) + dy * Math.cos(rad),
    };
  };
  const base = faceModel();
  base[468] = { x: 0.35, y: 0.48 };
  base[473] = { x: 0.65, y: 0.48 };

  const flat = extractFeatures(base, {}, 1000, 1000);
  const rotated = base.map((p) => rot(p, 15));
  const tilted = extractFeatures(rotated, {}, 1000, 1000);
  assert.ok(Math.abs(flat.geoR - tilted.geoR) < 0.005, `geo roll 불변: ${flat.geoR} vs ${tilted.geoR}`);
  assert.ok(Math.abs(flat.geoLowR - tilted.geoLowR) < 0.005, 'geoLow roll 불변');
  assert.ok(Math.abs(tilted.rollDeg - 15) < 1.5);
});

test('DEFAULT_PARAMS: 안전 관련 상수 확인', () => {
  // 벨 현상 방지: 깜빡임 직후 무시 시간과 디바운스는 안전-필수 값
  assert.ok(DEFAULT_PARAMS.postBlinkHoldMs >= 150);
  assert.ok(DEFAULT_PARAMS.debounceMs >= 100);
  assert.ok(DEFAULT_PARAMS.dwellMs >= 500);
  // 민감도 기본값은 하한(잡음)과 상한(도달 가능성) 사이
  assert.ok(DEFAULT_PARAMS.upSensitivity >= 0.3 && DEFAULT_PARAMS.upSensitivity <= 0.7);
});

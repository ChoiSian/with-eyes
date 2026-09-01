import test from 'node:test';
import assert from 'node:assert/strict';
import { Predictor, hasBatchim, particleForm } from '../js/predictor.js';
import { PARTICLES } from '../data/dictionary.js';

const DICT = [
  { w: '물', f: 95, pos: 'n', next: ['주세요'] },
  { w: '물티슈', f: 44, pos: 'n' },
  { w: '무릎', f: 58, pos: 'n' },
  { w: '주세요', f: 96 },
  { w: '화장실', f: 92, pos: 'n', next: ['가고'] },
  { w: '가고', f: 58, next: ['싶어요'] },
  { w: '싶어요', f: 92 },
  { w: '배고파요', f: 78 },
  { w: '배', f: 78, pos: 'n', next: ['아파요'] },
  { w: '아파요', f: 96 },
];

const newPredictor = () => new Predictor(DICT, { persist: false });

test('자모 접두사로 완성 후보 검색', () => {
  const p = newPredictor();
  const words = p.completions('ㅁ').map((c) => c.word);
  assert.ok(words.includes('물'));
  assert.ok(words.includes('무릎'));
});

test('조합 중인 음절 상태에서도 매칭 (부분 자모)', () => {
  const p = newPredictor();
  // '무' 까지 입력: ㅁ+ㅜ
  const words = p.completions('무').map((c) => c.word);
  assert.ok(words.includes('물'));
  assert.ok(words.includes('무릎'));
  // '물' 까지 입력하면 받침 ㄹ까지 매칭: 물티슈도 나와야 함
  const words2 = p.completions('물').map((c) => c.word);
  assert.ok(words2.includes('물티슈'));
});

test('받침/다음 초성 중의성: 배 -> 배고파요 와 배 모두 접근 가능', () => {
  const p = newPredictor();
  const words = p.completions('배').map((c) => c.word);
  assert.ok(words.includes('배고파요'));
});

test('빈도 높은 단어가 먼저', () => {
  const p = newPredictor();
  const words = p.completions('ㅁ').map((c) => c.word);
  assert.equal(words[0], '물'); // f=95 > 무릎 58 > 물티슈 44
});

test('다음 단어: 큐레이션 바이그램', () => {
  const p = newPredictor();
  const words = p.nextWords('화장실').map((c) => c.word);
  assert.ok(words.includes('가고'));
});

test('다음 단어: 명사 뒤 조사 (받침 규칙)', () => {
  const p = newPredictor();
  const sugg = p.nextWords('물', 8);
  const particles = sugg.filter((s) => s.particle).map((s) => s.word);
  assert.ok(particles.includes('이')); // 물 = 받침 있음 -> 이
  assert.ok(particles.includes('을'));
  assert.ok(!particles.includes('가'));
});

test('사용자 학습: 발화한 단어와 바이그램이 상위로', () => {
  const p = newPredictor();
  p.recordSentence('물 주세요');
  p.recordSentence('물 주세요');
  const next = p.nextWords('물').map((c) => c.word);
  assert.equal(next[0], '주세요');
  // 사전에 없는 새 단어 학습
  p.recordSentence('갈비탕 주세요');
  const comp = p.completions('갈').map((c) => c.word);
  assert.ok(comp.includes('갈비탕'));
});

test('suggest: 입력 중이면 완성, 아니면 다음 단어', () => {
  const p = newPredictor();
  const inWord = p.suggest({ currentWord: 'ㅁ', prevWord: null });
  assert.ok(inWord.length > 0 && inWord.every((s) => !s.particle));
  const between = p.suggest({ currentWord: '', prevWord: '화장실' });
  assert.ok(between.map((s) => s.word).includes('가고'));
});

test('hasBatchim / particleForm', () => {
  assert.equal(hasBatchim('물'), true);
  assert.equal(hasBatchim('배'), false);
  assert.equal(hasBatchim('ㄱ'), false); // 완성 음절이 아니면 false
  const objParticle = PARTICLES.find((p) => p.name === '을/를');
  assert.equal(particleForm(objParticle, '물'), '을');
  assert.equal(particleForm(objParticle, '배'), '를');
  const roParticle = PARTICLES.find((p) => p.name === '(으)로');
  assert.equal(particleForm(roParticle, '집'), '으로');
  assert.equal(particleForm(roParticle, '거실'), '로'); // ㄹ 받침 예외
  assert.equal(particleForm(roParticle, '위'), '로'); // 받침 없음
});

test('조사와 동형어 단어가 충돌하지 않는다', () => {
  const p = new Predictor([
    { w: '이', f: 48, pos: 'n' }, // 치아
    { w: '물', f: 95, pos: 'n' },
  ], { persist: false });
  const sugg = p.nextWords('물', 10);
  const forms = sugg.filter((s) => s.word === '이');
  // 조사 '이'(붙여쓰기)와 단어 '이'(띄어쓰기)가 별개 후보로 공존 가능
  assert.ok(forms.some((s) => s.particle === true));
});

test('importData: 같은 백업 두 번 가져와도 수치가 불어나지 않는다', () => {
  const p = newPredictor();
  p.recordSentence('갈비탕 주세요');
  const dump = p.exportData();
  const p2 = newPredictor();
  p2.importData(dump);
  p2.importData(dump); // 재가져오기
  const once = JSON.parse(p2.exportData());
  assert.equal(once.lex['갈비탕'][0], 1);
});

test('importData: 손상된 값은 조용히 건너뛴다', () => {
  const p = newPredictor();
  p.importData(JSON.stringify({
    lex: { '물': 5, '나쁨': ['a', 'b'], '좋음': [2, 1000] },
    bigrams: { '아': 'x' },
    __proto__: { hacked: true },
  }));
  // 유효한 항목만 반영, NaN 오염 없음
  const sugg = p.completions('좋');
  assert.ok(sugg.every((s) => Number.isFinite(s.score)));
  assert.equal(Object.prototype.hacked, undefined);
});

test('export / import 왕복', () => {
  const p = newPredictor();
  p.recordSentence('갈비탕 먹고 싶어요');
  const dump = p.exportData();
  const p2 = newPredictor();
  p2.importData(dump);
  assert.ok(p2.completions('갈').map((c) => c.word).includes('갈비탕'));
  assert.ok(p2.nextWords('갈비탕').map((c) => c.word).includes('먹고'));
});

test('NFD 입력도 NFC로 정규화되어 매칭', () => {
  const p = newPredictor();
  const nfd = '물'.normalize('NFD');
  assert.notEqual(nfd, '물'); // 실제로 다른 코드포인트인지 확인
  const words = p.completions(nfd).map((c) => c.word);
  assert.ok(words.includes('물티슈'));
});

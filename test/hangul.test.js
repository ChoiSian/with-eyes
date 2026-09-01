import test from 'node:test';
import assert from 'node:assert/strict';
import {
  HangulComposer,
  composeSyllable,
  decomposeSyllable,
  toJamoSeq,
} from '../js/hangul.js';

function type(composer, jamos) {
  for (const j of jamos) composer.input(j);
  return composer;
}

test('composeSyllable / decomposeSyllable 기본', () => {
  assert.equal(composeSyllable('ㄱ', 'ㅏ'), '가');
  assert.equal(composeSyllable('ㅎ', 'ㅏ', 'ㄴ'), '한');
  assert.equal(composeSyllable('ㄲ', 'ㅜ', 'ㅁ'), '꿈');
  assert.deepEqual(decomposeSyllable('값'), { cho: 'ㄱ', jung: 'ㅏ', jong: 'ㅄ' });
  assert.deepEqual(decomposeSyllable('가'), { cho: 'ㄱ', jung: 'ㅏ', jong: '' });
  assert.equal(decomposeSyllable('a'), null);
});

test('기본 음절 조합', () => {
  const c = type(new HangulComposer(), [...'ㅇㅏㄴㄴㅕㅇ']);
  assert.equal(c.value, '안녕');
});

test('겹모음 조합', () => {
  const c = type(new HangulComposer(), ['ㄱ', 'ㅗ', 'ㅏ']);
  assert.equal(c.value, '과');
  const c2 = type(new HangulComposer(), ['ㅇ', 'ㅜ', 'ㅣ', 'ㅅ', 'ㅏ']);
  assert.equal(c2.value, '위사');
});

test('겹받침 조합과 연음', () => {
  const c = type(new HangulComposer(), [...'ㄱㅏㅂㅅ']);
  assert.equal(c.value, '값');
  c.input('ㅏ');
  assert.equal(c.value, '갑사'); // 값 + ㅏ -> 갑사
});

test('받침 뒤 ㅇ이 오면 ㅇ이 새 초성이 된다', () => {
  const c = type(new HangulComposer(), [...'ㅁㅏㄴㅇㅏ']);
  assert.equal(c.value, '만아'); // 2벌식: ㄴ은 받침으로 남고 ㅇ이 새 음절 초성
});

test('받침 뒤 모음이 오면 받침이 초성으로 이동', () => {
  const c = type(new HangulComposer(), [...'ㄱㅏㄴㅏ']);
  assert.equal(c.value, '가나');
});

test('ㄸㅃㅉ 는 받침이 될 수 없다', () => {
  const c = type(new HangulComposer(), ['ㅂ', 'ㅏ', 'ㄸ', 'ㅏ']);
  assert.equal(c.value, '바따');
});

test('모음 단독 입력', () => {
  const c = type(new HangulComposer(), ['ㅏ', 'ㅗ']);
  assert.equal(c.value, 'ㅏㅗ');
  const c2 = type(new HangulComposer(), ['ㅗ', 'ㅏ']);
  assert.equal(c2.value, 'ㅘ'); // 단독 모음도 겹모음 조합
});

test('공백과 문장부호', () => {
  const c = type(new HangulComposer(), [...'ㄴㅔ', ' ', ...'ㅁㅏㅈㅇㅏㅇㅛ']);
  assert.equal(c.value, '네 맞아요');
});

test('백스페이스: 자모 단위', () => {
  const c = type(new HangulComposer(), [...'ㄱㅏㅂㅅ']);
  assert.equal(c.value, '값');
  c.backspace();
  assert.equal(c.value, '갑'); // 겹받침에서 마지막 요소만 제거
  c.backspace();
  assert.equal(c.value, '가');
  c.backspace();
  assert.equal(c.value, 'ㄱ');
  c.backspace();
  assert.equal(c.value, '');
  assert.equal(c.backspace(), false);
});

test('백스페이스: 확정된 글자로 되돌아가기', () => {
  const c = type(new HangulComposer(), [...'ㄱㅏㄴㅏ']);
  assert.equal(c.value, '가나');
  c.backspace();
  assert.equal(c.value, '가ㄴ');
  c.backspace();
  assert.equal(c.value, '가');
  c.backspace();
  assert.equal(c.value, 'ㄱ');
});

test('백스페이스: 두 번에 조합한 겹모음은 요소 단위로', () => {
  const c = type(new HangulComposer(), ['ㄱ', 'ㅗ', 'ㅏ']);
  assert.equal(c.value, '과');
  c.backspace();
  assert.equal(c.value, '고');
});

test('백스페이스: 한 번에 선택한 겹모음은 통째로 삭제', () => {
  // UI에서 ㅘ를 하나의 항목으로 직접 선택한 경우: 선택 1회 = 백스페이스 1회
  const c = type(new HangulComposer(), ['ㄱ', 'ㅘ']);
  assert.equal(c.value, '과');
  c.backspace();
  assert.equal(c.value, 'ㄱ');
  const c2 = type(new HangulComposer(), ['ㅢ']);
  assert.equal(c2.value, 'ㅢ');
  c2.backspace();
  assert.equal(c2.value, '');
});

test('백스페이스: 확정 글자로 되돌아가면 자모 단위', () => {
  // 확정된 '과'는 어떻게 입력했는지 알 수 없으므로 자모 단위로 지운다
  const c = type(new HangulComposer(), ['ㄱ', 'ㅘ', ' ']);
  assert.equal(c.value, '과 ');
  c.backspace(); // 공백
  c.backspace(); // ㅏ
  assert.equal(c.value, '고');
});

test('백스페이스: 공백은 통째로', () => {
  const c = type(new HangulComposer(), [...'ㄴㅔ', ' ']);
  c.backspace();
  assert.equal(c.value, '네');
  c.backspace();
  assert.equal(c.value, 'ㄴ');
});

test('toJamoSeq: 겹모음/겹받침 분해', () => {
  assert.deepEqual(toJamoSeq('과'), ['ㄱ', 'ㅗ', 'ㅏ']);
  assert.deepEqual(toJamoSeq('값'), ['ㄱ', 'ㅏ', 'ㅂ', 'ㅅ']);
  assert.deepEqual(toJamoSeq('꿈'), ['ㄲ', 'ㅜ', 'ㅁ']); // 쌍자음은 유지
  assert.deepEqual(toJamoSeq('안녕'), ['ㅇ', 'ㅏ', 'ㄴ', 'ㄴ', 'ㅕ', 'ㅇ']);
  assert.deepEqual(toJamoSeq('물 좀'), ['ㅁ', 'ㅜ', 'ㄹ', ' ', 'ㅈ', 'ㅗ', 'ㅁ']);
});

test('setText 후 이어서 입력', () => {
  const c = new HangulComposer();
  c.setText('물');
  c.input('ㅇ');
  c.input('ㅡ');
  c.input('ㄹ');
  assert.equal(c.value, '물을');
});

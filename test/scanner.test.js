import test from 'node:test';
import assert from 'node:assert/strict';
import { buildTree, SelectionCycle, cycleFromList } from '../js/scanner.js';

const items = (defs) => defs.map(([id, weight]) => ({ id, label: id, weight }));

test('buildTree: 단일 항목은 leaf', () => {
  const tree = buildTree(items([['a', 1]]));
  assert.equal(tree.id, 'a');
});

test('buildTree: 가중치 균형 분할', () => {
  // a=10 vs b+c+d=9 로 분할되어야 함
  const tree = buildTree(items([['a', 10], ['b', 4], ['c', 3], ['d', 2]]));
  assert.equal(tree.top.id, 'a');
  assert.deepEqual(tree.bottom.items.map((i) => i.id), ['b', 'c', 'd']);
});

test('고빈도 항목은 적은 질문으로 도달', () => {
  const list = items([['ㅇ', 17], ['ㄱ', 12], ['ㅅ', 9], ['ㄴ', 8], ['ㄷ', 7.5],
    ['ㅈ', 7.5], ['ㄹ', 6.5], ['ㅎ', 6.5], ['ㅁ', 5.5], ['ㅂ', 5],
    ['ㅊ', 3.5], ['ㅌ', 2], ['ㅍ', 1.6], ['ㅋ', 1.2], ['ㄲ', 1.0],
    ['ㅆ', 0.7], ['ㄸ', 0.6], ['ㅉ', 0.4], ['ㅃ', 0.3]]);
  const depthOf = (tree, id, d = 0) => {
    if (tree.items === undefined) return tree.id === id ? d : null;
    return depthOf(tree.top, id, d + 1) ?? depthOf(tree.bottom, id, d + 1);
  };
  const tree = buildTree(list);
  const dFreq = depthOf(tree, 'ㅇ');
  const dRare = depthOf(tree, 'ㅃ');
  assert.ok(dFreq < dRare, `고빈도 ${dFreq} < 저빈도 ${dRare} 이어야 함`);
  assert.ok(dFreq <= 3);
});

test('SelectionCycle: 응답으로 내려가 항목 확정', () => {
  const cycle = new SelectionCycle({
    top: items([['cmd1', 5], ['cmd2', 3]]),
    bottom: items([['ㄱ', 10], ['ㄴ', 8], ['ㄷ', 5]]),
  });
  // 아래 밴드(자모판) 선택
  let r = cycle.answer('down');
  assert.equal(r.done, false);
  const bands = cycle.bands;
  assert.ok(bands.top.length >= 1 && bands.bottom.length >= 1);
  // 계속 위만 선택하면 언젠가 확정
  let guard = 10;
  while (guard-- > 0) {
    r = cycle.answer('up');
    if (r.done) break;
  }
  assert.equal(r.done, true);
  assert.ok(['ㄱ', 'ㄴ', 'ㄷ'].includes(r.item.id));
});

test('SelectionCycle: back으로 되돌리기', () => {
  const cycle = new SelectionCycle({
    top: items([['a', 1], ['b', 1]]),
    bottom: items([['c', 1], ['d', 1]]),
  });
  cycle.answer('down');
  assert.equal(cycle.depth, 1);
  assert.equal(cycle.back(), true);
  assert.equal(cycle.depth, 0);
  assert.equal(cycle.back(), false); // 루트에서는 불가
});

test('SelectionCycle: reset', () => {
  const cycle = new SelectionCycle({
    top: items([['a', 1], ['b', 1]]),
    bottom: items([['c', 1], ['d', 1]]),
  });
  cycle.answer('up');
  cycle.reset();
  assert.equal(cycle.depth, 0);
  assert.deepEqual(cycle.bands.top.map((i) => i.id), ['a', 'b']);
});

test('cycleFromList: 목록에서 사이클 생성', () => {
  const cycle = cycleFromList(items([['x', 5], ['y', 3], ['z', 2]]));
  const r1 = cycle.answer('up');
  assert.equal(r1.done, true);
  assert.equal(r1.item.id, 'x'); // 고빈도가 위 밴드 단독
});

test('빈 밴드에 응답해도 죽지 않는다', () => {
  const cycle = new SelectionCycle({ top: items([['only', 1]]), bottom: [] });
  const r = cycle.answer('down');
  assert.equal(r.done, false);
  const r2 = cycle.answer('up');
  assert.equal(r2.done, true);
  assert.equal(r2.item.id, 'only');
});

test('모든 항목이 도달 가능하다', () => {
  const list = items([['a', 9], ['b', 5], ['c', 4], ['d', 2], ['e', 1], ['f', 1]]);
  const reach = (tree, acc) => {
    if (tree === null) return;
    if (tree.items === undefined) {
      acc.push(tree.id);
      return;
    }
    reach(tree.top, acc);
    reach(tree.bottom, acc);
  };
  const acc = [];
  reach(buildTree(list), acc);
  assert.deepEqual([...acc].sort(), ['a', 'b', 'c', 'd', 'e', 'f']);
});

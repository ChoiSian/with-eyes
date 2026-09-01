// 이지선다(위/아래) 공간 분할 선택 엔진.
// 항목 목록을 빈도 가중치로 균형 분할한 이진 트리를 만들고,
// "위를 본다 = 위 밴드 선택, 아래를 본다 = 아래 밴드 선택"으로 내려가
// 항목 하나가 남으면 선택이 확정된다.
//
// 항목: { id, label, weight, ...데이터 }
// 트리 노드: { top: node|item, bottom: node|item, items: [...] } (leaf는 항목 그대로)

function totalWeight(items) {
  return items.reduce((sum, it) => sum + (it.weight ?? 1), 0);
}

// 빈도순으로 정렬된 배열을 누적 가중치가 절반에 가장 가까운 지점에서
// 연속 분할한다. 앞(고빈도) 그룹이 위 밴드가 된다.
// 분할이 안정적이므로 항목 위치가 세션 간에 고정되어 운동 학습이 가능하다.
export function buildTree(items) {
  if (items.length === 0) return null;
  if (items.length === 1) return items[0];

  const total = totalWeight(items);
  let bestIdx = 1;
  let bestDiff = Infinity;
  let acc = 0;
  for (let i = 0; i < items.length - 1; i++) {
    acc += items[i].weight ?? 1;
    const diff = Math.abs(acc - (total - acc));
    if (diff < bestDiff) {
      bestDiff = diff;
      bestIdx = i + 1;
    }
  }
  return {
    top: buildTree(items.slice(0, bestIdx)),
    bottom: buildTree(items.slice(bestIdx)),
    items,
  };
}

function isLeaf(node) {
  return node !== null && node.items === undefined;
}

function nodeItems(node) {
  if (node === null) return [];
  return isLeaf(node) ? [node] : node.items;
}

// 하나의 선택 사이클. 루트를 명시적으로 두 그룹으로 나눠 시작할 수도 있다
// (예: 위 = 명령/제안 밴드, 아래 = 자모판).
export class SelectionCycle {
  constructor({ top, bottom }) {
    this.root = {
      top: buildTree(top),
      bottom: buildTree(bottom),
      items: [...top, ...bottom],
    };
    this.path = []; // 지나온 노드 스택 (되돌리기용)
    this.node = this.root;
  }

  // 현재 위/아래 밴드에 표시할 항목들
  get bands() {
    return {
      top: nodeItems(this.node.top),
      bottom: nodeItems(this.node.bottom),
    };
  }

  get depth() {
    return this.path.length;
  }

  // 위/아래 응답. 항목이 확정되면 {done: true, item}, 아니면 {done: false}.
  answer(dir) {
    const next = dir === 'up' ? this.node.top : this.node.bottom;
    if (next === null) return { done: false };
    if (isLeaf(next)) {
      return { done: true, item: next };
    }
    this.path.push(this.node);
    this.node = next;
    return { done: false };
  }

  // 한 단계 되돌리기 (긴 응시 취소). 루트면 false.
  back() {
    if (this.path.length === 0) return false;
    this.node = this.path.pop();
    return true;
  }

  reset() {
    this.path = [];
    this.node = this.root;
  }
}

// 단일 목록용 사이클 (빠른 말 카테고리, 지우기 메뉴 등)
export function cycleFromList(items) {
  const tree = buildTree(items);
  if (tree === null) throw new Error('빈 목록으로는 사이클을 만들 수 없습니다');
  if (isLeaf(tree)) {
    // 항목이 하나뿐이면 위 밴드에 그 항목, 아래는 비움
    return new SelectionCycle({ top: [tree], bottom: [] });
  }
  return new SelectionCycle({
    top: nodeItems(tree.top),
    bottom: nodeItems(tree.bottom),
  });
}

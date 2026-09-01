// 완전 로컬 단어 예측 엔진 (통계 언어 모델)
// - 자모 접두사 트라이: 조합 중인 자모열로 단어 완성 후보 검색
// - 유니그램(빈도) + 사용자 어휘 적응 + 최근성 보너스
// - 다음 단어 예측: 큐레이션 바이그램 + 학습된 사용자 바이그램
// 모든 학습 데이터는 localStorage에만 저장된다 (네트워크 없음).

import { toJamoSeq } from './hangul.js';
import { PARTICLES } from '../data/dictionary.js';

const LEX_KEY = 'aac.lex.v1';
const BIGRAM_KEY = 'aac.bigram.v1';
const LEX_CAP = 2000;
const BIGRAM_CAP = 5000;
const DAY_MS = 24 * 60 * 60 * 1000;
const SEP = '\u0001';

function safeGet(key) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function safeSet(key, obj) {
  try {
    localStorage.setItem(key, JSON.stringify(obj));
  } catch {
    /* 저장 불가 시 메모리에서만 유지 */
  }
}

// 마지막 글자에 받침이 있는지 (조사 활용형 결정)
export function hasBatchim(word) {
  const last = [...word].pop();
  if (!last) return false;
  const code = last.charCodeAt(0) - 0xac00;
  if (code < 0 || code > 11171) return false;
  return code % 28 !== 0;
}

// 받침이 ㄹ인지 ((으)로 예외 처리)
function endsWithRieul(word) {
  const last = [...word].pop();
  if (!last) return false;
  const code = last.charCodeAt(0) - 0xac00;
  if (code < 0 || code > 11171) return false;
  return code % 28 === 8; // ㄹ 받침
}

// 앞 단어에 맞는 조사 형태를 고른다.
export function particleForm(particle, prevWord) {
  if (particle.rieulForm && (endsWithRieul(prevWord) || !hasBatchim(prevWord))) {
    return particle.rieulForm;
  }
  return hasBatchim(prevWord) ? particle.forms[0] : particle.forms[1];
}

export class Predictor {
  constructor(dictionary, { persist = true } = {}) {
    this.persist = persist;
    this.entries = new Map(); // word -> {w, f, pos, next}
    this.trie = { children: new Map(), words: [] };
    this.userLex = persist ? safeGet(LEX_KEY) : {}; // word -> [count, lastUsedEpochMs]
    this.userBigrams = persist ? safeGet(BIGRAM_KEY) : {}; // prevnext -> count

    for (const entry of dictionary) {
      this.#addEntry(entry);
    }
    // 사용자가 학습시킨 단어 중 사전에 없는 것도 트라이에 추가
    for (const w of Object.keys(this.userLex)) {
      if (!this.entries.has(w)) this.#addEntry({ w, f: 1, learned: true });
    }
  }

  #addEntry(entry) {
    const word = entry.w.normalize('NFC');
    if (this.entries.has(word)) return;
    const normalized = { ...entry, w: word };
    this.entries.set(word, normalized);
    let node = this.trie;
    for (const jamo of toJamoSeq(word)) {
      if (!node.children.has(jamo)) {
        node.children.set(jamo, { children: new Map(), words: [] });
      }
      node = node.children.get(jamo);
      node.words.push(word);
    }
  }

  #score(word) {
    const entry = this.entries.get(word);
    const f = entry?.f ?? 1;
    let score = Math.log2(1 + f);
    const user = this.userLex[word];
    if (user) {
      score += 2 * Math.log2(1 + user[0]);
      const age = Date.now() - user[1];
      if (age < DAY_MS) score += 3.0;
      else if (age < 7 * DAY_MS) score += 1.5;
    }
    return score;
  }

  // 조합 중인 단어(완성 음절 + 조합 중 자모 포함)로 완성 후보를 찾는다.
  completions(partialWord, limit = 4) {
    const seq = toJamoSeq(partialWord.normalize('NFC'));
    if (seq.length === 0) return [];
    let node = this.trie;
    for (const jamo of seq) {
      node = node.children.get(jamo);
      if (!node) return [];
    }
    // 현재 단어와 같은 단어도 후보에 남긴다: 수락하면 단어 확정 + 자동 띄어쓰기
    const scored = node.words.map((w) => ({ word: w, score: this.#score(w) }));
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, limit);
  }

  // 직전 단어를 보고 다음 단어를 예측한다. 조사 후보도 포함한다.
  nextWords(prevWord, limit = 4) {
    const candidates = new Map(); // word -> {score, particle?}
    const add = (word, score, extra = {}) => {
      const cur = candidates.get(word);
      if (!cur || cur.score < score) candidates.set(word, { score, ...extra });
    };

    if (prevWord) {
      // 학습된 사용자 바이그램
      for (const [key, count] of Object.entries(this.userBigrams)) {
        const [prev, next] = key.split(SEP);
        if (prev === prevWord) add(next, this.#score(next) + 4 * Math.log2(1 + count));
      }
      // 큐레이션 바이그램
      const entry = this.entries.get(prevWord);
      if (entry?.next) {
        for (const next of entry.next) add(next, this.#score(next) + 2.0);
      }
      // 명사 뒤에는 조사 후보 (받침에 맞는 형태, 공백 없이 붙임)
      if (entry?.pos === 'n') {
        for (const p of PARTICLES) {
          add(particleForm(p, prevWord), Math.log2(1 + p.f) + 1.0, { particle: true });
        }
      }
    }

    // 전체 고빈도 유니그램으로 채우기
    if (candidates.size < limit * 2) {
      const top = [...this.entries.keys()]
        .map((w) => ({ w, s: this.#score(w) }))
        .sort((a, b) => b.s - a.s)
        .slice(0, limit * 3);
      for (const { w, s } of top) {
        if (w !== prevWord) add(w, s - 1.0);
      }
    }

    const out = [...candidates.entries()]
      .map(([word, { score, particle }]) => ({ word, score, particle: !!particle }))
      .sort((a, b) => b.score - a.score);
    return out.slice(0, limit);
  }

  // 현재 입력 상태에 맞는 제안 목록 (UI가 그대로 표시).
  suggest({ currentWord, prevWord }, limit = 4) {
    if (currentWord) return this.completions(currentWord, limit).map((c) => ({ ...c, particle: false }));
    return this.nextWords(prevWord, limit);
  }

  // 발화된 문장에서 사용자 어휘/바이그램을 학습한다 (발화 시에만 호출).
  recordSentence(text) {
    const words = text
      .normalize('NFC')
      .split(/\s+/)
      .map((w) => w.replace(/[.?!,]+$/, ''))
      .filter((w) => w.length > 0 && [...w].some((ch) => {
        const code = ch.charCodeAt(0) - 0xac00;
        return code >= 0 && code <= 11171;
      }));
    const now = Date.now();
    for (const w of words) {
      const cur = this.userLex[w] ?? [0, 0];
      this.userLex[w] = [cur[0] + 1, now];
      if (!this.entries.has(w)) this.#addEntry({ w, f: 1, learned: true });
    }
    for (let i = 0; i + 1 < words.length; i++) {
      const key = words[i] + SEP + words[i + 1];
      this.userBigrams[key] = (this.userBigrams[key] ?? 0) + 1;
    }
    this.#evict();
    if (this.persist) {
      safeSet(LEX_KEY, this.userLex);
      safeSet(BIGRAM_KEY, this.userBigrams);
    }
  }

  #evict() {
    const lexKeys = Object.keys(this.userLex);
    if (lexKeys.length > LEX_CAP) {
      lexKeys
        .sort((a, b) => this.userLex[a][1] - this.userLex[b][1])
        .slice(0, lexKeys.length - LEX_CAP)
        .forEach((k) => delete this.userLex[k]);
    }
    const biKeys = Object.keys(this.userBigrams);
    if (biKeys.length > BIGRAM_CAP) {
      biKeys
        .sort((a, b) => this.userBigrams[a] - this.userBigrams[b])
        .slice(0, biKeys.length - BIGRAM_CAP)
        .forEach((k) => delete this.userBigrams[k]);
    }
  }

  // 보호자용: 학습 데이터 백업/복원
  exportData() {
    return JSON.stringify({ lex: this.userLex, bigrams: this.userBigrams }, null, 2);
  }

  importData(json) {
    const data = JSON.parse(json);
    if (data.lex) {
      for (const [w, v] of Object.entries(data.lex)) {
        const cur = this.userLex[w] ?? [0, 0];
        this.userLex[w] = [cur[0] + v[0], Math.max(cur[1], v[1])];
        if (!this.entries.has(w)) this.#addEntry({ w, f: 1, learned: true });
      }
    }
    if (data.bigrams) {
      for (const [k, c] of Object.entries(data.bigrams)) {
        this.userBigrams[k] = (this.userBigrams[k] ?? 0) + c;
      }
    }
    this.#evict();
    if (this.persist) {
      safeSet(LEX_KEY, this.userLex);
      safeSet(BIGRAM_KEY, this.userBigrams);
    }
  }
}

// 한글 자모 조합 오토마타 (2벌식 IME 방식)
// 자모를 하나씩 입력받아 완성형 음절(U+AC00~)로 조합한다.

export const CHOSEONG = [...'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ'];
export const JUNGSEONG = [...'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'];
export const JONGSEONG = ['', ...'ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ'];

// 겹모음: [첫 모음][둘째 모음] -> 겹모음
const VOWEL_COMPOUNDS = {
  'ㅗㅏ': 'ㅘ', 'ㅗㅐ': 'ㅙ', 'ㅗㅣ': 'ㅚ',
  'ㅜㅓ': 'ㅝ', 'ㅜㅔ': 'ㅞ', 'ㅜㅣ': 'ㅟ',
  'ㅡㅣ': 'ㅢ',
};

// 겹받침: [첫 받침][둘째 받침] -> 겹받침
const JONG_COMPOUNDS = {
  'ㄱㅅ': 'ㄳ', 'ㄴㅈ': 'ㄵ', 'ㄴㅎ': 'ㄶ',
  'ㄹㄱ': 'ㄺ', 'ㄹㅁ': 'ㄻ', 'ㄹㅂ': 'ㄼ', 'ㄹㅅ': 'ㄽ',
  'ㄹㅌ': 'ㄾ', 'ㄹㅍ': 'ㄿ', 'ㄹㅎ': 'ㅀ',
  'ㅂㅅ': 'ㅄ',
};

const VOWEL_SPLIT = invertMap(VOWEL_COMPOUNDS);
const JONG_SPLIT = invertMap(JONG_COMPOUNDS);

function invertMap(map) {
  const out = {};
  for (const [pair, compound] of Object.entries(map)) out[compound] = [...pair];
  return out;
}

const HANGUL_BASE = 0xac00;

export function isConsonant(ch) {
  return CHOSEONG.includes(ch) || JONGSEONG.includes(ch);
}

export function isVowel(ch) {
  return JUNGSEONG.includes(ch);
}

export function isJamo(ch) {
  return isConsonant(ch) || isVowel(ch);
}

export function composeSyllable(cho, jung, jong = '') {
  const ci = CHOSEONG.indexOf(cho);
  const vi = JUNGSEONG.indexOf(jung);
  const ji = JONGSEONG.indexOf(jong);
  if (ci < 0 || vi < 0 || ji < 0) return null;
  return String.fromCharCode(HANGUL_BASE + (ci * 21 + vi) * 28 + ji);
}

export function decomposeSyllable(ch) {
  const code = ch.charCodeAt(0) - HANGUL_BASE;
  if (code < 0 || code > 11171) return null;
  return {
    cho: CHOSEONG[Math.floor(code / (21 * 28))],
    jung: JUNGSEONG[Math.floor(code / 28) % 21],
    jong: JONGSEONG[code % 28],
  };
}

// 문자열을 기본 자모 시퀀스로 분해한다 (겹모음/겹받침은 구성 요소로 분리).
// ㄲ ㅆ 등 쌍자음은 하나의 선택 항목이므로 그대로 유지한다.
// 예측 엔진의 접두사 매칭에 사용.
export function toJamoSeq(text) {
  const seq = [];
  for (const ch of text) {
    const parts = decomposeSyllable(ch);
    if (parts) {
      seq.push(parts.cho);
      seq.push(...(VOWEL_SPLIT[parts.jung] ?? [parts.jung]));
      if (parts.jong) seq.push(...(JONG_SPLIT[parts.jong] ?? [parts.jong]));
    } else if (VOWEL_SPLIT[ch]) {
      seq.push(...VOWEL_SPLIT[ch]);
    } else if (JONG_SPLIT[ch]) {
      seq.push(...JONG_SPLIT[ch]);
    } else {
      seq.push(ch);
    }
  }
  return seq;
}

// 조합 중인 음절 상태를 관리하고 전체 텍스트를 보관하는 오토마타.
export class HangulComposer {
  constructor() {
    this.committed = '';
    this.cho = null; // 초성 (자음)
    this.jung = null; // 중성 (모음)
    this.jong = null; // 종성 (자음)
  }

  get composing() {
    if (this.jung !== null) {
      if (this.cho !== null) {
        return composeSyllable(this.cho, this.jung, this.jong ?? '');
      }
      return this.jung; // 모음 단독
    }
    return this.cho ?? ''; // 자음 단독 또는 없음
  }

  get value() {
    return this.committed + this.composing;
  }

  clearComposing() {
    this.cho = null;
    this.jung = null;
    this.jong = null;
  }

  commitComposing() {
    this.committed += this.composing;
    this.clearComposing();
  }

  clear() {
    this.committed = '';
    this.clearComposing();
  }

  setText(text) {
    this.clear();
    this.committed = text;
  }

  // 자모 또는 일반 문자 하나를 입력한다.
  input(ch) {
    if (isVowel(ch)) {
      this.#inputVowel(ch);
    } else if (isConsonant(ch)) {
      this.#inputConsonant(ch);
    } else {
      // 공백, 문장부호 등: 조합 확정 후 그대로 추가
      this.commitComposing();
      this.committed += ch;
    }
  }

  #inputConsonant(ch) {
    if (this.jung === null) {
      if (this.cho === null) {
        if (CHOSEONG.includes(ch)) {
          this.cho = ch;
        } else {
          // 초성이 될 수 없는 자음(겹받침 문자 등)은 그대로 확정
          this.committed += ch;
        }
      } else {
        // 자음 뒤 자음: 앞 자음 확정, 새로 시작
        this.commitComposing();
        this.#inputConsonant(ch);
      }
      return;
    }

    // 모음까지 있는 상태
    if (this.cho === null) {
      // 모음 단독 뒤 자음: 모음 확정, 새로 시작
      this.commitComposing();
      this.#inputConsonant(ch);
      return;
    }

    if (this.jong === null) {
      if (JONGSEONG.includes(ch)) {
        this.jong = ch;
      } else {
        // ㄸ ㅃ ㅉ 은 받침 불가: 음절 확정 후 새 초성
        this.commitComposing();
        this.cho = ch;
      }
      return;
    }

    // 겹받침 시도
    const compound = JONG_COMPOUNDS[this.jong + ch];
    if (compound) {
      this.jong = compound;
    } else {
      this.commitComposing();
      this.#inputConsonant(ch);
    }
  }

  #inputVowel(ch) {
    if (this.jung === null) {
      // 초성만 있거나 아무것도 없는 상태
      this.jung = ch;
      return;
    }

    if (this.jong !== null) {
      // 연음: 받침(또는 겹받침의 마지막 요소)이 다음 음절 초성으로 이동
      const split = JONG_SPLIT[this.jong];
      let movedCho;
      if (split) {
        [this.jong, movedCho] = split;
      } else {
        movedCho = this.jong;
        this.jong = null;
      }
      if (!CHOSEONG.includes(movedCho)) {
        // 초성이 될 수 없는 받침이면 이동하지 않고 음절 확정 후 모음 단독
        this.jong = split ? JONG_COMPOUNDS[this.jong + movedCho] : movedCho;
        this.commitComposing();
        this.jung = ch;
        return;
      }
      this.commitComposing();
      this.cho = movedCho;
      this.jung = ch;
      return;
    }

    // 겹모음 시도
    const compound = VOWEL_COMPOUNDS[this.jung + ch];
    if (compound) {
      this.jung = compound;
    } else {
      this.commitComposing();
      this.jung = ch;
    }
  }

  // 자모 단위 백스페이스. 삭제할 것이 없으면 false.
  backspace() {
    if (this.jong !== null) {
      const split = JONG_SPLIT[this.jong];
      this.jong = split ? split[0] : null;
      return true;
    }
    if (this.jung !== null) {
      const split = VOWEL_SPLIT[this.jung];
      if (split) {
        this.jung = split[0];
      } else if (this.cho !== null) {
        this.jung = null;
      } else {
        this.jung = null; // 모음 단독 삭제
      }
      return true;
    }
    if (this.cho !== null) {
      this.cho = null;
      return true;
    }
    if (this.committed.length > 0) {
      // 마지막 확정 글자를 자모 상태로 되돌린 뒤 마지막 자모 제거
      const last = [...this.committed].pop();
      this.committed = [...this.committed].slice(0, -1).join('');
      const parts = decomposeSyllable(last);
      if (parts) {
        this.cho = parts.cho;
        this.jung = parts.jung;
        this.jong = parts.jong || null;
        return this.backspace();
      }
      if (isVowel(last)) {
        this.jung = last;
        return this.backspace();
      }
      if (isConsonant(last)) {
        this.cho = CHOSEONG.includes(last) ? last : null;
        if (this.cho === null) return true; // 겹받침 단독 문자였던 경우: 그냥 삭제
        return this.backspace();
      }
      return true; // 일반 문자(공백 등)는 통째로 삭제
    }
    return false;
  }
}

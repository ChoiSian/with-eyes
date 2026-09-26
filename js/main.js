// 앱 컨트롤러: 두 칸 터치 선택 + 삭제 버튼.
// 위/아래 큰 칸 중 원하는 글자가 있는 쪽을 누르면 그 칸이 절반으로 줄어들고,
// 하나가 남으면 입력된다. 삭제 버튼은 한 단계 뒤로/마지막 입력 되돌리기.

import { HangulComposer } from './hangul.js';
import { KoreanTTS } from './tts.js';
import { Predictor } from './predictor.js';
import { SelectionCycle, cycleFromList } from './scanner.js';
import { DICTIONARY } from '../data/dictionary.js';
import { QUICK_PHRASES } from '../data/quick-phrases.js';
import { CHO_FREQ, VOWEL_FREQ, batchimBlend } from '../data/jamo-freq.js';

const $ = (sel) => document.querySelector(sel);
const SETTINGS_KEY = 'aac.settings.v1';

const state = {
  composer: new HangulComposer(),
  predictor: new Predictor(DICTIONARY),
  tts: new KoreanTTS(),
  mode: 'main',
  modeArg: null,
  cycle: null,
  inputSuspended: false,
  undoStack: [],
  settingsOpen: false,
  settings: { ttsRate: 0.95 },
};

// ===== 소리 (이어콘) =====
let audioCtx = null;

function ensureAudio() {
  if (!audioCtx) {
    try {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    } catch { /* 소리 없이 진행 */ }
  }
  // iOS: 사용자 제스처 시점에 오디오/음성합성 잠금 해제
  audioCtx?.resume?.();
  state.tts.unlock();
}

function tone(freq, ms = 120, gainVal = 0.06) {
  if (!audioCtx) return;
  try {
    if (audioCtx.state === 'suspended') audioCtx.resume();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.frequency.value = freq;
    osc.type = 'sine';
    gain.gain.value = gainVal;
    osc.connect(gain).connect(audioCtx.destination);
    osc.start();
    gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + ms / 1000);
    osc.stop(audioCtx.currentTime + ms / 1000);
  } catch { /* 무시 */ }
}

const sounds = {
  up: () => tone(880, 110),
  down: () => tone(440, 110),
  select: () => { tone(660, 90); setTimeout(() => tone(990, 130), 90); },
  warn: () => tone(300, 220, 0.09),
  undo: () => { tone(500, 90); setTimeout(() => tone(350, 130), 90); },
};

// ===== 오버레이/토스트/에코 =====
function overlay(id, show) {
  $('#' + id).classList.toggle('show', show);
}

let toastTimer = 0;
function toast(msg, ms = 2500) {
  const el = $('#toast');
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), ms);
}

let echoTimer = 0;
function echo(label) {
  const el = $('#echo');
  el.querySelector('.echo-text').textContent = label;
  el.classList.add('show');
  clearTimeout(echoTimer);
  echoTimer = setTimeout(() => el.classList.remove('show'), 600);
}

// ===== 문장/단어 상태 =====
function wordContext() {
  const text = state.composer.value;
  const lastSpace = text.lastIndexOf(' ');
  const currentWord = text.slice(lastSpace + 1);
  const before = text.slice(0, lastSpace + 1).trim();
  const prevWord = before.split(/\s+/).filter(Boolean).pop() ?? null;
  return { currentWord, prevWord };
}

function captureComposer() {
  const c = state.composer;
  return { committed: c.committed, cho: c.cho, jung: c.jung, jong: c.jong, jungAtomic: c.jungAtomic };
}

function composerChanged(snap) {
  const c = state.composer;
  return snap.committed !== c.committed || snap.cho !== c.cho ||
    snap.jung !== c.jung || snap.jong !== c.jong;
}

// 문장을 실제로 바꾼 선택만 스냅샷으로 쌓는다.
// 삭제 버튼은 항상 '마지막으로 문장이 바뀌기 직전' 상태로 복원할 수 있다.
function pushSnapshot(snap) {
  state.undoStack.push(snap);
  if (state.undoStack.length > 20) state.undoStack.shift();
}

function restoreSnapshot() {
  const snap = state.undoStack.pop();
  if (!snap) return false;
  const c = state.composer;
  c.committed = snap.committed;
  c.cho = snap.cho;
  c.jung = snap.jung;
  c.jong = snap.jong;
  c.jungAtomic = snap.jungAtomic ?? false;
  return true;
}

// ===== 선택 항목 구성 =====
function jamoItems() {
  const { cho, jung } = state.composer;
  let weights;
  if (cho === null && jung === null) {
    // 초성 자리: 자음이 압도적으로 유력
    weights = { ...CHO_FREQ };
    for (const v of Object.keys(VOWEL_FREQ)) weights[v] = 0.15;
  } else if (cho !== null && jung === null) {
    // 모음 자리
    weights = { ...VOWEL_FREQ };
    for (const c of Object.keys(CHO_FREQ)) weights[c] = 0.15;
  } else {
    // 받침 또는 다음 음절 초성 자리
    weights = batchimBlend();
    for (const v of Object.keys(VOWEL_FREQ)) weights[v] = (weights[v] ?? 0) + 0.4;
  }
  return Object.entries(weights)
    .sort((a, b) => b[1] - a[1])
    .map(([ch, w]) => ({ id: 'j:' + ch, label: ch, weight: w, kind: 'jamo', jamo: ch }));
}

function currentSuggestions() {
  const { currentWord, prevWord } = wordContext();
  return state.predictor.suggest({ currentWord, prevWord }, 4);
}

function commandItems() {
  const items = [];
  const suggWeights = [12, 9, 7, 5];
  currentSuggestions().forEach((s, i) => {
    items.push({
      id: 's:' + s.word,
      label: s.particle ? '+' + s.word : s.word,
      kind: 'suggestion',
      word: s.word,
      particle: s.particle,
      weight: suggWeights[i] ?? 4,
      cls: 'suggestion',
    });
  });
  const text = state.composer.value;
  items.push({ id: 'a:quick', label: '⚡ 빠른 말', kind: 'action', action: 'quick', weight: 8, cls: 'action' });
  if (text.trim()) {
    items.push({ id: 'a:speak', label: '🔊 말하기', kind: 'action', action: 'speak', weight: 7, cls: 'action' });
  }
  items.push({
    id: 'a:space', label: '␣ 띄어쓰기', kind: 'action', action: 'space',
    weight: text ? 7 : 1.5, cls: 'action',
  });
  if (text) {
    items.push({ id: 'a:del', label: '⌫ 지우기', kind: 'action', action: 'delete', weight: 5, cls: 'action' });
  }
  return items;
}

function buildCycle() {
  const { mode, modeArg } = state;
  if (mode === 'main') {
    return new SelectionCycle({ top: commandItems(), bottom: jamoItems() });
  }
  if (mode === 'delete') {
    return cycleFromList([
      { id: 'd:jamo', label: '⌫ 한 글자씩', weight: 8, kind: 'del', del: 'jamo' },
      { id: 'd:cancel', label: '↩ 취소', weight: 5, kind: 'del', del: 'cancel' },
      { id: 'd:word', label: '⌫ 단어 지우기', weight: 4, kind: 'del', del: 'word' },
      { id: 'd:all', label: '🗑 전부 지우기', weight: 2, kind: 'del', del: 'all' },
    ]);
  }
  if (mode === 'quickcat') {
    const items = QUICK_PHRASES.map((c, i) => ({
      id: 'qc:' + i, label: c.category, weight: c.weight, kind: 'quickcat', idx: i,
    }));
    items.push({ id: 'qc:back', label: '↩ 취소', weight: 3, kind: 'quickcat', idx: -1 });
    return cycleFromList(items);
  }
  if (mode === 'quickphrase') {
    const cat = QUICK_PHRASES[modeArg];
    const items = cat.phrases.map((p, i) => ({
      id: 'qp:' + i, label: p.text, weight: p.weight, kind: 'quickphrase', text: p.text, cls: 'suggestion',
    }));
    items.push({ id: 'qp:back', label: '↩ 뒤로', weight: 3, kind: 'quickphrase', text: null });
    return cycleFromList(items);
  }
  if (mode === 'confirm-quick') {
    return new SelectionCycle({
      top: [{ id: 'y', label: `🔊 "${modeArg.text}" 말하기`, weight: 1, kind: 'confirm', yes: true }],
      bottom: [{ id: 'n', label: '↩ 아니오', weight: 1, kind: 'confirm', yes: false }],
    });
  }
  if (mode === 'confirm-speak') {
    return new SelectionCycle({
      top: [{ id: 'y', label: '🔊 지금 말하기', weight: 1, kind: 'confirm', yes: true }],
      bottom: [{ id: 'n', label: '↩ 아니오, 계속 쓰기', weight: 1, kind: 'confirm', yes: false }],
    });
  }
  if (mode === 'confirm-clear') {
    return new SelectionCycle({
      top: [{ id: 'n', label: '↩ 아니오, 그대로 두기', weight: 1, kind: 'confirm', yes: false }],
      bottom: [{ id: 'y', label: '🗑 네, 전부 지우기', weight: 1, kind: 'confirm', yes: true }],
    });
  }
  if (mode === 'post-speak') {
    return new SelectionCycle({
      top: [{ id: 'y', label: '🆕 지우고 새 문장', weight: 1, kind: 'confirm', yes: true }],
      bottom: [{ id: 'n', label: '✏️ 그대로 이어 쓰기', weight: 1, kind: 'confirm', yes: false }],
    });
  }
  throw new Error('알 수 없는 모드: ' + mode);
}

function setMode(mode, arg = null) {
  state.mode = mode;
  state.modeArg = arg;
  state.cycle = buildCycle();
  render();
}

// ===== 렌더링 =====
function renderSentence() {
  const el = $('#sentence');
  el.textContent = '';
  const committed = document.createElement('span');
  committed.textContent = state.composer.committed;
  el.appendChild(committed);
  const composing = state.composer.composing;
  if (composing) {
    const span = document.createElement('span');
    span.className = 'composing';
    span.textContent = composing;
    el.appendChild(span);
  }
  const cursor = document.createElement('span');
  cursor.className = 'cursor';
  el.appendChild(cursor);
  // 긴 문장은 스크롤로 항상 끝(커서)이 보이게
  const bar = el.parentElement;
  if (bar) bar.scrollTop = bar.scrollHeight;
}

const MODE_HINTS = {
  main: '글자나 단어를 고르세요',
  delete: '지우기 방법',
  quickcat: '빠른 말 — 분류',
  quickphrase: '빠른 말 — 문장',
  'confirm-quick': '이 문장을 말할까요?',
  'confirm-speak': '문장을 말할까요?',
  'confirm-clear': '정말 전부 지울까요?',
  'post-speak': '다 말했어요',
};

function renderBands() {
  const bands = state.cycle.bands;
  for (const [name, items] of [['top', bands.top], ['bottom', bands.bottom]]) {
    const el = $('#band-' + name);
    el.querySelectorAll('.tile').forEach((t) => t.remove());
    el.classList.toggle('single', items.length === 1);
    el.classList.toggle('dense', items.length > 20);
    el.classList.toggle('empty', items.length === 0);
    for (const item of items) {
      const tile = document.createElement('div');
      tile.className = 'tile' + (item.cls ? ' ' + item.cls : '');
      if (item.label.length > 4) tile.classList.add('small');
      tile.textContent = item.label;
      el.appendChild(tile);
    }
  }
  $('#mode-hint').textContent = MODE_HINTS[state.mode] ?? '';
}

function canGoBack() {
  return (state.cycle?.depth ?? 0) > 0 || state.mode !== 'main' || state.undoStack.length > 0;
}

function render() {
  renderSentence();
  renderBands();
  $('#btn-delete').classList.toggle('inactive', !canGoBack());
}

// ===== 동작 =====
function acceptSuggestion(item) {
  const c = state.composer;
  c.clearComposing();
  if (item.particle) {
    c.committed = c.committed.replace(/\s+$/, '') + item.word + ' ';
  } else {
    const lastSpace = c.committed.lastIndexOf(' ');
    c.committed = c.committed.slice(0, lastSpace + 1) + item.word + ' ';
  }
}

function deleteWord() {
  const c = state.composer;
  c.commitComposing();
  // 공백만 남은 경우도 비운다 (정규식이 비공백을 요구해 무시되는 일 방지)
  if (c.committed.trim() === '') c.committed = '';
  else c.committed = c.committed.replace(/\s*\S+\s*$/, '');
}

function cleanForSpeech(text) {
  return text
    .replace(/[ㄱ-ㅣ]/g, '') // 조합 안 된 낱자모는 발음 혼란만 줌
    .replace(/\s+/g, ' ')
    .trim();
}

async function speakText(spoken, { record } = { record: false }) {
  state.inputSuspended = true;
  $('#speaking-text').textContent = spoken;
  overlay('overlay-speaking', true);
  try {
    await state.tts.speak(spoken);
  } finally {
    overlay('overlay-speaking', false);
    state.inputSuspended = false;
  }
  if (record) state.predictor.recordSentence(spoken);
}

async function speakSentence() {
  state.composer.commitComposing();
  const spoken = cleanForSpeech(state.composer.value);
  if (!spoken) {
    toast('말할 내용이 없어요');
    setMode('main');
    return;
  }
  if (!state.tts.hasKoreanVoice) toast('한국어 음성이 없어 화면으로만 보여줍니다', 4000);
  await speakText(spoken, { record: true });
  setMode('post-speak');
}

function onSelect(item) {
  sounds.select();
  echo(item.label);
  const before = captureComposer();
  switch (item.kind) {
    case 'jamo':
      state.composer.input(item.jamo);
      setMode('main');
      break;
    case 'suggestion':
      acceptSuggestion(item);
      setMode('main');
      break;
    case 'action':
      if (item.action === 'space') {
        state.composer.input(' ');
        setMode('main');
      } else if (item.action === 'delete') {
        setMode('delete');
      } else if (item.action === 'speak') {
        setMode('confirm-speak');
      } else if (item.action === 'quick') {
        setMode('quickcat');
      }
      break;
    case 'del':
      if (item.del === 'jamo') {
        state.composer.backspace();
        setMode('main');
      } else if (item.del === 'word') {
        deleteWord();
        setMode('main');
      } else if (item.del === 'all') {
        setMode('confirm-clear');
      } else {
        setMode('main');
      }
      break;
    case 'quickcat':
      if (item.idx < 0) setMode('main');
      else setMode('quickphrase', item.idx);
      break;
    case 'quickphrase':
      if (item.text === null) setMode('quickcat');
      else setMode('confirm-quick', { text: item.text, catIdx: state.modeArg });
      break;
    case 'confirm':
      if (state.mode === 'confirm-quick') {
        if (item.yes) {
          const { text } = state.modeArg;
          setMode('main');
          speakText(text);
        } else {
          // 문장 목록(카테고리 유지)으로 돌아간다
          setMode('quickphrase', state.modeArg.catIdx);
        }
      } else if (state.mode === 'confirm-speak') {
        if (item.yes) speakSentence();
        else setMode('main');
      } else if (state.mode === 'confirm-clear') {
        if (item.yes) state.composer.clear();
        setMode('main');
      } else if (state.mode === 'post-speak') {
        if (item.yes) state.composer.clear();
        setMode('main');
      }
      break;
  }
  // 문장이 실제로 바뀐 선택만 삭제 버튼의 되돌리기 대상으로 기록
  if (composerChanged(before)) pushSnapshot(before);
}

// ===== 입력 =====
function choose(dir) {
  if (state.inputSuspended || state.settingsOpen) return;
  (dir === 'up' ? sounds.up : sounds.down)();
  const res = state.cycle.answer(dir);
  if (res.done) onSelect(res.item);
  else render();
}

// 삭제 버튼: 트리 한 단계 위 → 메뉴 취소 → 마지막 입력 되돌리기 순
function goBack() {
  if (state.inputSuspended || state.settingsOpen) return;
  sounds.undo();
  if (state.cycle.depth > 0) {
    state.cycle.back();
    render();
    toast('한 단계 되돌렸어요');
  } else if (state.mode !== 'main') {
    setMode('main');
    toast('취소했어요');
  } else if (restoreSnapshot()) {
    setMode('main');
    toast('마지막 입력을 지웠어요');
  } else {
    render();
    toast('지울 것이 없어요');
  }
}

function wireInput() {
  $('#band-top').addEventListener('click', () => choose('up'));
  $('#band-bottom').addEventListener('click', () => choose('down'));
  $('#btn-delete').addEventListener('click', () => goBack());

  // 키보드(데스크톱 테스트용): ↑ = 위 칸, ↓ = 아래 칸, Backspace = 삭제
  window.addEventListener('keydown', (e) => {
    if (e.target.closest?.('#settings') ||
        ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(e.target.tagName)) {
      return;
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      choose('up');
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      choose('down');
    } else if (e.key === 'Backspace') {
      e.preventDefault();
      goBack();
    }
  });

  // 첫 터치에서 오디오/음성 잠금 해제 (iOS)
  window.addEventListener('pointerdown', ensureAudio, { once: true });
}

// ===== 화면 꺼짐 방지 =====
let wakeLock = null;
async function acquireWakeLock() {
  try {
    wakeLock = await navigator.wakeLock?.request('screen');
  } catch { /* 지원 안 되면 무시 */ }
}
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') acquireWakeLock();
});

// ===== 설정 =====
function loadSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw) Object.assign(state.settings, JSON.parse(raw));
  } catch { /* 무시 */ }
  const rate = Number(state.settings.ttsRate);
  state.settings.ttsRate = Number.isFinite(rate) ? Math.min(1.6, Math.max(0.5, rate)) : 0.95;
  $('#set-rate').value = state.settings.ttsRate;
  $('#set-rate-val').textContent = state.settings.ttsRate;
}

function saveSettings() {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(state.settings));
  } catch { /* 무시 */ }
}

function wireSettingsUI() {
  $('#btn-settings').addEventListener('click', () => {
    state.settingsOpen = true;
    $('#settings').classList.add('show');
  });
  $('#btn-close-settings').addEventListener('click', () => {
    state.settingsOpen = false;
    $('#settings').classList.remove('show');
  });
  $('#set-rate').addEventListener('input', (e) => {
    state.settings.ttsRate = Number(e.target.value);
    $('#set-rate-val').textContent = state.settings.ttsRate;
    state.tts.rate = state.settings.ttsRate;
    saveSettings();
  });
  $('#btn-export').addEventListener('click', () => {
    const blob = new Blob([state.predictor.exportData()], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'sonkkeut-학습데이터.json';
    a.click();
    URL.revokeObjectURL(a.href);
  });
  $('#btn-import').addEventListener('click', () => $('#import-file').click());
  $('#import-file').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      state.predictor.importData(await file.text());
      toast('학습 데이터를 가져왔어요');
    } catch {
      toast('가져오기 실패: 올바른 파일이 아니에요');
    }
    e.target.value = '';
  });
}

// ===== 시작 =====
function checkTtsVoice() {
  const check = () => {
    if (state.tts.available && !state.tts.hasKoreanVoice) {
      toast('⚠️ 이 브라우저에 한국어 음성이 없어 소리로 읽어줄 수 없습니다', 6000);
    }
  };
  setTimeout(check, 1500);
}

function boot() {
  loadSettings();
  state.tts.rate = state.settings.ttsRate;
  checkTtsVoice();
  wireSettingsUI();
  wireInput();
  setMode('main');
  acquireWakeLock();
}

boot();

// 테스트/디버깅용 핸들 (콘솔에서 상태 확인 가능)
window.__aac = state;

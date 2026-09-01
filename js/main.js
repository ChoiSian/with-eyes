// 앱 컨트롤러: 화면 전환, 선택 사이클, 시선 이벤트 → 동작 연결.

import { HangulComposer } from './hangul.js';
import { KoreanTTS } from './tts.js';
import { Predictor } from './predictor.js';
import { SelectionCycle, cycleFromList } from './scanner.js';
import { EyeTracker, KeyboardTracker } from './eye-tracker.js';
import { DICTIONARY } from '../data/dictionary.js';
import { QUICK_PHRASES } from '../data/quick-phrases.js';
import { CHO_FREQ, VOWEL_FREQ, batchimBlend } from '../data/jamo-freq.js';

const $ = (sel) => document.querySelector(sel);
const SETTINGS_KEY = 'aac.settings.v1';

const state = {
  tracker: null,
  usingKeyboard: false,
  composer: new HangulComposer(),
  predictor: new Predictor(DICTIONARY),
  tts: new KoreanTTS(),
  mode: 'main',
  modeArg: null,
  cycle: null,
  paused: false,
  faceLost: false,
  inputSuspended: false,
  pausePattern: [],
  pausePatternAt: 0,
  undoStack: [],
  screen: 'screen-start',
  settingsOpen: false,
  settings: {
    dwellMs: 700, retractEnabled: true, ttsRate: 0.95, eyeMode: 'both',
    scanPeriodMs: 1500,
    // 위 응시 진입 임계값 = upSensitivity × 보정된 위 응시 크기. 낮을수록 민감.
    upSensitivity: 0.45,
  },
  // 단일 스위치 스캐닝: 하이라이트가 위/아래 밴드를 자동으로 오가고,
  // 환자가 위를 보면 '지금 켜져 있는 밴드'가 선택된다.
  scan: { highlight: 'top', lastFlip: 0, frozen: false, captured: null },
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

// ===== 화면/오버레이 =====
function showScreen(id) {
  document.querySelectorAll('.screen').forEach((s) => s.classList.remove('active'));
  $('#' + id).classList.add('active');
  state.screen = id;
  if (typeof updateFaceLostOverlay === 'function') updateFaceLostOverlay();
}

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
// 되돌리기(retract)는 항상 '마지막으로 문장이 바뀌기 직전' 상태로 복원된다.
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
  items.push({ id: 'a:rest', label: '😌 쉬기', kind: 'action', action: 'rest', weight: 1.2, cls: 'action' });
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
      top: [{ id: 'y', label: `🔊 "${modeArg}" 말하기`, weight: 1, kind: 'confirm', yes: true }],
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

function render() {
  renderSentence();
  renderBands();
  restartScan();
}

// ===== 단일 스위치 스캐닝 =====
function scanRunning() {
  return state.screen === 'screen-main' && !state.paused && !state.settingsOpen &&
    !state.inputSuspended && !(state.faceLost && !state.usingKeyboard) && state.cycle;
}

// 빈 밴드는 하이라이트에서 건너뛴다
function bandHasItems(name) {
  return state.cycle && state.cycle.bands[name === 'top' ? 'top' : 'bottom'].length > 0;
}

function renderScanHighlight() {
  const active = scanRunning();
  for (const name of ['top', 'bottom']) {
    const el = $('#band-' + name);
    const on = active && state.scan.highlight === name;
    el.classList.toggle('scan-on', on);
    el.classList.toggle('scan-off', active && !on);
    const cue = el.querySelector('.band-cue');
    cue.textContent = on ? '👁 지금 위를 보면 이 칸 선택!' : '잠시 후 이 칸 차례';
  }
}

function restartScan() {
  state.scan.highlight = bandHasItems('top') ? 'top' : 'bottom';
  state.scan.lastFlip = performance.now();
  state.scan.frozen = false;
  state.scan.captured = null;
  renderScanHighlight();
}

function flipScan() {
  const next = state.scan.highlight === 'top' ? 'bottom' : 'top';
  if (bandHasItems(next)) {
    state.scan.highlight = next;
    tone(next === 'top' ? 520 : 390, 60, 0.03); // 구분되는 짧은 틱
  }
  state.scan.lastFlip = performance.now();
  renderScanHighlight();
}

setInterval(() => {
  if (!scanRunning() || state.scan.frozen) return;
  if (performance.now() - state.scan.lastFlip >= state.settings.scanPeriodMs) flipScan();
}, 50);

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

function enterPause() {
  state.paused = true;
  state.pausePattern = [];
  $('#pause-pattern').textContent = '⬆ 0 / 3';
  overlay('overlay-pause', true);
  renderScanHighlight();
}

function resumeFromPause() {
  state.paused = false;
  overlay('overlay-pause', false);
  setMode('main');
  updateFaceLostOverlay();
  toast('다시 시작합니다');
}

// 쉬기 중 재개: 위 응시(선택 제스처)를 8초 안에 3번 연속
function handlePausePattern() {
  const now = Date.now();
  if (now - state.pausePatternAt > 8000) state.pausePattern = [];
  state.pausePatternAt = now;
  state.pausePattern.push('up');
  $('#pause-pattern').textContent =
    `⬆ ${state.pausePattern.length} / 3`;
  if (state.pausePattern.length >= 3) resumeFromPause();
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
      } else if (item.action === 'rest') {
        setMode('main');
        enterPause();
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
  // 문장이 실제로 바뀐 선택만 되돌리기 대상으로 기록
  if (composerChanged(before)) pushSnapshot(before);
}

// ===== 입력 라우팅 =====
let practiceResolver = null;

// 위 응시(선택 제스처) 한 가지만 입력으로 쓴다.
// 선택되는 밴드는 '제스처를 시작한 순간' 하이라이트돼 있던 밴드.
function onAnswer() {
  if (state.inputSuspended || state.settingsOpen) return;
  if (state.faceLost && !state.usingKeyboard) return;

  if (state.paused) {
    handlePausePattern();
    return;
  }

  if (state.screen === 'screen-practice') {
    if (practiceResolver) practiceResolver('up');
    return;
  }

  if (state.screen !== 'screen-main') return;

  const band = state.scan.captured ?? state.scan.highlight;
  state.scan.frozen = false;
  state.scan.captured = null;
  (band === 'top' ? sounds.up : sounds.down)();
  const res = state.cycle.answer(band === 'top' ? 'up' : 'down');
  if (res.done) onSelect(res.item);
  else render();
}

// 시선을 올리기 시작하면 하이라이트 전환을 멈춰서
// '내가 보던 칸'이 그대로 선택되도록 한다.
function onDwellStart() {
  if (!scanRunning()) return;
  state.scan.frozen = true;
  state.scan.captured = state.scan.highlight;
}

function onDwellAbort() {
  state.scan.frozen = false;
  state.scan.captured = null;
  state.scan.lastFlip = performance.now(); // 남은 시간을 새로 줘서 급한 전환 방지
}

function onRetract() {
  if (state.inputSuspended || state.paused || state.settingsOpen) return;
  if (state.screen !== 'screen-main') return;
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
    toast('마지막 입력을 되돌렸어요');
  }
}

// ===== 시선 시각화 =====
function bindGazeVisuals(tracker) {
  const needle = $('#gauge .needle');
  const zoneUp = $('#gauge .zone-up');
  const fillTop = $('#band-top .fill');
  const fillBottom = $('#band-bottom .fill');
  const bandTop = $('#band-top');
  const bandBottom = $('#band-bottom');

  tracker.addEventListener('gaze', (e) => {
    const { S, state: st, zone, progress, gated, upEnter } = e.detail;
    // 게이지: 아래(-2σ)부터 위 임계값 너머까지
    const max = upEnter * 1.6;
    const min = -2;
    const frac = 1 - (Math.min(max, Math.max(min, S)) - min) / (max - min);
    needle.style.top = `calc(${(frac * 100).toFixed(1)}% - 5px)`;
    needle.classList.toggle('gated', gated);
    zoneUp.style.top = '0';
    zoneUp.style.height = `${(((max - upEnter) / (max - min)) * 100).toFixed(1)}%`;

    // 체류 진행도는 '제스처 시작 시점에 켜져 있던' 밴드에 표시
    const inDwell = st === 'dwell' && zone === 'up';
    const band = state.scan.captured ?? state.scan.highlight;
    fillTop.style.transform = `scaleY(${inDwell && band === 'top' ? progress : 0})`;
    fillBottom.style.transform = `scaleY(${inDwell && band === 'bottom' ? progress : 0})`;
    const engaged = st === 'dwell' || st === 'debounce';
    bandTop.classList.toggle('hot-top', engaged && band === 'top');
    bandBottom.classList.toggle('hot-bottom', engaged && band === 'bottom');
  });
}

// ===== 트래커 이벤트 배선 =====
function wireTracker(tracker) {
  tracker.addEventListener('answer', () => onAnswer());
  tracker.addEventListener('dwellstart', () => onDwellStart());
  tracker.addEventListener('dwellabort', () => onDwellAbort());
  tracker.addEventListener('retract', () => onRetract());
  tracker.addEventListener('retractwarn', () => {
    sounds.warn();
    toast('계속 응시하면 방금 선택이 취소됩니다', 1500);
  });
  tracker.addEventListener('pausegesture', () => {
    if (!state.paused && !state.inputSuspended && !state.settingsOpen &&
        state.screen === 'screen-main') {
      enterPause();
    }
  });
  tracker.addEventListener('facelost', () => {
    state.faceLost = true;
    $('#track-dot').classList.remove('ok');
    $('#track-label').textContent = '얼굴 소실';
    updateFaceLostOverlay();
  });
  tracker.addEventListener('facelostlong', () => {
    sounds.warn();
  });
  tracker.addEventListener('rotationlock', (e) => {
    toast(`카메라 방향을 자동 보정했어요 (${e.detail.rotation}°)`, 3500);
  });
  tracker.addEventListener('facefound', () => {
    state.faceLost = false;
    $('#track-dot').classList.add('ok');
    $('#track-label').textContent = '추적 중';
    updateFaceLostOverlay();
  });
}

// 얼굴 소실 오버레이는 상태가 바뀔 때마다 다시 평가한다
// (다른 화면/쉬기 중에 소실돼도 메인으로 돌아오면 보이도록)
function updateFaceLostOverlay() {
  overlay('overlay-facelost',
    state.faceLost && !state.usingKeyboard &&
    state.screen === 'screen-main' && !state.paused);
}

// ===== 카메라 설정 화면 =====
let setupReady = false;

function wireSetupStatus(tracker) {
  let frames = 0;
  let lastFpsAt = performance.now();
  tracker.addEventListener('status', (e) => {
    const d = e.detail;
    frames++;
    const now = performance.now();
    if (now - lastFpsAt > 1000) {
      $('#fps-label').textContent = `${frames} fps`;
      frames = 0;
      lastFpsAt = now;
    }

    if (state.screen === 'screen-setup') {
      const faceEl = $('#st-face span');
      if (d.facePresent) {
        faceEl.textContent = '됨 ✓';
        faceEl.className = 'good';
        const distEl = $('#st-distance span');
        const okDist = d.interocularPx >= 80;
        distEl.textContent = `${Math.round(d.interocularPx)}px ${okDist ? '✓' : '(더 가까이)'}`;
        distEl.className = okDist ? 'good' : 'bad';
        const rollEl = $('#st-roll span');
        const okRoll = Math.abs(d.rollDeg) <= 25 || Math.abs(Math.abs(d.rollDeg) - 180) <= 25;
        const rotNote = d.rotation ? ` · 자동 회전 ${d.rotation}°` : '';
        rollEl.textContent = `${Math.round(d.rollDeg)}°${rotNote} ${okRoll ? '✓' : '(가능하면 카메라를 얼굴과 나란히)'}`;
        rollEl.className = okRoll ? 'good' : 'bad';
        setupReady = okDist;
      } else {
        faceEl.textContent = '안 됨';
        faceEl.className = 'bad';
        setupReady = false;
      }
      $('#btn-calib').disabled = !setupReady;
      if (setupReady) $('#setup-msg').textContent = '준비되면 아래 버튼으로 보정을 시작하세요.';
    }

    if (d.facePresent && !state.faceLost) {
      $('#track-dot').classList.add('ok');
      $('#track-label').textContent = '추적 중';
    }
  });
}

// ===== 보정 =====
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const CALIB_POSITIONS = { center: '50%', up: '10%' };
const CALIB_LABELS = { center: '가운데 점을 바라보세요', up: '위쪽 점을 바라보세요' };

async function runCalibrationFlow() {
  showScreen('screen-calib');
  const dot = $('#calib-dot');
  const all = { center: [], up: [], blinkCenter: [] };
  const order = ['center', 'up', 'center', 'up'];

  for (let i = 0; i < order.length; i++) {
    const target = order[i];
    $('#calib-label').textContent = CALIB_LABELS[target];
    $('#calib-progress').textContent = `${i + 1} / ${order.length}`;
    dot.style.top = CALIB_POSITIONS[target];
    dot.classList.remove('pulse');
    state.tts.speak(CALIB_LABELS[target]);
    await sleep(1400);
    dot.classList.add('pulse');
    const collected = await Promise.race([
      state.tracker.collectTarget(target, 3000, 700),
      sleep(9000).then(() => null),
    ]);
    dot.classList.remove('pulse');
    if (collected) {
      const key = collected.target;
      all[key].push(...collected.samples);
      if (key === 'center') all.blinkCenter.push(...collected.blinkSamples);
    } else {
      state.tracker.collecting = null;
    }
  }

  // 설정을 먼저 적용한 뒤 보정을 확정해야 약한 신호에 대한
  // dwell 연장(finishCalibration 내부)이 설정에 덮어써지지 않는다
  state.tracker.setParams({
    dwellMs: state.settings.dwellMs,
    retractEnabled: state.settings.retractEnabled,
    eyeMode: state.settings.eyeMode,
    upSensitivity: state.settings.upSensitivity,
  });
  const result = state.tracker.finishCalibration(all);
  if (!result.ok) {
    showScreen('screen-setup');
    $('#setup-msg').textContent = '⚠️ ' + result.message;
    sounds.warn();
    return;
  }
  if (result.calib.weakSignal && state.settings.dwellMs < 1000) {
    // 트래커가 올린 dwell을 설정에도 반영해 이후 슬라이더 조작에 덮어써지지 않게 한다
    state.settings.dwellMs = 1000;
    applySettingsToUI();
    saveSettings();
    toast('신호가 약해 응시 시간을 1초로 늘렸습니다', 4000);
  }
  await runPractice();
}

// ===== 연습 =====
// 1) 위를 보라고 할 때 인식되는지 (민감도)
// 2) 가만히 있으라고 할 때 오인식이 없는지 (특이도)
async function runPractice() {
  showScreen('screen-practice');
  let hits = 0;
  let falsePos = 0;
  $('#practice-result').textContent = '';

  const waitAnswer = (timeoutMs) => Promise.race([
    new Promise((resolve) => { practiceResolver = () => resolve(true); }),
    sleep(timeoutMs).then(() => false),
  ]);

  for (let i = 0; i < 4; i++) {
    $('#practice-arrow').textContent = '⬆️';
    $('#practice-prompt').textContent = '지금 위를 보세요!';
    const got = await waitAnswer(6000);
    practiceResolver = null;
    if (got) {
      hits++;
      $('#practice-result').textContent = `잘했어요! (${hits}/${i + 1})`;
    } else {
      sounds.warn();
      $('#practice-result').textContent = `인식되지 않았어요 (${hits}/${i + 1})`;
    }
    await sleep(900);
  }

  for (let i = 0; i < 2; i++) {
    $('#practice-arrow').textContent = '😌';
    $('#practice-prompt').textContent = '이번에는 가만히 정면을 보세요';
    const got = await waitAnswer(3500);
    practiceResolver = null;
    if (got) {
      falsePos++;
      sounds.warn();
      $('#practice-result').textContent = '가만히 있는데 선택이 인식됐어요';
    } else {
      $('#practice-result').textContent = '좋아요, 오인식 없음';
    }
    await sleep(700);
  }

  if (hits >= 3 && falsePos <= 1) {
    $('#practice-result').textContent = `위 응시 ${hits}/4 성공 — 시작합니다!`;
    await sleep(1400);
    enterMain();
  } else {
    $('#practice-result').textContent =
      `위 응시 ${hits}/4, 오인식 ${falsePos}/2 — 보정을 다시 하는 게 좋겠어요.`;
    await sleep(2200);
    showScreen('screen-setup');
    $('#setup-msg').textContent = '인식률이 낮았습니다. 조명/카메라 위치를 조정하고 다시 보정해 주세요.';
  }
}

// 사용 중 화면이 꺼지지 않게 (모바일/태블릿)
let wakeLock = null;
async function acquireWakeLock() {
  try {
    wakeLock = await navigator.wakeLock?.request('screen');
  } catch { /* 지원 안 되면 무시 */ }
}
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && state.screen === 'screen-main') {
    acquireWakeLock();
  }
});

function enterMain() {
  showScreen('screen-main');
  acquireWakeLock();
  const cam = $('#camera');
  const mini = $('#mini-cam');
  if (cam.srcObject) {
    mini.srcObject = cam.srcObject;
  } else {
    // 키보드 모드: 빈 비디오/게이지가 자리만 차지하지 않게 숨긴다
    mini.style.display = 'none';
    $('#gauge').style.display = 'none';
  }
  setMode('main');
}

// ===== 설정 =====
function loadSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw) Object.assign(state.settings, JSON.parse(raw));
  } catch { /* 무시 */ }
  // 손상된 저장값이 위험한 동작(예: dwell 0 = 모든 시선이 즉시 선택)을 만들지 않게 검증
  const s = state.settings;
  const num = (v, lo, hi, dflt) => (Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : dflt);
  s.dwellMs = num(s.dwellMs, 500, 2000, 700);
  s.scanPeriodMs = num(s.scanPeriodMs, 600, 4000, 1500);
  s.ttsRate = num(s.ttsRate, 0.5, 1.6, 0.95);
  s.upSensitivity = num(s.upSensitivity, 0.3, 0.7, 0.45);
  s.retractEnabled = s.retractEnabled !== false;
  if (!['both', 'left', 'right'].includes(s.eyeMode)) s.eyeMode = 'both';
  applySettingsToUI();
}

function saveSettings() {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(state.settings));
  } catch { /* 무시 */ }
}

function applySettingsToUI() {
  $('#set-dwell').value = state.settings.dwellMs;
  $('#set-dwell-val').textContent = state.settings.dwellMs + 'ms';
  $('#set-scan').value = state.settings.scanPeriodMs;
  $('#set-scan-val').textContent = (state.settings.scanPeriodMs / 1000).toFixed(1) + '초';
  // 슬라이더는 오른쪽 = 민감 이 되도록 반전해 표시
  $('#set-sens').value = Math.round(100 - state.settings.upSensitivity * 100);
  $('#set-sens-val').textContent = `${Math.round(100 - state.settings.upSensitivity * 100)}`;
  $('#set-retract').checked = state.settings.retractEnabled;
  $('#set-rate').value = state.settings.ttsRate;
  $('#set-rate-val').textContent = state.settings.ttsRate;
  $('#set-eye').value = state.settings.eyeMode;
}

function applySettings() {
  state.tts.rate = state.settings.ttsRate;
  if (state.tracker instanceof EyeTracker) {
    state.tracker.setParams({
      dwellMs: state.settings.dwellMs,
      retractEnabled: state.settings.retractEnabled,
      eyeMode: state.settings.eyeMode,
      upSensitivity: state.settings.upSensitivity,
    });
  }
  saveSettings();
}

function wireSettingsUI() {
  $('#btn-settings').addEventListener('click', () => {
    state.settingsOpen = true;
    $('#settings').classList.add('show');
    renderScanHighlight();
  });
  $('#btn-close-settings').addEventListener('click', () => {
    state.settingsOpen = false;
    $('#settings').classList.remove('show');
    restartScan();
  });
  $('#set-dwell').addEventListener('input', (e) => {
    state.settings.dwellMs = Number(e.target.value);
    $('#set-dwell-val').textContent = state.settings.dwellMs + 'ms';
    applySettings();
  });
  $('#set-scan').addEventListener('input', (e) => {
    state.settings.scanPeriodMs = Number(e.target.value);
    $('#set-scan-val').textContent = (state.settings.scanPeriodMs / 1000).toFixed(1) + '초';
    applySettings();
  });
  $('#set-sens').addEventListener('input', (e) => {
    const v = Number(e.target.value); // 오른쪽 = 민감
    state.settings.upSensitivity = (100 - v) / 100;
    $('#set-sens-val').textContent = `${v}`;
    applySettings();
  });
  $('#set-retract').addEventListener('change', (e) => {
    state.settings.retractEnabled = e.target.checked;
    applySettings();
  });
  $('#set-rate').addEventListener('input', (e) => {
    state.settings.ttsRate = Number(e.target.value);
    $('#set-rate-val').textContent = state.settings.ttsRate;
    applySettings();
  });
  $('#set-eye').addEventListener('change', (e) => {
    state.settings.eyeMode = e.target.value;
    applySettings();
    // 눈 구성이 바뀌면 기하 신호의 기준선이 달라지므로 반드시 다시 보정해야 한다
    if (state.tracker instanceof EyeTracker && state.tracker.calib) {
      state.settingsOpen = false;
      $('#settings').classList.remove('show');
      toast('사용할 눈이 바뀌어 다시 보정합니다', 3000);
      runCalibrationFlow();
    }
  });
  $('#btn-recalib').addEventListener('click', () => {
    state.settingsOpen = false;
    $('#settings').classList.remove('show');
    if (state.tracker instanceof EyeTracker) runCalibrationFlow();
    else toast('키보드 모드에서는 보정이 필요 없어요');
  });
  $('#btn-export').addEventListener('click', () => {
    const blob = new Blob([state.predictor.exportData()], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'nuneuro-학습데이터.json';
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
  $('#btn-resume').addEventListener('click', resumeFromPause);
}

// ===== 시작 =====
function checkTtsVoice() {
  const check = () => {
    if (state.tts.available && !state.tts.hasKoreanVoice) {
      $('#tts-warn').classList.add('show');
    } else {
      $('#tts-warn').classList.remove('show');
    }
  };
  setTimeout(check, 800);
  if (typeof speechSynthesis !== 'undefined') {
    speechSynthesis.addEventListener?.('voiceschanged', check);
  }
}

async function startWithCamera() {
  ensureAudio();
  showScreen('screen-setup');
  $('#setup-msg').textContent = '카메라와 얼굴 인식 모델을 불러오는 중… (최초 1회는 시간이 걸립니다)';
  const tracker = new EyeTracker(state.settings);
  state.tracker = tracker;
  try {
    await tracker.init($('#camera'));
  } catch (err) {
    // 모델 로드 실패 시 카메라 스트림을 정리하고 재시도 경로를 남긴다
    const video = $('#camera');
    video.srcObject?.getTracks?.().forEach((t) => t.stop());
    video.srcObject = null;
    state.tracker = null;
    showScreen('screen-start');
    toast('⚠️ 시작 실패: ' + (err?.message ?? err) +
      ' — 카메라 권한과 인터넷 연결(최초 1회 모델 다운로드)을 확인하고 다시 시도하세요.', 8000);
    sounds.warn();
    return;
  }
  wireTracker(tracker);
  wireSetupStatus(tracker);
  bindGazeVisuals(tracker);
  tracker.start();
  acquireWakeLock();
  if (tracker.loadStoredCalibration()) {
    $('#btn-skip-calib').style.display = '';
  }
  $('#setup-msg').textContent = '얼굴이 잘 보이는지 확인하세요.';
}

function startWithKeyboard() {
  ensureAudio();
  state.usingKeyboard = true;
  const tracker = new KeyboardTracker();
  state.tracker = tracker;
  wireTracker(tracker);
  tracker.start();
  $('#track-dot').classList.add('ok');
  $('#track-label').textContent = '키보드 모드';
  enterMain();
  toast('선택 = Space/↑ 또는 화면 탭 · 되돌리기 = Backspace 또는 길게 누르기 · 쉬기 = P', 6000);
}

function boot() {
  loadSettings();
  state.tts.rate = state.settings.ttsRate;
  checkTtsVoice();
  wireSettingsUI();
  $('#btn-start-camera').addEventListener('click', startWithCamera);
  $('#btn-start-keyboard').addEventListener('click', startWithKeyboard);
  $('#btn-calib').addEventListener('click', runCalibrationFlow);
  $('#btn-skip-calib').addEventListener('click', async () => {
    state.tracker.setParams(state.settings);
    await runPractice();
  });
  if (new URLSearchParams(location.search).get('keyboard') === '1') {
    startWithKeyboard();
  }
}

boot();

// 테스트/디버깅용 핸들 (콘솔에서 상태 확인 가능)
window.__aac = state;

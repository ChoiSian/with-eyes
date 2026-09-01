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
  settings: { dwellMs: 800, retractEnabled: true, ttsRate: 0.95, eyeMode: 'both' },
};

// ===== 소리 (이어콘) =====
let audioCtx = null;

function ensureAudio() {
  if (!audioCtx) {
    try {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    } catch { /* 소리 없이 진행 */ }
  }
}

function tone(freq, ms = 120, gainVal = 0.06) {
  if (!audioCtx) return;
  try {
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

function snapshot() {
  const c = state.composer;
  state.undoStack.push({ committed: c.committed, cho: c.cho, jung: c.jung, jong: c.jong });
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
  c.committed = c.committed.replace(/\s*\S+\s*$/, '');
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
  overlay('overlay-pause', true);
}

function resumeFromPause() {
  state.paused = false;
  overlay('overlay-pause', false);
  setMode('main');
  toast('다시 시작합니다');
}

function handlePausePattern(dir) {
  const now = Date.now();
  if (now - state.pausePatternAt > 8000) state.pausePattern = [];
  state.pausePatternAt = now;
  state.pausePattern.push(dir);
  if (state.pausePattern.length > 3) state.pausePattern.shift();
  if (state.pausePattern.join(',') === 'up,down,up') resumeFromPause();
}

function onSelect(item) {
  sounds.select();
  echo(item.label);
  switch (item.kind) {
    case 'jamo':
      snapshot();
      state.composer.input(item.jamo);
      setMode('main');
      break;
    case 'suggestion':
      snapshot();
      acceptSuggestion(item);
      setMode('main');
      break;
    case 'action':
      if (item.action === 'space') {
        snapshot();
        state.composer.input(' ');
        setMode('main');
      } else if (item.action === 'delete') {
        setMode('delete');
      } else if (item.action === 'speak') {
        setMode('confirm-speak');
      } else if (item.action === 'quick') {
        setMode('quickcat');
      } else if (item.action === 'rest') {
        enterPause();
        setMode('main');
      }
      break;
    case 'del':
      if (item.del === 'jamo') {
        snapshot();
        state.composer.backspace();
        setMode('main');
      } else if (item.del === 'word') {
        snapshot();
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
      else setMode('confirm-quick', item.text);
      break;
    case 'confirm':
      if (state.mode === 'confirm-quick') {
        if (item.yes) {
          const text = state.modeArg;
          setMode('main');
          speakText(text);
        } else {
          setMode('quickcat');
        }
      } else if (state.mode === 'confirm-speak') {
        if (item.yes) speakSentence();
        else setMode('main');
      } else if (state.mode === 'confirm-clear') {
        if (item.yes) {
          snapshot();
          state.composer.clear();
        }
        setMode('main');
      } else if (state.mode === 'post-speak') {
        if (item.yes) {
          snapshot();
          state.composer.clear();
        }
        setMode('main');
      }
      break;
  }
}

// ===== 입력 라우팅 =====
let practiceResolver = null;

function onAnswer(dir) {
  if (state.inputSuspended || state.settingsOpen) return;
  if (state.faceLost && !state.usingKeyboard) return;

  if (state.paused) {
    handlePausePattern(dir);
    return;
  }

  if (state.screen === 'screen-practice') {
    if (practiceResolver) practiceResolver(dir);
    return;
  }

  if (state.screen !== 'screen-main') return;

  (dir === 'up' ? sounds.up : sounds.down)();
  const res = state.cycle.answer(dir);
  if (res.done) onSelect(res.item);
  else render();
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
  const zoneDown = $('#gauge .zone-down');
  const fillTop = $('#band-top .fill');
  const fillBottom = $('#band-bottom .fill');
  const bandTop = $('#band-top');
  const bandBottom = $('#band-bottom');

  tracker.addEventListener('gaze', (e) => {
    const { S, state: st, zone, progress, gated, upEnter, downEnter } = e.detail;
    const max = upEnter * 1.6;
    const min = downEnter * 1.6;
    const frac = 1 - (Math.min(max, Math.max(min, S)) - min) / (max - min);
    needle.style.top = `calc(${(frac * 100).toFixed(1)}% - 5px)`;
    needle.classList.toggle('gated', gated);
    zoneUp.style.top = '0';
    zoneUp.style.height = `${(((max - upEnter) / (max - min)) * 100).toFixed(1)}%`;
    zoneDown.style.bottom = '0';
    zoneDown.style.top = 'auto';
    zoneDown.style.height = `${(((downEnter - min) / (max - min)) * 100).toFixed(1)}%`;

    const inDwell = st === 'dwell';
    fillTop.style.transform = `scaleY(${inDwell && zone === 'up' ? progress : 0})`;
    fillBottom.style.transform = `scaleY(${inDwell && zone === 'down' ? progress : 0})`;
    bandTop.classList.toggle('hot-top', zone === 'up' && (st === 'dwell' || st === 'debounce'));
    bandBottom.classList.toggle('hot-bottom', zone === 'down' && (st === 'dwell' || st === 'debounce'));
  });
}

// ===== 트래커 이벤트 배선 =====
function wireTracker(tracker) {
  tracker.addEventListener('answer', (e) => onAnswer(e.detail.dir));
  tracker.addEventListener('retract', () => onRetract());
  tracker.addEventListener('retractwarn', () => {
    sounds.warn();
    toast('계속 응시하면 방금 선택이 취소됩니다', 1500);
  });
  tracker.addEventListener('pausegesture', () => {
    if (!state.paused && state.screen === 'screen-main') enterPause();
  });
  tracker.addEventListener('facelost', () => {
    state.faceLost = true;
    $('#track-dot').classList.remove('ok');
    $('#track-label').textContent = '얼굴 소실';
    if (state.screen === 'screen-main' && !state.paused) overlay('overlay-facelost', true);
  });
  tracker.addEventListener('facelostlong', () => {
    sounds.warn();
  });
  tracker.addEventListener('facefound', () => {
    state.faceLost = false;
    $('#track-dot').classList.add('ok');
    $('#track-label').textContent = '추적 중';
    overlay('overlay-facelost', false);
  });
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
        const okRoll = Math.abs(d.rollDeg) <= 20 || Math.abs(Math.abs(d.rollDeg) - 180) <= 20;
        rollEl.textContent = `${Math.round(d.rollDeg)}° ${okRoll ? '✓' : '(카메라를 얼굴과 나란히)'}`;
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

const CALIB_POSITIONS = { center: '50%', up: '12%', down: '88%' };
const CALIB_LABELS = { center: '가운데 점을 바라보세요', up: '위쪽 점을 바라보세요', down: '아래쪽 점을 바라보세요' };

async function runCalibrationFlow() {
  showScreen('screen-calib');
  const dot = $('#calib-dot');
  const all = { center: [], up: [], down: [], blinkDown: [] };
  const order = ['center', 'up', 'down', 'center', 'up', 'down'];

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
      if (key === 'down') all.blinkDown.push(...collected.blinkSamples);
    } else {
      state.tracker.collecting = null;
    }
  }

  const result = state.tracker.finishCalibration(all);
  if (!result.ok) {
    showScreen('screen-setup');
    $('#setup-msg').textContent = '⚠️ ' + result.message;
    sounds.warn();
    return;
  }
  state.tracker.setParams(state.settings);
  if (result.calib.weakSignal) {
    toast('신호가 약해 응시 시간을 1초로 늘렸습니다', 4000);
  }
  await runPractice();
}

// ===== 연습 =====
async function runPractice() {
  showScreen('screen-practice');
  const dirs = ['up', 'down', 'up', 'down', 'down', 'up'];
  let correct = 0;
  $('#practice-result').textContent = '';
  for (let i = 0; i < dirs.length; i++) {
    const want = dirs[i];
    $('#practice-arrow').textContent = want === 'up' ? '⬆️' : '⬇️';
    $('#practice-prompt').textContent = want === 'up' ? '위를 보세요' : '아래를 보세요';
    const got = await new Promise((resolve) => { practiceResolver = resolve; });
    practiceResolver = null;
    if (got === want) {
      correct++;
      $('#practice-result').textContent = `잘했어요! (${correct}/${i + 1})`;
    } else {
      sounds.warn();
      $('#practice-result').textContent = `반대 방향이 인식됐어요 (${correct}/${i + 1})`;
    }
    await sleep(700);
  }
  if (correct >= 4) {
    $('#practice-result').textContent = `${dirs.length}개 중 ${correct}개 성공 — 시작합니다!`;
    await sleep(1400);
    enterMain();
  } else {
    $('#practice-result').textContent = `${dirs.length}개 중 ${correct}개 성공 — 보정을 다시 하는 게 좋겠어요.`;
    await sleep(2200);
    showScreen('screen-setup');
    $('#setup-msg').textContent = '인식률이 낮았습니다. 조명/카메라 위치를 조정하고 다시 보정해 주세요.';
  }
}

function enterMain() {
  showScreen('screen-main');
  const cam = $('#camera');
  if (cam.srcObject) $('#mini-cam').srcObject = cam.srcObject;
  setMode('main');
}

// ===== 설정 =====
function loadSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw) Object.assign(state.settings, JSON.parse(raw));
  } catch { /* 무시 */ }
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
    });
  }
  saveSettings();
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
  $('#set-dwell').addEventListener('input', (e) => {
    state.settings.dwellMs = Number(e.target.value);
    $('#set-dwell-val').textContent = state.settings.dwellMs + 'ms';
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
    $('#setup-msg').textContent =
      '⚠️ 카메라 또는 모델을 불러올 수 없습니다: ' + (err?.message ?? err) +
      ' — 카메라 권한을 허용했는지, 인터넷이 연결되어 있는지(최초 실행 시 모델 다운로드) 확인해 주세요.';
    sounds.warn();
    return;
  }
  wireTracker(tracker);
  wireSetupStatus(tracker);
  bindGazeVisuals(tracker);
  tracker.start();
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
  toast('↑/↓ = 선택, Backspace = 되돌리기, P = 쉬기', 5000);
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

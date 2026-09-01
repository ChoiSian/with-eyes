// Web Speech API 기반 한국어 음성 출력 (완전 로컬, 브라우저 내장 음성 사용)

export class KoreanTTS {
  constructor() {
    this.voice = null;
    this.rate = 0.95;
    this.pitch = 1.0;
    this.available = typeof speechSynthesis !== 'undefined';
    if (this.available) {
      this.#pickVoice();
      // 일부 브라우저는 목록이 비동기로 채워짐
      speechSynthesis.addEventListener?.('voiceschanged', () => this.#pickVoice());
    }
  }

  #pickVoice() {
    const voices = speechSynthesis.getVoices();
    const korean = voices.filter((v) => v.lang?.toLowerCase().startsWith('ko'));
    // localService 음성을 우선 선택 (오프라인 동작)
    this.voice =
      korean.find((v) => v.localService) ?? korean[0] ?? null;
  }

  get hasKoreanVoice() {
    return this.voice !== null;
  }

  // iOS 사파리 등은 사용자 제스처 안에서 한 번 speak해야 이후 발화가 허용된다.
  // 시작 버튼 클릭 시 호출해 잠금을 푼다.
  unlock() {
    if (!this.available || this.unlocked) return;
    try {
      const utter = new SpeechSynthesisUtterance(' ');
      utter.volume = 0;
      speechSynthesis.speak(utter);
      this.unlocked = true;
    } catch { /* 무시 */ }
  }

  // 문장을 읽는다. 완료 시 resolve.
  speak(text) {
    if (!this.available) return Promise.resolve(false);
    const cleaned = text.trim();
    if (!cleaned) return Promise.resolve(false);

    return new Promise((resolve) => {
      speechSynthesis.cancel(); // 이전 발화 중단
      const utter = new SpeechSynthesisUtterance(cleaned);
      utter.lang = 'ko-KR';
      if (this.voice) utter.voice = this.voice;
      utter.rate = this.rate;
      utter.pitch = this.pitch;
      // 음성 엔진이 콜백을 안 주는 경우에도 앱이 잠기지 않도록 워치독
      const watchdog = setTimeout(() => resolve(false), cleaned.length * 350 + 6000);
      const done = (ok) => {
        clearTimeout(watchdog);
        resolve(ok);
      };
      utter.onend = () => done(true);
      utter.onerror = () => done(false);
      speechSynthesis.speak(utter);
    });
  }

  stop() {
    if (this.available) speechSynthesis.cancel();
  }
}

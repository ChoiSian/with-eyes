// 빠른 말: 자주 쓰는 완성 문장. 카테고리 → 문장 2단계로 선택한다.
// 응급/고빈도 요구는 철자 입력 없이 몇 번의 시선 응답만으로 말할 수 있어야 한다.

export const QUICK_PHRASES = [
  {
    category: '아파요/몸',
    weight: 10,
    phrases: [
      { text: '아파요', weight: 10 },
      { text: '숨이 답답해요', weight: 9 },
      { text: '석션 해주세요', weight: 8 },
      { text: '가려워요, 긁어 주세요', weight: 6 },
      { text: '어지러워요', weight: 5 },
      { text: '속이 안 좋아요', weight: 5 },
    ],
  },
  {
    category: '요청',
    weight: 9,
    phrases: [
      { text: '화장실 가고 싶어요', weight: 10 },
      { text: '물 주세요', weight: 9 },
      { text: '자세 바꿔 주세요', weight: 9 },
      { text: '추워요, 이불 덮어 주세요', weight: 6 },
      { text: '더워요', weight: 5 },
      { text: '불 꺼 주세요', weight: 4 },
      { text: '불 켜 주세요', weight: 4 },
      { text: '티비 틀어 주세요', weight: 4 },
    ],
  },
  {
    category: '대답',
    weight: 8,
    phrases: [
      { text: '네, 맞아요', weight: 10 },
      { text: '아니에요', weight: 9 },
      { text: '잘 모르겠어요', weight: 6 },
      { text: '잠깐만요', weight: 6 },
      { text: '다시 말해 주세요', weight: 5 },
      { text: '조용히 해 주세요', weight: 3 },
    ],
  },
  {
    category: '마음/사람',
    weight: 6,
    phrases: [
      { text: '고마워요', weight: 9 },
      { text: '사랑해요', weight: 8 },
      { text: '미안해요', weight: 6 },
      { text: '좀 쉬고 싶어요', weight: 7 },
      { text: '가족 불러 주세요', weight: 6 },
      { text: '의사 선생님 불러 주세요', weight: 6 },
    ],
  },
];

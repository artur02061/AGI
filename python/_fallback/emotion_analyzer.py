"""
Python Fallback: EmotionAnalyzer

Зеркалит API Rust kristina_core.EmotionAnalyzer:
- analyze(text) → str ("positive"|"negative"|"curious"|"neutral")
- analyze_detailed(text) → (emotion, confidence, found_patterns)
"""

import re
from typing import List, Tuple


class EmotionAnalyzer:
    """Анализ эмоций — Python fallback для Rust Aho-Corasick"""

    def __init__(self):
        self._positive = [
            "спасибо", "отлично", "супер", "хорошо", "круто", "молодец",
            "замечательно", "класс", "здорово", "прекрасно", "великолепно",
            "восхитительно", "браво", "ура", "обожаю", "нравится", "люблю",
            "рад", "рада", "счастлив", "доволен", "довольна", "благодарю",
            "спс", "пасиб", "awesome", "nice", "great", "thanks", "cool",
            "👍", "😊", "😃", "❤️", "🎉", "💪", "🔥",
        ]
        self._negative = [
            "не работает", "ошибка", "плохо", "не получается", "проблема",
            "сломал", "баг", "глючит", "тормозит", "зависает", "ужасно",
            "отстой", "бесит", "раздражает", "не понимаю", "запутал",
            "неправильно", "некорректно", "фигня", "дерьмо", "не так",
            "broken", "error", "bug", "wrong", "bad", "fail",
            "😞", "😡", "😤", "💔", "😢", "🤬",
        ]
        self._curious = [
            "как", "что", "почему", "зачем", "когда", "где", "кто",
            "сколько", "можно ли", "а если", "расскажи", "объясни",
            "подскажи", "помоги", "покажи", "научи", "интересно",
            "how", "what", "why", "when", "where", "who",
            "🤔", "❓", "🧐",
        ]

        # Компилируем regex для быстрого мультипаттерна
        self._pos_re = self._build_pattern(self._positive)
        self._neg_re = self._build_pattern(self._negative)
        self._cur_re = self._build_pattern(self._curious)

    @staticmethod
    def _build_pattern(words: List[str]) -> re.Pattern:
        escaped = [re.escape(w) for w in words]
        return re.compile("|".join(escaped), re.IGNORECASE)

    def _count_matches(self, pattern: re.Pattern, text: str) -> List[str]:
        return pattern.findall(text)

    def analyze(self, text: str) -> str:
        pos_matches = self._count_matches(self._pos_re, text)
        neg_matches = self._count_matches(self._neg_re, text)
        cur_matches = self._count_matches(self._cur_re, text)

        pos_count = len(pos_matches)
        neg_count = len(neg_matches)
        cur_count = len(cur_matches)

        # Вопросительный знак — сильный сигнал
        if "?" in text:
            cur_count += 2

        if pos_count == 0 and neg_count == 0 and cur_count == 0:
            return "neutral"

        if neg_count > pos_count and neg_count >= cur_count:
            return "negative"
        elif pos_count > neg_count and pos_count >= cur_count:
            return "positive"
        elif cur_count > 0:
            return "curious"
        else:
            return "neutral"

    def analyze_detailed(self, text: str) -> Tuple[str, float, List[str]]:
        pos_matches = self._count_matches(self._pos_re, text)
        neg_matches = self._count_matches(self._neg_re, text)
        cur_matches = self._count_matches(self._cur_re, text)

        total = len(pos_matches) + len(neg_matches) + len(cur_matches)

        if total == 0:
            return ("neutral", 0.5, [])

        emotion = self.analyze(text)
        dominant_count = {
            "positive": len(pos_matches),
            "negative": len(neg_matches),
            "curious": len(cur_matches),
        }.get(emotion, 0)

        confidence = min(1.0, dominant_count / total) if total > 0 else 0.5

        all_matches = pos_matches + neg_matches + cur_matches
        return (emotion, confidence, all_matches)

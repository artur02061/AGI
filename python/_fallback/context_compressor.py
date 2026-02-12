"""
Python Fallback: ContextCompressor

Зеркалит API Rust kristina_core.ContextCompressor:
- compress_conversation(messages: List[(role, content, ts)]) → str
- extract_key_points(text) → List[str]
- summarize_episodes(episodes: List[(ts, user_input, importance)], max_length) → str
- estimate_tokens(text) → int
- truncate_to_tokens(text, max_tokens) → str
"""

import re
from typing import List, Tuple


class ContextCompressor:
    """Сжатие контекста — Python fallback для Rust"""

    IMPORTANT_WORDS = [
        "важно", "главное", "нужно", "проблема", "решение",
        "ошибка", "успешно", "не работает", "помоги", "критично",
        "срочно", "обязательно", "ключевой", "основной", "результат",
        "вывод", "итог", "причина", "следствие", "вопрос",
    ]

    def __init__(self, compression_ratio: float = 0.3):
        self.compression_ratio = compression_ratio
        self._important_re = re.compile(
            "|".join(re.escape(w) for w in self.IMPORTANT_WORDS),
            re.IGNORECASE,
        )

    def compress_conversation(self, messages: List[Tuple[str, str, str]]) -> str:
        """Сжимает историю. Вход: [(role, content, timestamp)]"""
        if not messages:
            return ""

        recent = messages[-10:]  # Последние 10
        parts = []
        for role, content, _ts in recent:
            if len(content) > 100:
                # Безопасная обрезка по символам
                truncated = content[:97] + "..."
            else:
                truncated = content
            parts.append(f"{role}: {truncated}")
        return "\n".join(parts)

    def extract_key_points(self, text: str) -> List[str]:
        """Извлекает ключевые предложения"""
        sentences = re.split(r'[.!?\n]+', text)

        scored = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            count = len(self._important_re.findall(s))
            if count > 0:
                scored.append((s, count))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored[:3]]

    def summarize_episodes(
        self,
        episodes: List[Tuple[str, str, int]],
        max_length: int = 500,
    ) -> str:
        """Суммаризует эпизоды. Вход: [(timestamp, user_input, importance)]"""
        if not episodes:
            return "Нет данных в памяти"

        sorted_eps = sorted(episodes, key=lambda x: x[2], reverse=True)

        parts = []
        current_length = 0

        for timestamp, user_input, _importance in sorted_eps[:5]:
            date = timestamp[:10] if len(timestamp) >= 10 else timestamp
            preview = user_input[:50]
            snippet = f"[{date}] {preview}"

            if current_length + len(snippet) > max_length:
                break

            parts.append(snippet)
            current_length += len(snippet)

        return "\n".join(parts)

    def estimate_tokens(self, text: str) -> int:
        """BPE-эвристика: ~4 chars/token EN, ~2 chars/token RU"""
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        non_ascii = len(text) - ascii_chars
        return (ascii_chars // 4) + (non_ascii // 2) + 1

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Обрезает текст до N токенов"""
        estimated = self.estimate_tokens(text)
        if estimated <= max_tokens:
            return text
        ratio = max_tokens / estimated
        target_chars = int(len(text) * ratio)
        return text[:target_chars]

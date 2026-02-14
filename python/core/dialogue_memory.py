"""
Кристина 7.5 — DialogueMemory (Безлимитная память диалога)

ПРОБЛЕМА:
  Раньше Кристина помнила только 3 последних сообщения в контексте.
  Через 10 минут — принудительный сброс темы.
  Пользователь говорит "как я говорил в начале" — а Кристина не помнит.

РЕШЕНИЕ — 3 механизма:

  1. SlidingSummary (Скользящее резюме)
     После каждых N сообщений старая часть сжимается в резюме.
     Резюме растёт ЛОГАРИФМИЧЕСКИ — 10 и 1000 сообщений дают
     примерно одинаковый размер (~500 токенов).

  2. SessionIndex (Индекс сессии)
     Каждое сообщение хранится с эмбеддингом в RAM.
     Семантический поиск по ЛЮБОМУ моменту разговора за O(N).
     При 1000 сообщений × 128 dim = ~500 KB — ничтожно.

  3. KeyFacts (Ключевые факты)
     Автоматическое извлечение имён, чисел, решений, тем.
     Факты не сжимаются и живут всю сессию.

РЕЗУЛЬТАТ:
  - Кристина помнит ВСЁ из текущей сессии
  - Контекст для LLM: ~1800 токенов (было ~200-300)
  - Память безлимитна (резюме логарифмическое)
"""

import math
import re
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from utils.logging import get_logger
import config

logger = get_logger("dialogue_memory")


# ═══════════════════════════════════════════════════════════════
#               ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════

def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Косинусное сходство двух векторов"""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return dot / (norm1 * norm2)


def _estimate_tokens(text: str) -> int:
    """Оценка количества токенов (heuristic)"""
    if not text:
        return 0
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    cyrillic_chars = len(text) - ascii_chars
    return ascii_chars // 4 + cyrillic_chars // 2


# ═══════════════════════════════════════════════════════════════
#               ИЗВЛЕЧЕНИЕ КЛЮЧЕВЫХ ФАКТОВ
# ═══════════════════════════════════════════════════════════════

# Паттерны для извлечения фактов из текста
_FACT_PATTERNS = [
    # Имена (с большой буквы после "меня зовут", "я —" и т.д.)
    (r'(?:меня зовут|я\s+—|я\s*-)\s+([А-ЯЁA-Z][а-яёa-z]+)', 'name'),
    # Числа с контекстом
    (r'(\d+)\s*(?:лет|года|год)', 'age'),
    (r'(\d+)\s*(?:рублей|руб|₽|\$|долларов|евро|€)', 'money'),
    # Решения / выводы
    (r'(?:решили?|договорились|итого|вывод)[:\s]+(.{10,80})', 'decision'),
    # Города / страны (после "живу в", "из")
    (r'(?:живу в|из|в городе)\s+([А-ЯЁA-Z][а-яёa-z]+)', 'location'),
    # Профессия
    (r'(?:работаю|я\s+(?:по профессии|программист|дизайнер|инженер|учитель|врач|студент))\s*([^\.,]{3,40})', 'profession'),
]

_ANAPHORA_PATTERNS = [
    "как я говорил", "как мы обсуждали", "помнишь",
    "в начале разговора", "раньше я", "ранее",
    "вернёмся к", "насчёт того", "по поводу",
    "об этом же", "продолжим", "как я уже",
    "мы уже", "я уже говорил", "ты уже",
]


def _extract_facts(text: str) -> List[Dict]:
    """Извлекает ключевые факты из текста"""
    facts = []
    for pattern, fact_type in _FACT_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            facts.append({
                'type': fact_type,
                'value': match.group(1).strip() if match.lastindex else match.group(0).strip(),
                'source': text[:60],
            })
    return facts


def _has_anaphora(text: str) -> bool:
    """Проверяет наличие анафорических ссылок (отсылки к прошлому)"""
    text_lower = text.lower()
    return any(p in text_lower for p in _ANAPHORA_PATTERNS)


def _extractive_summarize(messages: List[Dict], max_chars: int = 600) -> str:
    """
    Extractive summarization БЕЗ LLM.

    Берём самые информативные предложения:
    - С именами, числами, решениями
    - С вопросами (?)
    - Первое и последнее сообщение сессии
    """
    if not messages:
        return ""

    scored_sentences = []

    for msg in messages:
        text = msg.get('user', '') or ''
        if not text:
            continue

        # Оцениваем информативность
        score = 0.0

        # Числа — важно
        if re.search(r'\d+', text):
            score += 2.0

        # Имена собственные (слова с большой буквы)
        caps = re.findall(r'[А-ЯЁA-Z][а-яёa-z]{2,}', text)
        score += len(caps) * 1.5

        # Вопросы — важно
        if '?' in text:
            score += 1.5

        # Решения / выводы
        if any(w in text.lower() for w in ['решили', 'итого', 'вывод', 'договорились', 'нужно', 'план']):
            score += 3.0

        # Длина (средняя предпочтительнее)
        if 20 < len(text) < 200:
            score += 1.0

        scored_sentences.append((score, text[:150], msg.get('role', 'user')))

    # Сортируем по скору, берём лучшие
    scored_sentences.sort(key=lambda x: x[0], reverse=True)

    parts = []
    total_chars = 0
    for score, text, role in scored_sentences:
        if total_chars + len(text) > max_chars:
            break
        prefix = "П" if role == "user" else "К"
        parts.append(f"{prefix}: {text}")
        total_chars += len(text) + 3

    if not parts:
        # Fallback: берём первое и последнее
        if messages:
            first = messages[0].get('user', '')[:100]
            if first:
                parts.append(f"Начали с: {first}")
            if len(messages) > 1:
                last = messages[-1].get('user', '')[:100]
                if last:
                    parts.append(f"Последнее: {last}")

    return "; ".join(parts)


# ═══════════════════════════════════════════════════════════════
#               SESSION MESSAGE
# ═══════════════════════════════════════════════════════════════

@dataclass
class SessionMessage:
    """Одно сообщение в сессии"""
    role: str              # 'user' или 'assistant'
    text: str              # Полный текст сообщения
    embedding: Optional[List[float]]  # Вектор (128-dim)
    timestamp: datetime
    index: int             # Порядковый номер в сессии
    facts: List[Dict] = field(default_factory=list)  # Извлечённые факты


# ═══════════════════════════════════════════════════════════════
#               SLIDING SUMMARY
# ═══════════════════════════════════════════════════════════════

class SlidingSummary:
    """
    Скользящее резюме диалога.

    Каждые `window_size` сообщений старая часть сжимается в резюме.
    Резюме растёт логарифмически:
      - Первое сжатие: 6 сообщений → 150 слов
      - Второе: резюме + 6 сообщений → 200 слов
      - Третье: резюме + 6 сообщений → 230 слов (не 300!)
    """

    def __init__(
        self,
        window_size: int = 6,
        max_summary_tokens: int = 500,
        llm_summarizer=None,
    ):
        self.window_size = window_size
        self.max_summary_tokens = max_summary_tokens
        self._llm_summarizer = llm_summarizer  # async callable или None

        self.summary_text: str = ""
        self.summary_tokens: int = 0
        self.topic_history: List[str] = []
        self.compression_count: int = 0
        self.total_messages_compressed: int = 0

    def needs_compression(self, n_recent_messages: int) -> bool:
        """Пора ли сжимать?"""
        return n_recent_messages >= self.window_size

    async def compress(self, messages: List[Dict]) -> str:
        """
        Сжимает messages в резюме.

        Args:
            messages: [{'role': 'user'/'assistant', 'text': '...'}]

        Returns:
            Обновлённое резюме
        """
        if not messages:
            return self.summary_text

        self.compression_count += 1
        self.total_messages_compressed += len(messages)

        # Пробуем LLM-суммаризацию
        if self._llm_summarizer:
            try:
                new_summary = await self._llm_compress(messages)
                if new_summary and len(new_summary) > 20:
                    self.summary_text = new_summary
                    self.summary_tokens = _estimate_tokens(new_summary)
                    logger.info(
                        f"📝 SlidingSummary: LLM compress #{self.compression_count} "
                        f"({len(messages)} msgs → {self.summary_tokens} tokens)"
                    )
                    return self.summary_text
            except Exception as e:
                logger.debug(f"LLM summarization failed, using extractive: {e}")

        # Fallback: extractive summarization
        new_summary = self._extractive_compress(messages)
        self.summary_text = new_summary
        self.summary_tokens = _estimate_tokens(new_summary)

        logger.info(
            f"📝 SlidingSummary: extractive compress #{self.compression_count} "
            f"({len(messages)} msgs → {self.summary_tokens} tokens)"
        )
        return self.summary_text

    async def _llm_compress(self, messages: List[Dict]) -> str:
        """Сжимает через LLM (gemma3:4b)"""
        # Формируем текст для LLM
        dialogue_parts = []
        for msg in messages:
            role = "П" if msg['role'] == 'user' else "К"
            dialogue_parts.append(f"{role}: {msg['text'][:200]}")
        dialogue_text = "\n".join(dialogue_parts)

        prev_context = ""
        if self.summary_text:
            prev_context = f"\nПРЕДЫДУЩЕЕ РЕЗЮМЕ:\n{self.summary_text}\n"

        prompt = f"""Сожми диалог в краткое резюме (макс 4-5 предложений).
{prev_context}
ПРАВИЛА:
- Сохрани ВСЕ факты: имена, числа, решения, просьбы
- Объедини предыдущее резюме с новыми сообщениями
- Пиши кратко, по делу
- НЕ добавляй ничего от себя

НОВЫЕ СООБЩЕНИЯ:
{dialogue_text}

РЕЗЮМЕ:"""

        return await self._llm_summarizer(prompt)

    def _extractive_compress(self, messages: List[Dict]) -> str:
        """Extractive compression без LLM"""
        msg_dicts = [
            {'user': m['text'], 'role': m['role']}
            for m in messages
        ]
        new_part = _extractive_summarize(msg_dicts, max_chars=400)

        if self.summary_text:
            # Объединяем старое резюме + новое
            # Но ограничиваем общий размер
            max_old = max(200, self.max_summary_tokens * 2 - len(new_part))
            old_trimmed = self.summary_text[:max_old]
            return f"{old_trimmed} | {new_part}"
        else:
            return new_part

    def get_summary(self) -> str:
        """Возвращает текущее резюме"""
        return self.summary_text


# ═══════════════════════════════════════════════════════════════
#               SESSION INDEX (семантический поиск)
# ═══════════════════════════════════════════════════════════════

class SessionIndex:
    """
    In-memory индекс всех сообщений сессии с эмбеддингами.

    Позволяет найти любое сообщение по семантике:
    "найди где мы говорили о Python" → находит сообщения про Python

    Память: 1000 сообщений × 128 dim × 4 bytes = ~500 KB
    Скорость: cosine similarity для 1000 векторов < 1ms
    """

    def __init__(self, sentence_encoder=None):
        self._messages: List[SessionMessage] = []
        self._encoder = sentence_encoder  # SentenceEmbeddings.encode()
        self._msg_counter = 0

    def add(self, role: str, text: str) -> SessionMessage:
        """Добавляет сообщение в индекс"""
        embedding = None
        if self._encoder:
            try:
                embedding = self._encoder(text)
            except Exception as e:
                logger.debug(f"Embedding encode failed: {e}")

        facts = _extract_facts(text)

        msg = SessionMessage(
            role=role,
            text=text,
            embedding=embedding,
            timestamp=datetime.now(),
            index=self._msg_counter,
            facts=facts,
        )

        self._messages.append(msg)
        self._msg_counter += 1
        return msg

    def search(self, query: str, top_k: int = 3, min_score: float = 0.3) -> List[Tuple[SessionMessage, float]]:
        """
        Семантический поиск по всем сообщениям сессии.

        Returns:
            [(SessionMessage, score), ...] отсортированные по скору
        """
        if not self._messages or not self._encoder:
            return []

        try:
            query_embedding = self._encoder(query)
        except Exception as e:
            logger.debug(f"Query embedding failed: {e}")
            return []

        if not query_embedding:
            return []

        scored = []
        for msg in self._messages:
            if msg.embedding is None:
                continue

            score = _cosine_similarity(query_embedding, msg.embedding)

            # Бонус за свежесть (новые сообщения чуть важнее)
            recency_bonus = 0.05 * (msg.index / max(self._msg_counter, 1))
            score += recency_bonus

            if score >= min_score:
                scored.append((msg, score))

        # Сортируем по скору (убывание)
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_by_facts(self, fact_type: str = None) -> List[SessionMessage]:
        """Поиск по извлечённым фактам"""
        results = []
        for msg in self._messages:
            for fact in msg.facts:
                if fact_type is None or fact['type'] == fact_type:
                    results.append(msg)
                    break
        return results

    def get_all(self) -> List[SessionMessage]:
        return list(self._messages)

    def get_recent(self, n: int) -> List[SessionMessage]:
        return self._messages[-n:] if self._messages else []

    @property
    def size(self) -> int:
        return len(self._messages)


# ═══════════════════════════════════════════════════════════════
#               DIALOGUE MEMORY (объединяющий фасад)
# ═══════════════════════════════════════════════════════════════

class DialogueMemory:
    """
    Безлимитная память диалога.

    Объединяет:
    - SlidingSummary: скользящее резюме всей сессии
    - SessionIndex: семантический поиск по любому сообщению
    - KeyFacts: извлечённые факты (имена, числа, решения)

    Использование:
        dm = DialogueMemory(sentence_encoder=embeddings.encode)
        dm.add('user', 'Привет, меня зовут Артур')
        dm.add('assistant', 'Привет, Артур!')
        ...
        context = await dm.build_context('напомни как меня зовут')
    """

    def __init__(
        self,
        sentence_encoder=None,
        llm_summarizer=None,
        window_size: int = None,
        max_summary_tokens: int = None,
        session_search_top_k: int = None,
        session_search_threshold: float = None,
    ):
        cfg = config.config

        _window = window_size or getattr(cfg, 'sliding_summary_window', 6)
        _max_tokens = max_summary_tokens or getattr(cfg, 'sliding_summary_max_tokens', 500)
        _top_k = session_search_top_k or getattr(cfg, 'session_search_top_k', 3)
        _threshold = session_search_threshold or getattr(cfg, 'session_search_threshold', 0.3)

        self._summary = SlidingSummary(
            window_size=_window,
            max_summary_tokens=_max_tokens,
            llm_summarizer=llm_summarizer,
        )

        self._index = SessionIndex(sentence_encoder=sentence_encoder)
        self._search_top_k = _top_k
        self._search_threshold = _threshold

        # Буфер свежих сообщений (полный текст, для передачи в контекст)
        self._recent_buffer: List[Dict] = []
        self._recent_max = _window  # Столько же, сколько окно сжатия

        # Все факты сессии
        self._session_facts: List[Dict] = []

        # Статистика
        self._total_messages = 0
        self._total_compressions = 0
        self._total_searches = 0

        self._lock = threading.Lock()

        logger.info(
            f"🧠 DialogueMemory: window={_window}, "
            f"max_summary={_max_tokens}tok, "
            f"search_top_k={_top_k}"
        )

    def add(self, role: str, text: str):
        """
        Добавляет сообщение в память диалога.

        Вызывается из orchestrator._save_to_memory() для каждого сообщения.
        """
        with self._lock:
            self._total_messages += 1

            # Добавляем в индекс (с эмбеддингом)
            msg = self._index.add(role, text)

            # Сохраняем факты
            if msg.facts:
                self._session_facts.extend(msg.facts)

            # Добавляем в буфер свежих
            self._recent_buffer.append({
                'role': role,
                'text': text,
                'timestamp': datetime.now(),
            })

    async def maybe_compress(self):
        """
        Проверяет, нужно ли сжатие, и выполняет его.
        Вызывается после add() из оркестратора.
        """
        with self._lock:
            needs = self._summary.needs_compression(len(self._recent_buffer))

        if not needs:
            return

        with self._lock:
            # Забираем старые сообщения из буфера, оставляя последние 2
            to_compress = self._recent_buffer[:-2] if len(self._recent_buffer) > 2 else []
            if to_compress:
                self._recent_buffer = self._recent_buffer[-2:]
                self._total_compressions += 1

        if to_compress:
            await self._summary.compress(to_compress)

    def search_session(self, query: str) -> List[Tuple[SessionMessage, float]]:
        """Семантический поиск по всем сообщениям сессии"""
        self._total_searches += 1
        return self._index.search(
            query,
            top_k=self._search_top_k,
            min_score=self._search_threshold,
        )

    def get_recent_messages(self, n: int = 6) -> List[Dict]:
        """Возвращает последние N сообщений (полный текст)"""
        with self._lock:
            return list(self._recent_buffer[-n:])

    def get_summary(self) -> str:
        """Возвращает текущее скользящее резюме"""
        return self._summary.get_summary()

    def get_session_facts(self) -> List[Dict]:
        """Возвращает все извлечённые факты сессии"""
        return list(self._session_facts)

    def has_anaphora(self, text: str) -> bool:
        """Проверяет, есть ли ссылки на прошлое в тексте"""
        return _has_anaphora(text)

    async def build_context(self, user_input: str, max_tokens: int = 1800) -> str:
        """
        Строит полный контекст из всех источников памяти.

        Бюджет ~1800 токенов:
          - Скользящее резюме: ~500 токенов
          - Факты сессии: ~100 токенов
          - Релевантные сообщения (поиск): ~300 токенов
          - Последние сообщения (полный текст): ~600 токенов
          - Запас: ~300 токенов

        Returns:
            Строка контекста для LLM
        """
        parts = []
        used_tokens = 0

        # 1. Скользящее резюме сессии
        summary = self.get_summary()
        if summary:
            summary_tokens = _estimate_tokens(summary)
            budget = min(summary_tokens, 500)
            trimmed = summary[:budget * 3]  # ~3 chars per token
            parts.append(f"[Контекст сессии ({self._total_messages} сообщений)]: {trimmed}")
            used_tokens += _estimate_tokens(trimmed)

        # 2. Ключевые факты
        facts = self.get_session_facts()
        if facts:
            unique_facts = {}
            for f in facts:
                key = f"{f['type']}:{f['value']}"
                unique_facts[key] = f
            fact_strs = [f"{f['type']}: {f['value']}" for f in unique_facts.values()]
            facts_text = "; ".join(fact_strs[:10])  # Максимум 10 фактов
            if _estimate_tokens(facts_text) < 150:
                parts.append(f"[Факты]: {facts_text}")
                used_tokens += _estimate_tokens(facts_text)

        # 3. Семантический поиск по сессии (если есть ссылки на прошлое)
        if self._index.size > 3:
            # Всегда ищем для контекста (не только при анафоре)
            search_results = self.search_session(user_input)
            if search_results:
                search_parts = []
                for msg, score in search_results:
                    # Пропускаем совсем недавние (они и так в recent)
                    if msg.index >= self._total_messages - 4:
                        continue
                    role = "П" if msg.role == "user" else "К"
                    snippet = msg.text[:150]
                    search_parts.append(f"  [{msg.index}] {role}: {snippet}")

                if search_parts:
                    relevance_text = "\n".join(search_parts)
                    if used_tokens + _estimate_tokens(relevance_text) < max_tokens - 600:
                        parts.append(f"[Из ранее в разговоре]:\n{relevance_text}")
                        used_tokens += _estimate_tokens(relevance_text)

        # 4. Последние сообщения (полный текст — самый важный контекст)
        recent = self.get_recent_messages(n=6)
        if recent:
            recent_parts = []
            for msg in recent:
                role = "Пользователь" if msg['role'] == 'user' else "Кристина"
                text = msg['text'][:250]
                recent_parts.append(f"{role}: {text}")
            recent_text = "\n".join(recent_parts)
            parts.append(recent_text)

        return "\n\n".join(parts) if parts else ""

    def get_stats(self) -> Dict:
        """Статистика модуля"""
        return {
            'total_messages': self._total_messages,
            'index_size': self._index.size,
            'summary_tokens': self._summary.summary_tokens,
            'compressions': self._total_compressions,
            'searches': self._total_searches,
            'facts_count': len(self._session_facts),
            'recent_buffer': len(self._recent_buffer),
        }

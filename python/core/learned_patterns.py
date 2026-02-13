"""
Кристина 7.0 — LearnedPatterns (Самообучающийся мозг)

ПРИНЦИП:
  LLM = учитель. Каждый раз, когда LLM решает задачу, Кристина
  ЗАПОМИНАЕТ решение как алгоритм. Второй раз LLM не нужен.

  Это НЕ нейросеть. Это растущая база паттернов:
    "текст запроса" → {intent, agent, tool, args_template, response_template}

  Со временем покрытие растёт, и LLM вызывается всё реже.

АРХИТЕКТУРА:
  1. RoutingPattern  — "какой запрос → какой инструмент"
  2. ResponsePattern — "результат инструмента → как ответить пользователю"
  3. SlotPattern     — "как извлечь аргументы из текста"

ХРАНЕНИЕ:
  SQLite с FTS5 (полнотекстовый поиск) — быстро, персистентно, без зависимостей.
"""

import sqlite3
import json
import re
import time
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from collections import defaultdict

from utils.logging import get_logger
import config

logger = get_logger("learned_patterns")


class LearnedPatterns:
    """
    Самообучающаяся база паттернов Кристины.

    Принцип работы:
    1. LLM решает задачу (routing, response, slot extraction)
    2. Кристина ЗАПИСЫВАЕТ решение как паттерн
    3. При следующем похожем запросе — ищет паттерн СНАЧАЛА
    4. Если нашла (confidence >= порога) → отвечает БЕЗ LLM
    5. Если не нашла → спрашивает LLM → записывает новый паттерн

    Паттерны УСИЛИВАЮТСЯ при повторном успехе и
    ОСЛАБЛЯЮТСЯ при ошибках (пользователь недоволен).
    """

    def __init__(self, db_path: Path = None):
        self._db_path = db_path or (config.config.data_dir / "learned_patterns.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        self._create_tables()

        # Кэш часто используемых паттернов (в RAM для скорости)
        self._hot_cache: Dict[str, Dict] = {}
        self._cache_ttl = 300  # 5 минут

        stats = self.get_stats()
        logger.info(
            f"🧠 LearnedPatterns: routing={stats['routing']}, "
            f"response={stats['response']}, slots={stats['slots']}"
        )

    def _create_tables(self):
        """Создаёт таблицы для хранения паттернов"""
        cur = self._conn.cursor()

        # ── Routing patterns: запрос → intent + agent ──
        cur.execute("""
            CREATE TABLE IF NOT EXISTS routing_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT NOT NULL,
                keywords TEXT NOT NULL,
                intent TEXT NOT NULL,
                agent TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                successes INTEGER DEFAULT 1,
                failures INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                last_used REAL NOT NULL,
                source TEXT DEFAULT 'llm'
            )
        """)

        # ── Response patterns: intent + result_type → response template ──
        cur.execute("""
            CREATE TABLE IF NOT EXISTS response_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intent TEXT NOT NULL,
                result_type TEXT NOT NULL,
                template TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                successes INTEGER DEFAULT 1,
                failures INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                last_used REAL NOT NULL
            )
        """)

        # ── Slot patterns: intent → regex для извлечения аргументов ──
        cur.execute("""
            CREATE TABLE IF NOT EXISTS slot_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intent TEXT NOT NULL,
                slot_name TEXT NOT NULL,
                regex_pattern TEXT NOT NULL,
                slot_position INTEGER DEFAULT 0,
                examples TEXT DEFAULT '[]',
                confidence REAL DEFAULT 1.0,
                successes INTEGER DEFAULT 1,
                failures INTEGER DEFAULT 0,
                created_at REAL NOT NULL
            )
        """)

        # ── FTS5 для быстрого полнотекстового поиска по паттернам ──
        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS routing_fts
            USING fts5(keywords, content=routing_patterns, content_rowid=id)
        """)

        # Индексы
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_routing_intent ON routing_patterns(intent)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_routing_confidence ON routing_patterns(confidence DESC)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_response_intent ON response_patterns(intent)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_slots_intent ON slot_patterns(intent)
        """)

        self._conn.commit()

    # ═══════════════════════════════════════════════════════════════
    #                    ОБУЧЕНИЕ (ЗАПИСЬ ПАТТЕРНОВ)
    # ═══════════════════════════════════════════════════════════════

    def learn_routing(
        self,
        user_input: str,
        intent: str,
        agent: str,
        source: str = "llm",
    ):
        """
        Записывает паттерн роутинга: "такой запрос → такой intent/agent".

        Вызывается ПОСЛЕ каждого успешного LLM-роутинга.
        source: 'llm' (учитель) | 'user' (ручная коррекция) | 'rule' (из правил)
        """
        keywords = self._extract_keywords(user_input)

        # Проверяем: может уже есть похожий паттерн?
        existing = self._find_similar_routing(keywords, intent)
        if existing:
            # Усиливаем существующий
            self._reinforce_routing(existing["id"])
            return

        now = time.time()
        cur = self._conn.cursor()
        cur.execute("""
            INSERT INTO routing_patterns
            (pattern, keywords, intent, agent, confidence, created_at, last_used, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_input, keywords, intent, agent, 1.0, now, now, source))

        rowid = cur.lastrowid

        # Обновляем FTS индекс
        cur.execute("""
            INSERT INTO routing_fts (rowid, keywords)
            VALUES (?, ?)
        """, (rowid, keywords))

        self._conn.commit()
        logger.debug(f"📝 Learned routing: '{user_input[:50]}' → {intent} ({agent})")

    def learn_response(
        self,
        intent: str,
        tool_result: str,
        final_response: str,
    ):
        """
        Записывает паттерн ответа: "для такого intent с таким результатом → такой ответ".

        Вызывается ПОСЛЕ каждой успешной синтезации ответа LLM.
        """
        result_type = self._classify_result(tool_result)

        # Извлекаем шаблон из ответа LLM
        template = self._extract_template(intent, tool_result, final_response)

        existing = self._find_similar_response(intent, result_type)
        if existing:
            self._reinforce_response(existing["id"])
            return

        now = time.time()
        self._conn.execute("""
            INSERT INTO response_patterns
            (intent, result_type, template, created_at, last_used)
            VALUES (?, ?, ?, ?, ?)
        """, (intent, result_type, template, now, now))
        self._conn.commit()

        logger.debug(f"📝 Learned response: {intent}/{result_type}")

    def learn_slots(
        self,
        intent: str,
        user_input: str,
        extracted_args: Dict[str, Any],
    ):
        """
        Записывает паттерны извлечения аргументов.

        Пример: intent=create_file, input="создай файл wishes.txt с пожеланиями"
        → slot "filepath": regex=r'файл\s+([\w.]+)', value="wishes.txt"
        → slot "content": regex=r'с\s+(.+)$', value="пожеланиями"
        """
        for slot_name, slot_value in extracted_args.items():
            if not slot_value or not isinstance(slot_value, str):
                continue

            # Генерируем regex из примера
            regex = self._generate_slot_regex(user_input, slot_value, slot_name)
            if not regex:
                continue

            # Проверяем дубликат
            existing = self._conn.execute("""
                SELECT id FROM slot_patterns
                WHERE intent = ? AND slot_name = ? AND regex_pattern = ?
            """, (intent, slot_name, regex)).fetchone()

            if existing:
                self._conn.execute("""
                    UPDATE slot_patterns SET successes = successes + 1
                    WHERE id = ?
                """, (existing["id"],))
            else:
                examples = json.dumps([{
                    "input": user_input,
                    "value": str(slot_value),
                }], ensure_ascii=False)

                self._conn.execute("""
                    INSERT INTO slot_patterns
                    (intent, slot_name, regex_pattern, examples, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (intent, slot_name, regex, examples, time.time()))

        self._conn.commit()

    # ═══════════════════════════════════════════════════════════════
    #              ИСПОЛЬЗОВАНИЕ (ПОИСК ПАТТЕРНОВ)
    # ═══════════════════════════════════════════════════════════════

    def find_routing(self, user_input: str, min_confidence: float = 0.6) -> Optional[Dict]:
        """
        Ищет подходящий routing паттерн для запроса.

        Возвращает None если не нашёл (→ нужен LLM).
        Возвращает Dict если нашёл (→ LLM не нужен).
        """
        keywords = self._extract_keywords(user_input)
        if not keywords:
            return None

        # 1. Поиск через FTS5 (быстрый полнотекстовый)
        try:
            rows = self._conn.execute("""
                SELECT rp.id, rp.pattern, rp.intent, rp.agent,
                       rp.confidence, rp.successes, rp.failures,
                       routing_fts.rank
                FROM routing_fts
                JOIN routing_patterns rp ON routing_fts.rowid = rp.id
                WHERE routing_fts MATCH ?
                AND rp.confidence >= ?
                ORDER BY routing_fts.rank
                LIMIT 5
            """, (keywords, min_confidence)).fetchall()
        except Exception:
            rows = []

        if not rows:
            return None

        # 2. Ранжируем: confidence * successes / (failures + 1) * FTS_score
        best = None
        best_score = 0

        for row in rows:
            score = (
                row["confidence"]
                * (row["successes"] / (row["failures"] + 1))
            )
            if score > best_score:
                best_score = score
                best = dict(row)

        if not best:
            return None

        # 3. Обновляем last_used
        self._conn.execute("""
            UPDATE routing_patterns SET last_used = ? WHERE id = ?
        """, (time.time(), best["id"]))
        self._conn.commit()

        return {
            "intent": best["intent"],
            "agent": best["agent"],
            "confidence": best["confidence"],
            "pattern_id": best["id"],
            "source": "learned",
        }

    def find_response(self, intent: str, tool_result: str) -> Optional[str]:
        """
        Ищет шаблон ответа для intent + результата.

        Возвращает готовый ответ или None (→ нужен LLM для синтеза).
        """
        result_type = self._classify_result(tool_result)

        row = self._conn.execute("""
            SELECT id, template, confidence FROM response_patterns
            WHERE intent = ? AND result_type = ?
            AND confidence >= 0.6
            ORDER BY successes DESC
            LIMIT 1
        """, (intent, result_type)).fetchone()

        if not row:
            return None

        # Подставляем результат в шаблон
        try:
            response = row["template"].format(result=tool_result)
        except (KeyError, IndexError):
            response = row["template"].replace("{result}", tool_result)

        # Обновляем last_used
        self._conn.execute("""
            UPDATE response_patterns SET last_used = ? WHERE id = ?
        """, (time.time(), row["id"]))
        self._conn.commit()

        return response

    def find_slots(self, intent: str, user_input: str) -> Dict[str, str]:
        """
        Извлекает аргументы из текста запроса по выученным regex.

        Возвращает {"filepath": "wishes.txt", "content": "..."} или {}
        """
        rows = self._conn.execute("""
            SELECT slot_name, regex_pattern FROM slot_patterns
            WHERE intent = ? AND confidence >= 0.5
            ORDER BY successes DESC
        """, (intent,)).fetchall()

        slots = {}
        for row in rows:
            try:
                match = re.search(row["regex_pattern"], user_input, re.IGNORECASE)
                if match:
                    value = match.group(1) if match.groups() else match.group(0)
                    slots[row["slot_name"]] = value
            except re.error:
                continue

        return slots

    # ═══════════════════════════════════════════════════════════════
    #              ОБРАТНАЯ СВЯЗЬ (УСИЛЕНИЕ / ОСЛАБЛЕНИЕ)
    # ═══════════════════════════════════════════════════════════════

    def reinforce(self, pattern_id: int, table: str = "routing"):
        """Паттерн сработал правильно → усиливаем"""
        tbl = "routing_patterns" if table == "routing" else "response_patterns"
        self._conn.execute(f"""
            UPDATE {tbl}
            SET successes = successes + 1,
                confidence = MIN(1.0, confidence + 0.05),
                last_used = ?
            WHERE id = ?
        """, (time.time(), pattern_id))
        self._conn.commit()

    def weaken(self, pattern_id: int, table: str = "routing"):
        """Паттерн сработал неправильно → ослабляем"""
        tbl = "routing_patterns" if table == "routing" else "response_patterns"
        self._conn.execute(f"""
            UPDATE {tbl}
            SET failures = failures + 1,
                confidence = MAX(0.0, confidence - 0.15)
            WHERE id = ?
        """, (pattern_id,))
        self._conn.commit()

    # ═══════════════════════════════════════════════════════════════
    #              ВНУТРЕННИЕ МЕТОДЫ
    # ═══════════════════════════════════════════════════════════════

    def _extract_keywords(self, text: str) -> str:
        """Извлекает ключевые слова для FTS5 поиска"""
        stop_words = {
            "я", "ты", "он", "она", "мы", "вы", "они", "мне", "мой", "твой",
            "для", "меня", "тебя", "его", "неё",
            "в", "на", "и", "с", "по", "от", "к", "не", "что", "это", "как",
            "но", "а", "или", "да", "нет", "бы", "ли", "же", "вот", "так",
            "the", "is", "are", "a", "an", "in", "on", "for", "to", "of",
            "привет", "пожалуйста", "спасибо", "можешь",
        }
        words = []
        for word in re.findall(r'[а-яёa-z0-9]+', text.lower()):
            if len(word) > 2 and word not in stop_words:
                words.append(word)
        return " ".join(words[:15])

    def _find_similar_routing(self, keywords: str, intent: str) -> Optional[Dict]:
        """Ищет существующий routing паттерн с таким же intent и похожими keywords"""
        try:
            row = self._conn.execute("""
                SELECT rp.id, rp.keywords, rp.intent
                FROM routing_fts
                JOIN routing_patterns rp ON routing_fts.rowid = rp.id
                WHERE routing_fts MATCH ? AND rp.intent = ?
                LIMIT 1
            """, (keywords, intent)).fetchone()
            return dict(row) if row else None
        except Exception:
            return None

    def _find_similar_response(self, intent: str, result_type: str) -> Optional[Dict]:
        """Ищет существующий response паттерн"""
        row = self._conn.execute("""
            SELECT id FROM response_patterns
            WHERE intent = ? AND result_type = ?
            LIMIT 1
        """, (intent, result_type)).fetchone()
        return dict(row) if row else None

    def _reinforce_routing(self, pattern_id: int):
        """Усиливает routing паттерн"""
        self._conn.execute("""
            UPDATE routing_patterns
            SET successes = successes + 1,
                confidence = MIN(1.0, confidence + 0.03),
                last_used = ?
            WHERE id = ?
        """, (time.time(), pattern_id))
        self._conn.commit()

    def _reinforce_response(self, pattern_id: int):
        """Усиливает response паттерн"""
        self._conn.execute("""
            UPDATE response_patterns
            SET successes = successes + 1,
                confidence = MIN(1.0, confidence + 0.03),
                last_used = ?
            WHERE id = ?
        """, (time.time(), pattern_id))
        self._conn.commit()

    def _classify_result(self, result: str) -> str:
        """Классифицирует результат: success / error / empty"""
        if not result or not result.strip():
            return "empty"
        if result.startswith("ERROR") or "ошибка" in result.lower():
            return "error"
        return "success"

    def _extract_template(self, intent: str, tool_result: str, response: str) -> str:
        """
        Извлекает шаблон ответа из конкретного ответа LLM.

        Заменяет конкретный результат на {result} placeholder,
        чтобы шаблон можно было переиспользовать.
        """
        # Если результат инструмента содержится в ответе — заменяем на плейсхолдер
        template = response
        if tool_result and tool_result in response:
            template = response.replace(tool_result, "{result}")
        return template

    def _generate_slot_regex(
        self, user_input: str, slot_value: str, slot_name: str
    ) -> Optional[str]:
        """
        Генерирует regex для извлечения слота из текста.

        Пример: input="создай файл wishes.txt", value="wishes.txt"
        → regex: r'файл\s+([\w.]+)'

        Пример: input="удали файл test.py", value="test.py"
        → regex: r'файл\s+([\w.]+)'
        """
        escaped_value = re.escape(slot_value)

        # Находим позицию значения в тексте
        match = re.search(escaped_value, user_input, re.IGNORECASE)
        if not match:
            return None

        start = match.start()

        # Берём 1-2 слова перед значением как контекст
        prefix = user_input[:start].strip()
        prefix_words = prefix.split()

        if not prefix_words:
            return None

        # Используем последнее слово перед значением как якорь
        anchor = re.escape(prefix_words[-1])

        # Генерируем regex в зависимости от типа значения
        if re.match(r'^[\w.-]+\.\w+$', slot_value):
            # Это имя файла (содержит расширение)
            return rf'{anchor}\s+([\w\-. а-яёА-ЯЁ]+\.\w+)'
        elif slot_value.startswith("/") or slot_value.startswith("~"):
            # Это путь
            return rf'{anchor}\s+([/~][\w/\-. ]+)'
        else:
            # Общий текст — берём до конца строки или следующего ключевого слова
            return rf'{anchor}\s+(.+?)(?:\s*$)'

    # ═══════════════════════════════════════════════════════════════
    #              СТАТИСТИКА И ОБСЛУЖИВАНИЕ
    # ═══════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict[str, int]:
        """Статистика базы паттернов"""
        routing = self._conn.execute(
            "SELECT COUNT(*) as c FROM routing_patterns"
        ).fetchone()["c"]
        response = self._conn.execute(
            "SELECT COUNT(*) as c FROM response_patterns"
        ).fetchone()["c"]
        slots = self._conn.execute(
            "SELECT COUNT(*) as c FROM slot_patterns"
        ).fetchone()["c"]

        high_conf = self._conn.execute(
            "SELECT COUNT(*) as c FROM routing_patterns WHERE confidence >= 0.8"
        ).fetchone()["c"]

        return {
            "routing": routing,
            "response": response,
            "slots": slots,
            "high_confidence": high_conf,
        }

    def get_coverage_report(self) -> str:
        """Отчёт о покрытии: сколько запросов Кристина может решить без LLM"""
        stats = self.get_stats()
        total = stats["routing"]
        strong = stats["high_confidence"]

        # Уникальные intent-ы
        intents = self._conn.execute("""
            SELECT DISTINCT intent FROM routing_patterns WHERE confidence >= 0.7
        """).fetchall()

        report = f"Покрытие паттернов:\n"
        report += f"  Routing: {total} паттернов ({strong} сильных)\n"
        report += f"  Ответы: {stats['response']} шаблонов\n"
        report += f"  Слоты: {stats['slots']} regex\n"
        report += f"  Intent-ы без LLM: {', '.join(r['intent'] for r in intents)}\n"
        return report

    def cleanup_weak_patterns(self, min_confidence: float = 0.2, max_age_days: int = 30):
        """Удаляет слабые и старые паттерны"""
        cutoff = time.time() - (max_age_days * 86400)

        for table in ["routing_patterns", "response_patterns", "slot_patterns"]:
            self._conn.execute(f"""
                DELETE FROM {table}
                WHERE confidence < ? AND last_used < ?
            """, (min_confidence, cutoff) if "last_used" in table else (min_confidence, cutoff))

        # Перестраиваем FTS
        self._conn.execute("INSERT INTO routing_fts(routing_fts) VALUES('rebuild')")
        self._conn.commit()

        logger.info("🧹 Weak patterns cleaned up")

    def close(self):
        """Закрытие соединения с БД"""
        self._conn.close()

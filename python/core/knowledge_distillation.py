"""
Кристина 7.2 — Knowledge Distillation (Дистилляция знаний)

ЗАЧЕМ:
  Когда LLM решает задачу, Кристина запоминает не только ОТВЕТ,
  но и ПРОЦЕСС РАССУЖДЕНИЯ. Это ключевое отличие от простого кеширования.

  LLM: "Чтобы создать CSV-парсер:
        1) открыть файл
        2) разбить по разделителю
        3) обработать заголовки
        4) итерировать строки"

  Кристина запоминает ШАБЛОН рассуждения:
  "Чтобы создать [X]-парсер:
   1) открыть [источник]
   2) разбить по [формату]
   3) обработать [метаданные]
   4) итерировать [элементы]"

  В следующий раз для "создай JSON-парсер" — Кристина САМА
  применяет шаблон: открыть файл → разбить по JSON → ...

АРХИТЕКТУРА:
  ┌─────────────────────────────────────────────┐
  │ ReasoningChain                              │
  │   steps: [ThoughtStep, ThoughtStep, ...]    │
  │   intent: "create_parser"                   │
  │   variables: {X: "CSV", source: "файл"}     │
  │   template: generalized chain               │
  │   confidence: 0.85                          │
  └─────────────────────────────────────────────┘

  ThoughtStep:
    thought:     "Нужно открыть файл"
    action:      "read_file"
    observation:  "Файл прочитан, 100 строк"
    conclusion:  "Данные загружены, переходим к парсингу"

ХРАНЕНИЕ:
  SQLite — reasoning chains + templates
  FTS5 — быстрый поиск по ситуации

ОБУЧЕНИЕ:
  1. LLM решает задачу → парсим chain-of-thought
  2. Обобщаем: заменяем конкретные значения на переменные
  3. Сохраняем шаблон
  4. При похожем запросе: находим шаблон → подставляем переменные
"""

import sqlite3
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from utils.logging import get_logger
import config

logger = get_logger("knowledge_distillation")

# ═══════════════════════════════════════════════════════════════
#               ПАРСИНГ ЦЕПОЧЕК РАССУЖДЕНИЙ
# ═══════════════════════════════════════════════════════════════

# Паттерны для распознавания шагов в ответе LLM
STEP_PATTERNS = [
    # Нумерованные списки
    re.compile(r'^\s*(\d+)[.)]\s*(.+)', re.MULTILINE),
    # Маркированные списки
    re.compile(r'^\s*[-•*]\s*(.+)', re.MULTILINE),
    # "Шаг N:" формат
    re.compile(r'(?:шаг|step)\s*(\d+)\s*[:.]\s*(.+)', re.IGNORECASE | re.MULTILINE),
    # "Сначала..., затем..., потом..."
    re.compile(r'(?:сначала|первым делом|для начала)\s+(.+?)(?:\.|$)', re.IGNORECASE),
    re.compile(r'(?:затем|далее|потом|после этого)\s+(.+?)(?:\.|$)', re.IGNORECASE),
    re.compile(r'(?:наконец|в конце|в итоге)\s+(.+?)(?:\.|$)', re.IGNORECASE),
]

# Ключевые слова для обобщения (конкретное → переменная)
GENERALIZATION_PATTERNS = [
    # Имена файлов → {filename}
    (re.compile(r'[\w\-]+\.\w{1,5}'), "{filename}"),
    # Пути → {filepath}
    (re.compile(r'[/~][\w/\-.]+'), "{filepath}"),
    # Числа → {number}
    (re.compile(r'\b\d{2,}\b'), "{number}"),
    # URL → {url}
    (re.compile(r'https?://\S+'), "{url}"),
    # Языки программирования → {language}
    (re.compile(
        r'\b(python|javascript|typescript|java|rust|go|ruby|'
        r'php|c\+\+|swift|kotlin)\b', re.I
    ), "{language}"),
]


class KnowledgeDistillation:
    """
    Дистилляция знаний из LLM — учимся ДУМАТЬ, а не запоминать.

    Сохраняет цепочки рассуждений (chain-of-thought) из ответов LLM
    и применяет их к новым похожим задачам.

    Использование:
        kd = KnowledgeDistillation()

        # 1. LLM ответила — сохраняем рассуждение
        kd.distill(
            user_input="Создай CSV-парсер на Python",
            llm_response="Чтобы создать CSV-парсер:\n1) Откроем файл...",
            intent="create_code",
            result_success=True,
        )

        # 2. Похожий вопрос — ищем шаблон
        chain = kd.find_reasoning("Создай JSON-парсер")
        if chain:
            # Есть шаблон! Применяем с новыми переменными
            steps = chain["steps"]
            variables = chain["variables"]  # {format: "JSON"}

        # 3. Обратная связь
        kd.feedback(chain["chain_id"], useful=True)
    """

    def __init__(self, sentence_embeddings=None, db_path: Path = None):
        self._sentence = sentence_embeddings
        self._db_path = db_path or (config.config.data_dir / "knowledge_distillation.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        self._create_tables()

        stats = self.get_stats()
        logger.info(
            f"🧪 KnowledgeDistillation: {stats['chains']} цепочек, "
            f"{stats['templates']} шаблонов, "
            f"{stats['total_steps']} шагов"
        )

    def _create_tables(self):
        cur = self._conn.cursor()

        # Конкретные цепочки рассуждений
        cur.execute("""
            CREATE TABLE IF NOT EXISTS reasoning_chains (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT NOT NULL,
                intent TEXT NOT NULL,
                keywords TEXT NOT NULL,
                steps_json TEXT NOT NULL,
                variables_json TEXT DEFAULT '{}',
                confidence REAL DEFAULT 1.0,
                successes INTEGER DEFAULT 1,
                failures INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                last_used REAL NOT NULL
            )
        """)

        # Обобщённые шаблоны (generalized templates)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS reasoning_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intent_pattern TEXT NOT NULL,
                template_steps_json TEXT NOT NULL,
                variable_slots TEXT NOT NULL,
                example_inputs TEXT DEFAULT '[]',
                confidence REAL DEFAULT 1.0,
                usage_count INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)

        # FTS5 для поиска по ключевым словам
        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chains_fts
            USING fts5(keywords, content=reasoning_chains, content_rowid=id)
        """)

        # Индексы
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chains_intent ON reasoning_chains(intent)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chains_conf ON reasoning_chains(confidence DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_templates_intent ON reasoning_templates(intent_pattern)")

        self._conn.commit()

    # ═══════════════════════════════════════════════════════════════
    #               ДИСТИЛЛЯЦИЯ (сохранение рассуждений)
    # ═══════════════════════════════════════════════════════════════

    def distill(
        self,
        user_input: str,
        llm_response: str,
        intent: str,
        result_success: bool = True,
        extra_context: Dict = None,
    ) -> Optional[int]:
        """
        Дистиллирует знания из ответа LLM.

        1. Парсит цепочку рассуждений из текста
        2. Извлекает переменные
        3. Создаёт обобщённый шаблон
        4. Сохраняет в БД

        Returns:
            chain_id или None (если не удалось извлечь рассуждения)
        """
        # 1. Парсим шаги из ответа LLM
        steps = self._parse_reasoning_steps(llm_response)
        if not steps:
            return None

        # 2. Извлекаем переменные (конкретные значения)
        variables = self._extract_variables(user_input, llm_response)

        # 3. Извлекаем ключевые слова
        keywords = self._extract_keywords(user_input)

        # 4. Сохраняем конкретную цепочку
        now = time.time()
        cur = self._conn.cursor()
        cur.execute("""
            INSERT INTO reasoning_chains
            (user_input, intent, keywords, steps_json, variables_json,
             confidence, created_at, last_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_input, intent, keywords,
            json.dumps(steps, ensure_ascii=False),
            json.dumps(variables, ensure_ascii=False),
            1.0 if result_success else 0.5,
            now, now,
        ))
        chain_id = cur.lastrowid

        # Обновляем FTS
        cur.execute("""
            INSERT INTO chains_fts (rowid, keywords) VALUES (?, ?)
        """, (chain_id, keywords))

        # 5. Пытаемся создать/обновить обобщённый шаблон
        self._update_template(intent, steps, variables, user_input)

        self._conn.commit()

        logger.debug(
            f"🧪 Distilled: '{user_input[:50]}' → {len(steps)} steps, "
            f"{len(variables)} vars, chain_id={chain_id}"
        )

        return chain_id

    def _parse_reasoning_steps(self, text: str) -> List[Dict]:
        """
        Парсит цепочку рассуждений из текста LLM.

        Распознаёт форматы:
        - Нумерованные списки (1. 2. 3.)
        - Маркированные списки (- • *)
        - "Шаг N:" формат
        - "Сначала..., затем..., потом..."
        """
        steps = []

        # Пробуем нумерованный список
        numbered = re.findall(
            r'^\s*(\d+)[.)]\s*(.+?)$', text, re.MULTILINE
        )
        if len(numbered) >= 2:
            for num, step_text in numbered:
                steps.append({
                    "step": int(num),
                    "text": step_text.strip(),
                    "type": "action",
                })
            return steps

        # Пробуем маркированный список
        bulleted = re.findall(
            r'^\s*[-•*]\s*(.+?)$', text, re.MULTILINE
        )
        if len(bulleted) >= 2:
            for i, step_text in enumerate(bulleted, 1):
                steps.append({
                    "step": i,
                    "text": step_text.strip(),
                    "type": "action",
                })
            return steps

        # Пробуем последовательные маркеры
        sequential_markers = [
            (r'(?:сначала|первым делом|для начала)\s+(.+?)(?:\.|,|;|$)', "first"),
            (r'(?:затем|далее|потом|после этого|после)\s+(.+?)(?:\.|,|;|$)', "then"),
            (r'(?:наконец|в конце|в итоге|в результате)\s+(.+?)(?:\.|$)', "finally"),
        ]

        step_num = 0
        for pattern, step_type in sequential_markers:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                step_num += 1
                steps.append({
                    "step": step_num,
                    "text": match.strip(),
                    "type": step_type,
                })

        if len(steps) >= 2:
            return steps

        # Fallback: разбиваем по предложениям (если текст содержит логические шаги)
        sentences = re.split(r'[.!]\s+', text)
        action_sentences = [
            s.strip() for s in sentences
            if len(s.strip()) > 10 and any(
                kw in s.lower() for kw in
                ["нужно", "необходимо", "следует", "можно", "надо",
                 "создай", "открой", "запусти", "найди", "проверь"]
            )
        ]

        if len(action_sentences) >= 2:
            for i, sent in enumerate(action_sentences[:10], 1):
                steps.append({
                    "step": i,
                    "text": sent,
                    "type": "action",
                })
            return steps

        return []

    def _extract_variables(self, user_input: str, llm_response: str) -> Dict[str, str]:
        """Извлекает переменные (конкретные значения) из запроса"""
        variables = {}

        # Имена файлов
        files = re.findall(r'([\w\-]+\.\w{1,5})', user_input)
        for i, f in enumerate(files):
            key = "filename" if i == 0 else f"filename_{i+1}"
            variables[key] = f

        # Пути
        paths = re.findall(r'([/~][\w/\-.]+)', user_input)
        for i, p in enumerate(paths):
            key = "filepath" if i == 0 else f"filepath_{i+1}"
            variables[key] = p

        # Языки программирования
        langs = re.findall(
            r'\b(python|javascript|typescript|java|rust|go|ruby|'
            r'php|c\+\+|swift|kotlin)\b', user_input, re.I
        )
        if langs:
            variables["language"] = langs[0].lower()

        # Ключевые существительные (простая эвристика)
        words = user_input.lower().split()
        stop = {
            "создай", "сделай", "напиши", "найди", "покажи",
            "файл", "папку", "приложение", "для", "на", "в", "с",
            "как", "что", "это", "нужно", "можно", "пожалуйста",
        }
        meaningful = [w for w in words if w not in stop and len(w) > 3]
        if meaningful:
            variables["topic"] = meaningful[0]

        return variables

    def _extract_keywords(self, text: str) -> str:
        """Извлекает ключевые слова для FTS5"""
        stop_words = {
            "я", "ты", "он", "она", "мы", "вы", "они",
            "в", "на", "и", "с", "по", "от", "к", "не",
            "что", "это", "как", "но", "а", "или", "да", "нет",
            "можешь", "пожалуйста", "мне", "для", "меня",
        }
        words = re.findall(r'[а-яёa-z0-9]+', text.lower())
        keywords = [w for w in words if len(w) > 2 and w not in stop_words]
        return " ".join(keywords[:15])

    def _update_template(
        self,
        intent: str,
        steps: List[Dict],
        variables: Dict[str, str],
        user_input: str,
    ):
        """Создаёт или обновляет обобщённый шаблон"""
        # Обобщаем шаги: заменяем конкретные значения на {variable}
        template_steps = []
        for step in steps:
            text = step["text"]
            for var_name, var_value in variables.items():
                if var_value in text:
                    text = text.replace(var_value, "{" + var_name + "}")
            template_steps.append({
                "step": step["step"],
                "text": text,
                "type": step.get("type", "action"),
            })

        # Проверяем: есть ли уже шаблон для этого intent?
        existing = self._conn.execute("""
            SELECT id, example_inputs FROM reasoning_templates
            WHERE intent_pattern = ?
            ORDER BY usage_count DESC LIMIT 1
        """, (intent,)).fetchone()

        now = time.time()
        variable_slots = json.dumps(list(variables.keys()), ensure_ascii=False)

        if existing:
            # Обновляем: добавляем пример
            examples = json.loads(existing["example_inputs"])
            if user_input not in examples:
                examples.append(user_input)
                examples = examples[-20:]  # Храним максимум 20 примеров

            self._conn.execute("""
                UPDATE reasoning_templates
                SET example_inputs = ?, usage_count = usage_count + 1, updated_at = ?
                WHERE id = ?
            """, (json.dumps(examples, ensure_ascii=False), now, existing["id"]))
        else:
            # Создаём новый шаблон
            self._conn.execute("""
                INSERT INTO reasoning_templates
                (intent_pattern, template_steps_json, variable_slots,
                 example_inputs, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                intent,
                json.dumps(template_steps, ensure_ascii=False),
                variable_slots,
                json.dumps([user_input], ensure_ascii=False),
                now, now,
            ))

    # ═══════════════════════════════════════════════════════════════
    #               ПОИСК РАССУЖДЕНИЙ
    # ═══════════════════════════════════════════════════════════════

    def find_reasoning(
        self,
        user_input: str,
        intent: str = None,
        min_confidence: float = 0.6,
    ) -> Optional[Dict]:
        """
        Ищет подходящую цепочку рассуждений для запроса.

        1. Поиск по FTS5 (ключевые слова)
        2. Ранжирование по confidence + similarity
        3. Подстановка новых переменных

        Returns:
            Dict с полями:
            - chain_id: int
            - steps: List[Dict]
            - variables: Dict
            - confidence: float
            - source: "exact" | "template"
            Или None если ничего не нашли
        """
        keywords = self._extract_keywords(user_input)
        if not keywords:
            return None

        # 1. Поиск конкретной цепочки через FTS5
        try:
            rows = self._conn.execute("""
                SELECT rc.id, rc.user_input, rc.intent, rc.steps_json,
                       rc.variables_json, rc.confidence, rc.successes, rc.failures
                FROM chains_fts
                JOIN reasoning_chains rc ON chains_fts.rowid = rc.id
                WHERE chains_fts MATCH ?
                AND rc.confidence >= ?
                ORDER BY chains_fts.rank
                LIMIT 5
            """, (keywords, min_confidence)).fetchall()
        except Exception:
            rows = []

        # Фильтруем по intent если указан
        if intent and rows:
            filtered = [r for r in rows if r["intent"] == intent]
            if filtered:
                rows = filtered

        if rows:
            # Ранжируем
            best = max(rows, key=lambda r: (
                r["confidence"] * (r["successes"] / (r["failures"] + 1))
            ))

            # Обновляем last_used
            self._conn.execute(
                "UPDATE reasoning_chains SET last_used = ? WHERE id = ?",
                (time.time(), best["id"])
            )
            self._conn.commit()

            # Извлекаем новые переменные из текущего запроса
            new_variables = self._extract_variables(user_input, "")
            old_variables = json.loads(best["variables_json"])

            # Подставляем новые переменные в шаги
            steps = json.loads(best["steps_json"])
            adapted_steps = self._adapt_steps(steps, old_variables, new_variables)

            return {
                "chain_id": best["id"],
                "steps": adapted_steps,
                "variables": new_variables,
                "original_variables": old_variables,
                "confidence": best["confidence"],
                "source": "exact",
            }

        # 2. Поиск по шаблонам
        return self._find_by_template(user_input, intent, min_confidence)

    def _find_by_template(
        self,
        user_input: str,
        intent: str = None,
        min_confidence: float = 0.6,
    ) -> Optional[Dict]:
        """Ищет подходящий шаблон рассуждений"""
        query = "SELECT * FROM reasoning_templates WHERE confidence >= ?"
        params: list = [min_confidence]

        if intent:
            query += " AND intent_pattern = ?"
            params.append(intent)

        query += " ORDER BY usage_count DESC LIMIT 10"
        templates = self._conn.execute(query, params).fetchall()

        if not templates:
            return None

        # Если есть sentence_embeddings — ищем по сходству
        if self._sentence:
            best_template = None
            best_sim = 0.0

            for tpl in templates:
                examples = json.loads(tpl["example_inputs"])
                for example in examples:
                    sim = self._sentence.similarity(user_input, example)
                    if sim > best_sim:
                        best_sim = sim
                        best_template = tpl

            if best_template and best_sim >= 0.5:
                new_variables = self._extract_variables(user_input, "")
                steps = json.loads(best_template["template_steps_json"])
                adapted = self._adapt_template_steps(steps, new_variables)

                return {
                    "chain_id": best_template["id"],
                    "steps": adapted,
                    "variables": new_variables,
                    "confidence": best_sim * best_template["confidence"],
                    "source": "template",
                }
        else:
            # Без sentence_embeddings — берём первый подходящий
            if templates:
                tpl = templates[0]
                new_variables = self._extract_variables(user_input, "")
                steps = json.loads(tpl["template_steps_json"])
                adapted = self._adapt_template_steps(steps, new_variables)

                return {
                    "chain_id": tpl["id"],
                    "steps": adapted,
                    "variables": new_variables,
                    "confidence": tpl["confidence"] * 0.5,
                    "source": "template",
                }

        return None

    def _adapt_steps(
        self,
        steps: List[Dict],
        old_vars: Dict[str, str],
        new_vars: Dict[str, str],
    ) -> List[Dict]:
        """Адаптирует конкретные шаги: заменяет старые переменные на новые"""
        adapted = []
        for step in steps:
            text = step["text"]
            for var_name in old_vars:
                old_val = old_vars[var_name]
                new_val = new_vars.get(var_name, old_val)
                text = text.replace(old_val, new_val)
            adapted.append({**step, "text": text})
        return adapted

    def _adapt_template_steps(
        self,
        template_steps: List[Dict],
        variables: Dict[str, str],
    ) -> List[Dict]:
        """Подставляет переменные в шаблонные шаги"""
        adapted = []
        for step in template_steps:
            text = step["text"]
            for var_name, var_value in variables.items():
                text = text.replace("{" + var_name + "}", var_value)
            adapted.append({**step, "text": text})
        return adapted

    # ═══════════════════════════════════════════════════════════════
    #               ОБРАТНАЯ СВЯЗЬ
    # ═══════════════════════════════════════════════════════════════

    def feedback(self, chain_id: int, useful: bool, source: str = "exact"):
        """
        Обратная связь: была ли цепочка полезна.

        useful=True  → усиливаем (confidence += 0.05)
        useful=False → ослабляем (confidence -= 0.15)
        """
        if source == "exact":
            table = "reasoning_chains"
        else:
            table = "reasoning_templates"

        if useful:
            self._conn.execute(f"""
                UPDATE {table}
                SET successes = successes + 1,
                    confidence = MIN(1.0, confidence + 0.05)
                WHERE id = ?
            """, (chain_id,))
        else:
            self._conn.execute(f"""
                UPDATE {table}
                SET failures = failures + 1,
                    confidence = MAX(0.0, confidence - 0.15)
                WHERE id = ?
            """, (chain_id,))

        self._conn.commit()

    # ═══════════════════════════════════════════════════════════════
    #               СТАТИСТИКА
    # ═══════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict:
        chains = self._conn.execute(
            "SELECT COUNT(*) as c FROM reasoning_chains"
        ).fetchone()["c"]

        templates = self._conn.execute(
            "SELECT COUNT(*) as c FROM reasoning_templates"
        ).fetchone()["c"]

        total_steps = 0
        rows = self._conn.execute(
            "SELECT steps_json FROM reasoning_chains"
        ).fetchall()
        for row in rows:
            try:
                steps = json.loads(row["steps_json"])
                total_steps += len(steps)
            except (json.JSONDecodeError, TypeError):
                pass

        strong_chains = self._conn.execute(
            "SELECT COUNT(*) as c FROM reasoning_chains WHERE confidence >= 0.8"
        ).fetchone()["c"]

        # Уникальные intent-ы
        intents = self._conn.execute(
            "SELECT DISTINCT intent FROM reasoning_chains"
        ).fetchall()

        return {
            "chains": chains,
            "templates": templates,
            "total_steps": total_steps,
            "strong_chains": strong_chains,
            "intents": [r["intent"] for r in intents],
        }

    def get_reasoning_report(self) -> str:
        """Отчёт о накопленных знаниях"""
        stats = self.get_stats()
        report = "Дистилляция знаний:\n"
        report += f"  Цепочки рассуждений: {stats['chains']}\n"
        report += f"  Обобщённые шаблоны: {stats['templates']}\n"
        report += f"  Всего шагов: {stats['total_steps']}\n"
        report += f"  Сильные (conf≥0.8): {stats['strong_chains']}\n"
        report += f"  Intent-ы: {', '.join(stats['intents']) or 'нет'}\n"
        return report

    def cleanup(self, min_confidence: float = 0.2, max_age_days: int = 60):
        """Удаляет слабые и старые цепочки"""
        cutoff = time.time() - (max_age_days * 86400)

        self._conn.execute("""
            DELETE FROM reasoning_chains
            WHERE confidence < ? AND last_used < ?
        """, (min_confidence, cutoff))

        # Перестраиваем FTS
        try:
            self._conn.execute("INSERT INTO chains_fts(chains_fts) VALUES('rebuild')")
        except Exception:
            pass

        self._conn.commit()
        logger.info("🧹 Knowledge distillation: weak chains cleaned up")

    def close(self):
        self._conn.commit()
        self._conn.close()

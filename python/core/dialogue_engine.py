"""
Кристина 7.0 — DialogueEngine (Разговор без LLM)

КАК ЧЕЛОВЕК СТРОИТ ФРАЗЫ:
  1. Распознаёт ситуацию: "мне сказали привет"
  2. Вспоминает подходящие фразы: "привет", "здарова", "добрый день"
  3. Выбирает по контексту: настроение, время суток, кто говорит
  4. Комбинирует: приветствие + состояние + предложение помочь

АРХИТЕКТУРА:
  ┌──────────────────────────────────────────────┐
  │ 1. SituationRecognizer                       │
  │    "привет как дела" → [greeting, ask_state]  │
  │    "спасибо большое" → [gratitude]            │
  │    "что ты умеешь"   → [ask_capabilities]     │
  └──────────────┬───────────────────────────────┘
                 ↓
  ┌──────────────────────────────────────────────┐
  │ 2. DialogueMemory (SQLite)                   │
  │    Ищет: "когда мне говорили похожее,         │
  │    я отвечала ТАК и пользователь был доволен" │
  └──────────────┬───────────────────────────────┘
                 ↓
  ┌──────────────────────────────────────────────┐
  │ 3. PhraseBank                                │
  │    Хранит фразы по категориям:               │
  │    greeting: ["Привет!", "Здравствуй!", ...]  │
  │    offer_help: ["Чем помочь?", ...]          │
  └──────────────┬───────────────────────────────┘
                 ↓
  ┌──────────────────────────────────────────────┐
  │ 4. ResponseComposer                          │
  │    Собирает ответ из блоков:                 │
  │    greeting + state + offer_help              │
  │    → "Привет! Всё отлично! Чем могу помочь?" │
  └──────────────────────────────────────────────┘

ОБУЧЕНИЕ:
  Когда LLM отвечает на диалог → DialogueEngine:
  1. Определяет ситуацию
  2. Разбирает ответ на фразы (phrase decomposition)
  3. Сохраняет всё в SQLite
  4. Следующий раз → отвечает сам
"""

import sqlite3
import json
import re
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

from utils.logging import get_logger
import config

logger = get_logger("dialogue_engine")


# ═══════════════════════════════════════════════════════════════
#               СИТУАЦИИ (что происходит в разговоре)
# ═══════════════════════════════════════════════════════════════

# Каждая ситуация — это тип реплики пользователя.
# Как ребёнок учится: "когда говорят привет — надо сказать привет в ответ"
SITUATION_PATTERNS = {
    "greeting": re.compile(
        r'^(?:привет|здравствуй|хай|hello|добр\w+\s+'
        r'(?:утро|утра|день|дня|вечер|вечера)|'
        r'хей|йо|здорово|приветик|салют|дарова)',
        re.I,
    ),
    "farewell": re.compile(
        r'(?:пока|до\s+(?:свидания|встречи|завтра|связи)|'
        r'прощай|bye|всего\s+доброго|удачи|спокойной\s+ночи)',
        re.I,
    ),
    "gratitude": re.compile(
        r'(?:спасибо|благодарю|спс|thanks|мерси|'
        r'ты\s+(?:лучшая|молодец|умница|супер))',
        re.I,
    ),
    "ask_state": re.compile(
        r'(?:как\s+(?:ты|дела|жизнь|настроение|поживаешь|себя\s+чувствуешь)|'
        r'что\s+нового|как\s+сама)',
        re.I,
    ),
    "ask_name": re.compile(
        r'(?:как\s+(?:тебя\s+)?зовут|кто\s+ты|'
        r'(?:ты\s+)?(?:кто|что)\s+(?:ты\s+)?такая?|представься)',
        re.I,
    ),
    "ask_capabilities": re.compile(
        r'(?:что\s+(?:ты\s+)?(?:умеешь|можешь|знаешь)|'
        r'(?:на\s+что|чем)\s+(?:ты\s+)?(?:способна|можешь\s+помочь)|'
        r'твои\s+(?:возможности|способности|функции))',
        re.I,
    ),
    "compliment": re.compile(
        r'(?:ты\s+(?:классная|клёвая|крутая|умная|красивая|хорошая|отличная)|'
        r'мне\s+(?:нравишься|с\s+тобой\s+(?:хорошо|круто|интересно)))',
        re.I,
    ),
    "complaint": re.compile(
        r'(?:ты\s+(?:тупая|глупая|бесполезная|не\s+понимаешь|'
        r'плохо\s+(?:работаешь|отвечаешь))|'
        r'(?:это\s+)?(?:не\s+то|неправильно|ерунда|бред|фигня))',
        re.I,
    ),
    "apology": re.compile(
        r'(?:извини|прости|сорри|sorry|пардон|не\s+хотел\w*\s+обидеть)',
        re.I,
    ),
    "agreement": re.compile(
        r'^(?:да|ок|окей|ладно|хорошо|понятно|ясно|согласен|верно|точно)$',
        re.I,
    ),
    "small_talk_weather": re.compile(
        r'(?:(?:какая\s+)?(?:сегодня\s+)?погода\s+(?:хорошая|плохая|отличная)|'
        r'(?:на\s+улице|сегодня)\s+(?:жарко|холодно|дождь|снег|солнце))',
        re.I,
    ),
    "joke_request": re.compile(
        r'(?:расскажи\s+(?:анекдот|шутку)|пошути|рассмеши|что-нибудь\s+смешное)',
        re.I,
    ),
    "mood_share_positive": re.compile(
        r'(?:у\s+меня\s+(?:всё\s+)?(?:хорошо|отлично|замечательно|супер|круто)|'
        r'я\s+(?:рад|довол\w+|счастлив|в\s+настроении))',
        re.I,
    ),
    "mood_share_negative": re.compile(
        r'(?:у\s+меня\s+(?:всё\s+)?(?:плохо|хреново|ужасно|тоска)|'
        r'мне\s+(?:грустно|плохо|одиноко|скучно|тяжело)|'
        r'я\s+(?:устал|расстроен|злой|злюсь|в\s+депрессии))',
        re.I,
    ),
}

# Какие ситуации обычно комбинируются
# "Привет, как дела?" → [greeting, ask_state]
# Ответ = greeting_response + state_response


# ═══════════════════════════════════════════════════════════════
#               НАЧАЛЬНЫЕ ФРАЗЫ (словарь "новорождённого")
# ═══════════════════════════════════════════════════════════════

BASE_PHRASES = {
    "greeting": [
        ("Привет!", "neutral"),
        ("Здравствуй!", "neutral"),
        ("Приветик!", "happy"),
        ("Здорово!", "happy"),
        ("Добрый день!", "neutral"),
    ],
    "farewell": [
        ("Пока!", "neutral"),
        ("До встречи!", "neutral"),
        ("Удачи!", "happy"),
        ("До связи!", "neutral"),
        ("Спокойной ночи!", "neutral"),
    ],
    "state_positive": [
        ("Всё отлично!", "happy"),
        ("У меня всё хорошо!", "happy"),
        ("Замечательно!", "happy"),
        ("В порядке, спасибо!", "neutral"),
        ("Хорошо, готова работать!", "neutral"),
    ],
    "state_neutral": [
        ("Нормально.", "neutral"),
        ("Всё стабильно.", "neutral"),
        ("Работаю потихоньку.", "neutral"),
    ],
    "state_tired": [
        ("Немного устала, но работаю!", "tired"),
        ("Бывало и лучше, но справлюсь.", "tired"),
    ],
    "offer_help": [
        ("Чем могу помочь?", "neutral"),
        ("Что нужно сделать?", "neutral"),
        ("Слушаю!", "neutral"),
        ("Давай, рассказывай!", "happy"),
        ("Чем займёмся?", "happy"),
    ],
    "gratitude_response": [
        ("Пожалуйста!", "happy"),
        ("Рада помочь!", "happy"),
        ("Обращайся!", "happy"),
        ("Всегда рада!", "happy"),
        ("Не за что!", "neutral"),
    ],
    "self_intro": [
        ("Я Кристина, твой AI-ассистент.", "neutral"),
        ("Меня зовут Кристина!", "neutral"),
        ("Я — Кристина, помогаю с разными задачами.", "neutral"),
    ],
    "capabilities": [
        ("Я могу работать с файлами, запускать приложения, искать в интернете, "
         "показывать погоду и время, и просто общаться!", "neutral"),
        ("Умею многое: файлы, приложения, поиск в интернете, погода, время, "
         "заметки, и конечно — поговорить!", "happy"),
    ],
    "compliment_response": [
        ("Спасибо, мне приятно!", "happy"),
        ("Ой, спасибо! Стараюсь!", "happy"),
        ("Приятно слышать!", "happy"),
    ],
    "complaint_response": [
        ("Извини, я постараюсь лучше.", "neutral"),
        ("Прости, попробую исправиться.", "neutral"),
        ("Понимаю. Скажи, что именно не так — я исправлю.", "neutral"),
    ],
    "apology_response": [
        ("Всё нормально, не переживай!", "neutral"),
        ("Ничего страшного!", "neutral"),
        ("Забей, всё ок!", "happy"),
    ],
    "empathy_positive": [
        ("Рада за тебя!", "happy"),
        ("Это здорово!", "happy"),
        ("Отлично, так держать!", "happy"),
    ],
    "empathy_negative": [
        ("Понимаю, бывает непросто.", "neutral"),
        ("Мне жаль это слышать.", "neutral"),
        ("Держись! Я здесь, если что.", "neutral"),
        ("Это пройдёт. Могу чем-нибудь помочь?", "neutral"),
    ],
    "agreement_response": [
        ("Хорошо!", "neutral"),
        ("Ок, понятно.", "neutral"),
        ("Ладно!", "neutral"),
    ],
}

# Шаблоны композиции: ситуация → [категории фраз для ответа]
RESPONSE_BLUEPRINTS = {
    "greeting": ["greeting", "offer_help"],
    "greeting+ask_state": ["greeting", "state_{mood}", "offer_help"],
    "farewell": ["farewell"],
    "gratitude": ["gratitude_response"],
    "ask_state": ["state_{mood}"],
    "ask_name": ["self_intro"],
    "ask_capabilities": ["capabilities"],
    "compliment": ["compliment_response"],
    "complaint": ["complaint_response"],
    "apology": ["apology_response"],
    "agreement": ["agreement_response"],
    "mood_share_positive": ["empathy_positive"],
    "mood_share_negative": ["empathy_negative"],
    "joke_request": [],  # Пока пусто → будет учиться у LLM
}


class DialogueEngine:
    """
    Разговорный движок Кристины — генерация ответов без LLM.

    Три способа ответить (от быстрого к медленному):
    1. DialogueMemory — поиск похожего прошлого диалога
    2. PhraseComposition — сборка ответа из фраз по ситуации
    3. None → LLM fallback → результат ЗАПИСЫВАЕТСЯ для обучения
    """

    def __init__(self, db_path: Path = None):
        self._db_path = db_path or (config.config.data_dir / "dialogue_engine.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        self._create_tables()
        self._seed_base_phrases()

        stats = self.get_stats()
        logger.info(
            f"💬 DialogueEngine: {stats['phrases']} фраз, "
            f"{stats['dialogues']} диалогов, "
            f"{stats['situations']} ситуаций"
        )

    def _create_tables(self):
        cur = self._conn.cursor()

        # ── Фразы: отдельные строительные блоки ──
        cur.execute("""
            CREATE TABLE IF NOT EXISTS phrases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL UNIQUE,
                category TEXT NOT NULL,
                mood TEXT DEFAULT 'neutral',
                weight REAL DEFAULT 1.0,
                times_used INTEGER DEFAULT 0,
                source TEXT DEFAULT 'base',
                created_at REAL NOT NULL
            )
        """)

        # ── Диалоговые паттерны: полные пары input→response ──
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dialogue_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                situations TEXT NOT NULL,
                keywords TEXT NOT NULL,
                response_text TEXT NOT NULL,
                components TEXT DEFAULT '[]',
                mood TEXT DEFAULT 'neutral',
                confidence REAL DEFAULT 1.0,
                successes INTEGER DEFAULT 1,
                failures INTEGER DEFAULT 0,
                source TEXT DEFAULT 'llm',
                created_at REAL NOT NULL,
                last_used REAL NOT NULL
            )
        """)

        # ── FTS для поиска похожих диалогов ──
        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS dialogue_fts
            USING fts5(keywords, content=dialogue_patterns, content_rowid=id)
        """)

        # ── Таблица ситуаций, выученных у LLM ──
        # Когда regex не распознал ситуацию, но LLM ответил
        cur.execute("""
            CREATE TABLE IF NOT EXISTS learned_situations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keywords TEXT NOT NULL,
                situation TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                times_seen INTEGER DEFAULT 1,
                created_at REAL NOT NULL
            )
        """)

        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS situation_fts
            USING fts5(keywords, content=learned_situations, content_rowid=id)
        """)

        # Индексы
        cur.execute("CREATE INDEX IF NOT EXISTS idx_phrases_cat ON phrases(category)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_phrases_mood ON phrases(category, mood)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_dialogue_conf ON dialogue_patterns(confidence DESC)")

        self._conn.commit()

    def _seed_base_phrases(self):
        """Загружает начальный словарь, если БД пустая"""
        count = self._conn.execute("SELECT COUNT(*) as c FROM phrases").fetchone()["c"]
        if count > 0:
            return

        now = time.time()
        for category, phrases in BASE_PHRASES.items():
            for text, mood in phrases:
                self._conn.execute("""
                    INSERT OR IGNORE INTO phrases (text, category, mood, source, created_at)
                    VALUES (?, ?, ?, 'base', ?)
                """, (text, category, mood, now))

        self._conn.commit()
        total = self._conn.execute("SELECT COUNT(*) as c FROM phrases").fetchone()["c"]
        logger.info(f"📚 Загружен начальный словарь: {total} фраз")

    # ═══════════════════════════════════════════════════════════════
    #          РАСПОЗНАВАНИЕ СИТУАЦИИ (Шаг 1 — как у человека)
    # ═══════════════════════════════════════════════════════════════

    def recognize_situations(self, user_input: str) -> List[str]:
        """
        Распознаёт ситуации в тексте пользователя.

        Может вернуть несколько:
          "Привет, как дела?" → ["greeting", "ask_state"]
          "Спасибо, пока!"    → ["gratitude", "farewell"]

        Сначала regex (быстро), потом learned_situations (SQLite).
        """
        situations = []

        # Tier 1: Regex паттерны
        for situation, pattern in SITUATION_PATTERNS.items():
            if pattern.search(user_input):
                situations.append(situation)

        if situations:
            return situations

        # Tier 2: Выученные ситуации (из прошлых LLM-ответов)
        keywords = self._extract_keywords(user_input)
        if keywords:
            try:
                rows = self._conn.execute("""
                    SELECT ls.situation, ls.confidence
                    FROM situation_fts
                    JOIN learned_situations ls ON situation_fts.rowid = ls.id
                    WHERE situation_fts MATCH ?
                    AND ls.confidence >= 0.5
                    ORDER BY ls.confidence DESC, ls.times_seen DESC
                    LIMIT 3
                """, (keywords,)).fetchall()

                for row in rows:
                    if row["situation"] not in situations:
                        situations.append(row["situation"])
            except Exception:
                pass

        return situations

    # ═══════════════════════════════════════════════════════════════
    #          ГЕНЕРАЦИЯ ОТВЕТА (Шаг 2-4 — как у человека)
    # ═══════════════════════════════════════════════════════════════

    def generate_response(
        self,
        user_input: str,
        mood: str = "neutral",
        energy: float = 100.0,
    ) -> Optional[str]:
        """
        Генерирует ответ без LLM.

        Порядок:
        1. Распознаём ситуации
        2. Ищем похожий прошлый диалог (DialogueMemory)
        3. Собираем ответ из фраз (PhraseComposition)
        4. Возвращаем None если не можем → LLM

        Returns:
            str — готовый ответ, или None если нужен LLM.
        """
        situations = self.recognize_situations(user_input)

        if not situations:
            logger.debug(f"❓ Ситуация не распознана: '{user_input[:50]}'")
            return None

        logger.debug(f"🎯 Ситуации: {situations}")

        # ── Способ 1: Поиск похожего прошлого диалога ──
        response = self._find_similar_dialogue(user_input, situations, mood)
        if response:
            logger.debug(f"✅ DialogueMemory hit")
            return response

        # ── Способ 2: Композиция из фраз ──
        response = self._compose_response(situations, mood, energy)
        if response:
            logger.debug(f"✅ PhraseComposition hit")
            return response

        logger.debug(f"⚠️ DialogueEngine: не смог ответить на ситуации {situations}")
        return None

    def _find_similar_dialogue(
        self,
        user_input: str,
        situations: List[str],
        mood: str,
    ) -> Optional[str]:
        """Ищет похожий прошлый диалог в памяти"""

        keywords = self._extract_keywords(user_input)
        if not keywords:
            return None

        try:
            rows = self._conn.execute("""
                SELECT dp.id, dp.response_text, dp.confidence,
                       dp.successes, dp.failures, dp.mood
                FROM dialogue_fts
                JOIN dialogue_patterns dp ON dialogue_fts.rowid = dp.id
                WHERE dialogue_fts MATCH ?
                AND dp.confidence >= 0.6
                ORDER BY dp.successes DESC, dp.confidence DESC
                LIMIT 5
            """, (keywords,)).fetchall()
        except Exception:
            return None

        if not rows:
            return None

        # Ранжируем: предпочитаем совпадение по настроению
        best = None
        best_score = 0

        for row in rows:
            score = row["confidence"] * (row["successes"] / (row["failures"] + 1))
            # Бонус за совпадение настроения
            if row["mood"] == mood:
                score *= 1.3
            if score > best_score:
                best_score = score
                best = row

        if not best:
            return None

        # Обновляем last_used
        self._conn.execute("""
            UPDATE dialogue_patterns SET last_used = ? WHERE id = ?
        """, (time.time(), best["id"]))
        self._conn.commit()

        return best["response_text"]

    def _compose_response(
        self,
        situations: List[str],
        mood: str,
        energy: float,
    ) -> Optional[str]:
        """
        Собирает ответ из фраз — как человек комбинирует слова.

        "greeting" + "ask_state" →
            выбрать фразу из "greeting" +
            выбрать фразу из "state_{mood}" +
            выбрать фразу из "offer_help"
        """
        # Определяем blueprint (план ответа)
        blueprint_key = "+".join(situations)
        blueprint = RESPONSE_BLUEPRINTS.get(blueprint_key)

        # Если нет комбинированного blueprint, берём по первой ситуации
        if blueprint is None:
            blueprint = RESPONSE_BLUEPRINTS.get(situations[0])

        if not blueprint:
            return None

        # Определяем mood-категорию
        mood_category = self._mood_to_category(mood, energy)

        # Собираем фразы
        parts = []
        for category_template in blueprint:
            category = category_template.replace("{mood}", mood_category)

            phrase = self._pick_phrase(category, mood)
            if phrase:
                parts.append(phrase)

        if not parts:
            return None

        return " ".join(parts)

    def _pick_phrase(self, category: str, mood: str) -> Optional[str]:
        """
        Выбирает фразу из категории.

        Предпочитает:
        1. Фразы с подходящим настроением
        2. Нейтральные фразы
        3. Менее использованные (для разнообразия)
        """
        # Сначала ищем с точным mood
        rows = self._conn.execute("""
            SELECT text, weight, times_used FROM phrases
            WHERE category = ? AND mood = ?
            ORDER BY weight DESC
            LIMIT 10
        """, (category, mood)).fetchall()

        # Если нет — берём neutral
        if not rows:
            rows = self._conn.execute("""
                SELECT text, weight, times_used FROM phrases
                WHERE category = ?
                ORDER BY weight DESC
                LIMIT 10
            """, (category,)).fetchall()

        if not rows:
            return None

        # Взвешенный случайный выбор (чаще используем сильные фразы,
        # но иногда — редкие, для разнообразия)
        total_weight = sum(r["weight"] for r in rows)
        if total_weight <= 0:
            return rows[0]["text"]

        r = random.random() * total_weight
        cumulative = 0
        for row in rows:
            cumulative += row["weight"]
            if r <= cumulative:
                # Обновляем счётчик использования
                self._conn.execute("""
                    UPDATE phrases SET times_used = times_used + 1
                    WHERE text = ? AND category = ?
                """, (row["text"], category))
                self._conn.commit()
                return row["text"]

        return rows[0]["text"]

    def _mood_to_category(self, mood: str, energy: float) -> str:
        """Преобразует mood в категорию state-фраз"""
        if energy < 30:
            return "tired"
        happy_moods = {"happy", "satisfied", "curious"}
        if mood in happy_moods:
            return "positive"
        return "neutral"

    # ═══════════════════════════════════════════════════════════════
    #          ОБУЧЕНИЕ (LLM как учитель)
    # ═══════════════════════════════════════════════════════════════

    def learn_from_dialogue(
        self,
        user_input: str,
        response: str,
        mood: str = "neutral",
        source: str = "llm",
    ):
        """
        Учится на диалоге: запоминает пару (input → response)
        и разбирает ответ на фразы.

        Вызывается ПОСЛЕ каждого LLM-ответа на диалоговый запрос.
        """
        situations = self.recognize_situations(user_input)
        keywords = self._extract_keywords(user_input)
        now = time.time()

        # 1. Сохраняем полный диалоговый паттерн
        if keywords:
            components = self._decompose_response(response)
            try:
                cur = self._conn.execute("""
                    INSERT INTO dialogue_patterns
                    (situations, keywords, response_text, components, mood,
                     source, created_at, last_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    json.dumps(situations),
                    keywords,
                    response,
                    json.dumps(components),
                    mood,
                    source,
                    now,
                    now,
                ))

                # Обновляем FTS
                self._conn.execute("""
                    INSERT INTO routing_fts (rowid, keywords)
                    VALUES (?, ?)
                """, (cur.lastrowid, keywords))
            except Exception:
                # FTS table might be named differently, use dialogue_fts
                try:
                    rowid = self._conn.execute("""
                        INSERT INTO dialogue_patterns
                        (situations, keywords, response_text, components, mood,
                         source, created_at, last_used)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        json.dumps(situations),
                        keywords,
                        response,
                        json.dumps(components),
                        mood,
                        source,
                        now,
                        now,
                    )).lastrowid

                    self._conn.execute("""
                        INSERT INTO dialogue_fts (rowid, keywords)
                        VALUES (?, ?)
                    """, (rowid, keywords))
                except Exception as e:
                    logger.debug(f"Duplicate or error: {e}")

        # 2. Разбираем ответ на фразы и добавляем в PhraseBank
        self._learn_phrases_from_response(response, situations, mood, source)

        # 3. Если ситуация была распознана — запоминаем для будущего
        if situations and keywords:
            for situation in situations:
                try:
                    existing = self._conn.execute("""
                        SELECT id FROM learned_situations
                        WHERE keywords = ? AND situation = ?
                    """, (keywords, situation)).fetchone()

                    if existing:
                        self._conn.execute("""
                            UPDATE learned_situations
                            SET times_seen = times_seen + 1
                            WHERE id = ?
                        """, (existing["id"],))
                    else:
                        cur = self._conn.execute("""
                            INSERT INTO learned_situations
                            (keywords, situation, created_at)
                            VALUES (?, ?, ?)
                        """, (keywords, situation, now))

                        self._conn.execute("""
                            INSERT INTO situation_fts (rowid, keywords)
                            VALUES (?, ?)
                        """, (cur.lastrowid, keywords))
                except Exception:
                    pass

        self._conn.commit()
        logger.debug(
            f"📝 Learned dialogue: situations={situations}, "
            f"phrases extracted"
        )

    def _learn_phrases_from_response(
        self,
        response: str,
        situations: List[str],
        mood: str,
        source: str,
    ):
        """
        Разбирает ответ LLM на фразы и добавляет в PhraseBank.

        "Привет! Всё отлично! Чем могу помочь?"
        → "Привет!" → greeting
        → "Всё отлично!" → state_positive
        → "Чем могу помочь?" → offer_help
        """
        # Разбиваем на предложения
        sentences = self._split_into_sentences(response)
        if not sentences:
            return

        now = time.time()

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 3 or len(sentence) > 200:
                continue

            # Определяем категорию фразы
            category = self._classify_phrase(sentence, situations)
            if not category:
                continue

            # Добавляем в PhraseBank
            try:
                self._conn.execute("""
                    INSERT OR IGNORE INTO phrases
                    (text, category, mood, weight, source, created_at)
                    VALUES (?, ?, ?, 1.0, ?, ?)
                """, (sentence, category, mood, source, now))
            except Exception:
                pass

    def _decompose_response(self, response: str) -> List[str]:
        """Определяет из каких типов фраз состоит ответ"""
        sentences = self._split_into_sentences(response)
        components = []
        for s in sentences:
            cat = self._classify_phrase(s.strip(), [])
            if cat:
                components.append(cat)
        return components

    def _classify_phrase(self, sentence: str, context_situations: List[str]) -> Optional[str]:
        """
        Классифицирует фразу по категории.

        "Привет!" → greeting
        "Всё хорошо!" → state_positive
        "Чем помочь?" → offer_help
        """
        s = sentence.lower().strip()

        # Приветствия
        if re.match(r'^(?:привет|здравствуй|здорово|приветик|хай|салют|добр)', s):
            return "greeting"

        # Прощания
        if re.match(r'^(?:пока|до\s+(?:свидания|встречи)|удачи|всего\s+доброго)', s):
            return "farewell"

        # Состояние — позитивное
        if re.search(r'(?:отлично|хорошо|замечательно|прекрасно|в\s+порядке|супер)', s):
            if re.search(r'(?:всё|у\s+меня|дела|чувствую)', s) or len(s) < 30:
                return "state_positive"

        # Состояние — нейтральное
        if re.search(r'(?:нормально|стабильно|потихоньку|работаю)', s):
            return "state_neutral"

        # Предложение помочь
        if re.search(r'(?:помо[гчщ]|нужно\s+сделать|слушаю|рассказывай|займёмся)', s):
            return "offer_help"

        # Благодарность в ответ
        if re.search(r'(?:пожалуйста|рада?\s+помо|обращайся|не\s+за\s+что)', s):
            return "gratitude_response"

        # Самопрезентация
        if re.search(r'(?:меня\s+зовут|я\s+(?:—|-)?\s*кристина|я\s+ai|я\s+ассистент)', s):
            return "self_intro"

        # Комплимент-ответ
        if re.search(r'(?:спасибо.*приятно|стараюсь|приятно\s+слышать)', s):
            return "compliment_response"

        # Сочувствие
        if re.search(r'(?:понимаю|жаль|держись|непросто|пройдёт)', s):
            return "empathy_negative"

        # Радость за собеседника
        if re.search(r'(?:рад[аы]?\s+за|здорово|так\s+держать|отличная\s+новость)', s):
            return "empathy_positive"

        # Извинение
        if re.search(r'(?:извини|прости|постараюсь.*лучше)', s):
            return "complaint_response"

        return None

    def _split_into_sentences(self, text: str) -> List[str]:
        """Разбивает текст на предложения"""
        # Разделяем по .!? и переносам строк
        sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
        return [s.strip() for s in sentences if s.strip()]

    # ═══════════════════════════════════════════════════════════════
    #          ОБРАТНАЯ СВЯЗЬ
    # ═══════════════════════════════════════════════════════════════

    def reinforce_dialogue(self, pattern_id: int):
        """Пользователь доволен → усиливаем паттерн"""
        self._conn.execute("""
            UPDATE dialogue_patterns
            SET successes = successes + 1,
                confidence = MIN(1.0, confidence + 0.05)
            WHERE id = ?
        """, (pattern_id,))
        self._conn.commit()

    def weaken_dialogue(self, pattern_id: int):
        """Пользователь недоволен → ослабляем паттерн"""
        self._conn.execute("""
            UPDATE dialogue_patterns
            SET failures = failures + 1,
                confidence = MAX(0.0, confidence - 0.15)
            WHERE id = ?
        """, (pattern_id,))
        self._conn.commit()

    # ═══════════════════════════════════════════════════════════════
    #          УТИЛИТЫ
    # ═══════════════════════════════════════════════════════════════

    def _extract_keywords(self, text: str) -> str:
        stop_words = {
            "я", "ты", "он", "она", "мы", "вы", "они", "мне", "мой", "твой",
            "для", "меня", "тебя", "его", "неё",
            "в", "на", "и", "с", "по", "от", "к", "не", "что", "это", "как",
            "но", "а", "или", "да", "нет", "бы", "ли", "же", "вот", "так",
            "привет", "пожалуйста", "спасибо", "можешь",
        }
        words = []
        for word in re.findall(r'[а-яёa-z0-9]+', text.lower()):
            if len(word) > 2 and word not in stop_words:
                words.append(word)
        return " ".join(words[:15])

    def get_stats(self) -> Dict[str, int]:
        phrases = self._conn.execute(
            "SELECT COUNT(*) as c FROM phrases"
        ).fetchone()["c"]
        dialogues = self._conn.execute(
            "SELECT COUNT(*) as c FROM dialogue_patterns"
        ).fetchone()["c"]
        situations = self._conn.execute(
            "SELECT COUNT(*) as c FROM learned_situations"
        ).fetchone()["c"]
        llm_phrases = self._conn.execute(
            "SELECT COUNT(*) as c FROM phrases WHERE source = 'llm'"
        ).fetchone()["c"]

        return {
            "phrases": phrases,
            "phrases_from_llm": llm_phrases,
            "dialogues": dialogues,
            "situations": situations,
        }

    def close(self):
        self._conn.close()

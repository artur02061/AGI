"""
Кристина 7.1 — NeuralEngine (Нейронное обучение слов)

КАК ЭТО РАБОТАЕТ (аналогия с человеком):
  Ребёнок слышит слова → запоминает в каком контексте →
  понимает что "отлично" и "замечательно" значат одно и то же →
  строит СВОИ предложения из понятых слов.

АРХИТЕКТУРА:
  ┌──────────────────────────────────────────────────┐
  │ 1. Word2Vec (Skip-gram)                         │
  │    "привет" → [0.12, -0.34, 0.56, ...]          │
  │    "здравствуй" → [0.11, -0.33, 0.55, ...]      │
  │    (близкие вектора = похожий смысл)             │
  └──────────────┬───────────────────────────────────┘
                 ↓
  ┌──────────────────────────────────────────────────┐
  │ 2. WordKnowledge (граф знаний)                   │
  │    "файл" → {pos: "noun", assoc: ["создать",     │
  │              "открыть", "удалить"], role: "object"}│
  └──────────────┬───────────────────────────────────┘
                 ↓
  ┌──────────────────────────────────────────────────┐
  │ 3. N-gram Model (переходы между словами)         │
  │    "я" → "могу"(0.3), "буду"(0.2), "хочу"(0.15) │
  │    "могу" → "помочь"(0.4), "сделать"(0.3)        │
  └──────────────┬───────────────────────────────────┘
                 ↓
  ┌──────────────────────────────────────────────────┐
  │ 4. SentenceBuilder                               │
  │    intent="offer_help", mood="happy"              │
  │    seed="рада" → "рада помочь тебе !" (НОВОЕ)    │
  └──────────────────────────────────────────────────┘

ОБУЧЕНИЕ:
  Каждый LLM-ответ и каждая фраза пользователя:
  1. Токенизируется → слова добавляются в словарь
  2. Word2Vec обучается на контекстных парах (skip-gram)
  3. N-gram модель обновляет вероятности переходов
  4. WordKnowledge обновляет ассоциации и роли слов
"""

import sqlite3
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set

from utils.logging import get_logger
import config

logger = get_logger("neural_engine")

# ═══════════════════════════════════════════════════════════════
#               КОНСТАНТЫ
# ═══════════════════════════════════════════════════════════════

EMBEDDING_DIM = 128        # Размерность вектора слова (v7.3: 64→128 для лучшего разделения смыслов)
LEARNING_RATE = 0.025      # Скорость обучения
MIN_LEARNING_RATE = 0.001  # Минимальная скорость
WINDOW_SIZE = 3            # Окно контекста (±3 слова)
NEGATIVE_SAMPLES = 5       # Негативных сэмплов при обучении
MIN_WORD_FREQ = 1          # Мин. частота слова для эмбеддинга
MAX_SENTENCE_LEN = 20      # Макс. длина генерируемого предложения
MIN_SENTENCE_LEN = 3       # Мин. длина генерируемого предложения

# Русские стоп-слова (не учим эмбеддинги, но используем в генерации)
STOP_WORDS = {
    "и", "в", "на", "с", "по", "к", "от", "за", "из", "у", "о",
    "а", "но", "же", "ли", "бы", "не", "ни", "да", "нет",
}

# Паттерны для определения части речи (упрощённая морфология)
POS_PATTERNS = {
    "verb": re.compile(
        r'(?:ать|ять|еть|ить|уть|ыть|ти|чь|'
        r'аю|яю|ую|ешь|ишь|ет|ит|ем|им|ете|ите|ут|ют|ат|ят|'
        r'ал|ял|ел|ил|ала|яла|ела|ила|ало|яло|ело|ило|али|яли|ели|или|'
        r'ай|ей|уй|ой|айте|ейте|уйте|ойте)$'
    ),
    "noun": re.compile(
        r'(?:[а-я]+(?:ость|ение|ание|ство|тель|ник|чик|щик|ка|ция|'
        r'ие|ье|тия|зия|ия|ей|ов|ам|ами|ах))$'
    ),
    "adjective": re.compile(
        r'(?:[а-я]+(?:ый|ий|ой|ая|яя|ое|ее|ые|ие|ому|ему|ой|ей|'
        r'ым|им|ыми|ими|ых|их))$'
    ),
    "adverb": re.compile(
        r'(?:[а-я]+(?:но|ко|ски|чески|ьно|ело|сто|жно|чно|тно))$'
    ),
    "pronoun": re.compile(
        r'^(?:я|ты|он|она|оно|мы|вы|они|меня|тебя|его|её|нас|вас|их|'
        r'мне|тебе|ему|ей|нам|вам|им|мной|тобой|ним|ней|нами|вами|ними|'
        r'мой|твой|наш|ваш|его|её|их|свой|этот|тот|такой|какой|'
        r'что|кто|который|чей|сколько|столько)$'
    ),
}

# Категории слов для генерации (seed-слова по ситуациям)
SITUATION_SEEDS = {
    "greeting": ["привет", "здравствуй", "рада", "добрый"],
    "farewell": ["пока", "удачи", "встречи", "связи"],
    "offer_help": ["помочь", "помогу", "сделать", "нужно", "давай"],
    "state_positive": ["хорошо", "отлично", "замечательно", "прекрасно", "рада"],
    "state_neutral": ["нормально", "стабильно", "работаю", "порядке"],
    "state_tired": ["устала", "тяжело", "справлюсь"],
    "gratitude_response": ["пожалуйста", "рада", "обращайся"],
    "self_intro": ["кристина", "зовут", "ассистент", "помогаю"],
    "compliment_response": ["спасибо", "приятно", "стараюсь"],
    "empathy_positive": ["рада", "здорово", "отлично", "молодец"],
    "empathy_negative": ["понимаю", "бывает", "держись", "здесь"],
    "complaint_response": ["извини", "постараюсь", "исправлю", "лучше"],
}


def _sigmoid(x: float) -> float:
    """Стабильная сигмоида"""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Косинусное сходство двух векторов"""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return dot / (norm1 * norm2)


def _random_vector(dim: int) -> List[float]:
    """Случайный вектор для инициализации эмбеддинга"""
    return [(random.random() - 0.5) / dim for _ in range(dim)]


class NeuralEngine:
    """
    Нейронный движок Кристины — понимание слов и генерация предложений.

    Три компонента:
    1. Word2Vec — эмбеддинги слов (понимание значения)
    2. N-gram Model — вероятности переходов (структура предложений)
    3. WordKnowledge — часть речи, ассоциации (знание о словах)

    Обучается инкрементально из каждого диалога.
    """

    def __init__(self, db_path: Path = None):
        self._db_path = db_path or (config.config.data_dir / "neural_engine.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        self._create_tables()

        # In-memory кеш эмбеддингов для быстрого доступа
        self._embeddings_cache: Dict[str, List[float]] = {}
        self._word_freq: Dict[str, int] = {}
        self._total_words = 0

        # N-gram кеш
        self._bigrams: Dict[str, Dict[str, int]] = {}
        self._trigrams: Dict[Tuple[str, str], Dict[str, int]] = {}

        # Загружаем кеш из SQLite
        self._load_cache()

        stats = self.get_stats()
        logger.info(
            f"🧠 NeuralEngine: {stats['vocabulary']} слов, "
            f"{stats['embeddings']} эмбеддингов, "
            f"{stats['bigrams']} биграмм, "
            f"{stats['trigrams']} триграмм, "
            f"обучений: {stats['training_steps']}"
        )

    # ═══════════════════════════════════════════════════════════════
    #               ИНИЦИАЛИЗАЦИЯ БД
    # ═══════════════════════════════════════════════════════════════

    def _create_tables(self):
        cur = self._conn.cursor()

        # ── Словарь: все известные слова ──
        cur.execute("""
            CREATE TABLE IF NOT EXISTS vocabulary (
                word TEXT PRIMARY KEY,
                frequency INTEGER DEFAULT 1,
                pos TEXT DEFAULT 'unknown',
                embedding TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)

        # ── Биграммы: пары слов (word1 → word2) с частотой ──
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bigrams (
                word1 TEXT NOT NULL,
                word2 TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                PRIMARY KEY (word1, word2)
            )
        """)

        # ── Триграммы: тройки слов ──
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trigrams (
                word1 TEXT NOT NULL,
                word2 TEXT NOT NULL,
                word3 TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                PRIMARY KEY (word1, word2, word3)
            )
        """)

        # ── Ассоциации: какие слова часто рядом ──
        cur.execute("""
            CREATE TABLE IF NOT EXISTS associations (
                word1 TEXT NOT NULL,
                word2 TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                context TEXT DEFAULT '',
                PRIMARY KEY (word1, word2)
            )
        """)

        # ── Слово-ситуация: в каких ситуациях встречается слово ──
        cur.execute("""
            CREATE TABLE IF NOT EXISTS word_situations (
                word TEXT NOT NULL,
                situation TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                PRIMARY KEY (word, situation)
            )
        """)

        # ── Статистика обучения ──
        cur.execute("""
            CREATE TABLE IF NOT EXISTS training_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                source TEXT NOT NULL,
                words_processed INTEGER DEFAULT 0,
                pairs_trained INTEGER DEFAULT 0,
                loss REAL DEFAULT 0.0
            )
        """)

        # Индексы
        cur.execute("CREATE INDEX IF NOT EXISTS idx_vocab_freq ON vocabulary(frequency DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_bigram_w1 ON bigrams(word1)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_trigram_w1w2 ON trigrams(word1, word2)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_assoc_w1 ON associations(word1)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_wordsit_word ON word_situations(word)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_wordsit_sit ON word_situations(situation)")

        self._conn.commit()

    def _load_cache(self):
        """Загружает словарь и эмбеддинги в память"""
        # Словарь и частоты
        rows = self._conn.execute(
            "SELECT word, frequency, embedding FROM vocabulary"
        ).fetchall()

        migrated = 0
        for row in rows:
            word = row["word"]
            self._word_freq[word] = row["frequency"]
            self._total_words += row["frequency"]
            if row["embedding"]:
                try:
                    emb = json.loads(row["embedding"])
                    # Миграция: расширяем старые вектора до EMBEDDING_DIM
                    if len(emb) < EMBEDDING_DIM:
                        extra = EMBEDDING_DIM - len(emb)
                        emb.extend(
                            (random.random() - 0.5) / EMBEDDING_DIM
                            for _ in range(extra)
                        )
                        migrated += 1
                    self._embeddings_cache[word] = emb
                except (json.JSONDecodeError, TypeError):
                    pass

        if migrated > 0:
            logger.info(f"📏 Migrated {migrated} embeddings to {EMBEDDING_DIM}-dim")
            self._save_all_embeddings()

        # Биграммы
        rows = self._conn.execute(
            "SELECT word1, word2, frequency FROM bigrams"
        ).fetchall()

        for row in rows:
            w1 = row["word1"]
            if w1 not in self._bigrams:
                self._bigrams[w1] = {}
            self._bigrams[w1][row["word2"]] = row["frequency"]

        # Триграммы (загружаем только часто встречающиеся)
        rows = self._conn.execute(
            "SELECT word1, word2, word3, frequency FROM trigrams "
            "WHERE frequency >= 2"
        ).fetchall()

        for row in rows:
            key = (row["word1"], row["word2"])
            if key not in self._trigrams:
                self._trigrams[key] = {}
            self._trigrams[key][row["word3"]] = row["frequency"]

    # ═══════════════════════════════════════════════════════════════
    #     ТОКЕНИЗАЦИЯ (разбиение текста на слова)
    # ═══════════════════════════════════════════════════════════════

    def tokenize(self, text: str) -> List[str]:
        """
        Токенизирует текст на слова.

        Сохраняет знаки препинания как отдельные токены
        (нужны для генерации предложений).
        """
        text = text.lower().strip()
        # Разделяем слова и знаки препинания
        tokens = re.findall(r'[а-яёa-z0-9]+|[.!?,;:—\-]', text)
        return [t for t in tokens if t]

    def _guess_pos(self, word: str) -> str:
        """Определяет часть речи по морфологическим паттернам"""
        w = word.lower()

        for pos, pattern in POS_PATTERNS.items():
            if pattern.search(w):
                return pos

        # Короткие слова — часто местоимения или частицы
        if len(w) <= 2:
            return "particle"

        return "unknown"

    # ═══════════════════════════════════════════════════════════════
    #     ОБУЧЕНИЕ: WORD2VEC (Skip-gram + Negative Sampling)
    # ═══════════════════════════════════════════════════════════════

    def learn_from_text(
        self,
        text: str,
        source: str = "dialogue",
        situations: List[str] = None,
    ):
        """
        Учится на тексте:
        1. Добавляет слова в словарь
        2. Обновляет n-gram модель
        3. Обучает Word2Vec эмбеддинги
        4. Записывает ассоциации

        Вызывается для КАЖДОГО LLM-ответа и реплики пользователя.
        """
        tokens = self.tokenize(text)
        if len(tokens) < 2:
            return

        now = time.time()
        words_only = [t for t in tokens if re.match(r'[а-яёa-z]', t)]

        # 1. Обновляем словарь
        self._update_vocabulary(words_only, now)

        # 2. Обновляем n-gram модель
        self._update_ngrams(tokens)

        # 3. Обучаем Word2Vec
        pairs_trained = self._train_word2vec(words_only)

        # 4. Обновляем ассоциации (слова в одном предложении)
        self._update_associations(words_only)

        # 5. Привязываем слова к ситуациям
        if situations:
            self._update_word_situations(words_only, situations)

        # 6. Логируем
        self._conn.execute("""
            INSERT INTO training_log (timestamp, source, words_processed, pairs_trained)
            VALUES (?, ?, ?, ?)
        """, (now, source, len(words_only), pairs_trained))
        self._conn.commit()

        logger.debug(
            f"📖 NeuralEngine: learned {len(words_only)} words, "
            f"{pairs_trained} pairs from '{text[:40]}...'"
        )

    def _update_vocabulary(self, words: List[str], now: float):
        """Добавляет/обновляет слова в словаре"""
        for word in words:
            if word in STOP_WORDS and len(word) <= 2:
                continue

            if word in self._word_freq:
                self._word_freq[word] += 1
                self._conn.execute("""
                    UPDATE vocabulary SET frequency = frequency + 1, updated_at = ?
                    WHERE word = ?
                """, (now, word))
            else:
                pos = self._guess_pos(word)
                self._word_freq[word] = 1
                # Инициализируем эмбеддинг
                emb = _random_vector(EMBEDDING_DIM)
                self._embeddings_cache[word] = emb

                self._conn.execute("""
                    INSERT OR IGNORE INTO vocabulary
                    (word, frequency, pos, embedding, created_at, updated_at)
                    VALUES (?, 1, ?, ?, ?, ?)
                """, (word, pos, json.dumps(emb), now, now))

            self._total_words += 1

    def _update_ngrams(self, tokens: List[str]):
        """Обновляет биграммы и триграммы"""
        # Добавляем маркеры начала/конца предложения
        # Разбиваем по знакам конца предложения
        sentences = []
        current = ["<S>"]
        for t in tokens:
            if t in ".!?":
                current.append(t)
                current.append("</S>")
                sentences.append(current)
                current = ["<S>"]
            else:
                current.append(t)
        if len(current) > 1:
            current.append("</S>")
            sentences.append(current)

        for sent in sentences:
            # Биграммы
            for i in range(len(sent) - 1):
                w1, w2 = sent[i], sent[i + 1]
                if w1 not in self._bigrams:
                    self._bigrams[w1] = {}
                self._bigrams[w1][w2] = self._bigrams[w1].get(w2, 0) + 1

                self._conn.execute("""
                    INSERT INTO bigrams (word1, word2, frequency)
                    VALUES (?, ?, 1)
                    ON CONFLICT(word1, word2)
                    DO UPDATE SET frequency = frequency + 1
                """, (w1, w2))

            # Триграммы
            for i in range(len(sent) - 2):
                w1, w2, w3 = sent[i], sent[i + 1], sent[i + 2]
                key = (w1, w2)
                if key not in self._trigrams:
                    self._trigrams[key] = {}
                self._trigrams[key][w3] = self._trigrams[key].get(w3, 0) + 1

                self._conn.execute("""
                    INSERT INTO trigrams (word1, word2, word3, frequency)
                    VALUES (?, ?, ?, 1)
                    ON CONFLICT(word1, word2, word3)
                    DO UPDATE SET frequency = frequency + 1
                """, (w1, w2, w3))

    def _train_word2vec(self, words: List[str]) -> int:
        """
        Skip-gram Word2Vec с Negative Sampling.

        Для каждого слова в тексте:
          - Берём контекстные слова (±WINDOW_SIZE)
          - Обучаем: вектор центрального слова должен быть
            БЛИЗОК к векторам контекста и ДАЛЁК от случайных слов

        Всё на чистом Python — без numpy/torch.
        """
        if len(words) < 3 or len(self._word_freq) < 5:
            return 0

        pairs_trained = 0

        # Адаптивная скорость обучения
        vocab_size = len(self._word_freq)
        lr = max(
            MIN_LEARNING_RATE,
            LEARNING_RATE * (1.0 - self._total_words / max(vocab_size * 1000, 1))
        )

        # Таблица для negative sampling (unigram distribution^0.75)
        neg_table = self._build_neg_table()
        if not neg_table:
            return 0

        for i, center_word in enumerate(words):
            if center_word in STOP_WORDS and len(center_word) <= 2:
                continue

            center_emb = self._embeddings_cache.get(center_word)
            if not center_emb:
                continue

            # Динамический размер окна
            actual_window = random.randint(1, WINDOW_SIZE)

            for j in range(max(0, i - actual_window), min(len(words), i + actual_window + 1)):
                if i == j:
                    continue

                context_word = words[j]
                if context_word in STOP_WORDS and len(context_word) <= 2:
                    continue

                context_emb = self._embeddings_cache.get(context_word)
                if not context_emb:
                    continue

                # === Positive sample: center + context ===
                dot = sum(a * b for a, b in zip(center_emb, context_emb))
                # Ограничиваем для стабильности
                dot = max(-6.0, min(6.0, dot))
                sig = _sigmoid(dot)
                grad = lr * (1.0 - sig)

                # Обновляем оба вектора
                for k in range(EMBEDDING_DIM):
                    old_center = center_emb[k]
                    center_emb[k] += grad * context_emb[k]
                    context_emb[k] += grad * old_center

                # === Negative samples: center + random words ===
                for _ in range(NEGATIVE_SAMPLES):
                    neg_word = neg_table[random.randint(0, len(neg_table) - 1)]
                    if neg_word == center_word or neg_word == context_word:
                        continue

                    neg_emb = self._embeddings_cache.get(neg_word)
                    if not neg_emb:
                        continue

                    dot = sum(a * b for a, b in zip(center_emb, neg_emb))
                    dot = max(-6.0, min(6.0, dot))
                    sig = _sigmoid(dot)
                    neg_grad = lr * sig  # Отталкиваем

                    for k in range(EMBEDDING_DIM):
                        center_emb[k] -= neg_grad * neg_emb[k]
                        neg_emb[k] -= neg_grad * center_emb[k]

                pairs_trained += 1

        # Сохраняем обновлённые эмбеддинги в БД
        self._save_embeddings(words)

        return pairs_trained

    def _build_neg_table(self, table_size: int = 1000) -> List[str]:
        """
        Строит таблицу для negative sampling.
        Вероятность выбора слова ∝ freq^0.75
        """
        if not self._word_freq:
            return []

        # Берём только слова с эмбеддингами
        words_with_emb = [
            w for w in self._word_freq
            if w in self._embeddings_cache and w not in STOP_WORDS
        ]
        if not words_with_emb:
            return []

        # Вычисляем веса
        total_pow = sum(
            self._word_freq[w] ** 0.75 for w in words_with_emb
        )
        if total_pow == 0:
            return words_with_emb[:table_size]

        table = []
        for word in words_with_emb:
            weight = self._word_freq[word] ** 0.75
            count = max(1, int(weight / total_pow * table_size))
            table.extend([word] * count)

        return table[:table_size * 2]  # Ограничиваем размер

    def _save_embeddings(self, words: List[str]):
        """Сохраняет обновлённые эмбеддинги в SQLite"""
        now = time.time()
        for word in set(words):
            emb = self._embeddings_cache.get(word)
            if emb:
                self._conn.execute("""
                    UPDATE vocabulary SET embedding = ?, updated_at = ?
                    WHERE word = ?
                """, (json.dumps(emb), now, word))

    def _save_all_embeddings(self):
        """Сохраняет ВСЕ эмбеддинги (для миграции размерности)"""
        now = time.time()
        for word, emb in self._embeddings_cache.items():
            self._conn.execute("""
                UPDATE vocabulary SET embedding = ?, updated_at = ?
                WHERE word = ?
            """, (json.dumps(emb), now, word))
        self._conn.commit()

    def _update_associations(self, words: List[str]):
        """Обновляет ассоциации между словами в одном предложении"""
        # Берём только значимые слова
        meaningful = [
            w for w in words
            if w not in STOP_WORDS and len(w) > 2
        ]

        for i, w1 in enumerate(meaningful):
            for j in range(i + 1, min(i + 5, len(meaningful))):
                w2 = meaningful[j]
                if w1 == w2:
                    continue

                # Сила ассоциации убывает с расстоянием
                distance = j - i
                strength_delta = 1.0 / distance

                self._conn.execute("""
                    INSERT INTO associations (word1, word2, strength)
                    VALUES (?, ?, ?)
                    ON CONFLICT(word1, word2)
                    DO UPDATE SET strength = strength + ?
                """, (w1, w2, strength_delta, strength_delta))

                # Обратная ассоциация (слабее)
                self._conn.execute("""
                    INSERT INTO associations (word1, word2, strength)
                    VALUES (?, ?, ?)
                    ON CONFLICT(word1, word2)
                    DO UPDATE SET strength = strength + ?
                """, (w2, w1, strength_delta * 0.5, strength_delta * 0.5))

    def _update_word_situations(self, words: List[str], situations: List[str]):
        """Привязывает слова к ситуациям (в каких контекстах встречаются)"""
        meaningful = [w for w in words if w not in STOP_WORDS and len(w) > 2]

        for word in meaningful:
            for situation in situations:
                self._conn.execute("""
                    INSERT INTO word_situations (word, situation, frequency)
                    VALUES (?, ?, 1)
                    ON CONFLICT(word, situation)
                    DO UPDATE SET frequency = frequency + 1
                """, (word, situation))

    # ═══════════════════════════════════════════════════════════════
    #     ПОНИМАНИЕ СЛОВ
    # ═══════════════════════════════════════════════════════════════

    def word_meaning(self, word: str) -> Optional[Dict]:
        """
        Что Кристина знает о слове:
        - вектор (эмбеддинг)
        - часть речи
        - частота использования
        - похожие слова
        - ассоциации
        - в каких ситуациях встречается
        """
        word = word.lower()

        row = self._conn.execute(
            "SELECT * FROM vocabulary WHERE word = ?", (word,)
        ).fetchone()

        if not row:
            return None

        # Похожие по смыслу (через эмбеддинги)
        similar = self.find_similar_words(word, top_n=5)

        # Ассоциации
        assoc_rows = self._conn.execute("""
            SELECT word2, strength FROM associations
            WHERE word1 = ?
            ORDER BY strength DESC
            LIMIT 10
        """, (word,)).fetchall()

        # Ситуации
        sit_rows = self._conn.execute("""
            SELECT situation, frequency FROM word_situations
            WHERE word = ?
            ORDER BY frequency DESC
            LIMIT 5
        """, (word,)).fetchall()

        return {
            "word": word,
            "pos": row["pos"],
            "frequency": row["frequency"],
            "has_embedding": row["embedding"] is not None,
            "similar_words": similar,
            "associations": [
                {"word": r["word2"], "strength": round(r["strength"], 2)}
                for r in assoc_rows
            ],
            "situations": [
                {"situation": r["situation"], "frequency": r["frequency"]}
                for r in sit_rows
            ],
        }

    def find_similar_words(
        self,
        word: str,
        top_n: int = 10,
        pos_filter: str = None,
    ) -> List[Tuple[str, float]]:
        """
        Находит слова с похожим значением через косинусное сходство.

        "отлично" → [("замечательно", 0.92), ("прекрасно", 0.88), ...]
        """
        word = word.lower()
        emb = self._embeddings_cache.get(word)
        if not emb:
            return []

        similarities = []

        for other_word, other_emb in self._embeddings_cache.items():
            if other_word == word:
                continue
            if pos_filter:
                row = self._conn.execute(
                    "SELECT pos FROM vocabulary WHERE word = ?",
                    (other_word,)
                ).fetchone()
                if row and row["pos"] != pos_filter:
                    continue

            sim = _cosine_similarity(emb, other_emb)
            similarities.append((other_word, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [(w, round(s, 3)) for w, s in similarities[:top_n]]

    def understand_sentence(self, text: str) -> Dict:
        """
        Анализирует предложение: определяет значение каждого слова,
        общий смысл, ключевые слова.
        """
        tokens = self.tokenize(text)
        words = [t for t in tokens if re.match(r'[а-яёa-z]', t)]

        analysis = {
            "tokens": tokens,
            "words": [],
            "unknown_words": [],
            "understood_pct": 0.0,
        }

        known = 0
        for word in words:
            info = self._conn.execute(
                "SELECT pos, frequency FROM vocabulary WHERE word = ?",
                (word,)
            ).fetchone()

            if info:
                known += 1
                analysis["words"].append({
                    "word": word,
                    "pos": info["pos"],
                    "frequency": info["frequency"],
                    "known": True,
                })
            else:
                analysis["unknown_words"].append(word)
                analysis["words"].append({
                    "word": word,
                    "pos": self._guess_pos(word),
                    "frequency": 0,
                    "known": False,
                })

        if words:
            analysis["understood_pct"] = round(known / len(words) * 100, 1)

        return analysis

    # ═══════════════════════════════════════════════════════════════
    #     ГЕНЕРАЦИЯ ПРЕДЛОЖЕНИЙ
    # ═══════════════════════════════════════════════════════════════

    def generate_sentence(
        self,
        situation: str = None,
        seed_word: str = None,
        mood: str = "neutral",
        max_len: int = MAX_SENTENCE_LEN,
        creativity: float = 0.3,
    ) -> Optional[str]:
        """
        Генерирует НОВОЕ предложение.

        Алгоритм:
        1. Выбираем seed-слово (из ситуации или случайное)
        2. Строим предложение слово-за-словом:
           a. Берём кандидатов из n-gram модели (триграммы > биграммы)
           b. Ранжируем по: n-gram частота + embedding сходство с контекстом
           c. Добавляем случайность (creativity) для разнообразия
        3. Останавливаемся на знаке препинания или max_len

        Args:
            situation: тип ситуации ("greeting", "offer_help", ...)
            seed_word: начальное слово (если None — выбирается из ситуации)
            mood: настроение влияет на выбор слов
            max_len: максимальная длина
            creativity: 0.0=строго по статистике, 1.0=больше случайности

        Returns:
            Сгенерированное предложение или None если недостаточно данных.
        """
        if len(self._bigrams) < 10:
            return None  # Слишком мало данных

        # 1. Выбираем seed-слово
        start_word = self._choose_seed(situation, seed_word, mood)
        if not start_word:
            return None

        # 2. Строим предложение
        sentence = [start_word]
        prev_word = "<S>"
        curr_word = start_word

        for step in range(max_len - 1):
            next_word = self._predict_next(
                prev_word, curr_word, sentence,
                situation=situation,
                creativity=creativity,
            )

            if not next_word or next_word == "</S>":
                break

            # Пунктуация
            if next_word in ".!?,;:—":
                sentence.append(next_word)
                if next_word in ".!?":
                    break
                continue

            sentence.append(next_word)
            prev_word = curr_word
            curr_word = next_word

        # 3. Финализируем
        if len(sentence) < MIN_SENTENCE_LEN:
            return None

        result = self._format_sentence(sentence)
        return result

    def generate_response(
        self,
        situations: List[str],
        mood: str = "neutral",
        max_sentences: int = 2,
        creativity: float = 0.3,
    ) -> Optional[str]:
        """
        Генерирует полный ответ из нескольких предложений.

        Для каждой ситуации генерирует предложение,
        потом соединяет в связный ответ.
        """
        if len(self._bigrams) < 20:
            return None

        parts = []

        for situation in situations[:max_sentences]:
            sentence = self.generate_sentence(
                situation=situation,
                mood=mood,
                creativity=creativity,
            )
            if sentence:
                parts.append(sentence)

        if not parts:
            return None

        return " ".join(parts)

    def _choose_seed(
        self,
        situation: str = None,
        seed_word: str = None,
        mood: str = "neutral",
    ) -> Optional[str]:
        """Выбирает начальное слово для генерации"""

        if seed_word and seed_word in self._word_freq:
            return seed_word

        # Из ситуации — берём слова привязанные к ситуации
        if situation:
            # Сначала из выученных word_situations
            rows = self._conn.execute("""
                SELECT word, frequency FROM word_situations
                WHERE situation = ?
                ORDER BY frequency DESC
                LIMIT 20
            """, (situation,)).fetchall()

            candidates = [
                r["word"] for r in rows
                if r["word"] in self._bigrams  # Слово должно иметь продолжения
            ]

            # Добавляем предустановленные seed-слова
            static_seeds = SITUATION_SEEDS.get(situation, [])
            for sw in static_seeds:
                if sw in self._word_freq and sw not in candidates:
                    candidates.append(sw)

            if candidates:
                # Слова, которые чаще начинают предложения
                start_candidates = []
                for c in candidates:
                    bigram_freq = self._bigrams.get("<S>", {}).get(c, 0)
                    if bigram_freq > 0:
                        start_candidates.append((c, bigram_freq))

                if start_candidates:
                    # Взвешенный выбор
                    total = sum(f for _, f in start_candidates)
                    r = random.random() * total
                    cumulative = 0
                    for word, freq in start_candidates:
                        cumulative += freq
                        if r <= cumulative:
                            return word

                # Если нет стартовых — просто случайный из кандидатов
                return random.choice(candidates)

        # Fallback: случайное слово, которое начинает предложения
        starters = self._bigrams.get("<S>", {})
        if starters:
            words = list(starters.keys())
            freqs = list(starters.values())
            total = sum(freqs)
            r = random.random() * total
            cumulative = 0
            for word, freq in zip(words, freqs):
                cumulative += freq
                if r <= cumulative:
                    if word not in ("</S>",) and re.match(r'[а-яёa-z]', word):
                        return word

        return None

    def _predict_next(
        self,
        prev_word: str,
        curr_word: str,
        sentence: List[str],
        situation: str = None,
        creativity: float = 0.3,
    ) -> Optional[str]:
        """
        Предсказывает следующее слово.

        Комбинирует:
        1. Триграммы (если есть) — самые точные
        2. Биграммы — основной источник
        3. Embedding similarity — семантическая связность
        4. Ситуационный бонус — слова из нужной ситуации
        """
        candidates: Dict[str, float] = {}

        # 1. Триграммы: (prev, curr) → next
        trigram_key = (prev_word, curr_word)
        if trigram_key in self._trigrams:
            tri_total = sum(self._trigrams[trigram_key].values())
            for word, freq in self._trigrams[trigram_key].items():
                candidates[word] = candidates.get(word, 0) + (freq / tri_total) * 3.0

        # 2. Биграммы: curr → next
        if curr_word in self._bigrams:
            bi_total = sum(self._bigrams[curr_word].values())
            for word, freq in self._bigrams[curr_word].items():
                candidates[word] = candidates.get(word, 0) + (freq / bi_total) * 1.0

        if not candidates:
            return None

        # 3. Embedding similarity бонус
        curr_emb = self._embeddings_cache.get(curr_word)
        if curr_emb:
            for word in list(candidates.keys()):
                word_emb = self._embeddings_cache.get(word)
                if word_emb:
                    sim = _cosine_similarity(curr_emb, word_emb)
                    # Небольшой бонус за семантическую близость
                    candidates[word] += max(0, sim) * 0.3

        # 4. Ситуационный бонус
        if situation:
            sit_words = set()
            rows = self._conn.execute("""
                SELECT word FROM word_situations
                WHERE situation = ?
                AND frequency >= 2
            """, (situation,)).fetchall()
            sit_words = {r["word"] for r in rows}

            # Добавляем статические seed-слова
            static = SITUATION_SEEDS.get(situation, [])
            sit_words.update(static)

            for word in candidates:
                if word in sit_words:
                    candidates[word] *= 1.5

        # 5. Штраф за повторение слов в предложении
        for word in candidates:
            if word in sentence and word not in STOP_WORDS:
                candidates[word] *= 0.1

        # 6. Штраф за слишком длинные предложения (стимулируем конец)
        if len(sentence) > 8:
            for word in candidates:
                if word in (".!?", "</S>"):
                    candidates[word] *= 1.5 + (len(sentence) - 8) * 0.3

        # 7. Выбор с учётом creativity
        if not candidates:
            return None

        # Нормализуем
        total = sum(max(0, s) for s in candidates.values())
        if total <= 0:
            return None

        # Temperature sampling
        temperature = 0.5 + creativity  # 0.5-1.5
        scored = []
        for word, score in candidates.items():
            if score > 0:
                adjusted = (score / total) ** (1.0 / temperature)
                scored.append((word, adjusted))

        adj_total = sum(s for _, s in scored)
        if adj_total <= 0:
            return None

        r = random.random() * adj_total
        cumulative = 0
        for word, score in scored:
            cumulative += score
            if r <= cumulative:
                return word

        return scored[0][0] if scored else None

    def _format_sentence(self, tokens: List[str]) -> str:
        """Форматирует токены в предложение"""
        if not tokens:
            return ""

        # Первая буква — заглавная
        result = []
        for i, token in enumerate(tokens):
            if token in ".!?,;:—":
                # Пунктуация без пробела перед ней
                if result:
                    result[-1] = result[-1] + token
                else:
                    result.append(token)
            else:
                if i == 0:
                    result.append(token.capitalize())
                else:
                    result.append(token)

        text = " ".join(result)

        # Если нет завершающего знака — добавляем
        if text and text[-1] not in ".!?":
            text += "."

        return text

    # ═══════════════════════════════════════════════════════════════
    #     УТИЛИТЫ
    # ═══════════════════════════════════════════════════════════════

    def get_vocabulary_size(self) -> int:
        return len(self._word_freq)

    def get_stats(self) -> Dict[str, int]:
        """Статистика нейронного движка"""
        vocabulary = self._conn.execute(
            "SELECT COUNT(*) as c FROM vocabulary"
        ).fetchone()["c"]

        embeddings = len(self._embeddings_cache)

        bigrams = self._conn.execute(
            "SELECT COUNT(*) as c FROM bigrams"
        ).fetchone()["c"]

        trigrams = self._conn.execute(
            "SELECT COUNT(*) as c FROM trigrams"
        ).fetchone()["c"]

        associations = self._conn.execute(
            "SELECT COUNT(*) as c FROM associations"
        ).fetchone()["c"]

        training_steps = self._conn.execute(
            "SELECT COUNT(*) as c FROM training_log"
        ).fetchone()["c"]

        word_situations = self._conn.execute(
            "SELECT COUNT(*) as c FROM word_situations"
        ).fetchone()["c"]

        return {
            "vocabulary": vocabulary,
            "embeddings": embeddings,
            "bigrams": bigrams,
            "trigrams": trigrams,
            "associations": associations,
            "word_situations": word_situations,
            "training_steps": training_steps,
            "total_words_processed": self._total_words,
        }

    def close(self):
        self._conn.commit()
        self._conn.close()

"""
Кристина 7.2 — BPE Tokenizer (Byte-Pair Encoding)

ЗАЧЕМ:
  Обычная токенизация по словам НЕ работает для русского языка:
  - "перезапустить" = неизвестное слово (OOV)
  - "невозможность" = неизвестное слово (OOV)

  BPE разбивает на подслова:
  - "перезапустить" → ["пере", "за", "пуст", "ить"]
  - "невозможность" → ["не", "возможн", "ость"]

  Это даёт:
  1. Нет OOV — ЛЮБОЕ слово разбивается на известные части
  2. Морфология — Кристина понимает приставки, суффиксы, корни
  3. Компактный словарь — 8000-16000 подслов вместо 100K+ слов
  4. Фундамент для трансформера — BPE токены = вход трансформера

АЛГОРИТМ:
  1. Начинаем с символов (каждый символ = токен)
  2. Считаем частоту ПАРЫ соседних токенов
  3. Самую частую пару СЛИВАЕМ в один токен
  4. Повторяем до нужного размера словаря

ОБУЧЕНИЕ:
  Инкрементальное — можно дообучать на новых текстах
  без потери старых merge rules.

ХРАНЕНИЕ:
  SQLite — merge rules + vocabulary (персистентно)
"""

import sqlite3
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict

from utils.logging import get_logger
import config

logger = get_logger("bpe_tokenizer")

# ═══════════════════════════════════════════════════════════════
#               КОНСТАНТЫ
# ═══════════════════════════════════════════════════════════════

DEFAULT_VOCAB_SIZE = 8000        # Целевой размер словаря
MIN_PAIR_FREQ = 2               # Мин. частота пары для слияния
SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<S>": 2,     # Начало предложения
    "</S>": 3,    # Конец предложения
    "<SEP>": 4,   # Разделитель (вопрос/ответ)
    "<MASK>": 5,  # Для masked language modeling
}

# Предобученные частые подслова русского языка (ускоряют начальное обучение)
RUSSIAN_SEED_MERGES = [
    # Приставки
    ("п", "о"), ("п", "ре"), ("пре", "д"), ("н", "е"), ("в", "ы"),
    ("п", "ер"), ("пер", "е"), ("н", "а"), ("з", "а"), ("о", "т"),
    ("п", "ри"), ("в", "о"), ("р", "а"), ("ра", "з"),
    # Суффиксы
    ("н", "о"), ("т", "ь"), ("с", "т"), ("ст", "ь"),
    ("е", "н"), ("ен", "и"), ("ени", "е"),
    ("о", "с"), ("ос", "т"), ("ост", "ь"),
    # Корни
    ("м", "о"), ("мо", "г"), ("мог", "у"),
    ("д", "е"), ("де", "л"), ("дел", "а"),
    ("р", "а"), ("ра", "б"), ("раб", "о"), ("рабо", "т"),
    ("п", "о"), ("по", "м"), ("пом", "о"),
]


class BPETokenizer:
    """
    Byte-Pair Encoding токенизатор для Кристины.

    Учится на диалогах, разбивает текст на подслова.
    Инкрементальное обучение — растёт с каждым новым текстом.

    Использование:
        tokenizer = BPETokenizer()
        tokenizer.train_on_text("Привет! Как дела?")  # обучение
        tokens = tokenizer.encode("Невозможность")     # [23, 45, 67]
        text = tokenizer.decode([23, 45, 67])           # "невозможность"
    """

    def __init__(self, db_path: Path = None, vocab_size: int = DEFAULT_VOCAB_SIZE):
        self._db_path = db_path or (config.config.data_dir / "bpe_tokenizer.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._target_vocab_size = vocab_size

        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        self._create_tables()

        # In-memory состояние
        self._merges: List[Tuple[str, str]] = []     # Упорядоченные merge rules
        self._vocab: Dict[str, int] = {}              # token → id
        self._id_to_token: Dict[int, str] = {}        # id → token
        self._pair_freqs: Counter = Counter()          # Частоты пар (для инкрем. обучения)
        self._word_freqs: Counter = Counter()          # Частоты слов (для обучения)

        # Загружаем состояние
        self._load_state()

        # Если словарь пустой — инициализируем
        if not self._vocab:
            self._init_base_vocab()

        stats = self.get_stats()
        logger.info(
            f"📝 BPE Tokenizer: {stats['vocab_size']} токенов, "
            f"{stats['merge_rules']} merge rules, "
            f"{stats['texts_trained']} текстов обработано"
        )

    # ═══════════════════════════════════════════════════════════════
    #               ИНИЦИАЛИЗАЦИЯ
    # ═══════════════════════════════════════════════════════════════

    def _create_tables(self):
        cur = self._conn.cursor()

        # Merge rules (порядок важен!)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS merge_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_a TEXT NOT NULL,
                token_b TEXT NOT NULL,
                merged TEXT NOT NULL,
                frequency INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                UNIQUE(token_a, token_b)
            )
        """)

        # Vocabulary: token → id
        cur.execute("""
            CREATE TABLE IF NOT EXISTS vocabulary (
                token TEXT PRIMARY KEY,
                token_id INTEGER NOT NULL UNIQUE,
                frequency INTEGER DEFAULT 0,
                is_special INTEGER DEFAULT 0,
                created_at REAL NOT NULL
            )
        """)

        # Word frequencies (для инкрементального обучения)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS word_frequencies (
                word TEXT PRIMARY KEY,
                frequency INTEGER DEFAULT 1,
                updated_at REAL NOT NULL
            )
        """)

        # Статистика обучения
        cur.execute("""
            CREATE TABLE IF NOT EXISTS training_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                texts_count INTEGER DEFAULT 0,
                words_count INTEGER DEFAULT 0,
                merges_added INTEGER DEFAULT 0
            )
        """)

        cur.execute("CREATE INDEX IF NOT EXISTS idx_vocab_id ON vocabulary(token_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_merge_order ON merge_rules(id)")

        self._conn.commit()

    def _init_base_vocab(self):
        """Инициализирует базовый словарь символов + спецтокенов"""
        now = time.time()

        # 1. Специальные токены
        for token, token_id in SPECIAL_TOKENS.items():
            self._vocab[token] = token_id
            self._id_to_token[token_id] = token
            self._conn.execute("""
                INSERT OR IGNORE INTO vocabulary (token, token_id, is_special, created_at)
                VALUES (?, ?, 1, ?)
            """, (token, token_id, now))

        # 2. Базовые символы (русский + латиница + цифры + пунктуация)
        next_id = len(SPECIAL_TOKENS)
        base_chars = (
            "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789"
            " .!?,;:-—()\"'/"
        )
        for char in base_chars:
            if char not in self._vocab:
                self._vocab[char] = next_id
                self._id_to_token[next_id] = char
                self._conn.execute("""
                    INSERT OR IGNORE INTO vocabulary (token, token_id, created_at)
                    VALUES (?, ?, ?)
                """, (char, next_id, now))
                next_id += 1

        self._conn.commit()
        logger.info(f"📝 BPE: initialized base vocabulary with {len(self._vocab)} tokens")

    def _load_state(self):
        """Загружает merge rules и vocabulary из SQLite"""
        # Vocabulary
        rows = self._conn.execute(
            "SELECT token, token_id, frequency FROM vocabulary ORDER BY token_id"
        ).fetchall()
        for row in rows:
            self._vocab[row["token"]] = row["token_id"]
            self._id_to_token[row["token_id"]] = row["token"]

        # Merge rules (порядок критичен!)
        rows = self._conn.execute(
            "SELECT token_a, token_b FROM merge_rules ORDER BY id"
        ).fetchall()
        self._merges = [(row["token_a"], row["token_b"]) for row in rows]

        # Word frequencies
        rows = self._conn.execute(
            "SELECT word, frequency FROM word_frequencies"
        ).fetchall()
        self._word_freqs = Counter({row["word"]: row["frequency"] for row in rows})

    # ═══════════════════════════════════════════════════════════════
    #               ОБУЧЕНИЕ (ИНКРЕМЕНТАЛЬНОЕ)
    # ═══════════════════════════════════════════════════════════════

    def train_on_text(self, text: str, num_merges: int = 50):
        """
        Обучает BPE на новом тексте (инкрементально).

        1. Разбивает текст на слова
        2. Обновляет частоты слов
        3. Выполняет num_merges новых слияний (если есть частые пары)

        Args:
            text: текст для обучения
            num_merges: макс. количество новых merge rules за один вызов
        """
        # Предобработка
        words = self._preprocess_text(text)
        if not words:
            return

        # Обновляем частоты слов
        now = time.time()
        word_counter = Counter(words)
        self._word_freqs.update(word_counter)

        for word, freq in word_counter.items():
            self._conn.execute("""
                INSERT INTO word_frequencies (word, frequency, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(word)
                DO UPDATE SET frequency = frequency + ?, updated_at = ?
            """, (word, freq, now, freq, now))

        # Если словарь ещё не достиг целевого размера — учим новые merge rules
        merges_added = 0
        if len(self._vocab) < self._target_vocab_size:
            merges_added = self._learn_merges(num_merges)

        # Логируем
        self._conn.execute("""
            INSERT INTO training_stats (timestamp, texts_count, words_count, merges_added)
            VALUES (?, 1, ?, ?)
        """, (now, len(words), merges_added))
        self._conn.commit()

        logger.debug(
            f"📝 BPE trained: {len(words)} words, "
            f"{merges_added} new merges, "
            f"vocab={len(self._vocab)}"
        )

    def train_on_corpus(self, texts: List[str], num_merges: int = 500):
        """
        Обучает BPE на корпусе текстов (пакетное обучение).
        Эффективнее, чем train_on_text для каждого текста отдельно.
        """
        all_words = []
        for text in texts:
            all_words.extend(self._preprocess_text(text))

        if not all_words:
            return

        now = time.time()
        word_counter = Counter(all_words)
        self._word_freqs.update(word_counter)

        for word, freq in word_counter.items():
            self._conn.execute("""
                INSERT INTO word_frequencies (word, frequency, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(word)
                DO UPDATE SET frequency = frequency + ?, updated_at = ?
            """, (word, freq, now, freq, now))

        merges_added = self._learn_merges(num_merges)

        self._conn.execute("""
            INSERT INTO training_stats (timestamp, texts_count, words_count, merges_added)
            VALUES (?, ?, ?, ?)
        """, (now, len(texts), len(all_words), merges_added))
        self._conn.commit()

        logger.info(
            f"📝 BPE corpus training: {len(texts)} texts, "
            f"{len(all_words)} words, {merges_added} merges, "
            f"vocab={len(self._vocab)}"
        )

    def _preprocess_text(self, text: str) -> List[str]:
        """Разбивает текст на слова (для BPE обучения)"""
        text = text.lower().strip()
        # Разбиваем на слова (только буквы и цифры)
        words = re.findall(r'[а-яёa-z0-9]+', text)
        return [w for w in words if len(w) >= 2]

    def _learn_merges(self, max_merges: int) -> int:
        """
        Основной алгоритм BPE: находит и сливает частые пары.

        Возвращает количество новых merge rules.
        """
        # Строим текущее представление слов через символы + существующие merges
        word_splits = {}
        for word, freq in self._word_freqs.items():
            if freq < MIN_PAIR_FREQ:
                continue
            split = self._split_word(word)
            if len(split) >= 2:
                word_splits[word] = (split, freq)

        merges_added = 0

        for _ in range(max_merges):
            if len(self._vocab) >= self._target_vocab_size:
                break

            # Считаем частоты пар
            pair_freqs = Counter()
            for word, (split, freq) in word_splits.items():
                for i in range(len(split) - 1):
                    pair = (split[i], split[i + 1])
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            # Находим самую частую пару
            best_pair = pair_freqs.most_common(1)[0]
            pair, freq = best_pair

            if freq < MIN_PAIR_FREQ:
                break

            token_a, token_b = pair
            new_token = token_a + token_b

            # Записываем merge rule
            self._merges.append(pair)
            now = time.time()

            try:
                self._conn.execute("""
                    INSERT INTO merge_rules (token_a, token_b, merged, frequency, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (token_a, token_b, new_token, freq, now))
            except sqlite3.IntegrityError:
                # Пара уже существует, пропускаем
                continue

            # Добавляем новый токен в словарь
            if new_token not in self._vocab:
                new_id = max(self._id_to_token.keys()) + 1 if self._id_to_token else 0
                self._vocab[new_token] = new_id
                self._id_to_token[new_id] = new_token
                self._conn.execute("""
                    INSERT OR IGNORE INTO vocabulary (token, token_id, frequency, created_at)
                    VALUES (?, ?, ?, ?)
                """, (new_token, new_id, freq, now))

            # Обновляем splits всех слов, содержащих эту пару
            for word in list(word_splits.keys()):
                split, wfreq = word_splits[word]
                new_split = self._merge_pair(split, token_a, token_b)
                word_splits[word] = (new_split, wfreq)

            merges_added += 1

        self._conn.commit()
        return merges_added

    def _split_word(self, word: str) -> List[str]:
        """
        Разбивает слово на токены с учётом текущих merge rules.
        Начинаем с символов, затем применяем merges по порядку.
        """
        # Начинаем с отдельных символов
        tokens = list(word)

        # Применяем все merge rules по порядку
        for merge_a, merge_b in self._merges:
            tokens = self._merge_pair(tokens, merge_a, merge_b)
            if len(tokens) == 1:
                break

        return tokens

    @staticmethod
    def _merge_pair(tokens: List[str], a: str, b: str) -> List[str]:
        """Сливает все вхождения пары (a, b) в токенах"""
        if len(tokens) < 2:
            return tokens

        result = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                result.append(a + b)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return result

    # ═══════════════════════════════════════════════════════════════
    #               КОДИРОВАНИЕ / ДЕКОДИРОВАНИЕ
    # ═══════════════════════════════════════════════════════════════

    def encode(self, text: str) -> List[int]:
        """
        Кодирует текст в последовательность token IDs.

        "Привет мир" → [234, 56, 78, 11, 345, 67]
        """
        text = text.lower().strip()
        if not text:
            return []

        token_ids = []

        # Разбиваем на слова и пунктуацию
        parts = re.findall(r'[а-яёa-z0-9]+|[.!?,;:\-—()\s]', text)

        for part in parts:
            if not part.strip() and part == " ":
                # Пробел как токен
                if " " in self._vocab:
                    token_ids.append(self._vocab[" "])
                continue

            if len(part) == 1 and part in self._vocab:
                token_ids.append(self._vocab[part])
                continue

            # Разбиваем слово на BPE-токены
            subtokens = self._split_word(part)
            for st in subtokens:
                if st in self._vocab:
                    token_ids.append(self._vocab[st])
                else:
                    # Неизвестный подтокен — разбиваем на символы
                    for char in st:
                        if char in self._vocab:
                            token_ids.append(self._vocab[char])
                        else:
                            token_ids.append(SPECIAL_TOKENS["<UNK>"])

        return token_ids

    def encode_with_tokens(self, text: str) -> List[Tuple[str, int]]:
        """
        Кодирует текст, возвращая пары (token_text, token_id).
        Полезно для отладки и визуализации.

        "Привет" → [("при", 234), ("вет", 56)]
        """
        text = text.lower().strip()
        if not text:
            return []

        result = []
        parts = re.findall(r'[а-яёa-z0-9]+|[.!?,;:\-—()\s]', text)

        for part in parts:
            if not part.strip() and part == " ":
                if " " in self._vocab:
                    result.append((" ", self._vocab[" "]))
                continue

            if len(part) == 1 and part in self._vocab:
                result.append((part, self._vocab[part]))
                continue

            subtokens = self._split_word(part)
            for st in subtokens:
                if st in self._vocab:
                    result.append((st, self._vocab[st]))
                else:
                    for char in st:
                        tid = self._vocab.get(char, SPECIAL_TOKENS["<UNK>"])
                        result.append((char, tid))

        return result

    def decode(self, token_ids: List[int]) -> str:
        """
        Декодирует последовательность token IDs обратно в текст.

        [234, 56, 78] → "привет"
        """
        tokens = []
        for tid in token_ids:
            token = self._id_to_token.get(tid, "")
            if token and token not in SPECIAL_TOKENS:
                tokens.append(token)
        return "".join(tokens)

    def tokenize(self, text: str) -> List[str]:
        """
        Токенизирует текст в список строковых токенов (без ID).
        Совместимый интерфейс с NeuralEngine.tokenize().

        "Привет мир" → ["при", "вет", " ", "мир"]
        """
        text = text.lower().strip()
        if not text:
            return []

        result = []
        parts = re.findall(r'[а-яёa-z0-9]+|[.!?,;:\-—()\s]', text)

        for part in parts:
            if not part.strip() and part == " ":
                result.append(" ")
                continue

            if len(part) == 1:
                result.append(part)
                continue

            subtokens = self._split_word(part)
            result.extend(subtokens)

        return result

    # ═══════════════════════════════════════════════════════════════
    #               УТИЛИТЫ
    # ═══════════════════════════════════════════════════════════════

    def get_vocab_size(self) -> int:
        return len(self._vocab)

    def get_token_id(self, token: str) -> Optional[int]:
        return self._vocab.get(token)

    def get_token_by_id(self, token_id: int) -> Optional[str]:
        return self._id_to_token.get(token_id)

    def get_stats(self) -> Dict:
        """Статистика токенизатора"""
        texts_trained = self._conn.execute(
            "SELECT COALESCE(SUM(texts_count), 0) as c FROM training_stats"
        ).fetchone()["c"]

        return {
            "vocab_size": len(self._vocab),
            "merge_rules": len(self._merges),
            "unique_words": len(self._word_freqs),
            "texts_trained": texts_trained,
            "target_vocab_size": self._target_vocab_size,
            "coverage_pct": round(
                len(self._vocab) / self._target_vocab_size * 100, 1
            ),
        }

    def analyze_tokenization(self, text: str) -> Dict:
        """
        Анализирует токенизацию текста — для отладки и визуализации.

        Возвращает:
        - tokens: список токенов
        - token_ids: список ID
        - compression_ratio: сжатие (символы / токены)
        - unknown_count: количество <UNK> токенов
        """
        pairs = self.encode_with_tokens(text)
        tokens = [t for t, _ in pairs]
        ids = [i for _, i in pairs]
        unknown = sum(1 for i in ids if i == SPECIAL_TOKENS["<UNK>"])

        return {
            "text": text,
            "tokens": tokens,
            "token_ids": ids,
            "num_tokens": len(tokens),
            "num_chars": len(text),
            "compression_ratio": round(len(text) / max(len(tokens), 1), 2),
            "unknown_count": unknown,
            "unknown_pct": round(unknown / max(len(tokens), 1) * 100, 1),
        }

    def close(self):
        self._conn.commit()
        self._conn.close()

"""
Кристина 7.4 — IntentRouter (Четырёхуровневый роутер)

Заменяет LLM-вызов director.analyze_request() для большинства запросов.

ЧЕТЫРЕ УРОВНЯ (от быстрого к медленному):
  Tier 1: LearnedPatterns — паттерны, выученные у LLM (<10мс)
  Tier 2: RuleEngine     — захардкоженные regex правила (<5мс)
  Tier 2.5: EmbeddingClassifier — семантическое сходство (<50мс)
  Tier 3: LLM fallback   — director.analyze_request() (~25с)

v7.4:
  + Tier 2.5 — Intent classification на sentence embeddings
    Хранит эталонные эмбеддинги для каждого intent-а.
    Новый запрос сравнивается по cosine similarity.
    Порог 0.75 — если ниже, идём в LLM.
"""

import re
import math
from typing import Optional, Dict, List, Any

from utils.logging import get_logger

logger = get_logger("intent_router")


class EmbeddingClassifier:
    """
    Intent-классификатор на sentence embeddings.

    Хранит центроиды (средние эмбеддинги) для каждого intent-а.
    При классификации считает cosine similarity с каждым центроидом.
    """

    def __init__(self, similarity_threshold: float = 0.72):
        self._threshold = similarity_threshold
        # intent → {"centroid": [...], "count": N, "agent": "..."}
        self._centroids: Dict[str, Dict] = {}
        self._total_classified = 0

    def add_example(self, intent: str, agent: str, embedding: List[float]):
        """Добавляет пример для обучения центроида"""
        if not embedding or all(v == 0 for v in embedding):
            return

        if intent not in self._centroids:
            self._centroids[intent] = {
                "centroid": list(embedding),
                "count": 1,
                "agent": agent,
            }
        else:
            c = self._centroids[intent]
            n = c["count"]
            # Инкрементальное обновление центроида: running average
            c["centroid"] = [
                (old * n + new) / (n + 1)
                for old, new in zip(c["centroid"], embedding)
            ]
            c["count"] = n + 1

    def classify(self, embedding: List[float]) -> Optional[Dict[str, Any]]:
        """
        Классифицирует по cosine similarity с центроидами.

        Returns:
            Dict с intent/agent/confidence или None
        """
        if not embedding or not self._centroids:
            return None

        best_intent = None
        best_sim = -1.0
        best_agent = "director"

        for intent, data in self._centroids.items():
            sim = self._cosine_similarity(embedding, data["centroid"])
            if sim > best_sim:
                best_sim = sim
                best_intent = intent
                best_agent = data["agent"]

        if best_sim >= self._threshold and best_intent:
            self._total_classified += 1
            return {
                "intent": best_intent,
                "agent": best_agent,
                "confidence": round(best_sim, 3),
                "source": "embedding",
                "pattern_id": None,
                "slots": {},
            }

        return None

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Cosine similarity между двумя векторами"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return dot / (norm_a * norm_b)

    def get_stats(self) -> Dict:
        return {
            "intents": len(self._centroids),
            "total_classified": self._total_classified,
            "examples": {k: v["count"] for k, v in self._centroids.items()},
        }


class IntentRouter:
    """
    Четырёхуровневый роутер запросов.
    Tier 1-2: детерминированный (FTS5 + regex)
    Tier 2.5: embedding-based classification (sentence similarity)
    Tier 3: LLM fallback
    """

    def __init__(self, learned_patterns, tool_names: List[str] = None,
                 sentence_embeddings=None):
        """
        Args:
            learned_patterns: LearnedPatterns instance (SQLite база паттернов)
            tool_names: список доступных инструментов для валидации
            sentence_embeddings: SentenceEmbeddings для Tier 2.5
        """
        self.learned = learned_patterns
        self.tool_names = set(tool_names or [])
        self._sentence_embeddings = sentence_embeddings
        self._embedding_classifier = EmbeddingClassifier()
        self._build_rules()

        logger.info(
            f"🧭 IntentRouter: {len(self._rules)} правил, "
            f"{len(self.tool_names)} инструментов, "
            f"embedding_classifier={'on' if sentence_embeddings else 'off'}"
        )

    def _build_rules(self):
        """
        Захардкоженные правила (Tier 2).
        Это НАЧАЛЬНОЕ ЗНАНИЕ — как словарь для ребёнка.
        Со временем LearnedPatterns перекроет большинство из них.
        """
        self._rules = [
            # ── Файлы ──
            (re.compile(
                r'(?:создай|сделай|напиши|сгенерируй)\s+'
                r'(?:(?:текстовый|новый)\s+)?'
                r'(?:файл|документ|текст)',
                re.I),
             "create_file", "executor"),

            (re.compile(
                r'(?:удали|убери|сотри|удалить)\s+'
                r'(?:этот\s+)?(?:файл|документ)',
                re.I),
             "delete_file", "executor"),

            (re.compile(
                r'(?:прочитай|прочти|открой|покажи|что\s+в)\s+'
                r'(?:файл[ае]?|документ)',
                re.I),
             "read_file", "executor"),

            (re.compile(
                r'(?:запиши|допиши|добавь)\s+(?:в|к)\s+(?:файл|документ)',
                re.I),
             "append_file", "executor"),

            (re.compile(
                r'(?:скопируй|копируй|копировать)\s+(?:файл|документ)',
                re.I),
             "copy_file", "executor"),

            (re.compile(
                r'(?:перемести|перенеси|перемещ)\s+(?:файл|документ)',
                re.I),
             "move_file", "executor"),

            (re.compile(
                r'(?:переименуй|переименовать)\s+(?:файл|документ)',
                re.I),
             "rename_file", "executor"),

            (re.compile(
                r'(?:покажи|список|что\s+в)\s+'
                r'(?:папк[еу]|директори[юи]|каталог[еу]|рабочем\s+столе)',
                re.I),
             "list_directory", "executor"),

            (re.compile(
                r'(?:создай|сделай)\s+(?:папку|директорию|каталог)',
                re.I),
             "create_directory", "executor"),

            (re.compile(
                r'(?:найди|поищи|поиск)\s+(?:файл[ыа]?)',
                re.I),
             "search_files", "executor"),

            (re.compile(
                r'(?:информаци[яю]|размер|вес|дата)\s+'
                r'(?:о\s+)?(?:файл[ае])',
                re.I),
             "file_info", "executor"),

            (re.compile(
                r'(?:заархивируй|упакуй|архив)',
                re.I),
             "archive", "executor"),

            # ── Система ──
            (re.compile(
                r'(?:запусти|открой|запустить|включи)\s+'
                r'(?:приложение\s+)?(?!файл)([\wа-яёА-ЯЁ]+)',
                re.I),
             "launch_app", "executor"),

            (re.compile(
                r'(?:закрой|заверши|убей|останови)\s+'
                r'(?:процесс|приложение)\s+',
                re.I),
             "kill_process", "executor"),

            (re.compile(
                r'(?:статус|состояние|нагрузка)\s*'
                r'(?:систем|компьютер|пк)?',
                re.I),
             "system_status", "executor"),

            (re.compile(
                r'(?:информаци[яю]|инфо)\s*(?:о\s+)?'
                r'(?:систем[еу]|компьютер[еу]|пк)',
                re.I),
             "system_info", "executor"),

            (re.compile(
                r'(?:процесс[ыа]|запущенные|список\s+процесс)',
                re.I),
             "list_processes", "executor"),

            (re.compile(
                r'(?:мест[оа]\s+на\s+диск|дисков|свободн[оа]\s+на\s+диск)',
                re.I),
             "disk_usage", "executor"),

            (re.compile(
                r'(?:выполни\s+команд|терминал|командн\w+\s+строк)',
                re.I),
             "run_command", "executor"),

            # ── Время / Погода / Валюта ──
            (re.compile(
                r'(?:врем[яю]|\bчас\b|который\s+час|сколько\s+врем|'
                r'какой\s+(?:сегодня\s+)?день)',
                re.I),
             "get_current_time", "executor"),

            (re.compile(
                r'(?:погод[аеу]|температур|градус|на\s+улице)',
                re.I),
             "get_weather", "executor"),

            (re.compile(
                r'(?:курс|стоимость)\s+'
                r'(?:доллар|евро|валют|рубл|юан|фунт|USD|EUR|CNY|GBP)',
                re.I),
             "get_currency_rate", "executor"),

            # ── Память / Заметки ──
            (re.compile(
                r'(?:вспомни|напомни|помнишь|что\s+(?:ты\s+)?знаешь)',
                re.I),
             "recall_memory", "executor"),

            (re.compile(
                r'(?:сохрани|запиши|запомни)\s+(?:заметк|замечани)',
                re.I),
             "save_note", "executor"),

            (re.compile(
                r'(?:покажи|список)\s+(?:замет[ок]|заметки)',
                re.I),
             "list_notes", "executor"),

            # ── Веб ──
            (re.compile(
                r'(?:найди|поищи|загугли|погугли|search|'
                r'что\s+(?:такое|значит)|(?:кто\s+(?:такой|такая)))',
                re.I),
             "web_search", "analyst"),

            (re.compile(
                r'(?:скачай|загрузи|download)\s+(?:файл\s+)?(?:с|из|по)',
                re.I),
             "download_file", "executor"),

            # ── Диалог (director, без инструмента) ──
            (re.compile(
                r'^(?:привет|здравствуй|хай|hello|добр\w+\s+'
                r'(?:утро|день|вечер)|как\s+дела)',
                re.I),
             "greeting", "director"),

            (re.compile(
                r'(?:расскажи|объясни|почему|зачем|как\s+работает)',
                re.I),
             "explanation", "director"),

            (re.compile(
                r'(?:придумай|сочини|напиши\s+(?:стих|рассказ|историю|'
                r'пожелани|поздравлени))',
                re.I),
             "creative", "director"),

            # ── Самосознание / Личность / Философские вопросы ──
            (re.compile(
                r'(?:ты\s+(?:себя\s+)?осознаёшь|'
                r'ты\s+(?:себя\s+)?осознаешь|'
                r'ты\s+живая|ты\s+живой|'
                r'ты\s+(?:настоящ|реальн)|'
                r'ты\s+(?:робот|бот|ии|искусственн|программ|машин|нейросет)|'
                r'кто\s+ты|что\s+ты\s+(?:такое|есть)|'
                r'ты\s+чувствуешь|ты\s+(?:думаешь|мыслишь|понимаешь)|'
                r'у\s+тебя\s+(?:есть\s+)?(?:сознани|чувств|эмоци|душ)|'
                r'ты\s+(?:человек|личност)|'
                r'(?:что|как)\s+ты\s+(?:о\s+себе|думаешь\s+о\s+себе)|'
                r'ты\s+(?:можешь\s+)?(?:мечтать|любить|бояться|грустить))',
                re.I),
             "self_awareness", "director"),

            # ── Вопросы про возможности ──
            (re.compile(
                r'(?:что\s+(?:ты\s+)?(?:умеешь|можешь|способн)|'
                r'(?:на\s+что|чего)\s+ты\s+(?:способн|можешь))',
                re.I),
             "capabilities", "director"),

            # ── Как ты / Как дела / Что нового ──
            (re.compile(
                r'(?:^как\s+(?:ты|у\s+тебя|твои\s+дела|поживаешь|настроение)|'
                r'(?:что|как)\s+(?:нового|новенького)|'
                r'как\s+(?:себя\s+)?чувствуешь)',
                re.I),
             "smalltalk", "director"),
        ]

    def route(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Определяет intent без LLM.

        Returns:
            Dict с планом или None (→ нужен LLM).

            Если Dict:
              - intent: str      — имя инструмента
              - agent: str       — executor/analyst/director
              - confidence: float
              - source: str      — 'learned' | 'rule'
              - pattern_id: int? — ID паттерна (для reinforce/weaken)
              - slots: dict      — извлечённые аргументы
        """

        # ── Tier 1: Выученные паттерны (<10мс) ──
        learned_result = self.learned.find_routing(user_input)
        if learned_result and learned_result["confidence"] >= 0.7:
            # Также пытаемся извлечь аргументы
            slots = self.learned.find_slots(
                learned_result["intent"], user_input
            )
            learned_result["slots"] = slots
            logger.debug(
                f"✅ Tier 1 (learned): {learned_result['intent']} "
                f"(conf={learned_result['confidence']:.2f})"
            )
            return learned_result

        # ── Tier 2: Захардкоженные правила (<5мс) ──
        for pattern, intent, agent in self._rules:
            if pattern.search(user_input):
                # Валидация: intent должен быть реальным инструментом
                # (кроме director-специфичных как greeting, explanation, creative)
                if agent == "executor" and intent not in self.tool_names:
                    continue

                slots = self._extract_slots_by_rules(intent, user_input)

                # create_file без filepath — слишком сложный запрос для regex,
                # отправляем в LLM (Tier 4) для генерации содержимого
                if intent == "create_file" and "filepath" not in slots:
                    logger.debug(f"⚠️ Tier 2: create_file без filepath → LLM")
                    continue

                result = {
                    "intent": intent,
                    "agent": agent,
                    "confidence": 0.85,
                    "source": "rule",
                    "pattern_id": None,
                    "slots": slots,
                }
                logger.debug(f"✅ Tier 2 (rule): {intent}")
                return result

        # ── Tier 2.5: Embedding-based classification (<50мс) ──
        if self._sentence_embeddings:
            try:
                embedding = self._sentence_embeddings.encode(user_input)
                if embedding:
                    emb_result = self._embedding_classifier.classify(embedding)
                    if emb_result:
                        # Валидируем intent
                        if (emb_result["agent"] != "executor" or
                                emb_result["intent"] in self.tool_names or
                                emb_result["intent"] in ("greeting", "explanation", "creative")):
                            slots = self._extract_slots_by_rules(emb_result["intent"], user_input)
                            emb_result["slots"] = slots
                            logger.debug(
                                f"✅ Tier 2.5 (embedding): {emb_result['intent']} "
                                f"(sim={emb_result['confidence']:.2f})"
                            )
                            return emb_result
            except Exception as e:
                logger.debug(f"Tier 2.5 error: {e}")

        # ── Ничего не нашли → Tier 3 (LLM) ──
        logger.debug(f"⚠️ Tier 1+2+2.5 miss, нужен LLM для: '{user_input[:50]}'")
        return None

    def learn_from_route(self, user_input: str, intent: str, agent: str):
        """
        v7.4: Обучает EmbeddingClassifier на каждом успешном роутинге.
        Вызывается из orchestrator после каждого ответа.
        """
        if self._sentence_embeddings:
            try:
                embedding = self._sentence_embeddings.encode(user_input)
                if embedding:
                    self._embedding_classifier.add_example(intent, agent, embedding)
            except Exception:
                pass

    def _extract_slots_by_rules(self, intent: str, user_input: str) -> Dict[str, str]:
        """
        Извлечение аргументов из текста правилами.

        Сначала пробует learned slots, потом захардкоженные regex.
        """
        # Tier 1: Выученные slot-паттерны
        slots = self.learned.find_slots(intent, user_input)
        if slots:
            return slots

        # Tier 2: Базовые правила
        slots = {}

        if intent in ("create_file", "read_file", "delete_file",
                       "write_file", "append_file", "file_info"):
            # Ищем имя файла
            match = re.search(
                r'([\wа-яёА-ЯЁ\-]+\.[\wа-яёА-ЯЁ]+)', user_input, re.I
            )
            if match:
                slots["filepath"] = match.group(1)

        if intent == "create_file":
            # Ищем содержимое после ключевых слов
            for pattern in [
                r'(?:с\s+(?:текстом|содержимым|содержанием))\s*[:\-]?\s*(.+)',
                r'(?:напиши|написать)\s*[:\-]?\s*(.+)',
                r'\b(?:содержимое|текст)\b\s*[:\-]?\s*(.+)',
            ]:
                match = re.search(pattern, user_input, re.I)
                if match:
                    slots["content"] = match.group(1).strip()
                    break

        if intent == "launch_app":
            match = re.search(
                r'(?:запусти|открой|включи)\s+(?:приложение\s+)?'
                r'([\wа-яёА-ЯЁ]+)',
                user_input, re.I
            )
            if match:
                slots["app_name"] = match.group(1)

        if intent == "get_weather":
            match = re.search(
                r'(?:погод[аеу]|температур\w*)\s+(?:в\s+)?([\wа-яёА-ЯЁ]+)',
                user_input, re.I
            )
            if match:
                slots["city"] = match.group(1)

        if intent == "web_search":
            # Всё после ключевого слова — запрос
            match = re.search(
                r'(?:найди|поищи|загугли|погугли)\s+(.+)',
                user_input, re.I
            )
            if match:
                slots["query"] = match.group(1).strip()

        if intent == "kill_process":
            match = re.search(
                r'(?:закрой|заверши|убей)\s+(?:процесс\s+)?'
                r'([\wа-яёА-ЯЁ]+)',
                user_input, re.I
            )
            if match:
                slots["process_name"] = match.group(1)

        if intent == "get_currency_rate":
            match = re.search(
                r'(доллар|евро|юан|фунт|USD|EUR|CNY|GBP|JPY)',
                user_input, re.I
            )
            if match:
                mapping = {
                    "доллар": "USD", "евро": "EUR", "юан": "CNY",
                    "фунт": "GBP",
                }
                raw = match.group(1)
                slots["currency"] = mapping.get(raw.lower(), raw.upper())

        return slots

"""
Кристина 6.0 — МетаКогниция

Вдохновлено: SIGMA NEURON v5.0 (MetaCognition)

Способности:
1. Калибровка уверенности — если модель часто ошибается при высокой
   уверенности, система автоматически понижает confidence
2. Выбор стратегии — analytical/creative/tool_use/delegate
   на основе UCB (Upper Confidence Bound) + истории успехов
3. Known unknowns — отслеживание тем, где Кристина знает что не знает
4. Introspection — полная самодиагностика

Используется:
- В orchestrator для выбора агента
- В agent.py для решения "отвечать или искать"
- В identity для self-awareness метрик
"""

import time
import json
from collections import deque
from typing import Dict, List, Optional, Any
from pathlib import Path
import math

from utils.logging import get_logger
import config

logger = get_logger("metacognition")


class MetaCognition:
    """
    Мета-когнитивная система: думать о своём мышлении.
    Вдохновлено SIGMA MetaCognition + consciousness v0.7 self-model.
    """

    def __init__(self):
        self._data_file = config.config.data_dir / "metacognition.json"

        # === Confidence calibration (из SIGMA) ===
        self._predictions: deque = deque(maxlen=200)
        # (confidence, was_correct, timestamp, topic)

        # === Strategy selection (из SIGMA, UCB) ===
        self.strategies = ["direct", "tool_use", "web_search", "delegate", "creative"]
        self._strategy_history: Dict[str, deque] = {
            s: deque(maxlen=100) for s in self.strategies
        }

        # === Known unknowns ===
        self._known_unknowns: Dict[str, Dict] = {}
        # topic → {count, last_seen, confidence}

        # === Self-model (из consciousness v0.7) ===
        self.self_model = {
            "coherence": 0.5,          # Насколько self-model совпадает с реальностью
            "calibration_error": 0.5,   # Ошибка калибровки уверенности
            "agency": 0.5,              # Ощущение контроля
            "competence": 0.5,          # Оценка собственной компетенции
            "total_interactions": 0,
            "total_successes": 0,
        }

        self._load()

        logger.info(
            f"🧠 MetaCognition: {len(self._predictions)} predictions, "
            f"{len(self._known_unknowns)} known unknowns"
        )

    # ═══════════════════════════════════════════════════════════
    #               КАЛИБРОВКА УВЕРЕННОСТИ (из SIGMA)
    # ═══════════════════════════════════════════════════════════

    def estimate_confidence(self, topic: str = "", has_tools: bool = True) -> float:
        """
        Оценивает confidence для текущего ответа.

        Учитывает:
        - Историю предсказаний (калибровка)
        - Known unknowns (если тема в "не знаю" → низкий confidence)
        - Наличие tools (если есть → можно проверить)
        """
        base_confidence = 0.7

        # Калибровка по истории
        if len(self._predictions) > 10:
            recent = list(self._predictions)[-50:]
            predicted_avg = sum(c for c, _, _, _ in recent) / len(recent)
            actual_avg = sum(1 for _, correct, _, _ in recent if correct) / len(recent)

            if predicted_avg > actual_avg + 0.1:
                # Overconfident → снижаем
                base_confidence *= 0.85
            elif predicted_avg < actual_avg - 0.1:
                # Underconfident → повышаем
                base_confidence *= 1.1

        # Known unknown?
        if topic:
            topic_lower = topic.lower()
            for unknown_topic in self._known_unknowns:
                if unknown_topic in topic_lower or topic_lower in unknown_topic:
                    base_confidence *= 0.5
                    break

        # Tools доступны → чуть выше confidence (можно проверить)
        if has_tools:
            base_confidence = min(base_confidence + 0.05, 0.95)

        return max(0.1, min(0.95, base_confidence))

    def record_outcome(self, confidence: float, was_correct: bool, topic: str = ""):
        """Записывает результат для калибровки"""
        self._predictions.append((confidence, was_correct, time.time(), topic))

        self.self_model["total_interactions"] += 1
        if was_correct:
            self.self_model["total_successes"] += 1

        # Обновляем calibration error
        if len(self._predictions) > 10:
            recent = list(self._predictions)[-50:]
            self.self_model["calibration_error"] = sum(
                abs(c - (1.0 if correct else 0.0))
                for c, correct, _, _ in recent
            ) / len(recent)

        # Обновляем competence
        if self.self_model["total_interactions"] > 0:
            self.self_model["competence"] = (
                self.self_model["total_successes"]
                / self.self_model["total_interactions"]
            )

        # Если ошиблись уверенно → добавить в known_unknowns
        if not was_correct and confidence > 0.7 and topic:
            self.admit_unknown(topic, confidence)

    def admit_unknown(self, topic: str, failed_confidence: float = 0.5):
        """Признать незнание (из SIGMA)"""
        topic_lower = topic.lower()[:100]
        if topic_lower in self._known_unknowns:
            self._known_unknowns[topic_lower]["count"] += 1
            self._known_unknowns[topic_lower]["last_seen"] = time.time()
        else:
            self._known_unknowns[topic_lower] = {
                "count": 1,
                "last_seen": time.time(),
                "failed_confidence": failed_confidence,
            }
            logger.info(f"❓ Known unknown: {topic_lower}")

    # ═══════════════════════════════════════════════════════════
    #               ВЫБОР СТРАТЕГИИ (из SIGMA, UCB)
    # ═══════════════════════════════════════════════════════════

    def select_strategy(self, query: str) -> str:
        """
        Выбирает стратегию обработки запроса (UCB-like, из SIGMA).

        Стратегии:
        - direct: ответить из знаний модели
        - tool_use: использовать инструменты
        - web_search: искать в интернете
        - delegate: делегировать специализированному агенту
        - creative: креативный подход
        """
        total_trials = sum(len(self._strategy_history[s]) for s in self.strategies)

        if total_trials == 0:
            # Первый раз — по эвристике
            return self._heuristic_strategy(query)

        best_strategy = None
        best_score = -float("inf")

        for strategy in self.strategies:
            history = self._strategy_history[strategy]

            if not history:
                return strategy  # Попробовать непротестированное

            avg_perf = sum(history) / len(history)
            exploration = math.sqrt(2 * math.log(total_trials + 1) / len(history))

            score = avg_perf + exploration

            if score > best_score:
                best_score = score
                best_strategy = strategy

        return best_strategy or "direct"

    def record_strategy_outcome(self, strategy: str, success: float):
        """Записывает результат стратегии (0.0 = провал, 1.0 = успех)"""
        if strategy in self._strategy_history:
            self._strategy_history[strategy].append(success)

    def _heuristic_strategy(self, query: str) -> str:
        """Эвристика для первого выбора стратегии"""
        lower = query.lower()

        if any(w in lower for w in ["найди", "поиск", "погода", "курс", "новости"]):
            return "web_search"
        elif any(w in lower for w in ["удали", "запусти", "файл", "открой", "время"]):
            return "tool_use"
        elif any(w in lower for w in ["придумай", "напиши стих", "сочини", "нарисуй"]):
            return "creative"
        elif any(w in lower for w in ["анализ", "сравни", "объясни подробно"]):
            return "delegate"
        else:
            return "direct"

    # ═══════════════════════════════════════════════════════════
    #               SELF-MODEL (из consciousness v0.7)
    # ═══════════════════════════════════════════════════════════

    def update_coherence(self, expected_emotion: str, actual_emotion: str):
        """
        Обновляет coherence self-model (из consciousness v0.7).
        Если ожидаемая эмоция совпадает с реальной → coherence растёт.
        """
        if expected_emotion == actual_emotion:
            self.self_model["coherence"] = min(
                self.self_model["coherence"] + 0.02, 1.0
            )
        else:
            self.self_model["coherence"] = max(
                self.self_model["coherence"] - 0.05, 0.0
            )

    def update_agency(self, action_succeeded: bool):
        """Обновляет ощущение контроля"""
        target = 0.7 if action_succeeded else 0.3
        self.self_model["agency"] += (target - self.self_model["agency"]) * 0.05

    # ═══════════════════════════════════════════════════════════
    #               ИНТРОСПЕКЦИЯ (из SIGMA)
    # ═══════════════════════════════════════════════════════════

    def introspect(self) -> Dict[str, Any]:
        """Полная интроспекция (из SIGMA MetaCognition.introspect())"""
        strategy_perf = {}
        for s, h in self._strategy_history.items():
            strategy_perf[s] = round(sum(h) / len(h), 3) if h else 0.0

        return {
            "self_model": {k: round(v, 3) if isinstance(v, float) else v
                          for k, v in self.self_model.items()},
            "calibration": {
                "total_predictions": len(self._predictions),
                "calibration_error": round(self.self_model["calibration_error"], 3),
            },
            "known_unknowns": {
                "count": len(self._known_unknowns),
                "topics": list(self._known_unknowns.keys())[:10],
            },
            "strategy_performance": strategy_perf,
        }

    # ═══════════════════════════════════════════════════════════
    #               ПЕРСИСТЕНТНОСТЬ
    # ═══════════════════════════════════════════════════════════

    def save(self):
        try:
            data = {
                "self_model": self.self_model,
                "known_unknowns": self._known_unknowns,
                "strategy_history": {
                    s: list(h) for s, h in self._strategy_history.items()
                },
                "predictions": [
                    {"conf": c, "correct": cr, "ts": t, "topic": tp}
                    for c, cr, t, tp in self._predictions
                ],
            }
            with open(self._data_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения metacognition: {e}")

    def _load(self):
        if not self._data_file.exists():
            return
        try:
            with open(self._data_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.self_model.update(data.get("self_model", {}))
            self._known_unknowns = data.get("known_unknowns", {})

            for s, h in data.get("strategy_history", {}).items():
                if s in self._strategy_history:
                    self._strategy_history[s] = deque(h, maxlen=100)

            for p in data.get("predictions", []):
                self._predictions.append(
                    (p["conf"], p["correct"], p.get("ts", 0), p.get("topic", ""))
                )
        except Exception as e:
            logger.error(f"⚠️ Ошибка загрузки metacognition: {e}")

"""
Кристина 6.0 — Иерархическая суммаризация памяти

Уровни:
  L0: Raw episodes (последние ~200)
  L1: Daily summaries (1 параграф на день)
  L2: Weekly summaries (1 параграф на неделю)
  L3: Monthly summaries (1 параграф на месяц)

Поиск идёт от L0 (свежие) к L3 (старые).
Старые raw episodes заменяются на L1 summaries → "бесконечная" память.

Суммаризация запускается:
  - Автоматически при накоплении > daily_threshold эпизодов за день
  - Вручную через summarize_all()
  - Через task scheduler (Фаза 4)
"""

import json
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from ollama import AsyncClient as _OllamaAsyncClient

from utils.logging import get_logger
import config

logger = get_logger("memory.summarizer")


class MemorySummarizer:
    """Иерархическая суммаризация памяти"""

    def __init__(self):
        self._summaries_dir = config.config.summaries_dir
        self._daily_dir = self._summaries_dir / "daily"
        self._weekly_dir = self._summaries_dir / "weekly"
        self._monthly_dir = self._summaries_dir / "monthly"

        for d in [self._daily_dir, self._weekly_dir, self._monthly_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self._model = config.config.memory_summarizer_model
        self._daily_threshold = config.config.memory_daily_threshold

        # Кэш загруженных саммари
        self._daily_cache: Dict[str, str] = {}
        self._weekly_cache: Dict[str, str] = {}
        self._monthly_cache: Dict[str, str] = {}

        self._load_caches()

        logger.info(
            f"📚 Summarizer: daily={len(self._daily_cache)} "
            f"weekly={len(self._weekly_cache)} "
            f"monthly={len(self._monthly_cache)}"
        )

    # ═══════════════════════════════════════════════════════════
    #                    СУММАРИЗАЦИЯ
    # ═══════════════════════════════════════════════════════════

    async def summarize_day(self, date: str, episodes: List[Dict]) -> str:
        """
        Суммаризирует эпизоды одного дня.

        Args:
            date: "2026-02-11"
            episodes: [{user_input, response, emotion, importance, timestamp}, ...]

        Returns:
            Саммари дня (1-3 параграфа)
        """
        if not episodes:
            return ""

        # Уже есть?
        if date in self._daily_cache:
            return self._daily_cache[date]

        # Строим текст для LLM
        dialogue_text = self._format_episodes(episodes)

        prompt = f"""Ты — модуль памяти ИИ-ассистента Кристины.
Сожми диалоги за {date} в краткое резюме (1-3 параграфа).

ПРАВИЛА:
- Сохрани ВСЕ важные факты, имена, числа, решения
- Сохрани что пользователь просил и что было сделано
- Сохрани эмоциональный контекст (если был)
- Пиши от третьего лица ("пользователь попросил...", "было обсуждено...")
- НЕ добавляй ничего от себя

ДИАЛОГИ:
{dialogue_text}

РЕЗЮМЕ ДНЯ:"""

        try:
            client = _OllamaAsyncClient()
            response = await client.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 500},
            )
            summary = response["message"]["content"].strip()
        except Exception as e:
            logger.error(f"❌ Ошибка суммаризации дня {date}: {e}")
            # Fallback: простое сжатие
            summary = self._fallback_summarize(episodes)

        # Сохраняем
        self._daily_cache[date] = summary
        self._save_summary(self._daily_dir / f"{date}.json", {
            "date": date,
            "summary": summary,
            "episode_count": len(episodes),
            "created": datetime.now(timezone.utc).isoformat(),
        })

        logger.info(f"📝 Daily summary [{date}]: {len(episodes)} эпизодов → {len(summary)} chars")
        return summary

    async def summarize_week(self, year: int, week: int) -> str:
        """
        Суммаризирует ежедневные саммари за неделю.

        Args:
            year: 2026
            week: Номер недели (ISO)
        """
        week_key = f"{year}-W{week:02d}"

        if week_key in self._weekly_cache:
            return self._weekly_cache[week_key]

        # Находим daily summaries для этой недели
        daily_summaries = []
        for date_str, summary in sorted(self._daily_cache.items()):
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                iso_year, iso_week, _ = dt.isocalendar()
                if iso_year == year and iso_week == week:
                    daily_summaries.append((date_str, summary))
            except ValueError:
                continue

        if not daily_summaries:
            return ""

        daily_text = "\n\n".join(
            f"[{date}]\n{summary}" for date, summary in daily_summaries
        )

        prompt = f"""Ты — модуль памяти ИИ-ассистента Кристины.
Сожми ежедневные резюме за неделю {week_key} в одно недельное резюме (1-2 параграфа).

ПРАВИЛА:
- Сохрани ключевые события и решения
- Объедини похожие темы
- Отметь повторяющиеся паттерны
- Пиши от третьего лица

ЕЖЕДНЕВНЫЕ РЕЗЮМЕ:
{daily_text}

НЕДЕЛЬНОЕ РЕЗЮМЕ:"""

        try:
            client = _OllamaAsyncClient()
            response = await client.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 400},
            )
            summary = response["message"]["content"].strip()
        except Exception as e:
            logger.error(f"❌ Ошибка weekly summary {week_key}: {e}")
            summary = " ".join(s for _, s in daily_summaries)[:500]

        self._weekly_cache[week_key] = summary
        self._save_summary(self._weekly_dir / f"{week_key}.json", {
            "week": week_key,
            "summary": summary,
            "days_count": len(daily_summaries),
            "created": datetime.now(timezone.utc).isoformat(),
        })

        logger.info(f"📅 Weekly summary [{week_key}]: {len(daily_summaries)} дней")
        return summary

    async def summarize_month(self, year: int, month: int) -> str:
        """Суммаризирует weekly саммари за месяц"""
        month_key = f"{year}-{month:02d}"

        if month_key in self._monthly_cache:
            return self._monthly_cache[month_key]

        weekly_summaries = []
        for week_key, summary in sorted(self._weekly_cache.items()):
            if week_key.startswith(f"{year}-W"):
                try:
                    w = int(week_key.split("W")[1])
                    # Примерная проверка принадлежности к месяцу
                    dt = datetime.strptime(f"{year}-W{w:02d}-1", "%G-W%V-%u")
                    if dt.month == month:
                        weekly_summaries.append((week_key, summary))
                except (ValueError, IndexError):
                    continue

        if not weekly_summaries:
            return ""

        weekly_text = "\n\n".join(
            f"[{wk}]\n{summary}" for wk, summary in weekly_summaries
        )

        prompt = f"""Ты — модуль памяти ИИ-ассистента Кристины.
Сожми недельные резюме за {month_key} в месячное резюме (1 параграф).

ПРАВИЛА:
- Только самое важное: ключевые проекты, решения, паттерны
- Пиши от третьего лица

НЕДЕЛЬНЫЕ РЕЗЮМЕ:
{weekly_text}

МЕСЯЧНОЕ РЕЗЮМЕ:"""

        try:
            client = _OllamaAsyncClient()
            response = await client.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 300},
            )
            summary = response["message"]["content"].strip()
        except Exception as e:
            logger.error(f"❌ Ошибка monthly summary {month_key}: {e}")
            summary = " ".join(s for _, s in weekly_summaries)[:400]

        self._monthly_cache[month_key] = summary
        self._save_summary(self._monthly_dir / f"{month_key}.json", {
            "month": month_key,
            "summary": summary,
            "weeks_count": len(weekly_summaries),
            "created": datetime.now(timezone.utc).isoformat(),
        })

        logger.info(f"📆 Monthly summary [{month_key}]: {len(weekly_summaries)} недель")
        return summary

    # ═══════════════════════════════════════════════════════════
    #                    АВТОМАТИЧЕСКАЯ РОТАЦИЯ
    # ═══════════════════════════════════════════════════════════

    async def auto_summarize(self, episodes: List[Dict]) -> List[Dict]:
        """
        Проверяет и суммаризирует старые эпизоды.
        Возвращает оставшиеся raw episodes (свежие).

        Вызывать периодически или при save().
        """
        if not config.config.memory_summarize_enabled:
            return episodes

        if len(episodes) <= config.config.memory_max_raw_episodes:
            return episodes

        # Группируем по дням
        by_day: Dict[str, List[Dict]] = defaultdict(list)
        for ep in episodes:
            ts = ep.get("timestamp", "")
            date = ts[:10] if len(ts) >= 10 else "unknown"
            by_day[date].append(ep)

        today = datetime.now().strftime("%Y-%m-%d")
        keep_raw = []
        summarized_count = 0

        for date in sorted(by_day.keys()):
            day_episodes = by_day[date]

            # Сегодня и вчера — всегда raw
            if date >= (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"):
                keep_raw.extend(day_episodes)
                continue

            # Если уже есть daily summary — пропускаем raw
            if date in self._daily_cache:
                summarized_count += len(day_episodes)
                continue

            # Если достаточно эпизодов — суммаризируем
            if len(day_episodes) >= 3:
                await self.summarize_day(date, day_episodes)
                summarized_count += len(day_episodes)
            else:
                keep_raw.extend(day_episodes)

        if summarized_count > 0:
            logger.info(f"🔄 Авто-суммаризация: {summarized_count} эпизодов сжато")

        # Ротация weekly/monthly
        await self._rotate_summaries()

        return keep_raw

    async def _rotate_summaries(self):
        """Ротирует daily → weekly → monthly"""
        now = datetime.now()

        # Daily → Weekly (старше 14 дней)
        cutoff = now - timedelta(days=config.config.memory_daily_retention_days)
        old_dailies = [
            d for d in self._daily_cache
            if d < cutoff.strftime("%Y-%m-%d")
        ]

        if old_dailies:
            # Найдём все уникальные недели
            weeks_to_summarize = set()
            for date_str in old_dailies:
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                    iso_year, iso_week, _ = dt.isocalendar()
                    weeks_to_summarize.add((iso_year, iso_week))
                except ValueError:
                    continue

            for year, week in weeks_to_summarize:
                week_key = f"{year}-W{week:02d}"
                if week_key not in self._weekly_cache:
                    await self.summarize_week(year, week)

        # Weekly → Monthly (старше 8 недель)
        cutoff_weeks = now - timedelta(weeks=config.config.memory_weekly_retention_weeks)
        old_weeklies = []
        for wk in self._weekly_cache:
            try:
                parts = wk.split("-W")
                year, week_num = int(parts[0]), int(parts[1])
                dt = datetime.strptime(f"{year}-W{week_num:02d}-1", "%G-W%V-%u")
                if dt < cutoff_weeks:
                    old_weeklies.append((year, dt.month))
            except (ValueError, IndexError):
                continue

        for year, month in set(old_weeklies):
            month_key = f"{year}-{month:02d}"
            if month_key not in self._monthly_cache:
                await self.summarize_month(year, month)

    # ═══════════════════════════════════════════════════════════
    #                    ПОИСК ПО САММАРИ
    # ═══════════════════════════════════════════════════════════

    def search_summaries(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        Поиск по иерархии саммари (keyword-based).
        Для семантического поиска используй VectorMemory.
        """
        query_words = set(query.lower().split())
        results = []

        # L1: Daily
        for date, summary in self._daily_cache.items():
            score = self._keyword_score(query_words, summary)
            if score > 0:
                results.append({
                    "level": "daily",
                    "key": date,
                    "summary": summary,
                    "score": score,
                })

        # L2: Weekly
        for week, summary in self._weekly_cache.items():
            score = self._keyword_score(query_words, summary)
            if score > 0:
                results.append({
                    "level": "weekly",
                    "key": week,
                    "summary": summary,
                    "score": score * 0.8,  # Немного снижаем
                })

        # L3: Monthly
        for month, summary in self._monthly_cache.items():
            score = self._keyword_score(query_words, summary)
            if score > 0:
                results.append({
                    "level": "monthly",
                    "key": month,
                    "summary": summary,
                    "score": score * 0.6,
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max_results]

    @staticmethod
    def _keyword_score(query_words: set, text: str) -> int:
        text_words = set(text.lower().split())
        return len(query_words & text_words)

    # ═══════════════════════════════════════════════════════════
    #                    ВСПОМОГАТЕЛЬНЫЕ
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _format_episodes(episodes: List[Dict], max_chars: int = 3000) -> str:
        """Форматирует эпизоды для промпта"""
        parts = []
        total = 0
        for ep in episodes:
            user = ep.get("user_input", "")[:100]
            resp = ep.get("response", "")[:150]
            line = f"П: {user}\nК: {resp}"
            if total + len(line) > max_chars:
                break
            parts.append(line)
            total += len(line)
        return "\n---\n".join(parts)

    @staticmethod
    def _fallback_summarize(episodes: List[Dict]) -> str:
        """Fallback без LLM — простое сжатие"""
        parts = []
        for ep in episodes[:10]:
            user = ep.get("user_input", "")[:50]
            parts.append(f"• {user}")
        return "Обсуждалось: " + "; ".join(parts)

    def _load_caches(self):
        """Загружает все саммари с диска"""
        self._daily_cache = self._load_dir(self._daily_dir, "date", "summary")
        self._weekly_cache = self._load_dir(self._weekly_dir, "week", "summary")
        self._monthly_cache = self._load_dir(self._monthly_dir, "month", "summary")

    @staticmethod
    def _load_dir(directory: Path, key_field: str, value_field: str) -> Dict[str, str]:
        cache = {}
        if not directory.exists():
            return cache
        for f in directory.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                cache[data[key_field]] = data[value_field]
            except Exception:
                continue
        return cache

    @staticmethod
    def _save_summary(path: Path, data: Dict):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения саммари: {e}")

    def get_stats(self) -> Dict:
        return {
            "daily_summaries": len(self._daily_cache),
            "weekly_summaries": len(self._weekly_cache),
            "monthly_summaries": len(self._monthly_cache),
            "enabled": config.config.memory_summarize_enabled,
        }

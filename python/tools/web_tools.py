"""
Инструменты для работы с интернетом — JARVIS Edition v6.1

- Поиск в интернете (DuckDuckGo)
- Чтение веб-страниц
- Скачивание файлов
- Погода, время, курсы валют
"""

from pathlib import Path
from typing import Optional
from datetime import datetime
from tools.base import BaseTool, ToolSchema
from utils.logging import get_logger
from utils.validators import validate_url
import config

logger = get_logger("web_tools")

# Единая библиотека для веб-поиска
DDGS_AVAILABLE = False
DDGS_CLASS = None

try:
    from duckduckgo_search import DDGS
    DDGS_CLASS = DDGS
    DDGS_AVAILABLE = True
    logger.info("DuckDuckGo search ready")
except ImportError:
    logger.warning("duckduckgo-search не установлен: pip install duckduckgo-search")


# ═══════════════════════════════════════════════════════════════
#                         ПОИСК
# ═══════════════════════════════════════════════════════════════

class WebSearchTool(BaseTool):
    """Поиск в интернете"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="web_search",
            description="Ищет информацию в интернете через DuckDuckGo. Возвращает заголовки, ссылки и описания.",
            required_args=["query"],
            optional_args=["max_results"],
            arg_types={"query": str, "max_results": int},
            arg_descriptions={
                "query": "Поисковый запрос (например: 'новости ИИ 2025', 'курс биткоина', 'рецепт борща')",
                "max_results": "Максимальное количество результатов (по умолчанию 5)",
            },
            category="web",
            examples=[
                'web_search("новости ИИ 2025")',
                'web_search("погода москва")',
                'web_search("python async tutorial", max_results=10)',
            ],
        )

    async def execute(self, query: str, max_results: int = None) -> str:
        logger.info(f"Веб-поиск: {query}")

        if not DDGS_AVAILABLE or DDGS_CLASS is None:
            return (
                "Библиотека для веб-поиска не установлена.\n"
                "Установи: pip install duckduckgo-search\n"
                "После установки перезапусти программу."
            )

        max_res = max_results or config.WEB_SEARCH_MAX_RESULTS

        try:
            results = []
            ddgs = DDGS_CLASS()

            try:
                search_results = ddgs.text(query, max_results=max_res)
            except TypeError:
                search_results = list(ddgs.text(query))[:max_res]

            for r in search_results:
                title = r.get("title", "Без заголовка")
                url = r.get("href") or r.get("link", "")
                snippet = r.get("body", "") or r.get("snippet", "")

                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet[:200],
                })

            if not results:
                return f"Ничего не найдено по запросу: {query}"

            lines = [
                f"Найдено {len(results)} результатов по запросу: '{query}'",
                "",
            ]

            for i, r in enumerate(results, 1):
                lines.append(f"{i}. {r['title']}")
                lines.append(f"   URL: {r['url']}")
                if r['snippet']:
                    lines.append(f"   {r['snippet']}")
                lines.append("")

            logger.info(f"Найдено {len(results)} результатов")
            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Ошибка поиска: {type(e).__name__}: {e}")
            return f"Ошибка поиска: {e}\n\nПроверьте подключение к интернету."


# ═══════════════════════════════════════════════════════════════
#                     ВЕБ-СТРАНИЦЫ
# ═══════════════════════════════════════════════════════════════

class WebFetchTool(BaseTool):
    """Чтение веб-страницы"""

    def __init__(self):
        super().__init__()
        self.request_history = []

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="web_fetch",
            description="Читает содержимое веб-страницы по URL. Извлекает основной текст, убирая навигацию и скрипты.",
            required_args=["url"],
            arg_types={"url": str},
            arg_descriptions={
                "url": "Полный URL страницы (например: https://example.com/article)",
            },
            category="web",
            examples=[
                'web_fetch("https://habr.com/ru/articles/")',
                'web_fetch("https://en.wikipedia.org/wiki/Python")',
            ],
        )

    def _check_rate_limit(self) -> Optional[str]:
        """Проверяет rate limit"""
        now = datetime.now()
        self.request_history = [
            r for r in self.request_history
            if (now - r).total_seconds() < 60
        ]
        if len(self.request_history) >= config.WEB_RATE_LIMIT:
            return f"Превышен лимит запросов ({config.WEB_RATE_LIMIT}/мин). Подожди немного."
        self.request_history.append(now)
        return None

    async def execute(self, url: str) -> str:
        logger.info(f"Чтение URL: {url}")

        rate_error = self._check_rate_limit()
        if rate_error:
            return rate_error

        is_valid, reason = validate_url(url)
        if not is_valid:
            return f"Некорректный URL: {reason}"

        try:
            import httpx
            from bs4 import BeautifulSoup

            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(
                    url,
                    timeout=config.WEB_REQUEST_TIMEOUT,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
                )

            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Убираем мусор
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            text = soup.get_text(separator='\n', strip=True)
            lines = [line for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)

            max_size = config.WEB_FETCH_MAX_SIZE
            if len(text) > max_size:
                text = text[:max_size] + f"\n\n[...текст обрезан, полный размер: {len(text)} символов]"

            logger.info(f"Страница прочитана ({len(text)} символов)")
            return text

        except Exception as e:
            logger.error(f"Ошибка: {e}")
            return f"Ошибка загрузки страницы: {e}"


# ═══════════════════════════════════════════════════════════════
#                     СКАЧИВАНИЕ ФАЙЛОВ
# ═══════════════════════════════════════════════════════════════

class DownloadFileTool(BaseTool):
    """Скачивание файлов из интернета"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="download_file",
            description="Скачивает файл из интернета по URL и сохраняет на диск.",
            required_args=["url"],
            optional_args=["save_path", "filename"],
            arg_types={"url": str, "save_path": str, "filename": str},
            arg_descriptions={
                "url": "URL файла для скачивания",
                "save_path": "Путь для сохранения. По умолчанию — папка загрузок",
                "filename": "Имя файла. По умолчанию — из URL",
            },
            category="web",
            examples=[
                'download_file("https://example.com/report.pdf")',
                'download_file("https://example.com/data.csv", "/home/user/data/")',
            ],
        )

    async def execute(self, url: str, save_path: str = None, filename: str = None) -> str:
        logger.info(f"Скачивание: {url}")

        is_valid, reason = validate_url(url)
        if not is_valid:
            return f"Некорректный URL: {reason}"

        # Определяем путь сохранения
        if save_path:
            download_dir = Path(save_path).expanduser()
        else:
            download_dir = config.DOWNLOAD_DIR

        download_dir.mkdir(parents=True, exist_ok=True)

        # Определяем имя файла
        if not filename:
            from urllib.parse import urlparse, unquote
            parsed = urlparse(url)
            filename = unquote(parsed.path.split('/')[-1]) or "download"

        filepath = download_dir / filename

        try:
            import httpx

            max_size = config.DOWNLOAD_MAX_SIZE_MB * 1024 * 1024

            async with httpx.AsyncClient(follow_redirects=True) as client:
                async with client.stream(
                    "GET",
                    url,
                    timeout=60,
                    headers={'User-Agent': 'Mozilla/5.0'},
                ) as response:
                    response.raise_for_status()

                    # Проверяем размер
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > max_size:
                        size_mb = int(content_length) / (1024 * 1024)
                        return f"Файл слишком большой: {size_mb:.1f} MB (лимит: {config.DOWNLOAD_MAX_SIZE_MB} MB)"

                    # Скачиваем
                    total = 0
                    with open(filepath, 'wb') as f:
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            total += len(chunk)
                            if total > max_size:
                                filepath.unlink(missing_ok=True)
                                return f"Превышен лимит размера ({config.DOWNLOAD_MAX_SIZE_MB} MB)"
                            f.write(chunk)

            size_mb = total / (1024 * 1024)
            logger.info(f"Скачано: {filepath} ({size_mb:.1f} MB)")
            return f"Файл скачан: {filepath}\nРазмер: {size_mb:.1f} MB"

        except Exception as e:
            filepath.unlink(missing_ok=True)
            logger.error(f"Ошибка скачивания: {e}")
            return f"Ошибка скачивания: {e}"


# ═══════════════════════════════════════════════════════════════
#                    ПОГОДА / ВРЕМЯ / ВАЛЮТА
# ═══════════════════════════════════════════════════════════════

class GetWeatherTool(BaseTool):
    """Получение погоды"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="get_weather",
            description="Получает текущую погоду для города: температуру, ветер, влажность, состояние.",
            required_args=["city"],
            arg_types={"city": str},
            arg_descriptions={
                "city": "Название города (например: Москва, London, Санкт-Петербург, Tokyo)",
            },
            category="web",
            examples=[
                'get_weather("Москва")',
                'get_weather("London")',
                'get_weather("Санкт-Петербург")',
            ],
        )

    async def execute(self, city: str = "Moscow") -> str:
        logger.info(f"Получение погоды: {city}")

        try:
            import httpx

            url = f"https://wttr.in/{city}?format=j1&lang=ru"

            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=5)
            response.raise_for_status()

            data = response.json()
            current = data['current_condition'][0]

            result = f"Погода в {city}:\n"
            result += f"Температура: {current['temp_C']}°C (ощущается как {current['FeelsLikeC']}°C)\n"
            result += f"Состояние: {current['lang_ru'][0]['value']}\n"
            result += f"Влажность: {current['humidity']}%\n"
            result += f"Ветер: {current['windspeedKmph']} км/ч"

            logger.info(f"Погода получена для {city}")
            return result

        except Exception as e:
            logger.error(f"Ошибка получения погоды: {e}")
            return f"Не удалось получить погоду для {city}: {e}"


class GetCurrencyRateTool(BaseTool):
    """Курс валют (ЦБ РФ API)"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="get_currency_rate",
            description="Получает официальный курс валюты к рублю от ЦБ РФ. Поддерживает USD, EUR, CNY, GBP и другие.",
            required_args=["currency"],
            arg_types={"currency": str},
            arg_descriptions={
                "currency": "Код валюты (USD, EUR, CNY, GBP, JPY и т.д.)",
            },
            category="web",
            examples=[
                'get_currency_rate("USD")',
                'get_currency_rate("EUR")',
                'get_currency_rate("CNY")',
            ],
        )

    async def execute(self, currency: str = "USD") -> str:
        logger.info(f"Получение курса: {currency}")

        try:
            import httpx

            url = "https://www.cbr-xml-daily.ru/daily_json.js"

            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=5)
            response.raise_for_status()

            data = response.json()
            code = currency.upper()

            if code not in data['Valute']:
                available = ", ".join(list(data['Valute'].keys())[:10])
                return f"Валюта {currency} не найдена. Доступные: {available}..."

            valute = data['Valute'][code]
            rate = valute['Value'] / valute['Nominal']
            date = data['Date'][:10]

            result = f"Курс ЦБ РФ на {date}:\n"
            result += f"{valute['Name']}: {rate:.4f} руб.\n"
            result += f"(Номинал: {valute['Nominal']}, Значение: {valute['Value']:.4f})"

            if 'Previous' in valute:
                prev_rate = valute['Previous'] / valute['Nominal']
                diff = rate - prev_rate
                if diff > 0:
                    result += f"\n+{diff:.4f} руб. к предыдущему дню"
                elif diff < 0:
                    result += f"\n{diff:.4f} руб. к предыдущему дню"

            logger.info(f"Курс {currency} получен")
            return result

        except Exception as e:
            logger.error(f"Ошибка получения курса: {e}")
            return f"Ошибка получения курса: {e}"


class GetCurrentTimeTool(BaseTool):
    """Текущее время"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="get_current_time",
            description="Показывает текущее время, дату и день недели.",
            required_args=[],
            category="web",
            examples=['get_current_time()'],
        )

    async def execute(self) -> str:
        now = datetime.now()

        weekdays = {
            0: "понедельник", 1: "вторник", 2: "среда",
            3: "четверг", 4: "пятница", 5: "суббота", 6: "воскресенье",
        }

        weekday = weekdays[now.weekday()]

        result = f"Время: {now.strftime('%H:%M:%S')}\n"
        result += f"Дата: {now.strftime('%d.%m.%Y')} ({weekday})"

        return result

"""
Инструменты для работы с интернетом - ИСПРАВЛЕННАЯ ВЕРСИЯ
"""

from typing import Optional
from datetime import datetime
from tools.base import BaseTool, ToolSchema
from utils.logging import get_logger
from utils.validators import validate_url
import config

logger = get_logger("web_tools")

# ✅ ИСПРАВЛЕНИЕ: Проверяем доступность библиотек при импорте модуля
DDGS_AVAILABLE = False
DDGS_CLASS = None

try:
    from ddgs import DDGS as DDGS_NEW
    DDGS_CLASS = DDGS_NEW
    DDGS_AVAILABLE = True
    logger.info("✅ Используем библиотеку 'ddgs' для веб-поиска")
except ImportError:
    try:
        from duckduckgo_search import DDGS as DDGS_OLD
        DDGS_CLASS = DDGS_OLD
        DDGS_AVAILABLE = True
        logger.info("✅ Используем библиотеку 'duckduckgo_search' для веб-поиска")
    except ImportError:
        logger.warning("⚠️ Библиотеки для веб-поиска не установлены")
        logger.warning("⚠️ Установите: pip install ddgs  ИЛИ  pip install duckduckgo-search")

class WebSearchTool(BaseTool):
    """Поиск в интернете"""
    
    def __init__(self):
        super().__init__()
        self.rate_limiter = []
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="web_search",
            description="Ищет информацию в интернете через DuckDuckGo",
            required_args=["query"],
            arg_types={"query": str},
            examples=[
                'web_search("новости ИИ 2025")',
                'web_search("погода москва")',
                'web_search("курс доллара")'
            ]
        )
    
    async def execute(self, query: str) -> str:
        logger.info(f"🔍 Веб-поиск: {query}")
        
        # ✅ ИСПРАВЛЕНИЕ: Проверка доступности библиотеки перед использованием
        if not DDGS_AVAILABLE or DDGS_CLASS is None:
            error_msg = (
                "❌ Библиотеки для веб-поиска не установлены.\n\n"
                "Установите одну из них:\n"
                "  pip install ddgs\n"
                "или\n"
                "  pip install duckduckgo-search\n\n"
                "После установки перезапустите программу."
            )
            logger.error(error_msg)
            return error_msg
        
        try:
            results = []
            
            # Поиск через доступную библиотеку
            ddgs = DDGS_CLASS()
            
            # ✅ ИСПРАВЛЕНИЕ: Обработка разных версий API
            try:
                search_results = ddgs.text(
                    query, 
                    max_results=config.WEB_SEARCH_MAX_RESULTS
                )
            except TypeError:
                # Старая версия API
                search_results = list(ddgs.text(query))[:config.WEB_SEARCH_MAX_RESULTS]
            
            for r in search_results:
                # Обрабатываем разные форматы результатов
                title = r.get("title", "Без заголовка")
                url = r.get("href") or r.get("link", "")
                snippet = r.get("body", "") or r.get("snippet", "")
                
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet[:200]  # Ограничиваем длину
                })
            
            if not results:
                return f"❌ Ничего не найдено по запросу: {query}"
            
            # Форматируем результаты с описанием
            lines = [
                f"🔍 Найдено {len(results)} результатов по запросу: '{query}'",
                ""
            ]
            
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. {r['title']}")
                lines.append(f"   URL: {r['url']}")
                if r['snippet']:
                    lines.append(f"   {r['snippet']}")
                lines.append("")
            
            result_text = "\n".join(lines)
            
            logger.info(f"✅ Найдено {len(results)} результатов")
            
            return result_text
        
        except Exception as e:
            logger.error(f"❌ Ошибка поиска: {type(e).__name__}: {e}")
            return f"❌ Ошибка поиска: {e}\n\nПроверьте подключение к интернету и работоспособность DuckDuckGo."


class WebFetchTool(BaseTool):
    """Чтение веб-страницы"""
    
    def __init__(self):
        super().__init__()
        self.request_history = []
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="web_fetch",
            description="Читает содержимое веб-страницы по URL",
            required_args=["url"],
            arg_types={"url": str},
            examples=[
                'web_fetch("https://example.com/article")'
            ]
        )
    
    def _check_rate_limit(self) -> Optional[str]:
        """Проверяет rate limit"""
        
        now = datetime.now()
        
        # Убираем старые запросы (>1 минуты)
        self.request_history = [
            r for r in self.request_history
            if (now - r).total_seconds() < 60
        ]
        
        if len(self.request_history) >= config.WEB_RATE_LIMIT:
            return f"⚠️ Превышен лимит запросов ({config.WEB_RATE_LIMIT}/мин). Подожди немного."
        
        self.request_history.append(now)
        return None
    
    async def execute(self, url: str) -> str:
        logger.info(f"📄 Чтение URL: {url}")
        
        # Проверка rate limit
        rate_error = self._check_rate_limit()
        if rate_error:
            logger.warning(rate_error)
            return rate_error
        
        # Валидация URL
        is_valid, reason = validate_url(url)
        if not is_valid:
            logger.warning(f"❌ Некорректный URL: {reason}")
            return f"❌ {reason}"
        
        try:
            import httpx
            from bs4 import BeautifulSoup

            # Async запрос
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(
                    url,
                    timeout=config.WEB_REQUEST_TIMEOUT,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                )

            response.raise_for_status()

            # Парсинг
            soup = BeautifulSoup(response.content, 'html.parser')

            # Убираем мусор
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            # Извлекаем текст
            text = soup.get_text(separator='\n', strip=True)

            # Чистим пустые строки
            lines = [line for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)

            # Ограничиваем размер
            max_size = config.WEB_FETCH_MAX_SIZE
            if len(text) > max_size:
                text = text[:max_size] + f"\n\n[...текст обрезан, полный размер: {len(text)} символов]"

            logger.info(f"✅ Страница прочитана ({len(text)} символов)")

            return text

        except httpx.HTTPStatusError as e:
            logger.error(f"❌ Ошибка запроса: {e}")
            return f"❌ Ошибка загрузки страницы: {e}"

        except Exception as e:
            logger.error(f"❌ Ошибка обработки: {e}")
            return f"❌ Ошибка обработки страницы: {e}"


class GetWeatherTool(BaseTool):
    """Получение погоды"""
    
    def __init__(self):
        super().__init__()
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="get_weather",
            description="Получает текущую погоду для города",
            required_args=["city"],
            arg_types={"city": str},
            examples=[
                'get_weather("Москва")',
                'get_weather("London")',
                'get_weather("Санкт-Петербург")'
            ]
        )
    
    async def execute(self, city: str = "Moscow") -> str:
        logger.info(f"🌤️ Получение погоды: {city}")

        try:
            import httpx

            url = f"https://wttr.in/{city}?format=j1&lang=ru"

            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            current = data['current_condition'][0]
            
            result = f"🌤️ Погода в {city}:\n"
            result += f"🌡️ Температура: {current['temp_C']}°C (ощущается как {current['FeelsLikeC']}°C)\n"
            result += f"☁️ Состояние: {current['lang_ru'][0]['value']}\n"
            result += f"💧 Влажность: {current['humidity']}%\n"
            result += f"💨 Ветер: {current['windspeedKmph']} км/ч"
            
            logger.info(f"✅ Погода получена для {city}")
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Ошибка получения погоды: {e}")
            return f"❌ Не удалось получить погоду для {city}: {e}"


class GetCurrencyRateTool(BaseTool):
    """
    Курс валют (ЦБ РФ API)

    ПРИМЕЧАНИЕ: Этот инструмент НЕ зарегистрирован по умолчанию в main.py.
    Для активации добавь его в initialize_system() → tools.
    """
    
    def __init__(self):
        super().__init__()
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="get_currency_rate",
            description="Получает официальный курс валюты к рублю (ЦБ РФ)",
            required_args=["currency"],
            arg_types={"currency": str},
            examples=[
                'get_currency_rate("USD")',
                'get_currency_rate("EUR")',
                'get_currency_rate("CNY")'
            ]
        )
    
    async def execute(self, currency: str = "USD") -> str:
        logger.info(f"💱 Получение курса: {currency}")

        try:
            import httpx

            # API ЦБ РФ (JSON вместо XML)
            url = "https://www.cbr-xml-daily.ru/daily_json.js"

            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            code = currency.upper()
            
            if code not in data['Valute']:
                available = ", ".join(list(data['Valute'].keys())[:10])
                return f"❌ Валюта {currency} не найдена. Доступные: {available}"
            
            valute = data['Valute'][code]
            
            rate = valute['Value'] / valute['Nominal']
            date = data['Date'][:10]
            
            result = f"💱 Курс ЦБ РФ на {date}:\n"
            result += f"{valute['Name']}: {rate:.4f} руб.\n"
            result += f"(Номинал: {valute['Nominal']}, Значение: {valute['Value']:.4f})"
            
            # Динамика
            if 'Previous' in valute:
                prev_rate = valute['Previous'] / valute['Nominal']
                diff = rate - prev_rate
                
                if diff > 0:
                    result += f"\n📈 +{diff:.4f} руб. к предыдущему дню"
                elif diff < 0:
                    result += f"\n📉 {diff:.4f} руб. к предыдущему дню"
            
            logger.info(f"✅ Курс {currency} получен")
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Ошибка получения курса: {e}")
            return f"❌ Ошибка получения курса: {e}"


class GetCurrentTimeTool(BaseTool):
    """Текущее время"""
    
    def __init__(self):
        super().__init__()
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="get_current_time",
            description="Показывает текущее время и дату",
            required_args=[],
            examples=['get_current_time()']
        )
    
    async def execute(self) -> str:
        from datetime import datetime
        
        now = datetime.now()
        
        # День недели на русском
        weekdays = {
            0: "понедельник",
            1: "вторник",
            2: "среда",
            3: "четверг",
            4: "пятница",
            5: "суббота",
            6: "воскресенье"
        }
        
        weekday = weekdays[now.weekday()]
        
        result = f"🕐 Текущее время: {now.strftime('%H:%M:%S')}\n"
        result += f"📅 Дата: {now.strftime('%d.%m.%Y')} ({weekday})"
        
        return result

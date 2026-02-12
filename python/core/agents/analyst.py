"""
Analyst Agent — анализ данных и веб-поиск
"""

from typing import Dict, Any, List
import json

from core.agents.base_agent import BaseAgent
import config

class AnalystAgent(BaseAgent):
    """
    Аналитик — работа с данными и информацией
    
    Специализация:
    - Веб-поиск и анализ результатов
    - Извлечение информации из текста
    - Анализ данных
    - Обзор кода
    """
    
    def __init__(self, tools: Dict):
        # Получаем конфиг из config.py
        model_config = config.AGENT_MODELS["analyst"]
        
        super().__init__(
            name="analyst",
            model_config=model_config,
            capabilities=[
                "web_search",
                "web_fetch",
                "data_analysis",
                "information_extraction",
                "code_review",
                "summarization"
            ],
            description="Аналитик данных и веб-информации"
        )
        
        self.tools = tools
    
    async def execute(self, task: Dict[str, Any]) -> str:
        """Выполняет задачу Analyst"""
        
        task_type = task.get("type", "web_search")
        query = task.get("query", "")
        
        if task_type == "web_search":
            self.logger.info(f"🔍 Поиск: {query}")
            
            # Выполняем поиск
            if "web_search" in self.tools:
                search_results = await self.tools["web_search"](query)
            else:
                return "ERROR: Инструмент web_search недоступен"
            
            # Проверяем на ошибки
            if "ERROR" in search_results:
                return search_results
            
            if "Ничего не найдено" in search_results or not search_results.strip():
                return f"По запросу '{query}' ничего не найдено."
            
            # Анализируем результаты
            self.logger.info("📊 Анализ результатов...")
            
            analysis_prompt = (
                f"Пользователь спросил: '{query}'\n\n"
                f"Результаты поиска:\n{search_results}\n\n"
                "Твоя задача:\n"
                "1. Проанализируй найденную информацию\n"
                "2. Ответь на вопрос пользователя кратко и по существу\n"
                "3. Используй только информацию из результатов поиска\n"
                "4. Если информация недостаточна - так и скажи\n\n"
                "Отвечай кратко (3-5 предложений), на русском языке."
            )
            
            messages = [
                {
                    "role": "system",
                    "content": "Ты аналитик информации. Отвечай кратко и по делу."
                },
                {
                    "role": "user",
                    "content": analysis_prompt
                }
            ]
            
            try:
                analysis = await self._call_model(
                    messages, 
                    temperature=0.3, 
                    max_tokens=300
                )
                
                return analysis.strip()
            
            except Exception as e:
                self.logger.error(f"Ошибка анализа: {e}")
                return f"Найдены результаты, но анализ не удался:\n\n{search_results}"
        
        elif task_type == "web_fetch":
            # Чтение конкретной страницы
            url = task.get("url", "")
            
            if "web_fetch" in self.tools:
                return await self.tools["web_fetch"](url)
            else:
                return "ERROR: Инструмент web_fetch недоступен"
        
        elif task_type == "data_analysis":
            # Анализ данных
            data = task.get("data", "")
            return await self._analyze_data(data)
        
        else:
            return f"ERROR: Неизвестный тип задачи: {task_type}"
    
    async def _web_search_and_analyze(self, task: Dict) -> str:
        """Поиск в вебе + анализ результатов"""
        
        query = task.get("query", "")
        max_results = task.get("max_results", 3)
        
        if not query:
            return "ERROR: Не указан поисковый запрос"
        
        try:
            # Шаг 1: Веб-поиск
            self.logger.info(f"🔍 Поиск: {query}")
            
            search_tool = self.tools.get("web_search")
            if not search_tool:
                return "ERROR: Инструмент web_search недоступен"
            
            search_results = await search_tool(query)
            
            # Шаг 2: Анализ результатов
            self.logger.info("📊 Анализ результатов...")
            
            prompt = f"""Проанализируй результаты веб-поиска и извлеки ключевую информацию.

Запрос пользователя: "{query}"

Результаты поиска:
{search_results[:2000]}

ЗАДАЧА:
1. Извлеки самую важную информацию по запросу
2. Суммируй в 3-5 предложениях
3. Укажи источники (если есть URLs)
4. Отвечай на русском языке

Формат ответа:
[Краткое резюме информации]

Источники: [URLs если есть]"""
            
            messages = [
                {
                    "role": "system",
                    "content": "Ты аналитик, который извлекает и суммирует информацию из веб-результатов."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            analysis = await self._call_model(messages, temperature=0.5, max_tokens=500)
            
            return analysis.strip()
        
        except Exception as e:
            self.logger.error(f"Ошибка веб-анализа: {e}")
            return f"ERROR: {str(e)}"
    
    async def _analyze_data(self, task: Dict) -> str:
        """Анализирует структурированные данные"""
        
        data = task.get("data")
        question = task.get("question", "Проанализируй данные")
        
        if not data:
            return "ERROR: Нет данных для анализа"
        
        try:
            # Преобразуем данные в строку
            data_str = json.dumps(data, ensure_ascii=False, indent=2)[:1500]
            
            prompt = f"""Проанализируй следующие данные и ответь на вопрос.

Вопрос: {question}

Данные:
{data_str}

Дай краткий, информативный ответ на русском языке."""
            
            messages = [
                {"role": "system", "content": "Ты аналитик данных."},
                {"role": "user", "content": prompt}
            ]
            
            analysis = await self._call_model(messages, temperature=0.3, max_tokens=400)
            
            return analysis.strip()
        
        except Exception as e:
            self.logger.error(f"Ошибка анализа данных: {e}")
            return f"ERROR: {str(e)}"
    
    async def _summarize(self, task: Dict) -> str:
        """Суммаризация текста"""
        
        text = task.get("text", "")
        max_length = task.get("max_length", 200)
        
        if not text:
            return "ERROR: Нет текста для суммаризации"
        
        try:
            prompt = f"""Суммаризуй следующий текст в {max_length} словах или меньше.

Текст:
{text[:3000]}

Требования:
- Сохрани ключевую информацию
- Будь лаконичным
- Используй русский язык"""
            
            messages = [
                {"role": "system", "content": "Ты эксперт по суммаризации текстов."},
                {"role": "user", "content": prompt}
            ]
            
            summary = await self._call_model(messages, temperature=0.5, max_tokens=300)
            
            return summary.strip()
        
        except Exception as e:
            self.logger.error(f"Ошибка суммаризации: {e}")
            return f"ERROR: {str(e)}"
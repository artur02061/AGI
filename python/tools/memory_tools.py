"""
Инструменты для работы с памятью
"""

from tools.base import BaseTool, ToolSchema
from utils.logging import get_logger

logger = get_logger("memory_tools")

class RecallMemoryTool(BaseTool):
    """Поиск в эпизодической памяти"""
    
    def __init__(self, memory):
        super().__init__()
        self.memory = memory
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="recall_memory",
            description="Ищет информацию в краткосрочной памяти (последние диалоги)",
            required_args=["query"],
            arg_types={"query": str},
            examples=[
                'recall_memory("что мы обсуждали про погоду")',
                'recall_memory("игра которую я упоминал")'
            ]
        )
    
    async def execute(self, query: str) -> str:
        logger.info(f"Поиск в памяти: {query}")
        
        context = self.memory.get_relevant_context(query, max_items=5)
        
        if context == "Нет релевантного контекста.":
            return "🤷 Ничего не нашла в краткосрочной памяти по этому запросу"
        
        return f"📚 Из краткосрочной памяти:\n\n{context}"


class SearchMemoryTool(BaseTool):
    """Поиск в векторной памяти"""
    
    def __init__(self, vector_memory):
        super().__init__()
        self.vector_memory = vector_memory
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="search_memory",
            description="Ищет информацию в долговременной памяти (все прошлые разговоры)",
            required_args=["query"],
            optional_args=["timeframe"],
            arg_types={"query": str, "timeframe": str},
            examples=[
                'search_memory("проект о котором говорили")',
                'search_memory("курс доллара", "last_week")',
                'search_memory("игры в которые я играл")'
            ]
        )
    
    async def execute(self, query: str, timeframe: str = None) -> str:
        logger.info(f"Поиск в долговременной памяти: {query} (timeframe: {timeframe})")
        
        # Поиск с учётом времени
        if timeframe:
            results = self.vector_memory.search_by_timeframe(
                query,
                timeframe=timeframe,
                n_results=5
            )
        else:
            results = self.vector_memory.search(
                query,
                n_results=5,
                filter_metadata={"type": "dialogue"}
            )
        
        if not results:
            return "🤷 Ничего не нашла в долговременной памяти по этому запросу"
        
        # Форматируем результаты
        lines = ["🧠 Из долговременной памяти:", ""]
        
        for r in results:
            date = r['metadata'].get('date', 'н/д')
            text = r['text'][:150] + "..." if len(r['text']) > 150 else r['text']
            
            lines.append(f"📅 [{date}]")
            lines.append(f"   {text}")
            lines.append("")
        
        logger.info(f"✅ Найдено {len(results)} воспоминаний")
        
        return "\n".join(lines)
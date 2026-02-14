"""
Кристина 7.3 — Code Understanding (Понимание кода)

ЗАЧЕМ:
  Кристина помогает с кодом БЕЗ LLM:
  - Парсит Python-код через AST
  - Извлекает структуру: функции, классы, импорты
  - Находит паттерны и анти-паттерны
  - Создаёт code embeddings для поиска похожего кода
  - Объясняет что делает функция (на основе AST)

АРХИТЕКТУРА:
  ┌─────────────────────────────────────────────┐
  │ Source code (Python)                        │
  │         ↓                                    │
  │ ┌─────────────────┐                          │
  │ │  Python AST      │ → дерево разбора        │
  │ └────────┬────────┘                          │
  │          ↓                                    │
  │ ┌─────────────────────────────────────────┐  │
  │ │ CodeAnalyzer                            │  │
  │ │  - extract_functions()                  │  │
  │ │  - extract_classes()                    │  │
  │ │  - extract_imports()                    │  │
  │ │  - find_patterns()                      │  │
  │ │  - complexity_score()                   │  │
  │ └────────┬────────────────────────────────┘  │
  │          ↓                                    │
  │ ┌─────────────────────────────────────────┐  │
  │ │ CodeEmbedder                            │  │
  │ │  - code_to_vec() (bag-of-AST-nodes)     │  │
  │ │  - search_similar()                     │  │
  │ └─────────────────────────────────────────┘  │
  └─────────────────────────────────────────────┘

ЧИСТЫЙ PYTHON: использует только ast (стандартная библиотека).
"""

import ast
import json
import math
import re
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field

from utils.logging import get_logger
import config

logger = get_logger("code_understanding")


# ═══════════════════════════════════════════════════════════════
#               СТРУКТУРЫ ДАННЫХ
# ═══════════════════════════════════════════════════════════════


@dataclass
class FunctionInfo:
    """Информация о функции"""
    name: str
    args: List[str]
    returns: Optional[str] = None
    docstring: Optional[str] = None
    line_start: int = 0
    line_end: int = 0
    complexity: int = 1      # Cyclomatic complexity
    is_async: bool = False
    decorators: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)  # Вызываемые функции

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "args": self.args,
            "returns": self.returns,
            "docstring": self.docstring,
            "lines": f"{self.line_start}-{self.line_end}",
            "complexity": self.complexity,
            "is_async": self.is_async,
            "decorators": self.decorators,
            "calls": self.calls,
        }


@dataclass
class ClassInfo:
    """Информация о классе"""
    name: str
    bases: List[str]
    methods: List[FunctionInfo]
    docstring: Optional[str] = None
    line_start: int = 0
    line_end: int = 0
    attributes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "bases": self.bases,
            "methods": [m.to_dict() for m in self.methods],
            "docstring": self.docstring,
            "lines": f"{self.line_start}-{self.line_end}",
            "attributes": self.attributes,
        }


@dataclass
class CodePattern:
    """Обнаруженный паттерн/анти-паттерн"""
    name: str
    severity: str       # "info", "warning", "error"
    message: str
    line: int = 0
    suggestion: str = ""


@dataclass
class CodeAnalysis:
    """Результат анализа кода"""
    functions: List[FunctionInfo]
    classes: List[ClassInfo]
    imports: List[str]
    patterns: List[CodePattern]
    total_lines: int = 0
    complexity_score: float = 0.0
    summary: str = ""


# ═══════════════════════════════════════════════════════════════
#               AST АНАЛИЗАТОР
# ═══════════════════════════════════════════════════════════════


class CodeAnalyzer:
    """
    Анализирует Python-код через AST.

    Использование:
        analyzer = CodeAnalyzer()
        analysis = analyzer.analyze(source_code)
        print(analysis.summary)

        # Отдельные функции
        funcs = analyzer.extract_functions(source_code)
        classes = analyzer.extract_classes(source_code)
        complexity = analyzer.complexity_score(source_code)
    """

    def analyze(self, source: str) -> Optional[CodeAnalysis]:
        """Полный анализ исходного кода"""
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return CodeAnalysis(
                functions=[], classes=[], imports=[],
                patterns=[CodePattern(
                    name="syntax_error", severity="error",
                    message=f"Ошибка синтаксиса: {e}",
                    line=getattr(e, 'lineno', 0),
                )],
                total_lines=source.count('\n') + 1,
                summary=f"Ошибка парсинга: {e}",
            )

        functions = self._extract_functions(tree)
        classes = self._extract_classes(tree)
        imports = self._extract_imports(tree)
        patterns = self._find_patterns(tree, source)
        total_lines = source.count('\n') + 1
        complexity = self._total_complexity(functions, classes)

        summary = self._build_summary(functions, classes, imports, total_lines, complexity)

        return CodeAnalysis(
            functions=functions,
            classes=classes,
            imports=imports,
            patterns=patterns,
            total_lines=total_lines,
            complexity_score=complexity,
            summary=summary,
        )

    def _extract_functions(self, tree: ast.AST) -> List[FunctionInfo]:
        """Извлекает все функции верхнего уровня"""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Пропускаем методы классов (обрабатываются в _extract_classes)
                if self._is_top_level_or_nested(node, tree):
                    info = self._parse_function(node)
                    functions.append(info)
        return functions

    def _is_top_level_or_nested(self, func_node, tree) -> bool:
        """Проверяет что функция не является методом класса"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in ast.walk(node):
                    if item is func_node:
                        return False
        return True

    def _parse_function(self, node) -> FunctionInfo:
        """Парсит AST-узел функции"""
        # Arguments
        args = []
        for arg in node.args.args:
            args.append(arg.arg)

        # Return annotation
        returns = None
        if node.returns:
            returns = ast.dump(node.returns) if not isinstance(node.returns, ast.Constant) \
                else str(node.returns.value)
            # Упрощаем
            if isinstance(node.returns, ast.Name):
                returns = node.returns.id
            elif isinstance(node.returns, ast.Attribute):
                returns = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns.attr)

        # Docstring
        docstring = ast.get_docstring(node)

        # Decorators
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(
                    ast.unparse(dec) if hasattr(ast, 'unparse') else dec.attr
                )

        # Calls inside the function
        calls = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.add(child.func.attr)

        # Complexity
        complexity = self._cyclomatic_complexity(node)

        return FunctionInfo(
            name=node.name,
            args=args,
            returns=returns,
            docstring=docstring[:200] if docstring else None,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            complexity=complexity,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            decorators=decorators,
            calls=sorted(calls),
        )

    def _extract_classes(self, tree: ast.AST) -> List[ClassInfo]:
        """Извлекает все классы"""
        classes = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                info = self._parse_class(node)
                classes.append(info)
        return classes

    def _parse_class(self, node: ast.ClassDef) -> ClassInfo:
        """Парсит AST-узел класса"""
        # Bases
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(
                    ast.unparse(base) if hasattr(ast, 'unparse') else base.attr
                )

        # Methods
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(self._parse_function(item))

        # Docstring
        docstring = ast.get_docstring(node)

        # Attributes (from __init__ self.xxx = ...)
        attributes = set()
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name == '__init__':
                    for child in ast.walk(item):
                        if isinstance(child, ast.Assign):
                            for target in child.targets:
                                if isinstance(target, ast.Attribute) and \
                                   isinstance(target.value, ast.Name) and \
                                   target.value.id == 'self':
                                    attributes.add(target.attr)

        return ClassInfo(
            name=node.name,
            bases=bases,
            methods=methods,
            docstring=docstring[:200] if docstring else None,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            attributes=sorted(attributes),
        )

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Извлекает все импорты"""
        imports = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        return imports

    def _cyclomatic_complexity(self, node: ast.AST) -> int:
        """
        Вычисляет цикломатическую сложность.
        CC = 1 + число if/for/while/except/and/or/elif
        """
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.IfExp)):
                complexity += 1
            elif isinstance(child, (ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, (ast.While,)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.Assert,)):
                complexity += 1
        return complexity

    def _total_complexity(
        self,
        functions: List[FunctionInfo],
        classes: List[ClassInfo],
    ) -> float:
        """Средняя сложность по всем функциям"""
        all_funcs = list(functions)
        for cls in classes:
            all_funcs.extend(cls.methods)

        if not all_funcs:
            return 0.0

        return sum(f.complexity for f in all_funcs) / len(all_funcs)

    # ═══════════════════════════════════════════════════════════════
    #           ПАТТЕРНЫ И АНТИ-ПАТТЕРНЫ
    # ═══════════════════════════════════════════════════════════════

    def _find_patterns(self, tree: ast.AST, source: str) -> List[CodePattern]:
        """Находит паттерны и анти-паттерны"""
        patterns = []

        for node in ast.walk(tree):
            # 1. Слишком длинная функция (>50 строк)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = (node.end_lineno or node.lineno) - node.lineno
                if length > 50:
                    patterns.append(CodePattern(
                        name="long_function",
                        severity="warning",
                        message=f"Функция '{node.name}' слишком длинная ({length} строк)",
                        line=node.lineno,
                        suggestion="Разбей на несколько меньших функций",
                    ))

                # 2. Слишком много аргументов (>5)
                n_args = len(node.args.args)
                if n_args > 5:
                    patterns.append(CodePattern(
                        name="too_many_args",
                        severity="warning",
                        message=f"Функция '{node.name}' имеет {n_args} аргументов",
                        line=node.lineno,
                        suggestion="Используй dataclass или dict для группировки",
                    ))

                # 3. Высокая цикломатическая сложность
                cc = self._cyclomatic_complexity(node)
                if cc > 10:
                    patterns.append(CodePattern(
                        name="high_complexity",
                        severity="warning",
                        message=f"Функция '{node.name}' сложная (CC={cc})",
                        line=node.lineno,
                        suggestion="Упрости логику, вынеси ветвления",
                    ))

            # 4. Bare except
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                patterns.append(CodePattern(
                    name="bare_except",
                    severity="warning",
                    message="Используется bare except (ловит всё)",
                    line=node.lineno,
                    suggestion="Указывай конкретные исключения: except ValueError",
                ))

            # 5. Mutable default argument
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in node.args.defaults + node.args.kw_defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        patterns.append(CodePattern(
                            name="mutable_default",
                            severity="warning",
                            message=f"Мутабельный аргумент по умолчанию в '{node.name}'",
                            line=node.lineno,
                            suggestion="Используй None и создавай внутри функции",
                        ))

            # 6. Global statement
            if isinstance(node, ast.Global):
                patterns.append(CodePattern(
                    name="global_usage",
                    severity="info",
                    message=f"Используется global: {', '.join(node.names)}",
                    line=node.lineno,
                    suggestion="Избегай global, используй параметры или класс",
                ))

        return patterns

    def _build_summary(
        self,
        functions: List[FunctionInfo],
        classes: List[ClassInfo],
        imports: List[str],
        total_lines: int,
        complexity: float,
    ) -> str:
        """Строит человекочитаемое описание кода"""
        parts = [f"Код: {total_lines} строк"]

        if classes:
            class_names = ", ".join(c.name for c in classes)
            total_methods = sum(len(c.methods) for c in classes)
            parts.append(f"{len(classes)} класс(ов) [{class_names}], {total_methods} метод(ов)")

        if functions:
            func_names = ", ".join(f.name for f in functions[:5])
            if len(functions) > 5:
                func_names += f" и ещё {len(functions) - 5}"
            parts.append(f"{len(functions)} функций [{func_names}]")

        if imports:
            parts.append(f"{len(imports)} импортов")

        if complexity > 0:
            level = "низкая" if complexity < 5 else "средняя" if complexity < 10 else "высокая"
            parts.append(f"сложность: {complexity:.1f} ({level})")

        return ". ".join(parts)


# ═══════════════════════════════════════════════════════════════
#               CODE EMBEDDER (bag-of-AST)
# ═══════════════════════════════════════════════════════════════

# AST-node types for bag-of-AST encoding
AST_NODE_TYPES = [
    "FunctionDef", "AsyncFunctionDef", "ClassDef",
    "Return", "Assign", "AugAssign", "AnnAssign",
    "For", "AsyncFor", "While", "If", "With", "AsyncWith",
    "Raise", "Try", "Assert", "Import", "ImportFrom",
    "Global", "Nonlocal", "Expr", "Pass", "Break", "Continue",
    "BoolOp", "BinOp", "UnaryOp", "Lambda", "IfExp",
    "Dict", "Set", "ListComp", "SetComp", "DictComp", "GeneratorExp",
    "Await", "Yield", "YieldFrom",
    "Compare", "Call", "JoinedStr", "Attribute", "Subscript",
    "Starred", "Name", "List", "Tuple", "Slice",
]

AST_NODE_TO_IDX = {name: i for i, name in enumerate(AST_NODE_TYPES)}
CODE_EMBED_DIM = len(AST_NODE_TYPES) + 8  # AST nodes + structural features


class CodeEmbedder:
    """
    Создаёт вектор кода из AST-структуры (bag-of-AST-nodes).

    Каждая позиция = частота определённого типа AST-узла.
    Плюс 8 структурных признаков (сложность, глубина и т.д.).
    """

    def embed(self, source: str) -> Optional[List[float]]:
        """
        Кодирует Python-код в вектор фиксированной размерности.

        Returns:
            Вектор [CODE_EMBED_DIM] или None при ошибке
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        vec = [0.0] * CODE_EMBED_DIM

        # 1. Bag-of-AST-nodes: считаем частоту каждого типа узла
        total_nodes = 0
        for node in ast.walk(tree):
            node_type = type(node).__name__
            if node_type in AST_NODE_TO_IDX:
                vec[AST_NODE_TO_IDX[node_type]] += 1.0
                total_nodes += 1

        # Нормализуем частоты
        if total_nodes > 0:
            for i in range(len(AST_NODE_TYPES)):
                vec[i] /= total_nodes

        # 2. Структурные признаки (8 штук)
        base = len(AST_NODE_TYPES)

        # Количество строк (log-scale)
        n_lines = source.count('\n') + 1
        vec[base + 0] = math.log1p(n_lines) / 10.0

        # Количество функций
        n_funcs = sum(1 for n in ast.walk(tree)
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        vec[base + 1] = math.log1p(n_funcs) / 5.0

        # Количество классов
        n_classes = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
        vec[base + 2] = math.log1p(n_classes) / 3.0

        # Максимальная глубина вложенности
        max_depth = self._max_depth(tree)
        vec[base + 3] = min(max_depth / 10.0, 1.0)

        # Количество импортов
        n_imports = sum(1 for n in ast.walk(tree)
                        if isinstance(n, (ast.Import, ast.ImportFrom)))
        vec[base + 4] = math.log1p(n_imports) / 5.0

        # Доля циклов
        n_loops = sum(1 for n in ast.walk(tree)
                      if isinstance(n, (ast.For, ast.While, ast.AsyncFor)))
        vec[base + 5] = n_loops / max(total_nodes, 1)

        # Доля условий
        n_ifs = sum(1 for n in ast.walk(tree) if isinstance(n, ast.If))
        vec[base + 6] = n_ifs / max(total_nodes, 1)

        # Есть ли async
        n_async = sum(1 for n in ast.walk(tree)
                      if isinstance(n, (ast.AsyncFunctionDef, ast.AsyncFor, ast.Await)))
        vec[base + 7] = min(n_async / 5.0, 1.0)

        return vec

    def _max_depth(self, tree: ast.AST) -> int:
        """Максимальная глубина вложенности AST"""
        def _depth(node, current=0):
            max_d = current
            for child in ast.iter_child_nodes(node):
                max_d = max(max_d, _depth(child, current + 1))
            return max_d
        return _depth(tree)

    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Косинусная близость двух code embeddings"""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1) + 1e-10)
        norm2 = math.sqrt(sum(b * b for b in vec2) + 1e-10)
        return dot / (norm1 * norm2)


# ═══════════════════════════════════════════════════════════════
#               CODE UNDERSTANDING ENGINE
# ═══════════════════════════════════════════════════════════════


class CodeUnderstanding:
    """
    Центральный модуль понимания кода.

    Использование:
        cu = CodeUnderstanding()

        # Анализ кода
        analysis = cu.analyze_code(source_code)
        print(analysis.summary)

        # Объяснение функции
        explanation = cu.explain_function(source_code, "my_function")

        # Поиск похожего кода
        similar = cu.search_similar(source_code, top_k=3)

        # Паттерны
        patterns = cu.find_patterns(source_code)
    """

    def __init__(self, db_path: Path = None):
        self.analyzer = CodeAnalyzer()
        self.embedder = CodeEmbedder()

        # Хранилище code embeddings
        self._db_path = db_path or (config.config.data_dir / "code_understanding.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

        self._total_analyses = 0
        self._load_stats()

        logger.info(
            f"💻 CodeUnderstanding: embed_dim={CODE_EMBED_DIM}, "
            f"{self._total_analyses} analyses"
        )

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS code_snippets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                embedding TEXT NOT NULL,
                summary TEXT,
                created_at REAL NOT NULL,
                UNIQUE(source_hash)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS code_stats (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def _load_stats(self):
        row = self._conn.execute(
            "SELECT value FROM code_stats WHERE key = 'total_analyses'"
        ).fetchone()
        if row:
            self._total_analyses = int(row[0])

    def _save_stats(self):
        self._conn.execute("""
            INSERT INTO code_stats (key, value) VALUES ('total_analyses', ?)
            ON CONFLICT(key) DO UPDATE SET value = ?
        """, (str(self._total_analyses), str(self._total_analyses)))
        self._conn.commit()

    # ═══════════════════════════════════════════════════════════════
    #           ОСНОВНЫЕ ОПЕРАЦИИ
    # ═══════════════════════════════════════════════════════════════

    def analyze_code(self, source: str) -> Optional[CodeAnalysis]:
        """Полный анализ Python-кода"""
        analysis = self.analyzer.analyze(source)
        self._total_analyses += 1

        if self._total_analyses % 20 == 0:
            self._save_stats()

        return analysis

    def explain_function(self, source: str, func_name: str) -> Optional[str]:
        """
        Генерирует объяснение функции на основе AST.
        Работает без LLM!
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        # Ищем функцию
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == func_name:
                    return self._explain_function_node(node)

        return None

    def _explain_function_node(self, node) -> str:
        """Строит объяснение функции из AST"""
        parts = []

        # Тип
        if isinstance(node, ast.AsyncFunctionDef):
            parts.append(f"Асинхронная функция `{node.name}`")
        else:
            parts.append(f"Функция `{node.name}`")

        # Аргументы
        args = [a.arg for a in node.args.args if a.arg != 'self']
        if args:
            parts.append(f"Принимает: {', '.join(args)}")
        else:
            parts.append("Без аргументов")

        # Docstring
        docstring = ast.get_docstring(node)
        if docstring:
            first_line = docstring.split('\n')[0].strip()
            parts.append(f"Описание: {first_line}")

        # Что делает (по AST)
        actions = []
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value:
                actions.append("возвращает результат")
            if isinstance(child, (ast.For, ast.AsyncFor)):
                actions.append("использует цикл")
            if isinstance(child, ast.While):
                actions.append("содержит while-цикл")
            if isinstance(child, ast.If):
                actions.append("содержит условие")
            if isinstance(child, ast.Try):
                actions.append("обрабатывает исключения")
            if isinstance(child, ast.Yield):
                actions.append("является генератором")
            if isinstance(child, ast.Await):
                actions.append("ожидает async-операцию")
            if isinstance(child, ast.ListComp):
                actions.append("использует list comprehension")

        if actions:
            unique_actions = list(dict.fromkeys(actions))  # Remove duplicates, keep order
            parts.append("Действия: " + ", ".join(unique_actions[:5]))

        # Вызовы
        calls = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.add(child.func.attr)
        if calls:
            parts.append(f"Вызывает: {', '.join(sorted(calls)[:8])}")

        # Сложность
        cc = self.analyzer._cyclomatic_complexity(node)
        lines = (node.end_lineno or node.lineno) - node.lineno + 1
        parts.append(f"Размер: {lines} строк, сложность: {cc}")

        return ". ".join(parts)

    def index_code(self, name: str, source: str):
        """
        Индексирует код для поиска похожих фрагментов.
        """
        embedding = self.embedder.embed(source)
        if not embedding:
            return

        source_hash = str(hash(source))
        analysis = self.analyzer.analyze(source)
        summary = analysis.summary if analysis else ""

        self._conn.execute("""
            INSERT INTO code_snippets (name, source_hash, embedding, summary, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source_hash) DO UPDATE SET
                name = ?, embedding = ?, summary = ?, created_at = ?
        """, (
            name, source_hash, json.dumps(embedding), summary, time.time(),
            name, json.dumps(embedding), summary, time.time(),
        ))
        self._conn.commit()

    def search_similar(self, source: str, top_k: int = 3) -> List[Dict]:
        """Ищет похожие проиндексированные фрагменты кода"""
        query_vec = self.embedder.embed(source)
        if not query_vec:
            return []

        rows = self._conn.execute(
            "SELECT name, embedding, summary FROM code_snippets"
        ).fetchall()

        results = []
        for name, emb_json, summary in rows:
            try:
                emb = json.loads(emb_json)
                sim = self.embedder.similarity(query_vec, emb)
                results.append({
                    "name": name,
                    "similarity": round(sim, 4),
                    "summary": summary,
                })
            except (json.JSONDecodeError, TypeError):
                pass

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def find_patterns(self, source: str) -> List[Dict]:
        """Находит паттерны и анти-паттерны в коде"""
        analysis = self.analyze_code(source)
        if not analysis:
            return []

        return [
            {
                "name": p.name,
                "severity": p.severity,
                "message": p.message,
                "line": p.line,
                "suggestion": p.suggestion,
            }
            for p in analysis.patterns
        ]

    # ═══════════════════════════════════════════════════════════════
    #           СТАТИСТИКА
    # ═══════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict:
        snippet_count = self._conn.execute(
            "SELECT COUNT(*) FROM code_snippets"
        ).fetchone()[0]

        return {
            "total_analyses": self._total_analyses,
            "indexed_snippets": snippet_count,
            "embed_dim": CODE_EMBED_DIM,
        }

    def close(self):
        self._save_stats()
        self._conn.close()

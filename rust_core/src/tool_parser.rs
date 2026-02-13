//! ToolCallParser — парсер вызовов инструментов
//!
//! Разбирает строки вида: tool_name("arg1", "arg2", key="value")
//! Поддерживает:
//! - Позиционные и именованные аргументы
//! - Вложенные скобки и экранированные строки
//! - Извлечение ACTION:/FINAL_ANSWER: из текста
//! - Валидацию по списку известных инструментов

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use parking_lot::RwLock;
use std::collections::HashMap;

#[pyclass(frozen)]
pub struct ToolCallParser {
    known_tools: RwLock<Vec<String>>,
}

#[pymethods]
impl ToolCallParser {
    #[new]
    #[pyo3(signature = (known_tools=None))]
    fn new(known_tools: Option<Vec<String>>) -> Self {
        Self {
            known_tools: RwLock::new(known_tools.unwrap_or_default()),
        }
    }

    fn parse(&self, input: &str) -> PyResult<(String, Vec<String>, HashMap<String, String>)> {
        let input = input.trim();

        let paren_pos = input.find('(').ok_or_else(|| {
            PyValueError::new_err(format!(
                "Нет скобок в вызове: '{}'. Формат: tool_name(\"аргументы\")",
                input
            ))
        })?;

        let name = input[..paren_pos].trim().to_string();

        // Валидация по списку известных инструментов
        let known = self.known_tools.read();
        if !known.is_empty() && !known.contains(&name) {
            let available = known.join(", ");
            return Err(PyValueError::new_err(format!(
                "Инструмент '{}' не существует. Доступны: {}",
                name, available
            )));
        }
        drop(known);

        let rest = &input[paren_pos + 1..];
        let close_pos = find_matching_paren(rest).map_err(|e| PyValueError::new_err(e))?;

        let args_str = rest[..close_pos].trim();
        if args_str.is_empty() {
            return Ok((name, vec![], HashMap::new()));
        }

        let (args, kwargs) = parse_arguments(args_str);
        Ok((name, args, kwargs))
    }

    fn extract_action(&self, text: &str) -> Option<String> {
        for line in text.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("ACTION:") {
                let action = rest.trim();
                if !action.is_empty() {
                    return Some(action.to_string());
                }
            }
        }
        None
    }

    fn extract_final_answer(&self, text: &str) -> Option<String> {
        if let Some(pos) = text.find("FINAL_ANSWER:") {
            let answer = text[pos + "FINAL_ANSWER:".len()..].trim();
            if !answer.is_empty() {
                return Some(answer.to_string());
            }
        }
        None
    }

    fn is_final_answer(&self, text: &str) -> bool {
        text.contains("FINAL_ANSWER:")
    }

    fn is_action(&self, text: &str) -> bool {
        text.contains("ACTION:")
    }

    fn set_known_tools(&self, tools: Vec<String>) {
        *self.known_tools.write() = tools;
    }
}

// ── Парсер аргументов ──

fn find_matching_paren(s: &str) -> Result<usize, String> {
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut string_char = '"';
    let mut escape_next = false;

    for (i, ch) in s.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if ch == '\\' {
            escape_next = true;
            continue;
        }
        if in_string {
            if ch == string_char {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' | '\'' => {
                in_string = true;
                string_char = ch;
            }
            '(' => depth += 1,
            ')' => {
                if depth == 0 {
                    return Ok(i);
                }
                depth -= 1;
            }
            _ => {}
        }
    }

    Err("Не найдена закрывающая скобка".to_string())
}

fn split_args(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut in_string = false;
    let mut string_char = '"';
    let mut escape_next = false;
    let mut paren_depth: i32 = 0;

    for ch in s.chars() {
        if escape_next {
            current.push(ch);
            escape_next = false;
            continue;
        }
        if ch == '\\' {
            escape_next = true;
            current.push(ch);
            continue;
        }
        if in_string {
            current.push(ch);
            if ch == string_char {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' | '\'' => {
                in_string = true;
                string_char = ch;
                current.push(ch);
            }
            '(' => {
                paren_depth += 1;
                current.push(ch);
            }
            ')' => {
                paren_depth -= 1;
                current.push(ch);
            }
            ',' if paren_depth == 0 => {
                let part = current.trim().to_string();
                if !part.is_empty() {
                    parts.push(part);
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }

    let part = current.trim().to_string();
    if !part.is_empty() {
        parts.push(part);
    }
    parts
}

fn unquote(s: &str) -> String {
    let s = s.trim();
    let bytes = s.as_bytes();
    if bytes.len() >= 2 {
        let first = bytes[0];
        let last = bytes[bytes.len() - 1];
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            let inner = &s[1..s.len() - 1];
            return inner
                .replace("\\\"", "\"")
                .replace("\\'", "'")
                .replace("\\\\", "\\")
                .replace("\\n", "\n")
                .replace("\\t", "\t");
        }
    }
    s.to_string()
}

fn eq_in_string(s: &str, eq_pos: usize) -> bool {
    let mut in_string = false;
    let mut string_char = '"';

    for (i, ch) in s.char_indices() {
        if i == eq_pos {
            return in_string;
        }
        if !in_string && (ch == '"' || ch == '\'') {
            in_string = true;
            string_char = ch;
        } else if in_string && ch == string_char {
            in_string = false;
        }
    }
    false
}

fn parse_arguments(s: &str) -> (Vec<String>, HashMap<String, String>) {
    let mut args = Vec::new();
    let mut kwargs = HashMap::new();
    let parts = split_args(s);

    for part in parts {
        if let Some(eq_pos) = part.find('=') {
            if !eq_in_string(&part, eq_pos) {
                let key = part[..eq_pos].trim().to_string();
                let value = unquote(&part[eq_pos + 1..]);
                kwargs.insert(key, value);
                continue;
            }
        }
        args.push(unquote(&part));
    }

    (args, kwargs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_parse() {
        let parser = ToolCallParser::new(None);
        let (name, args, kwargs) = parser.parse("search(\"hello world\")").unwrap();
        assert_eq!(name, "search");
        assert_eq!(args, vec!["hello world"]);
        assert!(kwargs.is_empty());
    }

    #[test]
    fn test_kwargs_parse() {
        let parser = ToolCallParser::new(None);
        let (name, args, kwargs) = parser.parse("web_search(\"test\", lang=\"ru\")").unwrap();
        assert_eq!(name, "web_search");
        assert_eq!(args, vec!["test"]);
        assert_eq!(kwargs.get("lang").unwrap(), "ru");
    }

    #[test]
    fn test_no_args() {
        let parser = ToolCallParser::new(None);
        let (name, args, kwargs) = parser.parse("status()").unwrap();
        assert_eq!(name, "status");
        assert!(args.is_empty());
        assert!(kwargs.is_empty());
    }

    #[test]
    fn test_extract_action() {
        let parser = ToolCallParser::new(None);
        let text = "thinking...\nACTION: search(\"test\")\nmore text";
        assert_eq!(
            parser.extract_action(text),
            Some("search(\"test\")".to_string())
        );
    }

    #[test]
    fn test_final_answer() {
        let parser = ToolCallParser::new(None);
        let text = "FINAL_ANSWER: Ответ готов.";
        assert!(parser.is_final_answer(text));
        assert_eq!(
            parser.extract_final_answer(text),
            Some("Ответ готов.".to_string())
        );
    }

    #[test]
    fn test_unknown_tool() {
        let parser = ToolCallParser::new(Some(vec!["search".to_string()]));
        assert!(parser.parse("unknown(\"test\")").is_err());
    }
}

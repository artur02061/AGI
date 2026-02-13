//! ContextCompressor — сжатие контекста разговора
//!
//! Оптимизации:
//! - Aho-Corasick для детекции важных слов за O(n)
//! - Unicode-aware оценка токенов (BPE-эвристика для RU/EN)
//! - Безопасная обрезка по границам символов

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use aho_corasick::AhoCorasick;

const IMPORTANT_WORDS: &[&str] = &[
    "важно", "главное", "нужно", "проблема", "решение",
    "ошибка", "успешно", "не работает", "помоги", "критично",
    "срочно", "обязательно", "ключевой", "основной", "результат",
    "вывод", "итог", "причина", "следствие", "вопрос",
];

#[pyclass(frozen)]
pub struct ContextCompressor {
    #[allow(dead_code)]
    compression_ratio: f64,
    important_ac: AhoCorasick,
}

#[pymethods]
impl ContextCompressor {
    #[new]
    #[pyo3(signature = (compression_ratio=0.3))]
    fn new(compression_ratio: f64) -> Self {
        // Строим паттерны в lowercase для сопоставления с lowercase текстом
        Self {
            compression_ratio,
            important_ac: AhoCorasick::new(IMPORTANT_WORDS).unwrap(),
        }
    }

    /// Сжимает историю разговора.
    /// Принимает List[Tuple[str,str,str]] ИЛИ List[Dict] с ключами role/content/timestamp.
    fn compress_conversation(&self, messages: Bound<'_, pyo3::types::PyList>) -> PyResult<String> {
        let len = messages.len();
        if len == 0 {
            return Ok(String::new());
        }

        let start = if len > 10 { len - 10 } else { 0 };
        let mut parts = Vec::with_capacity(len - start);

        for i in start..len {
            let item = messages.get_item(i)?;

            let (role, content) = if let Ok(dict) = item.downcast::<PyDict>() {
                // List[dict] с ключами "role", "content"
                let r = dict
                    .get_item("role")?
                    .map(|v| v.extract::<String>())
                    .transpose()?
                    .unwrap_or_default();
                let c = dict
                    .get_item("content")?
                    .map(|v| v.extract::<String>())
                    .transpose()?
                    .unwrap_or_default();
                (r, c)
            } else if let Ok(tup) = item.downcast::<PyTuple>() {
                // List[Tuple[str, str, str]]
                let r: String = tup.get_item(0)?.extract()?;
                let c: String = tup.get_item(1)?.extract()?;
                (r, c)
            } else {
                // Попробуем как sequence
                let r: String = item.get_item(0)?.extract()?;
                let c: String = item.get_item(1)?.extract()?;
                (r, c)
            };

            let truncated = if content.chars().count() > 100 {
                let s: String = content.chars().take(97).collect();
                format!("{}...", s)
            } else {
                content
            };
            parts.push(format!("{}: {}", role, truncated));
        }
        Ok(parts.join("\n"))
    }

    /// Извлекает ключевые предложения по наличию важных слов
    fn extract_key_points(&self, text: &str) -> Vec<String> {
        let text_lower = text.to_lowercase();

        // Разбиваем на предложения
        let mut sentences = Vec::new();
        let mut start = 0;
        for (i, ch) in text.char_indices() {
            if ch == '.' || ch == '!' || ch == '?' || ch == '\n' {
                let sentence = text[start..i].trim();
                if !sentence.is_empty() {
                    sentences.push((sentence, start, i));
                }
                start = i + ch.len_utf8();
            }
        }
        // Последнее предложение
        let last = text[start..].trim();
        if !last.is_empty() {
            sentences.push((last, start, text.len()));
        }

        // Оцениваем каждое предложение
        let mut scored: Vec<(&str, usize)> = sentences
            .iter()
            .filter_map(|(sentence, start_byte, end_byte)| {
                let sentence_lower = &text_lower[*start_byte..*end_byte];
                let count = self.important_ac.find_iter(sentence_lower).count();
                if count > 0 {
                    Some((*sentence, count))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.1.cmp(&a.1));
        scored.into_iter().take(3).map(|(s, _)| s.to_string()).collect()
    }

    /// Суммаризует эпизоды. Вход: [(timestamp, user_input, importance)]
    #[pyo3(signature = (episodes, max_length=500))]
    fn summarize_episodes(
        &self,
        episodes: Vec<(String, String, i32)>,
        max_length: usize,
    ) -> String {
        if episodes.is_empty() {
            return "Нет данных в памяти".to_string();
        }

        let mut sorted = episodes;
        sorted.sort_by(|a, b| b.2.cmp(&a.2));

        let mut parts = Vec::new();
        let mut current_length = 0;

        for (timestamp, user_input, _importance) in sorted.iter().take(5) {
            let date: String = timestamp.chars().take(10).collect();
            let preview: String = user_input.chars().take(50).collect();
            let snippet = format!("[{}] {}", date, preview);

            if current_length + snippet.len() > max_length {
                break;
            }

            current_length += snippet.len();
            parts.push(snippet);
        }

        parts.join("\n")
    }

    /// BPE-эвристика: ~4 chars/token EN, ~2 chars/token RU
    fn estimate_tokens(&self, text: &str) -> usize {
        let mut ascii_chars = 0usize;
        let mut non_ascii = 0usize;
        for c in text.chars() {
            if c.is_ascii() {
                ascii_chars += 1;
            } else {
                non_ascii += 1;
            }
        }
        (ascii_chars / 4) + (non_ascii / 2) + 1
    }

    /// Обрезает текст до N токенов с учётом Unicode
    fn truncate_to_tokens(&self, text: &str, max_tokens: usize) -> String {
        let estimated = self.estimate_tokens(text);
        if estimated <= max_tokens {
            return text.to_string();
        }
        let ratio = max_tokens as f64 / estimated as f64;
        let char_count = text.chars().count();
        let target_chars = (char_count as f64 * ratio) as usize;
        text.chars().take(target_chars).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens_ascii() {
        let c = ContextCompressor::new(0.3);
        // "hello world" = 11 ASCII chars → 11/4 + 0/2 + 1 = 3
        assert_eq!(c.estimate_tokens("hello world"), 3);
    }

    #[test]
    fn test_estimate_tokens_russian() {
        let c = ContextCompressor::new(0.3);
        // "привет" = 6 non-ASCII chars → 0/4 + 6/2 + 1 = 4
        assert_eq!(c.estimate_tokens("привет"), 4);
    }

    // compress_conversation тест требует Python runtime (принимает PyList),
    // поэтому тестируется через integration test с maturin

    #[test]
    fn test_extract_key_points() {
        let c = ContextCompressor::new(0.3);
        let text = "Всё хорошо. Есть важная проблема с сетью. Погода солнечная.";
        let points = c.extract_key_points(text);
        assert!(!points.is_empty());
        assert!(points[0].contains("проблема"));
    }

    #[test]
    fn test_truncate() {
        let c = ContextCompressor::new(0.3);
        let long_text = "a".repeat(1000);
        let truncated = c.truncate_to_tokens(&long_text, 10);
        assert!(truncated.len() < long_text.len());
    }
}

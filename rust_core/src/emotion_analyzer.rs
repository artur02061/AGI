//! EmotionAnalyzer — анализ эмоций через Aho-Corasick
//!
//! Преимущество над Python regex:
//! - Aho-Corasick: O(n + m) вместо O(n * p) для p паттернов
//! - Единственный проход по тексту для всех паттернов
//! - Поддержка RU + EN + emoji

use pyo3::prelude::*;
use aho_corasick::AhoCorasick;

#[pyclass(frozen)]
pub struct EmotionAnalyzer {
    positive_ac: AhoCorasick,
    negative_ac: AhoCorasick,
    curious_ac: AhoCorasick,
    positive_patterns: Vec<String>,
    negative_patterns: Vec<String>,
    curious_patterns: Vec<String>,
}

#[pymethods]
impl EmotionAnalyzer {
    #[new]
    fn new() -> Self {
        let positive: Vec<&str> = vec![
            "спасибо", "отлично", "супер", "хорошо", "круто", "молодец",
            "замечательно", "класс", "здорово", "прекрасно", "великолепно",
            "восхитительно", "браво", "ура", "обожаю", "нравится", "люблю",
            "рад", "рада", "счастлив", "доволен", "довольна", "благодарю",
            "спс", "пасиб", "awesome", "nice", "great", "thanks", "cool",
            "\u{1f44d}", "\u{1f60a}", "\u{1f603}", "\u{2764}\u{fe0f}",
            "\u{1f389}", "\u{1f4aa}", "\u{1f525}",
        ];
        let negative: Vec<&str> = vec![
            "не работает", "ошибка", "плохо", "не получается", "проблема",
            "сломал", "баг", "глючит", "тормозит", "зависает", "ужасно",
            "отстой", "бесит", "раздражает", "не понимаю", "запутал",
            "неправильно", "некорректно", "фигня", "дерьмо", "не так",
            "broken", "error", "bug", "wrong", "bad", "fail",
            "\u{1f61e}", "\u{1f621}", "\u{1f624}", "\u{1f494}",
            "\u{1f622}", "\u{1f92c}",
        ];
        let curious: Vec<&str> = vec![
            "как", "что", "почему", "зачем", "когда", "где", "кто",
            "сколько", "можно ли", "а если", "расскажи", "объясни",
            "подскажи", "помоги", "покажи", "научи", "интересно",
            "how", "what", "why", "when", "where", "who",
            "\u{1f914}", "\u{2753}", "\u{1f9d0}",
        ];

        let positive_patterns: Vec<String> = positive.iter().map(|s| s.to_string()).collect();
        let negative_patterns: Vec<String> = negative.iter().map(|s| s.to_string()).collect();
        let curious_patterns: Vec<String> = curious.iter().map(|s| s.to_string()).collect();

        // Паттерны уже в lowercase — сопоставляем с lowercase текстом
        Self {
            positive_ac: AhoCorasick::new(&positive).unwrap(),
            negative_ac: AhoCorasick::new(&negative).unwrap(),
            curious_ac: AhoCorasick::new(&curious).unwrap(),
            positive_patterns,
            negative_patterns,
            curious_patterns,
        }
    }

    fn analyze(&self, text: &str) -> String {
        let text_lower = text.to_lowercase();

        let pos_count = self.positive_ac.find_iter(&text_lower).count();
        let neg_count = self.negative_ac.find_iter(&text_lower).count();
        let mut cur_count = self.curious_ac.find_iter(&text_lower).count();

        if text.contains('?') {
            cur_count += 2;
        }

        if pos_count == 0 && neg_count == 0 && cur_count == 0 {
            return "neutral".to_string();
        }

        if neg_count > pos_count && neg_count >= cur_count {
            "negative".to_string()
        } else if pos_count > neg_count && pos_count >= cur_count {
            "positive".to_string()
        } else if cur_count > 0 {
            "curious".to_string()
        } else {
            "neutral".to_string()
        }
    }

    fn analyze_detailed(&self, text: &str) -> (String, f64, Vec<String>) {
        let text_lower = text.to_lowercase();

        let pos_matches: Vec<String> = self.positive_ac
            .find_iter(&text_lower)
            .map(|m| self.positive_patterns[m.pattern().as_usize()].clone())
            .collect();
        let neg_matches: Vec<String> = self.negative_ac
            .find_iter(&text_lower)
            .map(|m| self.negative_patterns[m.pattern().as_usize()].clone())
            .collect();
        let cur_matches: Vec<String> = self.curious_ac
            .find_iter(&text_lower)
            .map(|m| self.curious_patterns[m.pattern().as_usize()].clone())
            .collect();

        let total = pos_matches.len() + neg_matches.len() + cur_matches.len();

        if total == 0 {
            return ("neutral".to_string(), 0.5, vec![]);
        }

        let emotion = self.analyze(text);
        let dominant_count = match emotion.as_str() {
            "positive" => pos_matches.len(),
            "negative" => neg_matches.len(),
            "curious" => cur_matches.len(),
            _ => 0,
        };

        let confidence = if total > 0 {
            (dominant_count as f64 / total as f64).min(1.0)
        } else {
            0.5
        };

        let mut all_matches = Vec::with_capacity(total);
        all_matches.extend(pos_matches);
        all_matches.extend(neg_matches);
        all_matches.extend(cur_matches);

        (emotion, confidence, all_matches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positive_emotion() {
        let analyzer = EmotionAnalyzer::new();
        assert_eq!(analyzer.analyze("Спасибо, отлично!"), "positive");
    }

    #[test]
    fn test_negative_emotion() {
        let analyzer = EmotionAnalyzer::new();
        assert_eq!(analyzer.analyze("Не работает, ошибка!"), "negative");
    }

    #[test]
    fn test_curious_emotion() {
        let analyzer = EmotionAnalyzer::new();
        assert_eq!(analyzer.analyze("Как это сделать?"), "curious");
    }

    #[test]
    fn test_neutral() {
        let analyzer = EmotionAnalyzer::new();
        assert_eq!(analyzer.analyze("абвгд"), "neutral");
    }

    #[test]
    fn test_detailed() {
        let analyzer = EmotionAnalyzer::new();
        let (emotion, confidence, matches) = analyzer.analyze_detailed("Спасибо, круто!");
        assert_eq!(emotion, "positive");
        assert!(confidence > 0.0);
        assert!(!matches.is_empty());
    }
}

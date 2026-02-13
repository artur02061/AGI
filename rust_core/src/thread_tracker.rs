//! ThreadTracker — отслеживание нитей разговора
//!
//! Отслеживает текущую тему разговора, определяет связанность
//! новых сообщений через:
//! - Совпадение темы/сущностей (substring match)
//! - Контекстные маркеры (Aho-Corasick: "помнишь", "продолжим", ...)
//! - Timeout: нить закрывается после timeout_secs бездействия

use pyo3::prelude::*;
use parking_lot::RwLock;
use chrono::{Utc, DateTime};
use aho_corasick::AhoCorasick;

// ── Внутренние структуры ──

struct CurrentThread {
    topic: String,
    entities: Vec<String>,
    started: DateTime<Utc>,
    messages: Vec<ThreadMessage>,
}

struct ThreadMessage {
    user: String,
    #[allow(dead_code)]
    assistant: String,
    timestamp: DateTime<Utc>,
}

#[derive(Clone)]
struct ArchivedThread {
    topic: String,
    duration_secs: f64,
    message_count: usize,
}

// ── Контекстные индикаторы (RU) ──

const CONTEXT_INDICATORS: &[&str] = &[
    "помнишь", "как мы говорили", "в той же теме",
    "продолжим", "вернёмся к", "насчёт того",
    "по поводу", "как я говорил", "об этом же",
];

// ── PyO3 класс ──

#[pyclass(frozen)]
pub struct ThreadTracker {
    timeout_secs: i64,
    current: RwLock<Option<CurrentThread>>,
    history: RwLock<Vec<ArchivedThread>>,
    context_ac: AhoCorasick,
}

fn archive_thread(thread: CurrentThread, history: &mut Vec<ArchivedThread>) {
    let duration = (Utc::now() - thread.started).num_seconds() as f64;
    history.push(ArchivedThread {
        topic: thread.topic,
        duration_secs: duration,
        message_count: thread.messages.len(),
    });
    if history.len() > 20 {
        let excess = history.len() - 20;
        history.drain(..excess);
    }
}

#[pymethods]
impl ThreadTracker {
    #[new]
    #[pyo3(signature = (timeout_secs=600))]
    fn new(timeout_secs: i64) -> Self {
        Self {
            timeout_secs,
            current: RwLock::new(None),
            history: RwLock::new(Vec::new()),
            context_ac: AhoCorasick::new(CONTEXT_INDICATORS).unwrap(),
        }
    }

    #[pyo3(signature = (topic, entities=None))]
    fn start_thread(&self, topic: &str, entities: Option<Vec<String>>) {
        let mut current = self.current.write();
        if let Some(thread) = current.take() {
            let mut history = self.history.write();
            archive_thread(thread, &mut history);
        }
        *current = Some(CurrentThread {
            topic: topic.to_string(),
            entities: entities.unwrap_or_default(),
            started: Utc::now(),
            messages: Vec::new(),
        });
    }

    fn add_message(&self, user_input: &str, response: &str) {
        let mut current = self.current.write();
        if let Some(ref mut thread) = *current {
            thread.messages.push(ThreadMessage {
                user: user_input.to_string(),
                assistant: response.to_string(),
                timestamp: Utc::now(),
            });
        }
    }

    fn update(&self, user_input: &str, response: &str) {
        let now = Utc::now();
        let mut current = self.current.write();

        // Проверяем timeout
        if let Some(ref thread) = *current {
            if let Some(last_msg) = thread.messages.last() {
                let elapsed = (now - last_msg.timestamp).num_seconds();
                if elapsed > self.timeout_secs {
                    let thread = current.take().unwrap();
                    let mut history = self.history.write();
                    archive_thread(thread, &mut history);
                }
            }
        }

        // Создаём нить если нет
        if current.is_none() {
            *current = Some(CurrentThread {
                topic: user_input.chars().take(50).collect(),
                entities: Vec::new(),
                started: now,
                messages: Vec::new(),
            });
        }

        if let Some(ref mut thread) = *current {
            thread.messages.push(ThreadMessage {
                user: user_input.to_string(),
                assistant: response.to_string(),
                timestamp: now,
            });
        }
    }

    fn is_related(&self, text: &str) -> bool {
        let current = self.current.read();
        let thread = match current.as_ref() {
            Some(t) => t,
            None => return false,
        };

        let elapsed = (Utc::now() - thread.started).num_seconds();
        if elapsed > self.timeout_secs {
            return false;
        }

        let text_lower = text.to_lowercase();

        // Проверяем совпадение темы
        if text_lower.contains(&thread.topic.to_lowercase()) {
            return true;
        }

        // Проверяем сущности
        for entity in &thread.entities {
            if text_lower.contains(&entity.to_lowercase()) {
                return true;
            }
        }

        // Проверяем контекстные маркеры
        self.context_ac.is_match(&text_lower)
    }

    fn get_context(&self) -> Option<String> {
        let current = self.current.read();
        let thread = current.as_ref()?;

        let elapsed = (Utc::now() - thread.started).num_seconds();
        if elapsed > self.timeout_secs {
            return None;
        }

        let mut parts = vec![format!("Текущая тема: {}", thread.topic)];

        if !thread.entities.is_empty() {
            let entities_str: Vec<&str> = thread.entities.iter().take(5).map(|s| s.as_str()).collect();
            parts.push(format!("Упоминается: {}", entities_str.join(", ")));
        }

        let recent_count = thread.messages.len().min(3);
        if recent_count > 0 {
            parts.push("\nПоследние сообщения:".to_string());
            let start = thread.messages.len() - recent_count;
            for msg in &thread.messages[start..] {
                let preview: String = msg.user.chars().take(60).collect();
                parts.push(format!("  Пользователь: {}", preview));
            }
        }

        Some(parts.join("\n"))
    }

    fn has_active_thread(&self) -> bool {
        let current = self.current.read();
        match current.as_ref() {
            Some(thread) => {
                let elapsed = (Utc::now() - thread.started).num_seconds();
                elapsed <= self.timeout_secs
            }
            None => false,
        }
    }

    fn get_current_topic(&self) -> Option<String> {
        let current = self.current.read();
        current.as_ref().map(|t| t.topic.clone())
    }

    #[pyo3(signature = (limit=5))]
    fn get_past_threads(&self, limit: usize) -> Vec<(String, f64, usize)> {
        let history = self.history.read();
        let start = if history.len() > limit {
            history.len() - limit
        } else {
            0
        };
        history[start..]
            .iter()
            .map(|t| (t.topic.clone(), t.duration_secs, t.message_count))
            .collect()
    }

    fn end_thread(&self) {
        let mut current = self.current.write();
        if let Some(thread) = current.take() {
            let mut history = self.history.write();
            archive_thread(thread, &mut history);
        }
    }

    fn get_stats(&self) -> HashMap<String, bool> {
        let current = self.current.read();
        let mut map = HashMap::new();
        map.insert("current_thread".to_string(), current.is_some());
        map
    }
}

use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_start_and_get_topic() {
        let tracker = ThreadTracker::new(600);
        tracker.start_thread("тестовая тема", None);
        assert_eq!(tracker.get_current_topic(), Some("тестовая тема".to_string()));
        assert!(tracker.has_active_thread());
    }

    #[test]
    fn test_is_related() {
        let tracker = ThreadTracker::new(600);
        tracker.start_thread("Rust программирование", Some(vec!["cargo".to_string()]));
        assert!(tracker.is_related("Расскажи про Rust программирование"));
        assert!(tracker.is_related("что там с cargo?"));
        assert!(tracker.is_related("помнишь, мы обсуждали?"));
    }

    #[test]
    fn test_end_thread_archives() {
        let tracker = ThreadTracker::new(600);
        tracker.start_thread("тема 1", None);
        tracker.add_message("привет", "здравствуй");
        tracker.end_thread();

        assert!(tracker.get_current_topic().is_none());
        let past = tracker.get_past_threads(5);
        assert_eq!(past.len(), 1);
        assert_eq!(past[0].0, "тема 1");
    }

    #[test]
    fn test_update_creates_thread() {
        let tracker = ThreadTracker::new(600);
        tracker.update("новое сообщение", "ответ");
        assert!(tracker.has_active_thread());
    }
}

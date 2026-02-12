//! MemoryEngine — управление памятью Кристины
//!
//! Три уровня:
//! - Working memory: текущий контекст (FIFO, ограничен по размеру)
//! - Episodic memory: история взаимодействий с keyword-индексом (xxh3)
//! - Semantic memory: факты key→value (DashMap, lock-free)
//!
//! Персистентность: JSON на диск (episodic.json, semantic.json)
//! Индексирование: xxh3 hash слов → inverted index для быстрого поиска

use pyo3::prelude::*;
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::path::PathBuf;
use chrono::{Utc, DateTime};
use xxhash_rust::xxh3::xxh3_64;

// ── Внутренние структуры ──

#[derive(Clone, Serialize, Deserialize)]
struct Episode {
    timestamp: String,
    user_input: String,
    response: String,
    emotion: String,
    importance: i32,
    keywords: Vec<String>,
}

#[derive(Clone)]
struct WorkingEntry {
    role: String,
    content: String,
    timestamp: String,
}

// ── Стоп-слова для извлечения ключевых слов (RU + EN) ──

const STOP_WORDS: &[&str] = &[
    "я", "ты", "он", "она", "мы", "вы", "они", "в", "на", "и",
    "с", "по", "для", "от", "к", "не", "что", "это", "как", "но",
    "the", "is", "are", "a", "an", "in", "on", "for", "to", "of",
];

#[inline]
fn word_hash(word: &str) -> u64 {
    xxh3_64(word.as_bytes())
}

fn extract_keywords(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|w| {
            let lower = w.to_lowercase();
            if lower.chars().count() > 3 && !STOP_WORDS.contains(&lower.as_str()) {
                Some(lower)
            } else {
                None
            }
        })
        .take(10)
        .collect()
}

// ── PyO3 класс ──

#[pyclass(frozen)]
pub struct MemoryEngine {
    dir: PathBuf,
    working_size: usize,
    max_episodic: usize,
    working: RwLock<Vec<WorkingEntry>>,
    episodic: RwLock<Vec<Episode>>,
    semantic: DashMap<String, String>,
    keyword_index: RwLock<HashMap<u64, Vec<usize>>>,
}

#[pymethods]
impl MemoryEngine {
    #[new]
    #[pyo3(signature = (memory_dir, working_size=10, max_episodic=1000))]
    fn new(memory_dir: &str, working_size: usize, max_episodic: usize) -> PyResult<Self> {
        let dir = PathBuf::from(memory_dir);
        std::fs::create_dir_all(&dir).ok();

        let engine = Self {
            dir,
            working_size,
            max_episodic,
            working: RwLock::new(Vec::new()),
            episodic: RwLock::new(Vec::new()),
            semantic: DashMap::new(),
            keyword_index: RwLock::new(HashMap::new()),
        };

        engine.load_from_disk();
        Ok(engine)
    }

    // ── Working Memory ──

    fn add_to_working(&self, role: &str, content: &str) {
        let mut working = self.working.write();
        working.push(WorkingEntry {
            role: role.to_string(),
            content: content.to_string(),
            timestamp: Utc::now().to_rfc3339(),
        });
        while working.len() > self.working_size {
            working.remove(0);
        }
    }

    fn get_working_memory(&self) -> Vec<(String, String, String)> {
        let working = self.working.read();
        working
            .iter()
            .map(|e| (e.role.clone(), e.content.clone(), e.timestamp.clone()))
            .collect()
    }

    fn clear_working(&self) {
        self.working.write().clear();
    }

    // ── Episodic Memory ──

    #[pyo3(signature = (user_input, response, emotion, importance=1))]
    fn add_episode(&self, user_input: &str, response: &str, emotion: &str, importance: i32) {
        let keywords = extract_keywords(user_input);
        let episode = Episode {
            timestamp: Utc::now().to_rfc3339(),
            user_input: user_input.to_string(),
            response: response.to_string(),
            emotion: emotion.to_string(),
            importance,
            keywords,
        };

        let mut episodic = self.episodic.write();
        let idx = episodic.len();
        episodic.push(episode);

        // Обновляем keyword index
        let combined = format!("{} {}", user_input, response);
        let mut ki = self.keyword_index.write();
        index_text(&mut ki, idx, &combined);

        // Проверяем необходимость ротации
        let needs_eviction = episodic.len() > self.max_episodic;
        drop(ki);
        drop(episodic);

        if needs_eviction {
            self.evict_episodes();
        }
    }

    #[pyo3(signature = (query, max_items=3))]
    fn get_relevant_context(&self, query: &str, max_items: usize) -> Vec<(String, String, i32)> {
        let episodic = self.episodic.read();
        let ki = self.keyword_index.read();

        let mut scores: HashMap<usize, i32> = HashMap::new();

        for word in query.split_whitespace() {
            let lower = word.to_lowercase();
            if lower.chars().count() <= 2 {
                continue;
            }
            let h = word_hash(&lower);
            if let Some(indices) = ki.get(&h) {
                for &idx in indices {
                    *scores.entry(idx).or_insert(0) += 1;
                }
            }
        }

        let mut results: Vec<(String, String, i32)> = scores
            .iter()
            .filter_map(|(&idx, &keyword_score)| {
                episodic.get(idx).map(|ep| {
                    let final_score = keyword_score * ep.importance;
                    let preview: String = ep.user_input.chars().take(80).collect();
                    (ep.timestamp.clone(), preview, final_score)
                })
            })
            .collect();

        results.sort_by(|a, b| b.2.cmp(&a.2));
        results.truncate(max_items);
        results
    }

    // ── Semantic Memory ──

    fn add_semantic(&self, key: &str, value: &str) {
        self.semantic.insert(key.to_string(), value.to_string());
    }

    fn get_semantic(&self, key: &str) -> Option<String> {
        self.semantic.get(key).map(|v| v.value().clone())
    }

    // ── Персистентность ──

    fn save(&self) {
        let episodic_path = self.dir.join("episodic.json");
        let semantic_path = self.dir.join("semantic.json");

        if let Ok(data) = serde_json::to_string_pretty(&*self.episodic.read()) {
            let _ = std::fs::write(&episodic_path, data);
        }

        let semantic_map: HashMap<String, String> = self.semantic
            .iter()
            .map(|r| (r.key().clone(), r.value().clone()))
            .collect();
        if let Ok(data) = serde_json::to_string_pretty(&semantic_map) {
            let _ = std::fs::write(&semantic_path, data);
        }
    }

    fn load(&self) {
        self.load_from_disk();
    }

    fn get_stats(&self) -> (usize, usize, usize) {
        (
            self.working.read().len(),
            self.episodic.read().len(),
            self.semantic.len(),
        )
    }
}

// ── Приватные методы ──

impl MemoryEngine {
    fn load_from_disk(&self) {
        let episodic_path = self.dir.join("episodic.json");
        if episodic_path.exists() {
            if let Ok(data) = std::fs::read_to_string(&episodic_path) {
                if let Ok(episodes) = serde_json::from_str::<Vec<Episode>>(&data) {
                    let mut ep = self.episodic.write();
                    *ep = episodes;
                    let mut ki = self.keyword_index.write();
                    rebuild_index(&mut ki, &ep);
                }
            }
        }

        let semantic_path = self.dir.join("semantic.json");
        if semantic_path.exists() {
            if let Ok(data) = std::fs::read_to_string(&semantic_path) {
                if let Ok(map) = serde_json::from_str::<HashMap<String, String>>(&data) {
                    self.semantic.clear();
                    for (k, v) in map {
                        self.semantic.insert(k, v);
                    }
                }
            }
        }
    }

    fn evict_episodes(&self) {
        let mut episodic = self.episodic.write();
        let remove_count = std::cmp::max(1, self.max_episodic / 10);
        let now = Utc::now();

        let mut scored: Vec<(usize, f64)> = episodic
            .iter()
            .enumerate()
            .map(|(i, ep)| {
                let age_hours = ep
                    .timestamp
                    .parse::<DateTime<Utc>>()
                    .map(|ts| (now - ts).num_seconds() as f64 / 3600.0)
                    .unwrap_or(1.0)
                    .max(1.0);
                (i, ep.importance as f64 / age_hours)
            })
            .collect();

        // Сортируем по eviction score (наименее ценные первыми)
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut to_remove: Vec<usize> = scored
            .iter()
            .take(remove_count)
            .map(|&(i, _)| i)
            .collect();
        to_remove.sort_unstable_by(|a, b| b.cmp(a)); // Обратный порядок для безопасного удаления

        for idx in to_remove {
            if idx < episodic.len() {
                episodic.remove(idx);
            }
        }

        let mut ki = self.keyword_index.write();
        rebuild_index(&mut ki, &episodic);
    }
}

// ── Standalone helpers ──

fn index_text(ki: &mut HashMap<u64, Vec<usize>>, idx: usize, text: &str) {
    for word in text.split_whitespace() {
        let lower = word.to_lowercase();
        if lower.chars().count() > 2 {
            let h = word_hash(&lower);
            ki.entry(h).or_default().push(idx);
        }
    }
}

fn rebuild_index(ki: &mut HashMap<u64, Vec<usize>>, episodes: &[Episode]) {
    ki.clear();
    for (i, ep) in episodes.iter().enumerate() {
        let combined = format!("{} {}", ep.user_input, ep.response);
        index_text(ki, i, &combined);
    }
}

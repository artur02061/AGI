//! EmbeddingCache — lock-free кэш эмбеддингов
//!
//! Оптимизации vs Python fallback:
//! - DashMap: конкурентный доступ без GIL
//! - AtomicU64: lock-free счётчики hits/misses
//! - xxh3: ~10x быстрее md5 для хэширования текста
//! - LRU eviction: удаляет 10% наименее используемых

use pyo3::prelude::*;
use dashmap::DashMap;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use xxhash_rust::xxh3::xxh3_64;

#[inline]
fn text_hash(text: &str) -> String {
    format!("{:016x}", xxh3_64(text.as_bytes()))
}

#[pyclass(frozen)]
pub struct EmbeddingCache {
    cache: DashMap<String, Vec<f32>>,
    access_count: DashMap<String, u64>,
    max_size: usize,
    cache_path: PathBuf,
    hits: AtomicU64,
    misses: AtomicU64,
}

#[pymethods]
impl EmbeddingCache {
    #[new]
    #[pyo3(signature = (cache_dir, max_size=10000))]
    fn new(cache_dir: &str, max_size: usize) -> PyResult<Self> {
        let dir = PathBuf::from(cache_dir);
        std::fs::create_dir_all(&dir).ok();

        let cache = Self {
            cache: DashMap::new(),
            access_count: DashMap::new(),
            max_size,
            cache_path: dir.join("embedding_cache.json"),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        };

        cache.load_from_disk();
        Ok(cache)
    }

    fn get(&self, text: &str) -> Option<Vec<f32>> {
        let h = text_hash(text);
        if let Some(entry) = self.cache.get(&h) {
            self.access_count
                .entry(h)
                .and_modify(|c| *c += 1)
                .or_insert(1);
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(entry.value().clone())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    fn put(&self, text: &str, embedding: Vec<f32>) {
        let h = text_hash(text);
        if self.cache.len() >= self.max_size {
            self.evict_lru();
        }
        self.cache.insert(h.clone(), embedding);
        self.access_count.insert(h, 1);
    }

    fn contains(&self, text: &str) -> bool {
        let h = text_hash(text);
        self.cache.contains_key(&h)
    }

    #[pyo3(name = "len")]
    fn py_len(&self) -> usize {
        self.cache.len()
    }

    fn get_stats(&self) -> (usize, u64, u64) {
        (
            self.cache.len(),
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
        )
    }

    fn save(&self) {
        let map: HashMap<String, Vec<f32>> = self.cache
            .iter()
            .map(|r| (r.key().clone(), r.value().clone()))
            .collect();
        if let Ok(data) = serde_json::to_string(&map) {
            let _ = std::fs::write(&self.cache_path, data);
        }
    }

    fn clear(&self) {
        self.cache.clear();
        self.access_count.clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }
}

impl EmbeddingCache {
    fn load_from_disk(&self) {
        if !self.cache_path.exists() {
            return;
        }
        if let Ok(data) = std::fs::read_to_string(&self.cache_path) {
            if let Ok(map) = serde_json::from_str::<HashMap<String, Vec<f32>>>(&data) {
                for (k, v) in map {
                    self.cache.insert(k.clone(), v);
                    self.access_count.insert(k, 0);
                }
            }
        }
    }

    fn evict_lru(&self) {
        let evict_count = std::cmp::max(1, self.max_size / 10);
        let mut entries: Vec<(String, u64)> = self.access_count
            .iter()
            .map(|r| (r.key().clone(), *r.value()))
            .collect();
        entries.sort_by_key(|(_, count)| *count);

        for (key, _) in entries.into_iter().take(evict_count) {
            self.cache.remove(&key);
            self.access_count.remove(&key);
        }
    }
}

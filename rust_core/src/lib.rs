//! Кристина 6.0 — Высокопроизводительное Rust-ядро
//!
//! PyO3 модуль, предоставляющий:
//! - MemoryEngine: управление памятью (working/episodic/semantic)
//! - EmbeddingCache: lock-free кэш эмбеддингов
//! - EmotionAnalyzer: Aho-Corasick анализ эмоций
//! - ToolCallParser: парсер вызовов инструментов
//! - ContextCompressor: сжатие контекста
//! - ThreadTracker: отслеживание нитей разговора
//! - cosine_similarity / batch_cosine_similarity: векторные операции

use pyo3::prelude::*;

mod similarity;
mod memory_engine;
mod embedding_cache;
mod emotion_analyzer;
mod tool_parser;
mod context_compressor;
mod thread_tracker;

#[pymodule]
fn kristina_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<memory_engine::MemoryEngine>()?;
    m.add_class::<embedding_cache::EmbeddingCache>()?;
    m.add_class::<emotion_analyzer::EmotionAnalyzer>()?;
    m.add_class::<tool_parser::ToolCallParser>()?;
    m.add_class::<context_compressor::ContextCompressor>()?;
    m.add_class::<thread_tracker::ThreadTracker>()?;
    m.add_function(wrap_pyfunction!(similarity::cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(similarity::batch_cosine_similarity, m)?)?;
    Ok(())
}

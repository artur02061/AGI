//! Векторные операции: cosine similarity + batch с Rayon параллелизмом
//!
//! Оптимизации:
//! - f64 аккумуляторы для точности при f32 входах
//! - SIMD-friendly tight loops (auto-vectorization при opt-level=3)
//! - Rayon параллелизм для batch > 32 документов
//! - select_nth_unstable для O(n) partial sort вместо O(n log n)
//! - GIL release во время вычислений

use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::Ordering;

fn partial_cmp_f32_desc(a: &f32, b: &f32) -> Ordering {
    b.partial_cmp(a).unwrap_or(Ordering::Equal)
}

/// Косинусное сходство двух векторов.
/// Возвращает 0.0 при несовпадении размерностей или нулевых нормах.
#[pyfunction]
pub fn cosine_similarity(py: Python<'_>, a: Vec<f32>, b: Vec<f32>) -> f32 {
    py.allow_threads(|| cosine_similarity_impl(&a, &b))
}

#[inline]
fn cosine_similarity_impl(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for i in 0..a.len() {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-8 {
        0.0
    } else {
        (dot / denom) as f32
    }
}

/// Batch cosine similarity: query vs N документов.
/// Возвращает top_k пар (index, similarity), отсортированных по убыванию.
/// Использует Rayon для параллелизма при > 32 документах.
#[pyfunction]
#[pyo3(signature = (query, documents, top_k=5))]
pub fn batch_cosine_similarity(
    py: Python<'_>,
    query: Vec<f32>,
    documents: Vec<Vec<f32>>,
    top_k: usize,
) -> Vec<(usize, f32)> {
    py.allow_threads(|| batch_cosine_impl(&query, &documents, top_k))
}

fn batch_cosine_impl(query: &[f32], documents: &[Vec<f32>], top_k: usize) -> Vec<(usize, f32)> {
    if query.is_empty() || documents.is_empty() {
        return vec![];
    }

    // Предвычисляем норму query один раз
    let q_norm: f64 = query.iter().map(|x| (*x as f64) * (*x as f64)).sum();
    let q_norm = q_norm.sqrt();
    if q_norm < 1e-8 {
        return vec![];
    }

    let compute_sim = |(i, doc): (usize, &Vec<f32>)| -> Option<(usize, f32)> {
        if doc.len() != query.len() {
            return None;
        }
        let mut dot = 0.0f64;
        let mut d_norm = 0.0f64;
        for j in 0..query.len() {
            let q = query[j] as f64;
            let d = doc[j] as f64;
            dot += q * d;
            d_norm += d * d;
        }
        let d_norm = d_norm.sqrt();
        if d_norm < 1e-8 {
            return None;
        }
        Some((i, (dot / (q_norm * d_norm)) as f32))
    };

    const PARALLEL_THRESHOLD: usize = 32;

    let mut results: Vec<(usize, f32)> = if documents.len() >= PARALLEL_THRESHOLD {
        documents
            .par_iter()
            .enumerate()
            .filter_map(compute_sim)
            .collect()
    } else {
        documents
            .iter()
            .enumerate()
            .filter_map(compute_sim)
            .collect()
    };

    // Partial sort: O(n) вместо O(n log n) для top-k
    if results.len() > top_k {
        results.select_nth_unstable_by(top_k, |a, b| partial_cmp_f32_desc(&a.1, &b.1));
        results.truncate(top_k);
    }
    results.sort_by(|a, b| partial_cmp_f32_desc(&a.1, &b.1));
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity_impl(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity_impl(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_batch_top_k() {
        let query = vec![1.0, 0.0, 0.0];
        let docs = vec![
            vec![1.0, 0.0, 0.0], // sim = 1.0
            vec![0.0, 1.0, 0.0], // sim = 0.0
            vec![0.5, 0.5, 0.0], // sim ~= 0.707
        ];
        let results = batch_cosine_impl(&query, &docs, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // index 0 first (highest sim)
    }
}

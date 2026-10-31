//! Байтовый BPE-токенайзер — тот же алгоритм, что в `microllm/tokenizer.py`, но на Rust.
//!
//! Главное требование: **результат совпадает с Python байт в байт** (те же слияния в том же порядке).
//! Поэтому при равных счётчиках выбираем пару, которая раньше встретилась в тексте — как это делает
//! `Counter.most_common(1)` в Python.

use regex::Regex;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

type Pair = (u32, u32);

/// Предварительное разбиение на «слова» — то же регулярное выражение, что в Python.
fn pretoken_regex() -> Regex {
    Regex::new(r"\s?\w+|\s?[^\w\s]+|\s+").unwrap()
}

fn merge_ids(ids: &[u32], pair: Pair, new_id: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && (ids[i], ids[i + 1]) == pair {
            out.push(new_id);
            i += 2;
        } else {
            out.push(ids[i]);
            i += 1;
        }
    }
    out
}

#[wasm_bindgen]
pub struct Tokenizer {
    merges: Vec<Pair>,
    rank: HashMap<Pair, u32>,
    vocab: Vec<Vec<u8>>,
    cache: HashMap<String, Vec<u32>>,
    pretoken: Regex,
}

impl Tokenizer {
    fn from_pairs(merges: Vec<Pair>) -> Tokenizer {
        let mut vocab: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        let mut rank = HashMap::with_capacity(merges.len());
        for (i, &(a, b)) in merges.iter().enumerate() {
            let mut bytes = vocab[a as usize].clone();
            bytes.extend_from_slice(&vocab[b as usize]);
            vocab.push(bytes);
            rank.insert((a, b), i as u32);
        }
        Tokenizer { merges, rank, vocab, cache: HashMap::new(), pretoken: pretoken_regex() }
    }

    fn encode_word(&mut self, word: &str) -> Vec<u32> {
        if let Some(hit) = self.cache.get(word) {
            return hit.clone();
        }
        let mut ids: Vec<u32> = word.bytes().map(u32::from).collect();
        while ids.len() >= 2 {
            // слияние с наименьшим номером — в том порядке, в каком училось
            let best = ids
                .windows(2)
                .filter_map(|w| self.rank.get(&(w[0], w[1])).map(|&r| (r, (w[0], w[1]))))
                .min_by_key(|&(r, _)| r);
            match best {
                Some((r, pair)) => ids = merge_ids(&ids, pair, 256 + r),
                None => break,
            }
        }
        self.cache.insert(word.to_string(), ids.clone());
        ids
    }
}

#[wasm_bindgen]
impl Tokenizer {
    /// Обучение: ищем самую частую пару, склеиваем, повторяем. Считаем по уникальным словам с частотами.
    pub fn train(text: &str, vocab_size: usize) -> Result<Tokenizer, JsError> {
        if vocab_size < 256 {
            return Err(JsError::new("vocab_size должен быть не меньше 256 (по токену на байт)"));
        }
        let re = pretoken_regex();
        // слова в порядке первого появления + частоты
        let mut index: HashMap<&str, usize> = HashMap::new();
        let mut freq: Vec<u64> = Vec::new();
        let mut pieces: Vec<Vec<u32>> = Vec::new();
        for m in re.find_iter(text) {
            let w = m.as_str();
            match index.get(w) {
                Some(&i) => freq[i] += 1,
                None => {
                    index.insert(w, pieces.len());
                    freq.push(1);
                    pieces.push(w.bytes().map(u32::from).collect());
                }
            }
        }

        let mut merges: Vec<Pair> = Vec::new();
        for _ in 0..(vocab_size - 256) {
            // (счётчик, порядок первого появления пары)
            let mut stats: HashMap<Pair, (u64, usize)> = HashMap::new();
            let mut order = 0usize;
            for (ids, &f) in pieces.iter().zip(&freq) {
                for w in ids.windows(2) {
                    let e = stats.entry((w[0], w[1])).or_insert_with(|| {
                        order += 1;
                        (0, order)
                    });
                    e.0 += f;
                }
            }
            // максимум по счётчику; при равенстве — пара, появившаяся раньше (как в Python)
            let best = stats
                .iter()
                .max_by(|a, b| a.1 .0.cmp(&b.1 .0).then(b.1 .1.cmp(&a.1 .1)))
                .map(|(&p, &(c, _))| (p, c));
            let (pair, count) = match best {
                Some(x) => x,
                None => break,
            };
            if count < 2 {
                break; // пары больше не повторяются
            }
            let new_id = 256 + merges.len() as u32;
            merges.push(pair);
            for ids in pieces.iter_mut() {
                if ids.len() >= 2 {
                    *ids = merge_ids(ids, pair, new_id);
                }
            }
        }
        Ok(Tokenizer::from_pairs(merges))
    }

    /// Загрузить из JSON того же формата, что пишет Python: `{"kind":"bpe","merges":[[a,b],…]}`.
    pub fn from_json(json: &str) -> Result<Tokenizer, JsError> {
        let v: serde_json::Value = serde_json::from_str(json).map_err(|e| JsError::new(&e.to_string()))?;
        let arr = v["merges"].as_array().ok_or_else(|| JsError::new("нет поля merges"))?;
        let mut merges = Vec::with_capacity(arr.len());
        for m in arr {
            let a = m[0].as_u64().ok_or_else(|| JsError::new("плохая пара"))? as u32;
            let b = m[1].as_u64().ok_or_else(|| JsError::new("плохая пара"))? as u32;
            merges.push((a, b));
        }
        Ok(Tokenizer::from_pairs(merges))
    }

    pub fn to_json(&self) -> String {
        let pairs: Vec<[u32; 2]> = self.merges.iter().map(|&(a, b)| [a, b]).collect();
        serde_json::json!({ "kind": "bpe", "merges": pairs }).to_string()
    }

    pub fn encode(&mut self, text: &str) -> Vec<u32> {
        let words: Vec<String> = self.pretoken.find_iter(text).map(|m| m.as_str().to_string()).collect();
        let mut out = Vec::new();
        for w in words {
            out.extend(self.encode_word(&w));
        }
        out
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &i in ids {
            if let Some(b) = self.vocab.get(i as usize) {
                bytes.extend_from_slice(b);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn vocab_size(&self) -> usize {
        256 + self.merges.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_any_language() {
        let text = "привет мир, hello world! 你好 🙂 привет привет привет";
        let mut t = Tokenizer::train(text, 300).ok().unwrap();
        let ids = t.encode(text);
        assert_eq!(t.decode(&ids), text);
        assert!(ids.len() < text.len());
    }

    #[test]
    fn json_roundtrip() {
        let t = Tokenizer::train("абвабвабвабв abab abab", 280).ok().unwrap();
        let t2 = Tokenizer::from_json(&t.to_json()).ok().unwrap();
        assert_eq!(t.merges, t2.merges);
    }
}

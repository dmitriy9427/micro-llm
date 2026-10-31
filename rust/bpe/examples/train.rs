//! `cargo run --release --example train -- corpus.txt 1024 out.json` — обучить и сохранить токенайзер.
use microllm_bpe::Tokenizer;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&args[1]).expect("не прочитал корпус");
    let vocab: usize = args[2].parse().unwrap();
    let t0 = Instant::now();
    let tok = Tokenizer::train(&text, vocab).ok().unwrap();
    eprintln!("обучение: {:.2} с, словарь {}", t0.elapsed().as_secs_f64(), tok.vocab_size());
    std::fs::write(&args[3], tok.to_json()).unwrap();
}

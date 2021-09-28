use std::cmp::Ordering;
use mako::{tokenization::Tokenizer, vocab::Vocab};
use serde::{Deserialize};
use tch::Tensor;

#[derive(Clone)]
pub struct LoadingState<T: Tokenizer>{
    pub vocab: Vocab,
    pub tokenizer: T
}

#[derive(Deserialize)]
pub struct QAExample {
    question: String,
    context: Vec<String>,
    answer_index: i32,
    contains_answer: bool,
}

#[derive(Clone, Debug)]
pub struct ProcessedQAExample {
    pub(crate) question: Vec<u32>,
    pub(crate) context: Vec<Vec<u32>>,
    pub(crate) answer: i32,
}

pub fn loading_function<T: Tokenizer>(string: &str, state: Option<&LoadingState<T>>) -> ProcessedQAExample {
    let qa_example: QAExample = serde_json::from_str(string).expect("Failed to load example");

    ProcessedQAExample {
        question: {
            let tokens = state.unwrap().tokenizer.tokenize(&qa_example.question.replace("\\", "").replace('"', ""));
            state.unwrap().vocab.indexes_from_tokens(&tokens).unwrap()
        },
        context: {
            let tokens = state.unwrap().tokenizer.batch_tokenize(qa_example.context.iter().map(|s| {s.replace("\\", "").replace('"', "")}).collect());
            state.unwrap().vocab.batch_indexes_from_tokens(&tokens).unwrap()
        },
        answer: if qa_example.contains_answer {qa_example.answer_index} else {-1}
    }
}

pub fn sorting_function(example1: &ProcessedQAExample, example2: &ProcessedQAExample) -> Ordering {
    let example1_len = example1.context.iter().map(|f| {f.len()}).max().unwrap();
    let example2_len = example2.context.iter().map(|f| {f.len()}).max().unwrap();
    example1_len.cmp(&example2_len)
}

pub fn filter_function(example: &ProcessedQAExample) -> bool {
    example.answer >= 0
}

pub fn collate_function(batch: Vec<ProcessedQAExample>) -> (Tensor, Tensor, Tensor) {
    fn convert_to_i32(v: Vec<u32>) -> Vec<i32> {
        v.iter().map(|e| {*e as i32}).collect()
    }

    // Make questions tensor
    let question = {
        let max_size = batch.iter().map(|t| {t.question.len()}).max().unwrap();
        let padded_vec: Vec<Vec<i32>> = batch.iter().map(|t| {
            let mut question = t.question.clone();
            question.extend(vec![0; max_size - t.question.len()]); 
            convert_to_i32(question)
        }).collect();
        Tensor::of_slice2(&padded_vec)
    };

    // Make context tensor
    let context = {
        let context_sentences = batch.iter().map(|t| {t.context.len()}).max().unwrap();
        let sentence_len = batch.iter().map(|t| {t.context.iter().map(|l| {l.len()}).max().unwrap()}).max().unwrap();
        let padded_vec: Vec<Vec<Vec<i32>>> = batch.iter().map(|example| {
            // Pad existing sentences
            let mut sentences = example.context.iter().map(|sentence| {
                let mut sentence = sentence.clone();
                sentence.extend(vec![0; sentence_len - sentence.len()]);
                convert_to_i32(sentence)
            }).collect::<Vec<Vec<i32>>>();
            // Add more context sentences
            sentences.extend(vec![vec![0; sentence_len]; context_sentences - example.context.len()]);
            sentences
        }).collect();
        let unstacked: Vec<Tensor> = padded_vec.iter().map(|v| {Tensor::of_slice2(v)}).collect();
        Tensor::stack(&unstacked, 0)
    };

    // Make answer tensor
    let answer = {
        let answers: Vec<i32> = batch.iter().map(|e| {e.answer}).collect();
        Tensor::of_slice(&answers)
    };

    (question, context, answer)
}
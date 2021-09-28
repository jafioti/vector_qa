use condor::modules::{NNModule, TransformerAggregator};
use hypernonsense::multiindex::MultiIndex;
use mako::tokenization::Tokenizer;
use mako::vocab::load_wordpiece_vocab;
use rand::thread_rng;
use tch::{Device, IndexOp, Kind, nn};
use text_io::read;

use crate::{EMBED_SIZE, HEADS, LAYERS, AGGREGATION_SIZE, DROPOUT};

use super::model::QAModel;
use super::data::{collate_function, ProcessedQAExample};

pub fn eval() {
    let mut vs = nn::VarStore::new(Device::cuda_if_available());
    let local_vocab = load_wordpiece_vocab();
    let mut model = QAModel::from_encoders(
        TransformerAggregator::new(&(&vs.root() / "question"), EMBED_SIZE, HEADS, LAYERS, AGGREGATION_SIZE, local_vocab.num_tokens as i64, 1000, DROPOUT), 
        TransformerAggregator::new(&(&vs.root() / "context"), EMBED_SIZE, HEADS, LAYERS, AGGREGATION_SIZE, local_vocab.num_tokens as i64, 1000, DROPOUT),
        DROPOUT);
    vs.load("model64acc.pt").expect("Failed to load model!");
    let context = vec!["We open at 12 AM and close at 5 PM.", "We serve pizza in small, medium and large sizes.", "We are located at 125 Harveys Lake Dr.", "The Panzer IV was a feared main battle tank in the steppe of the Soviet Union at the tail end of WW2."];
    loop {
        println!("Input:");
        // Get input
        let input: String = read!("{}\n");
        println!("Top Result: {}", eval_example(&mut model, &input, context.clone()));
    }
}

pub fn eval_example(model: &mut QAModel, question: &str, context: Vec<&str>) -> String {
    model.eval();
    let context_sentences = context.clone();
    // Vectorize question and context
    let tokenizer = mako::tokenization::WordpieceTokenizer::load();
    let vocab = mako::vocab::load_wordpiece_vocab();
    let example = ProcessedQAExample {
        question: {
            let tokens = tokenizer.tokenize(&question.replace("\\", "").replace('"', ""));
            vocab.indexes_from_tokens(&tokens).unwrap()
        },
        context: {
            let tokens = tokenizer.batch_tokenize(context.iter().map(|s| {s.replace("\\", "").replace('"', "")}).collect());
            vocab.batch_indexes_from_tokens(&tokens).unwrap()
        },
        answer: 0
    };
    // Create tensor
    let (mut question, mut context, _) = collate_function(vec![example.clone()]);
    question = question.to_device(Device::cuda_if_available());
    context = context.to_device(Device::cuda_if_available());

    // Feed through model
    let encoded_question = model.context_encoder.forward(&question);
    let (batch_size, context_sentences_size, context_len) = context.size3().unwrap();
    let encoded_context = model.context_encoder.forward(&context.reshape(&[batch_size * context_sentences_size, context_len]));
    let encoded_context = encoded_context.reshape(&[batch_size, context_sentences_size, encoded_context.size()[1]]);
    let mut dot_prod = encoded_context.matmul(&encoded_question.unsqueeze(-1)).squeeze_dim(-1);

    // Make index
    let mut multi_index: MultiIndex<usize> = MultiIndex::new(512, example.context.len() as u8, example.context.len() as u8, &mut thread_rng());
    for i in 0..context_sentences_size {
        multi_index.add(i as usize, &Vec::<f32>::from(encoded_context.i(0).i(i)));
    }
    let index_output = multi_index.nearest(&Vec::<f32>::from(encoded_question.i(0)), context_sentences_size as usize, |point, key| {
        let mut sum = 0.;
        let vector = &Vec::<f32>::from(encoded_context.i(0).i(*key as i64));
        #[allow(clippy::needless_range_loop)]
        for i in 0..point.len() {
            sum += point[i] * vector[i];
        }
        -sum
    });
    println!("Index output: {:?}", index_output.iter().map(|f|{f.key}).collect::<Vec<usize>>());
    println!("Output: {:?}", Vec::<f32>::from(dot_prod.i(0)));
    dot_prod = dot_prod.argmax(-1, false).to_kind(Kind::Int);
    let index = Vec::<i32>::from(dot_prod)[0];
    context_sentences[index as usize].to_string()
}
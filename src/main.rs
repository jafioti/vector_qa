mod model;
mod data;
mod eval;

use condor::{modules::{LanguageModel, ModuleCopy, TransformerAggregator}, utils::{DecayingOptimizer, ExponentialAverage, count_parameters, readable_number, test_progress_bar, train_progress_bar}};
use model::QAModel;
use data::{LoadingState, ProcessedQAExample, loading_function, sorting_function, filter_function, collate_function};
use eval::eval;
use mako::{dataloader::ThreadedDataloader, tokenization::{self, Tokenizer}, vocab::load_wordpiece_vocab};
use tch::{Device, Kind, nn::{self, AdamW, OptimizerConfig}};
use clap::Clap;

use crate::eval::eval_example;

const BATCH_SIZE: usize = 4;
const BATCH_AGGREGATIONS: usize = 5;
const LEARNING_RATE: f64 = 0.0001;
const LR_DECAY: f64 = 1.;
const EPOCHS: usize = 50;
const LAYERS: i64 = 6;
const HEADS: i64 = 16;
const EMBED_SIZE: i64 = 512;
const AGGREGATION_SIZE: i64 = 512;
const DROPOUT: f64 = 0.2;

fn main() {
    println!("|VECTOR QUESTION ANSWERING MODEL|");
    let args = Args::parse();
    if args.eval {
        eval();
    } else {
        train(args);
    }
}

fn train(args: Args) {
    // Create dataset
    println!("Builing Dataset...");
    let mut train_dataset = ThreadedDataloader::new(
        &["/home/jafioti/Datasets/Squad2.0/processed_train.txt"],
        BATCH_SIZE,
        None,
        Some(10000),
        10000,
        Some(LoadingState{
            tokenizer: tokenization::WordpieceTokenizer::load(),
            vocab: load_wordpiece_vocab()
        }),
        loading_function,
        Some(sorting_function),
        Some(filter_function),
    );

    let mut test_dataset = ThreadedDataloader::new(
        &["/home/jafioti/Datasets/Squad2.0/processed_dev.txt"],
        BATCH_SIZE,
        None,
        None,
        10000,
        Some(LoadingState{
            tokenizer: tokenization::WordpieceTokenizer::load(),
            vocab: load_wordpiece_vocab()
        }),
        loading_function,
        Some(sorting_function),
        Some(filter_function),
    );
    println!("Train Dataset: {} examples", train_dataset.len());
    println!("Test Dataset: {} examples", test_dataset.len());

    println!("Building Model...");
    let local_vocab = load_wordpiece_vocab();
    let device = Device::cuda_if_available();
    // Build main model
    let mut vs = nn::VarStore::new(device);
    let mut model = QAModel::from_encoders(
        TransformerAggregator::new(&(&vs.root() / "question"), EMBED_SIZE, HEADS, LAYERS, AGGREGATION_SIZE, local_vocab.num_tokens as i64, 1000, DROPOUT), 
        TransformerAggregator::new(&(&vs.root() / "context"), EMBED_SIZE, HEADS, LAYERS, AGGREGATION_SIZE, local_vocab.num_tokens as i64, 1000, DROPOUT),
        DROPOUT);
    // Build and load language model
    if args.use_lm {
        let mut lm_vs = nn::VarStore::new(device);
        let language_model = LanguageModel::new(&lm_vs.root(), EMBED_SIZE, HEADS, LAYERS, local_vocab.num_tokens as i64, 1000, DROPOUT);
        lm_vs.load("lm.pt").expect("Failed to load language model!");
        // Copy over langauge model
        model.context_encoder.encoder.copy(&language_model.transformer).expect("Failed to copy Context Encoder parameters!");
        model.question_encoder.encoder.copy(&language_model.transformer).expect("Failed to copy Question Encoder parameters!");
        drop(language_model);
        drop(lm_vs);
    }
    // Load model
    if args.load {
        vs.load("model.pt").expect("Failed to load model!");
    }
    let mut opt = DecayingOptimizer::new(
        nn::AdamW::default().build(&(vs), LEARNING_RATE).expect("Failed to build optimizer"),
        LEARNING_RATE,
        LR_DECAY);
    println!("Model Parameters: {}", readable_number(count_parameters(&vs) as i64));

    // Training
    let mut best_loss = f64::MAX;
    for epoch in 0..EPOCHS {
        println!("\n\nEpoch {}", epoch + 1);
        println!("Training...");
        let (train_loss, train_acc) = train_epoch(&mut model, &mut train_dataset, &mut opt);
        println!("Train Loss: {} Train Acc: {}", train_loss, train_acc);
        println!("Testing...");
        let (test_loss, test_acc) = test_epoch(&mut model, &mut test_dataset);
        println!("Test Loss: {} Test Acc: {}", test_loss, test_acc);
        opt.step_lr();
        println!("Learning Rate: {}", opt.get_lr());
        
        // Save model
        if test_loss < best_loss {
            println!("Saving...");
            vs.save("model.pt").expect("Failed to save model");
            best_loss = test_loss;
        }
    }

    // Run eval
    println!("Eval: {}", eval_example(&mut model, "Location", 
        vec!["We are located at 143 Harvey's Lake Drive", 
        "We serve pizza in small, medium and large sizes. Our toppings include carrots, pepperoni, spinach, and broccoli.", 
        "We are open 10 AM to 3 PM on weekdays. We are open 8 AM to 10 PM on weekends."]));
}

fn train_epoch<T: Tokenizer + Send + Sync>(model: &mut QAModel, dataset: &mut ThreadedDataloader<ProcessedQAExample, LoadingState<T>>, optimizer: &mut DecayingOptimizer<AdamW>) -> (f64, f64) {
    model.train();
    let bar = train_progress_bar((dataset.len() / BATCH_SIZE) as u64);
    let mut loss_avg = ExponentialAverage::new();
    let mut acc_avg = ExponentialAverage::new();
    for (i, batch) in dataset.enumerate() {
        // Process batch into tensors
        let (mut question, mut context, mut answer) = collate_function(batch);
        question = question.to_device(Device::cuda_if_available());
        context = context.to_device(Device::cuda_if_available());
        answer = answer.to_device(Device::cuda_if_available());
        // Run through model
        let output = model.forward_qc(&question, &context);

        // Get loss and backprop
        let loss = output.log_softmax(1, Kind::Float).nll_loss(&answer.to_kind(Kind::Int64));
        loss.backward();
        loss_avg.update(f64::from(loss));
        if i % BATCH_AGGREGATIONS == 0 {
            optimizer.step();
            optimizer.zero_grad();
        }

        // Get accuracy
        acc_avg.update(f64::from(output.accuracy_for_logits(&answer.to_kind(Kind::Int64))));

        bar.inc(1);
        bar.set_message(format!("Loss: {:.2}", loss_avg.value));
    }
    (loss_avg.value, acc_avg.value)
}

fn test_epoch<T: Tokenizer + Send + Sync>(model: &mut QAModel, dataset: &mut ThreadedDataloader<ProcessedQAExample, LoadingState<T>>) -> (f64, f64) {
    model.eval();
    let mut losses = vec![];
    let mut accuracies: Vec<f64> = vec![];
    let bar = test_progress_bar((dataset.len() / BATCH_SIZE) as u64);
    for batch in dataset {
        // Process batch into tensors
        let (mut question, mut context, mut answer) = collate_function(batch);
        question = question.to_device(Device::cuda_if_available());
        context = context.to_device(Device::cuda_if_available());
        answer = answer.to_device(Device::cuda_if_available());
        // Run through model
        let output = model.forward_qc(&question, &context);

        // Get loss
        let loss = output.cross_entropy_for_logits(&answer.to_kind(Kind::Int64));
        let loss_num = f64::from(loss);
        losses.push(loss_num);

        // Get accuracy
        accuracies.push(f64::from(output.accuracy_for_logits(&answer.to_kind(Kind::Int64))));
        bar.inc(1);
    }
    (losses.iter().sum::<f64>() / losses.len() as f64, accuracies.iter().sum::<f64>() / accuracies.len() as f64)
}

/// Vector QA Model Training
#[derive(Clap, Debug)]
#[clap(name = "Vector QA Model Training")]
struct Args {
    /// Whether or not to attempt to load a pretrained language model as each encoder
    #[clap(short, long)]
    use_lm: bool,

    /// Whether or not to attempt to load the model before training
    #[clap(short, long)]
    load: bool,

    /// Whether or not to only run eval
    #[clap(short, long)]
    eval: bool,
}
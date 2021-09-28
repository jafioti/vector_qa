use tch::{IndexOp, Kind, Tensor, nn};
use condor::modules::{TransformerAggregator, NNModule};

pub struct QAModel {
    pub question_encoder: TransformerAggregator,
    pub context_encoder: TransformerAggregator,
    dropout: f64,
    train: bool
}

impl QAModel {
    pub fn from_encoders(question_encoder: TransformerAggregator, context_encoder: TransformerAggregator, dropout: f64) -> Self {
        QAModel {
            question_encoder,
            context_encoder,
            dropout,
            train: true
        }
    }

    pub fn forward_qc(&self, q: &Tensor, c: &Tensor) -> Tensor {
        // q shape: (batch size, seq len), c shape: (batch size, n sentences, seq len)
        let (batch_size, context_sentences, context_len) = c.size3().unwrap();
        // Run context sentences through encoder
        let c = self.context_encoder.forward(&c.reshape(&[batch_size * context_sentences, context_len]));
        // Run questions through encoder
        let q = self.question_encoder.forward(q);

        // Unflatten encoded contexts
        let c = c.reshape(&[batch_size, context_sentences, c.size()[1]]).dropout(self.dropout, self.train);

        // q: (batch size, n embed), c: (batch size, context sentences, n embed)

        // Distance calculation
        c.matmul(&q.unsqueeze(-1)).squeeze_dim(-1)
        //-Tensor::cdist(&c, &q.unsqueeze(1), 2., Some(1)).squeeze_dim(-1)
    }

    pub fn train(&mut self) {
        self.question_encoder.train();
        self.context_encoder.train();
        self.train = true;
    }

    pub fn eval(&mut self) {
        self.question_encoder.eval();
        self.context_encoder.eval();
        self.train = false;
    }
}
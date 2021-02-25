import torch
import spacy
from spacy.tokenizer import Tokenizer
from torch import nn
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence


class BaselineLSTMClassifier(nn.Module):
    def __init__(self, device, vocab_size, embedding_size=512, number_rnn=2, bidirectional=True, number_layers=2,
                 layer_size=256, minimum_layer_size=8, dropout_rate=0.0):
        super().__init__()
        # Token
        self.model = spacy.load('en_core_web_lg')
        self.tokenizer = Tokenizer(self.model.vocab)
        self.vocab_size = vocab_size
        self.vocab = {None: 1}
        self.device = device
        # Embedding
        self.embedding_size = embedding_size
        # RNN
        self.number_rnn = number_rnn
        self.bidirectional = bidirectional
        # FCN
        self.number_layers = number_layers
        self.layer_size = layer_size
        self.minimum_layer_size = minimum_layer_size
        self.dropout_rate = dropout_rate

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, self.embedding_size)

        # RNN Layer
        self.rnn = nn.LSTM(self.embedding_size, self.layer_size, num_layers=self.number_rnn,
                           bidirectional=self.bidirectional, dropout=self.dropout_rate, batch_first=True)
        self.transform_rnn_output = BaselineLSTMClassifier.transform_rnn_output_bidirectional if self.bidirectional \
            else BaselineLSTMClassifier.transform_rnn_output_nbidirectional

        # Fully Connected Layer
        self.layer_size = self.layer_size * 2 if self.bidirectional else self.layer_size
        previous_layer_size = self.layer_size // 2
        fully_connected = [nn.Linear(self.layer_size, previous_layer_size)]
        for _ in range(1, self.number_layers):
            _fc_output_size = max(minimum_layer_size, previous_layer_size // 2)
            fully_connected.append(nn.Linear(previous_layer_size, _fc_output_size))
            previous_layer_size = _fc_output_size
        self.fc = nn.ModuleList(fully_connected)

        # Dropouts
        self.dropout = nn.Dropout(self.dropout_rate)

        # Classifier
        self.classifier = nn.Linear(previous_layer_size, 1)

    def add_encoding(self, x):
        for _x in x:
            for __x in _x:
                if __x not in self.vocab:
                    if len(self.vocab) < (self.vocab_size - 1):
                        self.vocab[__x] = max(self.vocab.values() or [0]) + 1
                    else:
                        return

    def encode(self, x):
        return pad_sequence(list(map(lambda _x: torch.tensor(list(map(lambda __x: self.vocab.get(__x, 1), _x))), x)),
                            batch_first=True)

    @staticmethod
    def transform_rnn_output_bidirectional(x):
        return torch.cat((x[-2, :, :], x[-1, :, :]), dim=1)

    @staticmethod
    def transform_rnn_output_nbidirectional(x):
        return x[-1, :, :]

    def forward(self, x):
        x = list(map(lambda _x: self.tokenizer(_x), x))
        if len(self.vocab) < (self.vocab_size - 1):
            self.add_encoding(x)
        x = self.encode(x).to(self.device)
        x = self.embedding(x)
        _, (x, cell) = self.rnn(x)
        x = self.transform_rnn_output(x)
        for fc_layer in self.fc:
            x = torch.relu(fc_layer(x))
            x = self.dropout(x)

        x = torch.tanh(self.classifier(x))
        return torch.squeeze(x, dim=1)

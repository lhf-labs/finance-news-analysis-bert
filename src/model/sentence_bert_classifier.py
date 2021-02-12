import torch
from torch import nn
from model.sentence_encoder import sentence_encoder


class SentenceBERTClassifier(nn.Module):
    def __init__(self, dataset, sentence_model, device, number_layers=2, layer_size=256, minimum_layer_size=8,
                 dropout_rate=0.0):
        super().__init__()
        self.dataset = dataset
        self.embedding_model = sentence_encoder(sentence_model)
        self.device = device
        self.number_layers = number_layers
        self.layer_size = layer_size
        self.minimum_layer_size = minimum_layer_size
        self.dropout_rate = dropout_rate
        self.input_size = len(self.embedding_model.encode(""))

        # Fully Connected Layer
        fully_connected = [nn.Linear(self.input_size, self.layer_size)]
        previous_layer_size = self.layer_size
        for _ in range(1, self.number_layers):
            _fc_output_size = max(minimum_layer_size, previous_layer_size // 2)
            fully_connected.append(nn.Linear(previous_layer_size, _fc_output_size))
            previous_layer_size = _fc_output_size
        self.fc = nn.ModuleList(fully_connected)

        # Dropouts
        self.dropout = nn.Dropout(self.dropout_rate)

        # Classifier
        if dataset == 'finance':
            self.classifier = nn.Linear(previous_layer_size, 1)
            self.activation = SentenceBERTClassifier.activation_finance
        else:
            self.classifier = nn.Linear(previous_layer_size, 3)
            self.activation = SentenceBERTClassifier.activation_phrasebank

    @staticmethod
    def activation_finance(x):
        x = torch.tanh(x)
        return torch.squeeze(x, dim=1)

    @staticmethod
    def activation_phrasebank(x):
        x = torch.softmax(x, dim=1)
        return x

    def forward(self, x):
        x = torch.Tensor(self.embedding_model.encode(x)).to(self.device)
        for fc_layer in self.fc:
            x = torch.relu(fc_layer(x))
            x = self.dropout(x)
        x = self.activation(self.classifier(x))
        return x

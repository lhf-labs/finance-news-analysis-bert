import torch
from torch import nn
from model.sentence_encoder import sentence_encoder


class SentenceBERTClassifier(nn.Module):
    def __init__(self, sentence_model, device, number_layers=2, layer_size=256, minimum_layer_size=8,
                 dropout_rate=0.0):
        super().__init__()
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
        self.classifier = nn.Linear(previous_layer_size, 1)

    def forward(self, x):
        x = torch.Tensor(self.embedding_model.encode(x)).to(self.device)
        for fc_layer in self.fc:
            x = torch.relu(fc_layer(x))
            x = self.dropout(x)

        x = torch.tanh(self.classifier(x))
        return torch.squeeze(x, dim=1)

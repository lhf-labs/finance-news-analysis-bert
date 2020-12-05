import torch
from torch import nn
from transformers import BertTokenizer, BertModel


class BaselineBERTClassifier(nn.Module):
    def __init__(self, model, device, number_layers=2, layer_size=256, minimum_layer_size=8,
                 dropout_rate=0.0):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.model = BertModel.from_pretrained(model)
        self.device = device
        self.number_layers = number_layers
        self.layer_size = layer_size
        self.minimum_layer_size = minimum_layer_size
        self.dropout_rate = dropout_rate
        self.input_size = self.model(**self.tokenizer("", return_tensors="pt"))[1].shape[1]

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
        x = self.tokenizer(x, return_tensors="pt", padding=True).to(self.device)
        x = self.model(**x)[1]
        for fc_layer in self.fc:
            x = torch.relu(fc_layer(x))
            x = self.dropout(x)

        x = torch.tanh(self.classifier(x))
        return torch.squeeze(x, dim=1)

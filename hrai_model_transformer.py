# main.py
import torch
import torch.nn as nn
import numpy as np

class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes, num_heads, num_layers, dropout=0.1, new_embed_dim=24):
        super(TransformerModel, self).__init__()
        self.input_map = nn.Linear(input_size, new_embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=new_embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(new_embed_dim, num_classes)

    def forward(self, x):
        x = self.input_map(x)
        transformed = self.transformer_encoder(x)
        output = self.classifier(transformed.mean(dim=1))
        return output

def load_model(model_path, input_size, num_classes, num_heads, num_layers, new_embed_dim):
    model = TransformerModel(input_size, num_classes, num_heads, num_layers, new_embed_dim=new_embed_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, input_vector):
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_vector).unsqueeze(0).unsqueeze(0)  # Добавляем размерность батча и последовательности
        output = model(input_tensor)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()

# Загрузка модели
model_path = 'best_model.pth'  # Путь к сохраненной модели
input_size = 23
num_classes = 2
num_heads = 2
num_layers = 2
new_embed_dim = 24
model = load_model(model_path, input_size, num_classes, num_heads, num_layers, new_embed_dim)

# Пример входного вектора данных
example_input = np.random.rand(input_size)  # Случайный вектор

# Выполнение предсказания
prediction = predict(model, example_input)
print("Результат предсказания:", prediction)

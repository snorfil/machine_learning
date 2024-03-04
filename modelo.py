from transformers import AutoTokenizer, AutoModel
import torch


sentences = ["This is an example sentence", "Each sentence is converted"]

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
torch.load('ruta_para_guardar_modelo.pth')

# Guardar el modelo y los embeddings en formato compatible con PyTorch
torch.save(model, 'ruta_para_guardar_modelo.pth')
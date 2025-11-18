from transformers import AutoTokenizer, AutoModel
import torch

class EmbeddingGenerator:
    def __init__(self):
        self.model_name = "BAAI/bge-small-en-v1.5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

    def create_embedding(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        return embeddings[0].tolist()

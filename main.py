import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, AdamW
import pandas as pd

class TripleImportanceDataset(Dataset):
    def __init__(self, tsv_file, relation2text_file, tokenizer):
        self.df = pd.read_csv(tsv_file, sep='\t', header=None, names=['head', 'relation', 'tail', 'importance'])
        self.relation2text = pd.read_csv(relation2text_file, sep='\t', header=None, names=['relation', 'text']).set_index('relation')['text'].to_dict()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        triple = self.df.iloc[idx]
        text = f"{triple['head']} {self.relation2text[triple['relation']]} {triple['tail']}"
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        return {'input_ids': encoding['input_ids'].squeeze(), 'attention_mask': encoding['attention_mask'].squeeze(), 'importance': triple['importance']}

def train(model, loader, device, optimizer):
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['importance'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    tsv_file = 'output/train_importance_scores.tsv'
    relation2text_file = 'data/YAGO3-10/relation2text.txt'
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    dataset = TripleImportanceDataset(tsv_file, relation2text_file, tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=1)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    epochs = 3
    for epoch in range(epochs):
        train_loss = train(model, loader, device, optimizer)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f}")

if __name__ == "__main__":
    main()

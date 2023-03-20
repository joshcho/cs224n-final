import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, AdamW
import pandas as pd
from tqdm import tqdm
import datetime
import os

class TripleImportanceDataset(Dataset):
    def __init__(self, tsv_file, relation2text_file, tokenizer):
        self.df = pd.read_csv(tsv_file, sep='\t', header=None, names=['head', 'relation', 'tail', 'importance'])
        self.df = self.df[(self.df['importance'] != -1) & (self.df['importance'] != -2)]
        self.df = self.df[:3000]
        # print(self.df['importance'].mean())
        self.relation2text = pd.read_csv(relation2text_file, sep='\t', header=None, names=['relation', 'text']).set_index('relation')['text'].to_dict()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        triple = self.df.iloc[idx]
        text = f"{triple['head']} {self.relation2text[triple['relation']]} {triple['tail']}"
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze(), triple['importance']

def train(model, loader, device, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(loader)

def predict(model, loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            input_ids, attention_mask, _ = [x.to(device) for x in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            pred = outputs.logits.squeeze().cpu().numpy()
            predictions.extend(pred)
    return predictions

def main():
    # Load data
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    relation2text_file = 'preprocess-rust/data/YAGO3-10/relation2text.txt'
    train_tsv_file = 'preprocess-rust/output/train_sample_importance_scores.tsv'
    test_tsv_file = 'preprocess-rust/output/test_importance_scores.tsv'
    dev_tsv_file = 'preprocess-rust/output/dev_importance_scores.tsv'

    train_dataset = TripleImportanceDataset(train_tsv_file, relation2text_file, tokenizer)
    test_dataset = TripleImportanceDataset(test_tsv_file, relation2text_file, tokenizer)
    dev_dataset = TripleImportanceDataset(dev_tsv_file, relation2text_file, tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)
    dev_loader = DataLoader(dev_dataset, batch_size=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=1)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    epochs = 3
    for epoch in range(epochs):
        train_loss = train(model, train_loader, device, optimizer)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f}")

    test_loss = evaluate(model, test_loader, device)
    print(test_loss)

    predictions = predict(model, dev_loader, device)
    print("Predictions:", predictions)

    # Save the fine-tuned model
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_directory = "finetuned_models"
    os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist
    model_save_path = f"{save_directory}/finetuned_distilbert_{timestamp}.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()

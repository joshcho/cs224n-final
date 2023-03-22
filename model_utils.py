import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW

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
            input_ids, attention_mask = [x.to(device) for x in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            pred = outputs.logits.squeeze().cpu().numpy()
            pred = pred.reshape(-1)
            predictions.extend(pred)
    return predictions

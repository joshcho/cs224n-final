import pandas as pd
from torch.utils.data import Dataset
from utils import power_decay

class TripleDataset(Dataset):
    def __init__(self, tsv_file, relation2text_file, tokenizer):
        self.df = pd.read_csv(tsv_file, sep='\t', header=None, usecols=[0, 1, 2], names=['head', 'relation', 'tail'])
        self.relation2text = pd.read_csv(relation2text_file, sep='\t', header=None, names=['relation', 'text']).set_index('relation')['text'].to_dict()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        triple = self.df.iloc[idx]
        text = f"{triple['head']} {self.relation2text[triple['relation']]} {triple['tail']}"
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()

    def text(self, idx):
        triple = self.df.iloc[idx]
        return f"{triple['head']} {self.relation2text[triple['relation']]} {triple['tail']}"

class TripleImportanceDataset(Dataset):
    def __init__(self, tsv_file, relation2text_file, tokenizer, decay_factor, importance_column):
        self.df = pd.read_csv(tsv_file, sep='\t', header=None, names=['head', 'relation', 'tail', 'index', 'char-index', 'index-with-infobox'])
        self.df = self.df[(self.df[importance_column] != -1) & (self.df[importance_column] != -2)]
        self.relation2text = pd.read_csv(relation2text_file, sep='\t', header=None, names=['relation', 'text']).set_index('relation')['text'].to_dict()
        self.tokenizer = tokenizer
        self.decay_factor = decay_factor
        self.importance_column = importance_column

    def __getitem__(self, idx):
        triple = self.df.iloc[idx]
        text = f"{triple['head']} {self.relation2text[triple['relation']]} {triple['tail']}"
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        importance = power_decay(triple[self.importance_column], self.decay_factor)
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze(), importance

    def __len__(self):
        return len(self.df)

    def text(self, idx):
        triple = self.df.iloc[idx]
        return f"{triple['head']} {self.relation2text[triple['relation']]} {triple['tail']}"

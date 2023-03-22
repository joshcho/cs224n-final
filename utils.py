from transformers import (BertForSequenceClassification, BertTokenizerFast,
                          DistilBertForSequenceClassification, DistilBertTokenizerFast,
                          RobertaForSequenceClassification, RobertaTokenizerFast)

def power_decay(index, decay_factor):
    return decay_factor**index

def get_pretrained_model_name(model):
    return "roberta-base" if model == "roberta" else f'{model}-base-uncased'

def get_model_and_tokenizer(model_name):
    if model_name == 'bert':
        return BertForSequenceClassification, BertTokenizerFast
    elif model_name == 'distilbert':
        return DistilBertForSequenceClassification, DistilBertTokenizerFast
    elif model_name == 'roberta':
        return RobertaForSequenceClassification, RobertaTokenizerFast
    else:
        raise ValueError("Invalid model name. Choose from 'bert', 'distilbert', or 'roberta'.")

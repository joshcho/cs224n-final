import torch
import argparse
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataset import TripleDataset, TripleImportanceDataset
from model_utils import train, evaluate
from impute import convert_and_impute
from utils import get_model_and_tokenizer, get_pretrained_model_name
from datetime import datetime
import os

def main(args):
    # Check if either training or a model path is specified for imputation
    if not args.no_impute:
        assert args.wiki_no_infobox_train or args.wiki_no_infobox_path, "Either training or a model path must be specified for wiki_no_infobox when imputation occurs."
        assert args.wiki_with_infobox_train or args.wiki_with_infobox_path, "Either training or a model path must be specified for wiki_with_infobox when imputation occurs."
        assert args.wiki_char_index_train or args.wiki_char_index_path, "Either training or a model path must be specified for wiki_char_index when imputation occurs."
    else:
        assert not (args.wiki_no_infobox_path or args.wiki_with_infobox_path or args.wiki_char_index_path), "Paths to saved models are not necessary when just training (no imputation)."

    # Load data
    model_class, tokenizer_class = get_model_and_tokenizer(args.model)
    pretrained_model_name = get_pretrained_model_name(args.model)
    tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)
    relation2text_file = 'preprocess-rust/data/YAGO3-10/relation2text.txt'
    train_tsv_file = 'preprocess-rust/output/train_all_indices.tsv'
    test_tsv_file = 'preprocess-rust/output/test_all_indices.tsv'
    dev_tsv_file = 'preprocess-rust/output/dev_all_indices.tsv'

    models_to_train = []
    if args.wiki_no_infobox_train:
        models_to_train.append('wiki_no_infobox')
    if args.wiki_with_infobox_train:
        models_to_train.append('wiki_with_infobox')
    if args.wiki_char_index_train:
        models_to_train.append('wiki_char_index')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_paths = {
        'wiki_no_infobox': args.wiki_no_infobox_path,
        'wiki_with_infobox': args.wiki_with_infobox_path,
        'wiki_char_index': args.wiki_char_index_path
    }

    decay_factors = {
        'wiki_no_infobox': 0.95,
        'wiki_with_infobox': 0.95,
        'wiki_char_index': 0.999
    }

    importance_columns = {
        'wiki_no_infobox': 'index',
        'wiki_with_infobox': 'index-with-infobox',
        'wiki_char_index': 'char-index'
    }

    if not os.path.exists('finetuned_models'):
        os.makedirs('finetuned_models')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_name in models_to_train:
        print(f"Start training {model_name} for {args.num_epochs} epochs:")
        decay_factor = decay_factors[model_name]
        importance_column = importance_columns[model_name]
        train_dataset = TripleImportanceDataset(train_tsv_file, relation2text_file, tokenizer, decay_factor, importance_column)
        test_dataset = TripleImportanceDataset(test_tsv_file, relation2text_file, tokenizer, decay_factor, importance_column)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        model = model_class.from_pretrained(pretrained_model_name, num_labels=1)
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=1e-5)

        for epoch in range(args.num_epochs):
            train_loss = train(model, train_loader, device, optimizer)
            print(f"{model_name} - Epoch: {epoch}, Train Loss: {train_loss}")

        test_loss = evaluate(model, test_loader, device)
        print(f"{model_name} - Test Loss: {test_loss}")

        model_path = f"finetuned_models/{args.model}_{model_name}_finetuned_{timestamp}.pth"
        torch.save(model.state_dict(), model_path)
        assert model_paths[model_name] == ''
        model_paths[model_name] = model_path

    # Impute missing importance scores
    if not args.no_impute:
        convert_and_impute(train_tsv_file, test_tsv_file, dev_tsv_file, model_class, tokenizer, device, relation2text_file, args, model_paths, decay_factors, importance_columns)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or load a model and make predictions.')
    parser.add_argument('--model', type=str, choices=['bert', 'distilbert', 'roberta'], required=True,
                        help='The model to use: bert, distilbert, or roberta.')
    parser.add_argument('--wiki_no_infobox_train', action='store_true', help='Train wiki_no_infobox model.')
    parser.add_argument('--wiki_no_infobox_path', type=str, default='', help='Path to the wiki_no_infobox finetuned model file.')
    parser.add_argument('--wiki_with_infobox_train', action='store_true', help='Train wiki_with_infobox model.')
    parser.add_argument('--wiki_with_infobox_path', type=str, default='', help='Path to the wiki_with_infobox finetuned model file.')
    parser.add_argument('--wiki_char_index_train', action='store_true', help='Train wiki_char_index model.')
    parser.add_argument('--wiki_char_index_path', type=str, default='', help='Path to the wiki_char_index finetuned model file.')
    parser.add_argument('--no_impute', action='store_true', help='Skip imputation.')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')

    args = parser.parse_args()

    main(args)

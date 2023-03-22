import torch
import argparse
from dataset import TripleDataset, TripleImportanceDataset
from model_utils import train, evaluate, predict
from impute import convert_and_impute
from utils import get_model_and_tokenizer, get_pretrained_model_name

def main(args):
    # Load data
    model_class, tokenizer_class = get_model_and_tokenizer(args.model)
    pretrained_model_name = get_pretrained_model_name(args.model)
    tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)
    relation2text_file = 'preprocess-rust/data/YAGO3-10/relation2text.txt'
    train_tsv_file = 'preprocess-rust/output/train_indices.tsv'
    test_tsv_file = 'preprocess-rust/output/test_indices.tsv'
    dev_tsv_file = 'preprocess-rust/output/dev_indices.tsv'

    models_to_train = []
    if args.wiki_no_infobox_train:
        models_to_train.append(('wiki_no_infobox', 0.95, 'index'))
    if args.wiki_with_infobox_train:
        models_to_train.append(('wiki_with_infobox', 0.95, 'index'))
    if args.wiki_char_index_train:
        models_to_train.append(('wiki_char_index', 0.99, 'char-index'))

    for model_name, decay_factor, importance_column in models_to_train:
        train_dataset = TripleImportanceDataset(train_tsv_file, relation2text_file, tokenizer, decay_factor, importance_column)
        test_dataset = TripleImportanceDataset(test_tsv_file, relation2text_file, tokenizer, decay_factor, importance_column)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        model = model_class.from_pretrained(pretrained_model_name)
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=1e-5)

        for epoch in range(args.num_epochs):
            train_loss = train(model, train_loader, device, optimizer)
            print(f"{model_name} - Epoch: {epoch}, Train Loss: {train_loss}")

        test_loss = evaluate(model, test_loader, device)
        print(f"{model_name} - Test Loss: {test_loss}")

        torch.save(model.state_dict(), f"{model_name}_finetuned.pth")

    # Impute missing importance scores
    if not args.no_impute:
        convert_and_impute(train_tsv_file, test_tsv_file, dev_tsv_file, model_class, tokenizer, device, args)

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

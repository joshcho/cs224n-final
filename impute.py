import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import power_decay, get_pretrained_model_name

def convert_and_impute(train_tsv, test_tsv, dev_tsv, model_class, tokenizer, device, args):
    dfs = {
        'train': pd.read_csv(train_tsv, sep='\t', header=None, names=['head', 'relation', 'tail', 'index', 'char-index']),
        'test': pd.read_csv(test_tsv, sep='\t', header=None, names=['head', 'relation', 'tail', 'index', 'char-index']),
        'dev': pd.read_csv(dev_tsv, sep='\t', header=None, names=['head', 'relation', 'tail', 'index', 'char-index'])
    }

    model_paths = {
        'wiki_no_infobox': args.wiki_no_infobox_path,
        'wiki_with_infobox': args.wiki_with_infobox_path,
        'wiki_char_index': args.wiki_char_index_path
    }

    decay_factors = {
        'wiki_no_infobox': 0.95,
        'wiki_with_infobox': 0.95,
        'wiki_char_index': 0.99
    }

    importance_columns = {
        'wiki_no_infobox': 'index',
        'wiki_with_infobox': 'index',
        'wiki_char_index': 'char-index'
    }

    for model_name, model_path in model_paths.items():
        print(f"{model_name} - Imputing:")
        model = model_class.from_pretrained(get_pretrained_model_name(args.model))
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        for dataset_name, df in dfs.items():
            missing_indices = df[df[model_name].isin([-1, -2])].index
            missing_data = TripleImportanceDataset(df.iloc[missing_indices], relation2text_file, tokenizer, decay_factors[model_name], importance_columns[model_name])
            missing_loader = DataLoader(missing_data, batch_size=16)
            imputed_scores = predict(model, missing_loader, device)
            df.loc[missing_indices, model_name] = imputed_scores

            # Apply power decay for non-missing importance scores
            non_missing_indices = df[~df[model_name].isin([-1, -2])].index
            df.loc[non_missing_indices, model_name] = df.loc[non_missing_indices, model_name].apply(lambda x: power_decay(x, decay_factors[model_name]))

    # Combine the 3 importance scores and remove 'index' and 'char-index' columns
    for dataset_name, df in dfs.items():
        print(f"Combining imputed values for {dataset_name}.")
        df = df.drop(columns=['index', 'char-index'])

        # Save the imputed dataset
        df.to_csv(f"{dataset_name}_imputed.tsv", sep='\t', index=False, header=False)

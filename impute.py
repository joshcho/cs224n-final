import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from datetime import datetime
import os
import tempfile
from utils import power_decay, get_pretrained_model_name
from dataset import TripleDataset, TripleImportanceDataset
from model_utils import predict

def convert_and_impute(train_tsv, test_tsv, dev_tsv, model_class, tokenizer, device, relation2text_file, args, model_paths, decay_factors, importance_columns):
    dfs = {
        'train': pd.read_csv(train_tsv, sep='\t', header=None, names=['head', 'relation', 'tail', 'index', 'char-index']),
        'test': pd.read_csv(test_tsv, sep='\t', header=None, names=['head', 'relation', 'tail', 'index', 'char-index']),
        'dev': pd.read_csv(dev_tsv, sep='\t', header=None, names=['head', 'relation', 'tail', 'index', 'char-index'])
    }

    for model_name, model_path in model_paths.items():
        # If the model path is not specified, use the provided model_paths from training
        assert model_path != ''

        model = model_class.from_pretrained(get_pretrained_model_name(args.model), num_labels=1)
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        for dataset_name, df in dfs.items():
            print(f"Imputing {model_name}, {dataset_name}:")
            importance_column = importance_columns[model_name]
            decay_factor = decay_factors[model_name]
            missing_indices = df[df[importance_column].isin([-1, -2])].index
            imputed_scores = []
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
                temp_df = df.iloc[missing_indices]
                temp_df.to_csv(temp_file.name, sep='\t', index=False, header=None)
                missing_data = TripleDataset(temp_file.name, relation2text_file, tokenizer)
                missing_loader = DataLoader(missing_data, batch_size=args.batch_size)
                imputed_scores = predict(model, missing_loader, device)
            df.loc[missing_indices, model_name] = imputed_scores

            # Apply power decay for non-missing importance scores
            non_missing_indices = df[~df[importance_column].isin([-1, -2])].index
            df.loc[non_missing_indices, model_name] = df.loc[non_missing_indices, importance_column].apply(lambda x: power_decay(x, decay_factor))
            dfs[dataset_name] = df

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists('output'):
        os.makedirs('output')
    # Combine the 3 importance scores and remove 'index' and 'char-index' columns
    for dataset_name, df in dfs.items():
        output_path = f"output/{dataset_name}_imputed_{timestamp}.tsv"
        print(f"Combining imputed values for {dataset_name} to {output_path}.")
        df = df.drop(columns=['index', 'char-index'])

        # Save the imputed dataset
        df.to_csv(output_path, sep='\t', index=False, header=True)

# Importance-augmented Knowledge Graphs (IKAG)
This project aims to augment knowledge graphs with importance weights, leveraging pre-trained language models and Wikipedia data. By assigning importance weights to the relationships in a knowledge graph, we can better represent the prioritization of relationships between entities.

## Getting Started
To use the IAKG project, follow these steps:

1. Install the required Python packages:
`pip install -r requirements.txt`
2. Prepare your dataset files in the appropriate format (TSV files). Make sure you have the following files:

- preprocess-rust/data/YAGO3-10/relation2text.txt
- preprocess-rust/output/train\_indices.tsv
- preprocess-rust/output/test\_indices.tsv
- preprocess-rust/output/dev\_indices.tsv

You can get these files from preprocess-rust directory, which is a Rust project that generates the indices using Wikipedia. You can download YAGO3-10 from [HuggingFace](https://huggingface.co/datasets/VLyb/YAGO3-10).

Train or load models using the main.py script. Use the following command-line arguments:

- --model: Choose the pre-trained language model to use: bert, distilbert, or roberta.
- --wiki\_no\_infobox\_train: Train the wiki\_no\_infobox model.
- --wiki\_no\_infobox\_path: Path to the wiki\_no\_infobox finetuned model file.
- --wiki\_with\_infobox\_train: Train the wiki\_with\_infobox model.
- --wiki\_with\_infobox\_path: Path to the wiki\_with\_infobox finetuned model file.
- --wiki\_char\_index\_train: Train the wiki\_char\_indexmodel.
- --wiki\_char\_index\_path: Path to the wiki\_char\_indexfinetuned model file.
- --no\_impute: Skip the imputation step.
- --num\_epochs: Number of epochs for training (default is 3).
- --batch\_size`: Batch size for training (default is 16).

Example usage for training bert model:

`python main.py --model bert --wiki\_no\_infobox\_train --wiki\_with\_infobox\_train --wiki\_char\_index\_train`

After training or loading the models, the script will impute the missing importance scores (unless --no\_impute is specified) and save the results to the appropriate files.

## Project Structure

The main components of the IAKG project are:

- main.py: The main script for training or loading models, and performing imputation.
- dataset.py: Contains the TripleDataset and TripleImportanceDataset classes for handling the data.
- model\_utils.py: Contains utility functions for training and evaluating models.
- impute.py: Contains the convert\_and\_impute function for converting and imputing importance scores.
- utils.py: Contains utility functions for getting the model and tokenizer classes, and the pre-trained model name.
- preprocess-rust: Contains a Rust project that does preprocessing for getting Wikipedia articles and computing

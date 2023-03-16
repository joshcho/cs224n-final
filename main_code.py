import os
import concurrent.futures
import json
from bs4 import BeautifulSoup
from urllib.request import urlopen
from tqdm import tqdm
from collections import defaultdict
import urllib.parse
from threading import Lock

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = [line.strip().split('\t')[:3] for line in f.readlines()]
    return data

def load_tail_texts(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        tail_texts = {line.split('\t')[0]: line.split('\t')[1].strip() for line in lines}
    return tail_texts

def compute_importance_score(tail_position, decay_factor=0.95, default_importance_score=-1):
    return decay_factor ** tail_position if tail_position >= 0 else default_importance_score

def process_head_and_tails(head, relations_and_tails, output_file, file_lock):
    base_url = "https://en.wikipedia.org/wiki/"
    fullurl = base_url + urllib.parse.quote(head)

    output_lines = []

    try:
        html_content = urlopen(fullurl).read()
        soup = BeautifulSoup(html_content, "html.parser")
        main_content = soup.find('div', {'id': 'mw-content-text'})

        if main_content is not None:
            links = main_content.find_all('a')

            tail_positions = {tail: -1 for _, tail in relations_and_tails}
            remaining_tails = {tail for _, tail in relations_and_tails}

            for i, link in enumerate(links):
                if link.has_attr('href'):
                    href = link['href']
                    if href.startswith('/wiki/') and ':' not in href:
                        tail_candidate = href[6:]

                        if tail_candidate in remaining_tails:
                            tail_positions[tail_candidate] = i
                            remaining_tails.remove(tail_candidate)

                        if not remaining_tails:
                            break

            for tail, position in tail_positions.items():
                relation = next(r for r, t in relations_and_tails if t == tail)
                importance_score = compute_importance_score(position, decay_factor=0.95)
                output_lines.append(f'{head}\t{relation}\t{tail}\t{importance_score}')
        else:
            for relation, tail in relations_and_tails:
                output_lines.append(f'{head}\t{relation}\t{tail}\t{-2}')

    except Exception as e:
        # print(f"Error while fetching the Wikipedia page for {head}: {e}")
        for relation, tail in relations_and_tails:
            output_lines.append(f'{head}\t{relation}\t{tail}\t{-2}')

    with file_lock:
        with open(output_file, 'a') as f:
            f.write('\n'.join(output_lines) + '\n')

def process_data(data, output_path, dataset_name, num_workers=16):
    output_file = os.path.join(output_path, f"{dataset_name}_importance_scores.tsv")
    os.makedirs(output_path, exist_ok=True)

    grouped_data = defaultdict(list)
    for head, relation, tail in data:
        grouped_data[head].append((relation, tail))

    total_count = len(grouped_data)
    file_lock = Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        input_tuples = [(head, relations_and_tails, output_file, file_lock) for head, relations_and_tails in grouped_data.items()]
        results = list(tqdm(executor.map(lambda args: process_head_and_tails(*args), input_tuples), total=total_count))

def main():
    output_path = 'output'
    datasets = ['train', 'test', 'dev']
    #tail_texts = load_tail_texts("data/YAGO3-10/entity2text.txt")

    for dataset_name in datasets:
        print(f"Loading data/YAGO3-10/{dataset_name}.tsv")
        data = load_data(f"data/YAGO3-10/{dataset_name}.tsv")
        print(f"Processing data/YAGO3-10/{dataset_name}.tsv")
        process_data(data, output_path, dataset_name)

if __name__ == "__main__":
    main()

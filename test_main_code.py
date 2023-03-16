import tempfile
import os
import wikipediaapi
import pandas as pd
from main_code import get_ordered_links, compute_importance_score, process_data, process_head_and_tails, load_tail_texts, load_data
import pytest

def test_get_ordered_links():
    wiki = wikipediaapi.Wikipedia('en')
    page = wiki.page("Python (programming language)")
    ordered_links = get_ordered_links(page)
    assert len(ordered_links) > 0, "get_ordered_links() should return a non-empty list"
    assert ordered_links[0] != "", "The first element in ordered_links should not be empty"

def test_compute_importance_score():
    assert compute_importance_score(-1) == 0, "compute_importance_score() should return 0 when tail_position is -1"
    assert compute_importance_score(0) == 1, "compute_importance_score() should return 1 when tail_position is 0"
    assert compute_importance_score(1) > 0, "compute_importance_score() should return a positive value when tail_position is positive"

def test_process_head_and_tails(tmpdir):
    wiki = wikipediaapi.Wikipedia('en', extract_format=wikipediaapi.ExtractFormat.HTML)
    head = "Barack_Obama"
    relations_and_tails = [("spouse", "Michelle_Obama"), ("birthPlace", "Honolulu")]
    tail_texts = {"Michelle_Obama": "Michelle Obama", "Honolulu": "Honolulu"}

    temp = tmpdir.join("temp.txt")
    process_head_and_tails(head, relations_and_tails, wiki, temp)

    with open(temp, 'r') as f:
        lines = f.read().splitlines()

    assert len(lines) == 2
    assert lines[0].startswith(f'{head}\t{relations_and_tails[0][0]}\t{relations_and_tails[0][1]}\t')
    assert lines[1].startswith(f'{head}\t{relations_and_tails[1][0]}\t{relations_and_tails[1][1]}\t')
    assert 0 <= float(lines[0].split('\t')[-1]) <= 1
    assert 0 <= float(lines[1].split('\t')[-1]) <= 1

def test_process_data(tmp_path):
    data_path = "data/YAGO3-10/test_sample.tsv"
    output_path = tmp_path / "output"
    os.makedirs(output_path, exist_ok=True)
    wiki = wikipediaapi.Wikipedia('en')
    #tail_texts = load_tail_texts("data/YAGO3-10/entity2text.txt")
    data = load_data(data_path)
    process_data(data, wiki, output_path, "test")
    output_file = output_path / "test_importance_scores.tsv"
    assert os.path.exists(output_file), "process_data() should generate an output file"

    df = pd.read_csv(output_file, delimiter='\t', names=['head', 'relation', 'tail', 'importance_score'])
    assert not df.empty, "Output DataFrame should not be empty"
    assert len(df.columns) == 4, "Output DataFrame should have 4 columns"
    assert df['importance_score'].isnull().sum() == 0, "There should not be any null importance scores in the output"

if __name__ == "__main__":
    pytest.main(["-v", "test_main_code.py"])

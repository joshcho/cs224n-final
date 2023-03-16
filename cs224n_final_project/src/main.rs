use csv::ReaderBuilder;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::{blocking::Client, Url};
use scraper::{Html, Selector};
use std::{fs::File, io::Write, path::Path};
use serde_derive::Deserialize;
use std::collections::HashMap;
use std::sync::{Mutex, Arc};
use rayon::prelude::*;

#[derive(Debug, Deserialize, Clone)]
struct InputRow {
    head: String,
    relation: String,
    tail: String,
}

fn main() {
    let file_path = "data/YAGO3-10/train.tsv";
    let output_path = Path::new("output/train_importance_scores.tsv");
    let output_file = Arc::new(Mutex::new(File::create(output_path).unwrap()));

    let mut reader = ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_path(file_path)
        .unwrap();

    let mut data = Vec::new();

    for result in reader.records() {
        let record = result.unwrap();
        let input_row = InputRow {
            head: record[0].to_string(),
            relation: record[1].to_string(),
            tail: record[2].to_string(),
        };
        data.push(input_row);
    }

    let mut batches: HashMap<String, Vec<InputRow>> = HashMap::new();

    for input_row in data.clone() {
        batches
            .entry(input_row.head.clone())
            .or_insert_with(Vec::new)
            .push(input_row.clone());
    }

    let data_batches: Vec<Vec<InputRow>> = batches.into_iter().map(|(_, batch)| batch).collect();
    let data_batches = Arc::new(data_batches);
    let pb = Arc::new(Mutex::new(ProgressBar::new(data_batches.len() as u64)));
    pb.lock().unwrap().set_style(ProgressStyle::default_bar().template("[{elapsed_precise}] {bar:40} {pos}/{len} ETA: {eta} {msg}").unwrap());

    data_batches.par_iter().for_each(|batch| {
        let head = batch[0].head.clone();
        let pb = pb.clone();
        let output_file = output_file.clone();

        match process_batch(&head, &batch, &output_file, pb) {
            Ok(()) => (),
            Err(e) => eprintln!("Error: {:?}", e),
        }
    });

    let pb = pb.lock().unwrap();
    pb.finish_with_message("Finished processing batches.");
}

fn process_batch(head: &str, batch: &[InputRow], output_file: &Mutex<File>, pb: Arc<Mutex<ProgressBar>>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client = Client::new();
    let base_url = "https://en.wikipedia.org/wiki/";
    let full_url = base_url.to_string() + &head.replace(" ", "_");
    let url = Url::parse(&full_url).map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
    let resp = client.get(url.clone()).send().map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

    if resp.status().is_success() {
        let html = resp.text().map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
        let document = Html::parse_document(&html);
        let selector = Selector::parse("div#mw-content-text a").unwrap();
        let links = document.select(&selector);

        let mut preprocessed_links: Vec<String> = Vec::new();
        for link in links {
            if let Some(href) = link.value().attr("href") {
                if href.starts_with("/wiki/") && !href.contains(':') {
                    preprocessed_links.push(href[6..].to_string());
                }
            }
        }

        for row in batch {
            let mut pos = 0;
            let mut found = false;
            for (index, link) in preprocessed_links.iter().enumerate() {
                if link == &row.tail {
                    pos = index;
                    found = true;
                    break;
                }
            }
            let score = if found {
                0.95f32.powf(pos as f32)
            } else {
                -1.0
            };

            {
                let mut file = output_file.lock().unwrap();
                writeln!(file, "{}\t{}\t{}\t{}", row.head, row.relation, row.tail, score)?;
            }
        }
    } else {
        for row in batch {
            let score = -2.0;
            {
                let mut file = output_file.lock().unwrap();
                writeln!(file, "{}\t{}\t{}\t{}", row.head, row.relation, row.tail, score)?;
            }
        }
    }

    {
        let pb = pb.lock().unwrap();
        pb.inc(1);
    }

    Ok(())
}

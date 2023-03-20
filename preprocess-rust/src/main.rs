use csv::ReaderBuilder;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::{blocking::Client, Url};
use markup5ever::interface::tree_builder::TreeSink;
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
    let output_path = Path::new("output/train_indices.tsv");
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

fn fetch_wikipedia_page(head: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let client = Client::new();
    let base_url = "https://en.wikipedia.org/wiki/";
    let full_url = base_url.to_string() + &head.replace(" ", "_");
    let url = Url::parse(&full_url).map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
    let resp = client.get(url.clone()).send().map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

    if resp.status().is_success() {
        let html = resp.text().map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
        Ok(html)
    } else {
        Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Failed to fetch Wikipedia page",
        )))
    }
}

fn remove_aux(document: &mut Html) {
    let sections_to_remove = [
        "style",
        "link",
        ".infobox",
        ".shortdescription",
        ".sidebar",
        "#References",
        ".hatnote",
        "#External_links",
        "#Further_reading",
    ];

    for section in &sections_to_remove {
        let selector = Selector::parse(section).unwrap();
        for element in document.clone().select(&selector) {
            document.remove_from_parent(&element.id());
        }
    }
}

fn get_links_and_indices(document: &Html) -> Vec<(String, usize)> {
    let mw_parser_output_selector = Selector::parse("div#mw-content-text").unwrap();
    let a_selector = Selector::parse("a").unwrap();
    let mw_parser_output = document.select(&mw_parser_output_selector).next().unwrap();

    let document_text = mw_parser_output.text().collect::<String>();
    let mut links_and_indices = Vec::new();

    for a_element in mw_parser_output.select(&a_selector) {
        if let Some(href) = a_element.value().attr("href") {
            if href.starts_with("/wiki/") && !href.contains(':') {
                let link_str = &href[6..];
                let link_text = a_element.text().collect::<String>();
                if let Some(index) = document_text.find(&link_text) {
                    links_and_indices.push((link_str.to_string(), index));
                }
            }
        }
    }

    links_and_indices
}

fn process_link_and_batch(
    link_str: &str,
    link_char_index: usize,
    batch: &mut Vec<InputRow>,
    output_file: &Mutex<File>,
    index: &mut i32,
) {
    if let Some(row_pos) = batch.iter().position(|row| row.tail == link_str) {
        let row = batch.remove(row_pos);
        let mut file = output_file.lock().unwrap();
        writeln!(
            file,
            "{}\t{}\t{}\t{}\t{}",
            row.head, row.relation, row.tail, *index, link_char_index
        )
        .unwrap();
    }
    *index += 1;
}

fn write_remaining_batch(batch: &[InputRow], output_file: &Mutex<File>, index: i32) {
    for row in batch {
        let mut file = output_file.lock().unwrap();
        writeln!(file, "{}\t{}\t{}\t{}\t{}", row.head, row.relation, row.tail, index, -1).unwrap();
    }
}

fn process_batch(head: &str, batch: &[InputRow], output_file: &Mutex<File>, pb: Arc<Mutex<ProgressBar>>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    match fetch_wikipedia_page(head) {
        Ok(html) => {
            let mut document = Html::parse_document(&html);
            remove_aux(&mut document);
            let links_indices = get_links_and_indices(&document);

            let mut batch = batch.to_vec();
            let mut index: i32 = 0;

            for (link_str, link_char_index) in links_indices {
                process_link_and_batch(&link_str, link_char_index, &mut batch, output_file, &mut index);
                if batch.is_empty() {
                    break;
                }
            }

            write_remaining_batch(&batch, output_file, -1);
        }
        Err(_) => {
            write_remaining_batch(&batch, output_file, -2);
        }
    }

    {
        let pb = pb.lock().unwrap();
        pb.inc(1);
    }

    Ok(())
}

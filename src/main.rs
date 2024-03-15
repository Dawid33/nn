use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::path::PathBuf;

use log::warn;

mod graph;
mod mlp;

struct Dataset {
    predictors: Vec<Vec<f64>>,
    response: Vec<f64>,
}

fn let_er_rip(d: Dataset) {
    let gd = graph::GraphDealer::new();

    // Inner loop, quit on ctrl+c and have ability to resume from where it left off.
    {
        let g = gd.get_next_graph();
        let mlp_template = mlp::MLPTemplate::new(g, 1);
        let mut mlp = mlp_template.build_simple();

        for _ in 0..100000 {
            mlp.train_batch(dataset);
        }
        eval_and_dump();
    }
}

fn eval_and_dump() {}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        panic!("Incorrect number of arguments. Please supply a CSV file path and the name of the column to use as a response variable.");
    }

    let file_path = PathBuf::from(&args[1]);
    if file_path.exists() {
        let f = File::open(file_path).expect(format!("Failed to open file {}", args[1]).as_str());
        let mut csv_reader = csv::ReaderBuilder::new().has_headers(true).from_reader(f);
        let headers = csv_reader.headers().expect("Failed reading headers.");
        let reponse_index = headers.iter().position(|x| args[2] == x).unwrap();

        let mut p = Vec::new();
        let mut r = Vec::new();
        for result in csv_reader.records() {
            let mut record: Vec<f64> = result.unwrap().iter().map(|x| x.parse().unwrap()).collect();
            r.push(record.remove(reponse_index));
            p.push(record);
        }
    } else {
        panic!("File '{}' does not exist.", args[0]);
    }
}

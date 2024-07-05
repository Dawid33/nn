use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::{collections::BTreeMap, process::Termination};

use graph::{render, save_graph, Graph};
use log::warn;
use mlp::TrainMetrics;
use num_bigint::BigUint;

use crate::mlp::MLPTemplate;

mod graph;
mod mlp;

struct Params {
    learning_rate: Vec<f64>,
}

struct Dataset {
    name: String,
    data: Vec<(Vec<f64>, Vec<f64>)>,
    params: Params,
}

fn dump_results(inner: u64, id: BigUint, g: &Graph, t: TrainMetrics) {
    let path = PathBuf::from(format!("graphs/{}/{}", inner, id));
    std::fs::create_dir_all(&path).unwrap();

    let graph_path = path.join("adjacency_matrix.txt");
    save_graph(graph_path, &g);

    let graph_path = path.join(format!("graph.dot"));
    let mut f = File::create(graph_path).unwrap();
    render(&g, &mut f);

    let results_path = path.join(format!("output.csv"));
    t.to_file(File::create(results_path).unwrap())
}

fn let_er_rip(d: Dataset) {
    let mut gd = match graph::GraphDealer::from_file("partitions.toml") {
        Ok(g) => g,
        Err(_) => graph::GraphDealer::new(
            d.data.get(0).unwrap().0.len() as u64,
            1,
            &[2],
            "partitions.toml".to_string(),
        ),
    };

    println!("Looking for graphs...");
    loop {
        let (id, g) = gd.get_next_graph().unwrap();
        let inner = g.total - g.input - g.output;
        print!("Found graph id {}/{}.", inner, id);
        let mlp_template = MLPTemplate::new(g.clone(), 1);
        let mut mlp = mlp_template.build_simple();
        let result = mlp.train(&d, 1);
        println!(" Trained and dumped to {}/{}", inner, id);
        dump_results(inner, id, &g, result);
    }
}

fn main() {
    if std::path::PathBuf::from("out.txt").exists() {
        std::fs::remove_file("out.txt").unwrap();
    }
    if std::path::PathBuf::from("output.csv").exists() {
        std::fs::remove_file("output.csv").unwrap();
    }
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        panic!("Incorrect number of arguments. Please supply a CSV file path and the name of the column to use as a response variable.");
    }

    let file_path = PathBuf::from(&args[1]);
    let dataset = if file_path.exists() {
        let f = File::open(&file_path).expect(format!("Failed to open file {}", args[1]).as_str());
        let mut csv_reader = csv::ReaderBuilder::new().has_headers(true).from_reader(f);
        let headers = csv_reader.headers().expect("Failed reading headers.");
        let reponse_index = headers.iter().position(|x| args[2] == x).unwrap();
        println!("headers: {:?}", headers);

        let mut data = Vec::new();
        for result in csv_reader.records() {
            let mut r = Vec::new();
            let mut record: Vec<f64> = result.unwrap().iter().map(|x| x.parse().unwrap()).collect();
            r.push(record.remove(reponse_index));
            data.push((record, r));
        }
        Dataset {
            name: String::from(file_path.file_name().unwrap().to_str().unwrap()),
            data,
            params: Params {
                learning_rate: vec![0.005],
            },
        }
    } else {
        panic!("File '{}' does not exist.", args[0]);
    };

    let_er_rip(dataset);
}

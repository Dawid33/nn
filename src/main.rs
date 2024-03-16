use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::path::PathBuf;

use graph::{render, save_graph, Graph};
use log::warn;
use num_bigint::BigUint;

mod graph;
mod mlp;

struct Dataset {
    name: String,
    predictors: Vec<Vec<f64>>,
    response: Vec<f64>,
}

fn dump_graph(id: BigUint, g: &Graph) {
    let inner = g.total - g.input - g.output;
    let path = PathBuf::from(format!("graphs/{}/{}", inner, id));
    std::fs::create_dir_all(&path).unwrap();

    let graph_path = path.join("adjacency_matrix.txt");
    save_graph(graph_path, &g);

    let graph_path = path.join(format!("graph.dot"));
    let mut f = File::create(graph_path).unwrap();
    render(&g, &mut f);
}

fn let_er_rip(d: Dataset) {
    let mut gd = graph::GraphDealer::new(
        d.predictors.get(0).unwrap().len() as u64,
        1,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    );
    println!("GraphDealer {:?}", gd);

    // Inner loop, quit on ctrl+c and have ability to resume from where it left off.
    loop {
        let (id, g) = gd.get_next_graph().unwrap();
        let inner = g.total - g.input - g.output;
        println!("Dumping graph id {}/{}", inner, id);
        dump_graph(id, &g);

        // let mlp_template = mlp::MLPTemplate::new(g, 1);
        // let mut mlp = mlp_template.build_simple();

        // for _ in 0..100000 {
        // mlp.train_batch(dataset);
        // }
        // eval_and_dump();
    }
}

fn eval_and_dump() {}

fn main() {
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

        let mut p = Vec::new();
        let mut r = Vec::new();
        for result in csv_reader.records() {
            let mut record: Vec<f64> = result.unwrap().iter().map(|x| x.parse().unwrap()).collect();
            r.push(record.remove(reponse_index));
            p.push(record);
        }
        Dataset {
            name: String::from(file_path.file_name().unwrap().to_str().unwrap()),
            predictors: p,
            response: r,
        }
    } else {
        panic!("File '{}' does not exist.", args[0]);
    };

    let_er_rip(dataset);
}

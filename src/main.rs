use rand::prelude::*;
use std::path::PathBuf;

use base::Graph;
use rand_chacha::ChaCha8Rng;

mod base;

type ActivationFn = fn(f64) -> f64;

#[derive(Clone)]
struct Perceptron {
    weights: Vec<f64>,
    activation_function: ActivationFn,
}

pub fn identity(x: f64) -> f64 {
    x
}

pub fn step_wise(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

pub fn linear(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

impl Perceptron {
    pub fn new(inputs: u8) -> Self {
        if inputs < 1 {
            panic!("Perceptron can't have less than one input");
        }
        let mut weights = Vec::new();
        for _ in 0..inputs {
            weights.push(0.0);
        }
        Self {
            weights,
            activation_function: identity,
        }
    }

    pub fn calc(&self, input: Vec<f64>) -> f64 {
        if input.len() != self.weights.len() {
            panic!("Input not equal to number of weights.");
        }

        let mut result: f64 = 0.0;
        for i in 0..self.weights.len() {
            result += self.weights.get(i).unwrap() * input.get(i).unwrap();
        }
        let func = self.activation_function;
        func(result)
    }
}

struct MLPTemplate {
    graph: Graph,
    deps: Vec<Vec<u8>>,
    perceptrons: Vec<Perceptron>,
    seed: u64,
}

impl MLPTemplate {
    pub fn new(graph: Graph, seed: u64) -> Self {
        let mut deps: Vec<Vec<u8>> = Vec::new();

        for _ in graph.input as usize..graph.adjacency_matrix.len() {
            deps.push(Vec::new())
        }

        // Build a list of nodes that each node depends on.
        for i in graph.input as usize..graph.adjacency_matrix.len() {
            for (j, y) in graph.adjacency_matrix.get(i).unwrap().iter().enumerate() {
                if *y == 1 {
                    deps.get_mut(i - graph.input as usize)
                        .unwrap()
                        .push(j as u8);
                }
            }
        }

        let mut perceptrons = Vec::new();
        for d in &deps {
            perceptrons.push(Perceptron::new(d.len() as u8))
        }

        Self {
            graph,
            perceptrons,
            deps,
            seed,
        }
    }

    pub fn build_simple(&self) -> MLP {
        let mut perceptrons = self.perceptrons.clone();
        let deps = self.deps.clone();
        let graph = self.graph.clone();
        let mut results = Vec::new();
        for _ in 0..perceptrons.len() {
            results.push(0.0);
        }

        // Set parameters for simple mlp
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        for p in &mut perceptrons {
            p.activation_function = step_wise;
            for w in &mut p.weights {
                *w = rng.gen_range(0.0..1.0) as f64;
            }
        }

        MLP {
            perceptrons,
            deps,
            graph,
        }
    }
}

struct MLP {
    perceptrons: Vec<Perceptron>,
    deps: Vec<Vec<u8>>,
    graph: Graph,
}

impl MLP {
    pub fn train(&mut self, input: &[f64], expected_output: &[f64]) {
        let output = self.calc(input);
    }

    pub fn calc(&mut self, input: &[f64]) -> Vec<f64> {
        let mut results: Vec<f64> = Vec::new();
        for _ in 0..self.perceptrons.len() {
            results.push(0.0);
        }
        for (i, p) in &mut self.perceptrons.iter().enumerate() {
            let mut p_input: Vec<f64> = Vec::new();
            let deps = self.deps.get(i).unwrap();
            for d in deps {
                if d < &self.graph.input {
                    p_input.push(*input.get(*d as usize).unwrap());
                } else {
                    p_input.push(*results.get((d - &self.graph.input) as usize).unwrap());
                }
            }
            let o = results.get_mut(i).unwrap();
            *o = p.calc(p_input);
        }

        results.split_off((self.graph.total - self.graph.output - 1) as usize)
    }
}

fn main() {
    // println!("Generating Matrices... ");
    // base::gen_matrices(1, 2);
    // println!("Done");

    let g = base::read_graph(PathBuf::from("graphs/2/1/adjacency_matrix.txt"));
    let mlp_template = MLPTemplate::new(g, 1);
    let mut mlp = mlp_template.build_simple();
    println!("{:?}", mlp.calc(&[0.0]));
}

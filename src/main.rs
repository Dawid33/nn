use rand::{prelude::*, Error};
use std::path::PathBuf;

use base::Graph;
use rand_chacha::ChaCha8Rng;

mod base;

type ActivationFn = fn(f64) -> f64;

#[derive(Clone, Debug)]
struct Perceptron {
    pub weights: Vec<f64>,
    pub activation_function: ActivationFn,
    pub learning_rate: f64,
    pub bias: f64,
}

pub fn identity(x: f64) -> f64 {
    x
}

pub fn sigmoid(x: f64) -> f64 {
    return 1.0 / (1.0 + -x.exp());
}

// compute the derivative of the sigmoid function ASSUMING that the input "x"
// has already been passed through the sigmoid  activation function
pub fn sigmoid_deriv(x: f64) -> f64 {
    return x * (1.0 - x);
}

pub fn linear(x: f64) -> f64 {
    if x >= 0.0 {
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
            learning_rate: 0.01,
            bias: -1.0,
        }
    }

    pub fn calc(&self, input: &[f64]) -> f64 {
        if input.len() != self.weights.len() {
            panic!("Input not equal to number of weights.");
        }

        let mut result: f64 = 0.0;
        for i in 0..self.weights.len() {
            result += self.weights.get(i).unwrap() * input.get(i).unwrap();
        }
        result += self.bias;
        let func = self.activation_function;
        func(result)
    }

    pub fn train(&mut self, input: &[f64], expected_output: f64) {
        let o = self.calc(input);
        for (i, w) in &mut self.weights.iter_mut().enumerate() {
            *w = *w + self.learning_rate * ((expected_output - o) * input.get(i).unwrap());
        }
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
            p.activation_function = sigmoid;
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
        let mut output = self.calc(input);
        let out_nodes = output.split_off((output.len() - self.graph.output as usize) as usize);
        // let mut errors = Vec::new();
        // errors.push(out_nodes.get(i).unwrap() - expected_output.get(i).unwrap());
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
            *o = p.calc(&p_input);
        }

        return results;
    }

    pub fn predict(&mut self, input: &[f64]) -> Vec<f64> {
        let mut results = self.calc(input);
        results.split_off((results.len() - self.graph.output as usize) as usize)
    }

    pub fn _print(&self) {
        for x in &self.perceptrons {
            println!("{:?}", x);
        }
    }
}

fn main() {
    println!("Generating Matrices... ");
    base::gen_matrices(2, 1);
    println!("Done");

    let g = base::read_graph(PathBuf::from("graphs/3/99/adjacency_matrix.txt"));
    let mlp_template = MLPTemplate::new(g, 1);
    let mut mlp = mlp_template.build_simple();
    let dataset = &[
        (&[1.0, 1.0], &[1.0]),
        (&[0.0, 0.0], &[0.0]),
        (&[0.0, 1.0], &[0.0]),
        (&[1.0, 0.0], &[0.0]),
    ];

    for _ in 0..1 {
        for (input, expected) in dataset {
            mlp.train(*input, *expected);
        }
    }

    let input = &[1.0, 1.0];
    let expected = &[1.0];
    let output = mlp.predict(input);
    println!(
        "input: {:?}, expected: {:?}, output: {:?}",
        input, expected, output
    );
}

fn train_perceptron(dataset: &[(&[f64; 2], f64)]) -> Perceptron {
    let mut p = Perceptron::new(2);
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    for w in &mut p.weights {
        *w = rng.gen_range(0.0..1.0) as f64;
    }
    p.activation_function = linear;
    p.learning_rate = 0.01;
    for _ in 0..100 {
        for (input, expected) in dataset {
            p.train(*input, *expected);
        }
    }
    p
}

#[test]
fn test_perceptron_and() {
    let dataset = &[
        (&[1.0, 1.0], 1.0),
        (&[0.0, 0.0], 0.0),
        (&[0.0, 1.0], 0.0),
        (&[1.0, 0.0], 0.0),
    ];
    let p = train_perceptron(dataset);
    assert_eq!(p.calc(&[0.0, 0.0]), 0.0);
    assert_eq!(p.calc(&[1.0, 0.0]), 0.0);
    assert_eq!(p.calc(&[0.0, 1.0]), 0.0);
    assert_eq!(p.calc(&[1.0, 1.0]), 1.0);
}

#[test]
fn test_perceptron_or() {
    let dataset = &[
        (&[1.0, 1.0], 1.0),
        (&[1.0, 0.0], 1.0),
        (&[0.0, 1.0], 1.0),
        (&[0.0, 0.0], 0.0),
    ];
    let p = train_perceptron(dataset);
    assert_eq!(p.calc(&[0.0, 0.0]), 0.0);
    assert_eq!(p.calc(&[1.0, 0.0]), 1.0);
    assert_eq!(p.calc(&[1.0, 1.0]), 1.0);
    assert_eq!(p.calc(&[0.0, 1.0]), 1.0);
}

use rand::{prelude::*, Error};
use std::{collections::VecDeque, iter::zip, path::PathBuf};

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
    return 1.0 / (1.0 + (-x).exp());
}

// compute the derivative of the sigmoid function ASSUMING that the input "x"
// has already been passed through the sigmoid  activation function
pub fn sigmoid_deriv(x: f64) -> f64 {
    return sigmoid(x) * (1.0 - sigmoid(x));
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
            activation_function: sigmoid,
            learning_rate: 0.01,
            bias: 1.0,
        }
    }

    pub fn weighted_input(&self, input: &[f64]) -> f64 {
        if input.len() != self.weights.len() {
            panic!("Input not equal to number of weights.");
        }

        let mut result: f64 = 0.0;
        for i in 0..self.weights.len() {
            result += self.weights.get(i).unwrap() * input.get(i).unwrap();
        }
        result += self.bias;
        result
    }

    pub fn activation(&self, input: &[f64]) -> f64 {
        let f = self.activation_function;
        f(self.weighted_input(input))
    }

    pub fn train(&mut self, input: &[f64], expected_output: f64) {
        let o = self.activation(input);

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
            deps.push(Vec::new());
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
        println!("preceptrons {:?}", perceptrons);

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
            learning_rate: 0.01,
            graph,
        }
    }
}

struct MLP {
    perceptrons: Vec<Perceptron>,
    deps: Vec<Vec<u8>>,
    learning_rate: f64,
    graph: Graph,
}

impl MLP {
    pub fn train(&mut self, dataset: &[(&[f64], &[f64])]) {
        for batch in 0..3 {
            println!("===== BATCH {} =====", batch);
            let mut mae = 0.0;
            let mut cost_deriv_weight: Vec<Vec<f64>> = Vec::with_capacity(self.perceptrons.len());
            let mut cost_deriv_bias: Vec<f64> = Vec::with_capacity(self.perceptrons.len());
            for _ in &self.perceptrons {
                cost_deriv_bias.push(0.0);
                cost_deriv_weight.push(Vec::new());
            }
            for (row, (input, expected_output)) in dataset.iter().enumerate() {
                println!("===== ROW {} ===== {:?}", row, input);
                let mut b = String::new();
                let (all_activations, all_weighted_input) = self.calc(input);
                println!("===== TRAIN =====");
                let mut inner_activations = all_activations.clone();
                let output_activations = inner_activations
                    .split_off((all_activations.len() - self.graph.output as usize) as usize);

                let mut inner_weighted_input = all_weighted_input.clone();
                let output_weighted_input = inner_weighted_input
                    .split_off((all_weighted_input.len() - self.graph.output as usize) as usize);

                {
                    // Compute cost function over all training samples, stored in mae
                    let mut errors = Vec::new();
                    for (o, e) in output_activations.iter().zip(*expected_output) {
                        errors.push((o - e).powi(2));
                    }

                    let mut length = 0.0;
                    for e in errors {
                        length += e;
                    }
                    length = length.sqrt();
                    mae += length.powi(2);
                }

                let mut past_delta_times_weight: Vec<Vec<f64>> = Vec::new();
                for _ in &self.perceptrons {
                    past_delta_times_weight.push(Vec::new());
                }
                let mut node_stack: VecDeque<usize> = VecDeque::new();
                for (i, ((o, e), z)) in output_activations
                    .iter()
                    .zip(*expected_output)
                    .zip(output_weighted_input)
                    .enumerate()
                {
                    let node_index = (self.perceptrons.len() - self.graph.output as usize) + i;

                    // o - e is the derivative of the cost function with respect to
                    // the neuron activations of the last layer
                    let derivative = o - e;
                    let delta = derivative * sigmoid_deriv(z);
                    *cost_deriv_bias.get_mut(node_index).unwrap() += delta;
                    b.push_str(format!("Node: {}, ", node_index).as_str());
                    b.push_str(format!("activation: {}, ", z).as_str());
                    b.push_str(format!("delta: {}, ", delta).as_str());

                    // Get outputs ancestors
                    let out_node_deps = self.deps.get(node_index).unwrap();
                    let mut weights = Vec::new();
                    for (i, a) in out_node_deps.iter().enumerate() {
                        if a < &self.graph.input {
                            weights.push(input.get(*a as usize).unwrap() * delta);
                        } else {
                            weights.push(
                                all_activations
                                    .get((*a - self.graph.input) as usize)
                                    .unwrap()
                                    * delta,
                            );
                            let weight = self
                                .perceptrons
                                .get(node_index)
                                .unwrap()
                                .weights
                                .get(i)
                                .unwrap();
                            past_delta_times_weight
                                .get_mut((*a - self.graph.input) as usize)
                                .unwrap()
                                .push(weight * delta);
                            node_stack.push_back((*a - self.graph.input) as usize);
                        }
                    }

                    let existing_w = cost_deriv_weight.get_mut(node_index).unwrap();
                    if existing_w.is_empty() {
                        *existing_w = weights;
                    } else {
                        for (e, w) in existing_w.into_iter().zip(weights) {
                            *e += w;
                        }
                    }
                    println!("    {}", b);
                    b.clear();
                }

                while let Some(node_index) = node_stack.pop_front() {
                    let weighted_input = *all_weighted_input.get(node_index).unwrap();
                    b.push_str(format!("Node: {}, ", node_index).as_str());
                    b.push_str(format!("weighted_input: {}, ", weighted_input).as_str());

                    // Get outputs of decendants
                    let deltas_from_future = past_delta_times_weight.get(node_index).unwrap();
                    b.push_str(format!("delta x weight: {:?}, ", deltas_from_future).as_str());
                    let mut propagated_error_sum = 0.0;
                    for d in deltas_from_future {
                        propagated_error_sum += *d;
                    }
                    let delta = propagated_error_sum * sigmoid_deriv(weighted_input);
                    *cost_deriv_bias.get_mut(node_index).unwrap() += delta;

                    let out_node_deps = self.deps.get(node_index).unwrap();
                    let mut weights = Vec::new();
                    for (i, a) in out_node_deps.iter().enumerate() {
                        if a < &self.graph.input {
                            weights.push(input.get(*a as usize).unwrap() * delta);
                        } else {
                            weights.push(
                                all_activations
                                    .get((*a - self.graph.input) as usize)
                                    .unwrap()
                                    * delta,
                            );
                            let weight = self
                                .perceptrons
                                .get(node_index)
                                .unwrap()
                                .weights
                                .get(i)
                                .unwrap();
                            past_delta_times_weight
                                .get_mut((a - self.graph.input) as usize)
                                .unwrap()
                                .push(weight * delta);
                            node_stack.push_back((*a - self.graph.input) as usize);
                        }
                    }
                    let existing_w = cost_deriv_weight.get_mut(node_index).unwrap();
                    if existing_w.is_empty() {
                        *existing_w = weights;
                    } else {
                        for (e, w) in existing_w.into_iter().zip(weights) {
                            *e += w;
                        }
                    }
                }
                // Update weights
                // println!("cost_deriv_weight {:?}", cost_deriv_weight);
                // println!("cost_deriv_bias   {:?}", cost_deriv_bias);
                // println!();
                println!("    {}", b);
                b.clear();
            }
            mae /= 2.0 * dataset.len() as f64;
            // println!("cost_deriv_weight {:?}", cost_deriv_weight);
            // println!("cost_deriv_bias   {:?}", cost_deriv_bias);
            println!("mae: {}", mae);

            for (p, (w, b)) in self
                .perceptrons
                .iter_mut()
                .zip(zip(cost_deriv_weight, cost_deriv_bias))
            {
                for (old, new) in p.weights.iter_mut().zip(w) {
                    *old += (self.learning_rate / dataset.len() as f64) * new;
                }
                p.bias += p.bias + ((self.learning_rate / dataset.len() as f64) * b);
            }
            println!("");
        }
    }

    pub fn calc(&mut self, input: &[f64]) -> (Vec<f64>, Vec<f64>) {
        println!("===== CALC =====");
        let mut activations: Vec<f64> = Vec::new();
        let mut weighted_inputs: Vec<f64> = Vec::new();
        for _ in 0..self.perceptrons.len() {
            activations.push(0.0);
            weighted_inputs.push(0.0);
        }
        for (i, p) in &mut self.perceptrons.iter().enumerate() {
            let mut b = String::new();
            b.push_str(format!("Node: {}, ", i).as_str());
            let mut p_input: Vec<f64> = Vec::new();
            let deps = self.deps.get(i).unwrap();
            for d in deps {
                if d < &self.graph.input {
                    p_input.push(*input.get(*d as usize).unwrap());
                } else {
                    p_input.push(*activations.get((d - &self.graph.input) as usize).unwrap());
                }
            }
            let o = activations.get_mut(i).unwrap();
            *o = p.activation(&p_input);
            b.push_str(format!("Inputs: {:?}, ", p_input).as_str());
            b.push_str(format!("Outputs: {}", o).as_str());
            let o = weighted_inputs.get_mut(i).unwrap();
            *o = p.weighted_input(&p_input);
            println!("{}", b);
        }
        return (activations, weighted_inputs);
    }

    pub fn predict(&mut self, input: &[f64]) -> Vec<f64> {
        let (mut results, _) = self.calc(input);
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

    let g = base::read_graph(PathBuf::from("graphs/1/1/adjacency_matrix.txt"));
    let mlp_template = MLPTemplate::new(g, 1);
    let mut mlp = mlp_template.build_simple();
    let dataset = &[
        (&[1.0, 1.0][..], &[1.0][..]),
        (&[0.0, 0.0][..], &[0.0][..]),
        (&[0.0, 1.0][..], &[0.0][..]),
        (&[1.0, 0.0][..], &[0.0][..]),
    ];

    mlp.train(dataset);
    let input = &[1.0, 1.0];
    let expected = &[1.0];
    let output = mlp.predict(input);
    println!(
        "input: {:?}, expected: {:?}, output: {:?}",
        input, expected, output
    );
    let input = &[1.0, 0.0];
    let expected = &[0.0];
    let output = mlp.predict(input);
    println!(
        "input: {:?}, expected: {:?}, output: {:?}",
        input, expected, output
    );
    let input = &[0.0, 1.0];
    let expected = &[0.0];
    let output = mlp.predict(input);
    println!(
        "input: {:?}, expected: {:?}, output: {:?}",
        input, expected, output
    );
    let input = &[0.0, 0.0];
    let expected = &[0.0];
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
    assert_eq!(p.activation(&[0.0, 0.0]), 0.0);
    assert_eq!(p.activation(&[1.0, 0.0]), 0.0);
    assert_eq!(p.activation(&[0.0, 1.0]), 0.0);
    assert_eq!(p.activation(&[1.0, 1.0]), 1.0);
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
    assert_eq!(p.activation(&[0.0, 0.0]), 0.0);
    assert_eq!(p.activation(&[1.0, 0.0]), 1.0);
    assert_eq!(p.activation(&[1.0, 1.0]), 1.0);
    assert_eq!(p.activation(&[0.0, 1.0]), 1.0);
}

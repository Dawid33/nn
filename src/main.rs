use rand::prelude::*;
use std::fs::OpenOptions;
use std::io::Write;
use std::{collections::VecDeque, iter::zip, path::PathBuf};

use base::Graph;
use rand_chacha::ChaCha8Rng;

mod base;

// http://neuralnetworksanddeeplearning.com/chap2.html
fn dumpln(s: &str) {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("out.txt")
        .unwrap();
    write!(file, "{}\n", s).unwrap();
}

type ActivationFn = fn(f64) -> f64;

#[derive(Clone, Debug)]
struct Perceptron {
    pub weights: Vec<f64>,
    pub activation_function: ActivationFn,
    pub activation_partial_dervivative: ActivationFn,
    pub learning_rate: f64,
    pub bias: f64,
}

pub fn identity(x: f64) -> f64 {
    x
}

pub fn sigmoid(x: f64) -> f64 {
    return 1.0 / (1.0 + (-x).exp());
}

pub fn sigmoid_deriv(x: f64) -> f64 {
    return sigmoid(x) * (1.0 - sigmoid(x));
}

pub fn linear(x: f64) -> f64 {
    if x >= 0.0 {
        x
    } else {
        0.0
    }
}

pub fn linear_deriv(x: f64) -> f64 {
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
            activation_function: linear,
            activation_partial_dervivative: linear_deriv,
            learning_rate: 0.9,
            bias: -1.0,
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

        // Build a list of nodes that each node depends on i.e. all the nodes pointing to that node.
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
            p.activation_partial_dervivative = sigmoid_deriv;
            for w in &mut p.weights {
                *w = rng.gen();
            }
        }

        let len = perceptrons.len();
        for p in &mut perceptrons[len - self.graph.output as usize..len] {
            p.activation_function = sigmoid;
            p.activation_partial_dervivative = sigmoid_deriv;
        }

        MLP {
            perceptrons,
            deps,
            learning_rate: 0.5,
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

#[derive(Debug)]
struct WeightChange {
    pub weight: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
}

impl WeightChange {
    pub fn new(perceptrons: &Vec<Perceptron>) -> Self {
        let mut weight: Vec<Vec<f64>> = Vec::with_capacity(perceptrons.len());
        let mut bias: Vec<f64> = Vec::with_capacity(perceptrons.len());
        for i in 0..perceptrons.len() {
            bias.push(0.0);
            let mut weights = Vec::new();
            for _ in 0..perceptrons.get(i).unwrap().weights.len() {
                weights.push(0.0);
            }
            weight.push(weights);
        }
        Self { weight, bias }
    }
    pub fn add(&mut self, other: WeightChange) {
        for (weights, other_weights) in self.weight.iter_mut().zip(other.weight.iter()) {
            for (w, other_w) in weights.iter_mut().zip(other_weights.iter()) {
                *w += other_w;
            }
        }
        for (b, other_b) in self.bias.iter_mut().zip(other.bias.iter()) {
            *b += other_b;
        }
    }
}

impl MLP {
    // Goal of backprop is to get the rate of change of each weight in the
    // neural network.
    fn backprop(&mut self, input: &[f64], expected: &[f64]) -> (WeightChange, f64) {
        let mut change = WeightChange::new(&self.perceptrons);
        let (activations, weighted_inputs) = self.calc(input);

        // Get a list of activations from the output nodes and a list from all other nodes.
        let mut inner_activations = activations.clone();
        let output_activations =
            inner_activations.split_off((activations.len() - self.graph.output as usize) as usize);

        // Get a list of weighted inputs from the output nodes and a list from all other nodes.
        let mut inner_weighted_input = weighted_inputs.clone();
        let output_weighted_input = inner_weighted_input
            .split_off((weighted_inputs.len() - self.graph.output as usize) as usize);

        // Keep a stack of past weight changes
        let mut past_error_times_weight: Vec<Vec<f64>> = Vec::new();
        for _ in &self.perceptrons {
            past_error_times_weight.push(Vec::new());
        }

        let partial_mae = {
            // Compute cost function over all training samples, stored in mae
            let mut total = 0.0;
            for (e, o) in expected.iter().zip(output_activations.iter()) {
                total += (e - o).powi(2);
            }
            total
        };

        let mut node_stack: VecDeque<usize> = VecDeque::new();

        let apply_error_for_weight_change = |node_index: usize,
                                             error: f64,
                                             node_stack: &mut VecDeque<usize>,
                                             past_error_times_weight: &mut Vec<Vec<f64>>|
         -> Vec<f64> {
            let node_deps = self.deps.get(node_index).unwrap();
            let mut weight_change = Vec::new();
            for (i, ancestor) in node_deps.iter().enumerate() {
                // If the ancestor is an input to the neural network then we just update that
                // weight, otherwise we also keep track of this nodes delta and push the ancestor
                // onto the stack for processing.
                if ancestor < &self.graph.input {
                    weight_change.push(input.get(*ancestor as usize).unwrap() * error);
                } else {
                    let ancestor_node_index = (*ancestor - self.graph.input) as usize;
                    weight_change.push(activations.get(ancestor_node_index).unwrap() * error);
                    // Get the weight that connects the ancestors output to the current node
                    let weight = self
                        .perceptrons
                        .get(node_index)
                        .unwrap()
                        .weights
                        .get(i)
                        .unwrap();

                    past_error_times_weight
                        .get_mut(ancestor_node_index)
                        .unwrap()
                        .push(weight * error);

                    node_stack.push_back(ancestor_node_index);
                }
            }
            weight_change
        };

        // Process output nodes first
        for (i, ((activation, expected_activation), weighted_output)) in output_activations
            .iter()
            .zip(expected)
            .zip(output_weighted_input)
            .enumerate()
        {
            // Index of the current perceptron
            let node_index = (self.perceptrons.len() - self.graph.output as usize) + i;

            let partial_deriv = self
                .perceptrons
                .get(node_index)
                .unwrap()
                .activation_partial_dervivative;

            // This is the derivative of the cost function with respect to
            // the neuron activations of the last layer
            let cost_deriv = activation - expected_activation;
            // This is the error of the neuron.
            let error = cost_deriv * partial_deriv(weighted_output);
            dumpln(&format!(
                "Node {}:  Last layer error: {}",
                node_index + self.graph.input as usize,
                error,
            ));
            change.bias[node_index] = error;
            change.weight[node_index] = apply_error_for_weight_change(
                node_index,
                error,
                &mut node_stack,
                &mut past_error_times_weight,
            );
        }

        // Backpropagate the error, processing each ancestor.
        while let Some(node_index) = node_stack.pop_front() {
            // Weighted input for this current node.
            let weighted_input = *weighted_inputs.get(node_index).unwrap();
            let partial_deriv = self
                .perceptrons
                .get(node_index)
                .unwrap()
                .activation_partial_dervivative;

            let errors_from_decendants = past_error_times_weight.get(node_index).unwrap();
            let mut total = 0.0;
            for e in errors_from_decendants {
                total += e;
            }
            let error = total * partial_deriv(weighted_input);
            dumpln(&format!(
                "Node {}:  Inner layer error: {}",
                node_index + self.graph.input as usize,
                error,
            ));
            change.bias[node_index] = error;
            change.weight[node_index] = apply_error_for_weight_change(
                node_index,
                error,
                &mut node_stack,
                &mut past_error_times_weight,
            );
        }
        return (change, partial_mae);
    }

    pub fn train_batch(&mut self, batch: &[(&[f64], &[f64])]) -> f64 {
        // Computes the amount of change that needs to be applied to each perceptron
        // based on this batch by using stochastic gradient descent.
        let mut change = WeightChange::new(&self.perceptrons);
        let mut mae = 0.0;
        for (_, (input, expected)) in batch.iter().enumerate() {
            let (batch_change, partial_mae) = self.backprop(input, expected);
            mae += partial_mae;
            dumpln(&format!("Weights delta {:?}", batch_change.weight));
            change.add(batch_change);
        }
        mae /= 2.0 * batch.len() as f64;
        dumpln(&format!("Total weights delta {:?}", change.weight));
        dumpln(&format!("Total bias delta {:?}", change.bias));
        dumpln(&format!("BATCH MAE: {}", mae));

        // Apply updates to each perceptron
        for (perceptron, (weights_change, bias_change)) in self
            .perceptrons
            .iter_mut()
            .zip(zip(change.weight, change.bias))
        {
            for (old_weight, weight_change) in perceptron.weights.iter_mut().zip(weights_change) {
                *old_weight += (self.learning_rate / batch.len() as f64) * weight_change;
            }
            perceptron.bias += (self.learning_rate / batch.len() as f64) * bias_change;
        }

        dumpln("");
        return mae;
    }

    pub fn calc(&mut self, input: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut activations: Vec<f64> = Vec::new();
        let mut weighted_inputs: Vec<f64> = Vec::new();
        for _ in 0..self.perceptrons.len() {
            activations.push(0.0);
            weighted_inputs.push(0.0);
        }
        for (i, p) in &mut self.perceptrons.iter().enumerate() {
            let mut b = String::new();
            b.push_str(format!("Node: {}, ", i + self.graph.input as usize).as_str());
            let mut perceptron_input: Vec<f64> = Vec::new();
            // Get and iterate over a nodes ancestors in order to find the inputs to the current
            // node.
            let deps = self.deps.get(i).unwrap();
            for d in deps {
                if d < &self.graph.input {
                    perceptron_input.push(*input.get(*d as usize).unwrap());
                } else {
                    perceptron_input
                        .push(*activations.get((d - &self.graph.input) as usize).unwrap());
                }
            }
            let o = activations.get_mut(i).unwrap();
            *o = p.activation(&perceptron_input);
            b.push_str(format!("Inputs: {:?}, ", perceptron_input).as_str());
            b.push_str(format!("Outputs: {}", o).as_str());
            let o = weighted_inputs.get_mut(i).unwrap();
            *o = p.weighted_input(&perceptron_input);
            // dumpln(&b);
        }
        // dumpln("");
        return (activations, weighted_inputs);
    }

    pub fn predict(&mut self, input: &[f64]) -> Vec<f64> {
        let (mut results, _) = self.calc(input);
        results.split_off((results.len() - self.graph.output as usize) as usize)
    }

    pub fn dump_weights(&self) {
        for (i, x) in self.perceptrons.iter().enumerate() {
            dumpln(&format!("{}: {:?}", i, x.weights));
        }
    }
}

fn main() {
    if std::path::PathBuf::from("out.txt").exists() {
        std::fs::remove_file("out.txt").unwrap();
    }
    println!("Generating Matrices... ");
    base::gen_matrices(2, 1);
    println!("Done");

    let g = base::read_graph(PathBuf::from("graphs/2/15/adjacency_matrix.txt"));
    let mlp_template = MLPTemplate::new(g, 1);
    let mut mlp = mlp_template.build_simple();
    let dataset = &[
        (&[1.0, 1.0][..], &[0.0][..]),
        (&[0.0, 0.0][..], &[0.0][..]),
        (&[0.0, 1.0][..], &[1.0][..]),
        (&[1.0, 0.0][..], &[1.0][..]),
    ];

    for _ in 0..100 {
        mlp.train_batch(dataset);
    }

    let input = &[0.0, 1.0];
    let expected = &[1.0];
    let output = mlp.predict(input);
    println!(
        "input: {:?}, expected: {:?}, output: {:?}",
        input, expected, output
    );

    let input = &[1.0, 0.0];
    let expected = &[1.0];
    let output = mlp.predict(input);
    println!(
        "input: {:?}, expected: {:?}, output: {:?}",
        input, expected, output
    );

    let input = &[1.0, 1.0];
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

mod unit_tests {
    use super::*;
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
}

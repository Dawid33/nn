use num_bigint::BigUint;
use simple_error::SimpleError;

use std::borrow::Cow;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

type Node = isize;
type Edge = (isize, isize);
pub struct Edges {
    edges: Vec<Edge>,
    input: u64,
    output: u64,
    total: u64,
}

#[derive(Clone)]
pub struct Graph {
    pub adjacency_matrix: Vec<Vec<u8>>,
    pub input: u64,
    pub output: u64,
    pub total: u64,
}

pub fn render<W: Write>(g: &Graph, output: &mut W) {
    let mut e = Vec::new();
    for (x, rows) in g.adjacency_matrix.iter().enumerate() {
        for (y, value) in rows.iter().enumerate() {
            if *value > 0 {
                e.push((y as isize, x as isize));
            }
        }
    }
    dot::render(
        &Edges {
            edges: e,
            input: g.input,
            output: g.output,
            total: g.total,
        },
        output,
    )
    .unwrap()
}

impl<'a> dot::Labeller<'a, Node, Edge> for Edges {
    fn graph_id(&'a self) -> dot::Id<'a> {
        dot::Id::new("dag").unwrap()
    }

    fn node_id(&'a self, n: &Node) -> dot::Id<'a> {
        if *n < self.input as isize {
            return dot::Id::new(format!("in{}", n)).unwrap();
        } else if *n >= (self.total - self.output) as isize {
            return dot::Id::new(format!("N{}_out", n)).unwrap();
        } else {
            return dot::Id::new(format!("N{}", n)).unwrap();
        }
    }
}

impl<'a> dot::GraphWalk<'a, Node, Edge> for Edges {
    fn nodes(&self) -> dot::Nodes<'a, Node> {
        let mut nodes = Vec::with_capacity(self.edges.len());
        for (s, t) in &self.edges {
            nodes.push(*s);
            nodes.push(*t);
        }
        nodes.sort();
        nodes.dedup();
        Cow::Owned(nodes)
    }

    fn edges(&'a self) -> dot::Edges<'a, Edge> {
        Cow::Borrowed(&self.edges[..])
    }

    fn source(&self, e: &Edge) -> Node {
        e.0
    }

    fn target(&self, e: &Edge) -> Node {
        e.1
    }
}

pub fn save_graph(path: PathBuf, g: &Graph) {
    let mut result = String::from(format!("{} {}\n", g.input, g.output));

    for x in &g.adjacency_matrix {
        let mut iter = x.iter();

        if let Some(value) = iter.next() {
            result.push_str(format!("{}", value).as_str());
        }

        while let Some(value) = &iter.next() {
            result.push_str(format!(" {}", value).as_str());
        }
        result.push('\n');
    }

    std::fs::write(path, result).unwrap();
}

pub fn read_graph(path: PathBuf) -> Graph {
    let raw = std::fs::read_to_string(path).unwrap();
    let mut matrix: Vec<Vec<u8>> = Vec::new();
    let mut iter = raw.split('\n');

    let mut input = 0;
    let mut output = 0;

    if let Some(start_line) = &iter.next() {
        let mut nums = start_line.split(' ');
        if let Some(value) = nums.next() {
            input = u64::from_str_radix(value, 10).unwrap();
        }

        if let Some(value) = nums.next() {
            output = u64::from_str_radix(value, 10).unwrap();
        }
    }

    for line in iter {
        if line.is_empty() {
            break;
        }

        let mut row = Vec::new();
        for value in line.split(' ') {
            row.push(u8::from_str_radix(value, 10).unwrap());
        }
        matrix.push(row);
    }

    Graph {
        total: matrix.len() as u64,
        adjacency_matrix: matrix,
        input,
        output,
    }
}

pub struct GraphDealer {}

impl GraphDealer {
    pub fn new() -> Self {
        Self {}
    }
    pub fn get_next_graph(&mut self) -> Graph {}
}

// Step 1: Get all possible permutations of a graph by getting getting the
// number of cells, incrementing a number from zero up to that number of cells
// in bits and setting the lower triangle with that bit representation.
//
// Step 2: Set the bits, checking for edge cases that make the graph invalid.
// - inputs cannot connect to inputs
// - inputs cannot connect directly to outputs
//
// Step 3: Check if the graph is connected and reject it if its not.
pub fn gen_matrices(input: u64, output: u64) {
    for i in 1..2 {
        std::fs::create_dir_all(format!("graphs/{}", i)).unwrap();
        let path = PathBuf::from(format!("graphs/{}", i));
        println!("Generating graphs with {} inner nodes", i);
        if let Err(_) = gen_matrices_inner(input, i, output, path) {
            break;
        }
    }
}
pub fn gen_matrices_inner(
    input: u64,
    inner: u64,
    output: u64,
    path: PathBuf,
) -> Result<(), Box<dyn Error>> {
    let n = inner + input + output;

    if input < 1 {
        // panic!("Not enough input vertices specified.");
    }

    if output < 1 {
        panic!("Not enough output vertices specified.");
    }

    // Initialize an n x n  matrix
    let mut empty: Vec<Vec<u8>> = Vec::new();
    let mut row: Vec<u8> = Vec::new();
    for _ in 0..n {
        row.push(0);
    }
    for _ in 0..n {
        empty.push(row.clone());
    }

    // Calculate the max number of edges in the graph, i.e. the number of
    // cells in the lower triangle of the matrix
    let mut edges = 0;
    for i in 1..n {
        edges += i;
    }

    // The total num of matrices is all the permuatations of the edges.
    let mut num_of_matrices = BigUint::from_slice(&[1]);
    num_of_matrices = num_of_matrices << edges as u32;
    let mut index = BigUint::from_slice(&[0]);

    let mut actually_correct_cnt = 1;
    let matrix_creation_start = std::time::Instant::now();

    'outer: loop {
        if index >= &num_of_matrices - 1 as u32 {
            break;
        }
        index += 1 as u32;

        // if std::time::Instant::now().duration_since(matrix_creation_start)
        //     > std::time::Duration::from_secs(30)
        // {
        //     println!(
        //         "Matrix Creation for {} inner nodes is taking more than 30 seconds, aborting.",
        //         inner
        //     );
        //     return Err(Box::new(SimpleError::new(
        //         "Took too long generating matrices.",
        //     )));
        // }

        let mut m = empty.clone();
        let mut cnt = 0;

        // Iterate over the lower triangle and set it to the current permutation
        // of bits in `index`
        for x in 1..n {
            for y in 0..x {
                let cell = m.get_mut(x as usize).unwrap().get_mut(y as usize).unwrap();
                if index.bit(cnt) {
                    *cell = 1;
                } else {
                    *cell = 0;
                }

                // // Check if we are processing the edge of an input node.
                if y < input {
                    // If input is trying to connect with an output OR if input
                    // is trying to connect with another input then we abort
                    // this permutation.

                    // x >= n instead of x > n because x is zero indexed whereas
                    // n - output starts at 1 since it n denotes the number
                    // of vertices.

                    if (x >= n - output || x < input) && *cell == 1 {
                        continue 'outer;
                    }
                }
                if x >= n - output {
                    // if output is trying to connect with another output then
                    // we abort this permutation.
                    if y >= n - output && *cell == 1 {
                        continue 'outer;
                    }
                }
                cnt += 1;
            }
        }

        // Search over the graph such that we start the search from every input
        // vertex and finish at some output vertex. Check off which nodes have
        // been traversed during this search and if its < n then the graph isn't
        // connected.

        // TODO: Important! make sure that there are some nodes in common
        //       between all paths from input to some output. It is otherwise
        //       possible to have a disconnected graph where each input reaches
        //       some output but the paths don't cross.

        // Start from all input vertices
        let mut traversed_nodes: Vec<usize> = Vec::new();
        let mut open_nodes: Vec<usize> = Vec::new();
        for i in 0..input {
            open_nodes.push(i as usize);
        }

        while let Some(node) = open_nodes.pop() {
            // let row = m.get(node).unwrap();
            let mut has_something = false;
            for (i, row) in m.iter().enumerate() {
                let cell = row.get(node).unwrap();
                if *cell == 1 {
                    has_something = true;
                    if !open_nodes.contains(&i) {
                        open_nodes.push(i);
                    }
                }
            }

            // If node has no outward connections and it is not an output
            // node then we CULL THE UNCONNECTED
            if !has_something && node < (n - output) as usize {
                continue 'outer;
            }

            if !traversed_nodes.contains(&node) {
                traversed_nodes.push(node);
            }
        }

        if traversed_nodes.len() < n as usize {
            continue 'outer;
        }

        // Dump matrix to file system
        {
            let path = path.join(format!("{}", actually_correct_cnt));
            actually_correct_cnt += 1;
            std::fs::create_dir_all(&path).unwrap();

            let g = Graph {
                adjacency_matrix: m,
                input,
                output,
                total: n,
            };

            let graph_path = path.join("adjacency_matrix.txt");
            save_graph(graph_path, &g);

            let graph_path = path.join(format!("graph.dot"));
            if graph_path.exists() {
                continue 'outer;
            }
            let mut f = File::create(graph_path).unwrap();
            render(&g, &mut f);
        }
    }
    // let path = path.join("index.txt");
    // std::fs::write(
    //     path,
    //     format!("total_graphs={}", actually_correct_cnt).as_bytes(),
    // )
    // .unwrap();
    return Ok(());
}

pub fn print_matrix(m: &Vec<Vec<u8>>) {
    for x in m {
        for y in x {
            print!(" {}", y);
        }
        println!("");
    }
}

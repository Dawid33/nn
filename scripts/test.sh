#!/bin/bash

# cargo test --color always 2>&1 | less -R +F 
cargo run --color always xor.csv result 2>&1 | less -R +F

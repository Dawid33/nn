#!/bin/bash

cargo test --color on 2>&1 | less -R +F 

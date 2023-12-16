#!/bin/bash

cargo build 2>&1 | less -R +F

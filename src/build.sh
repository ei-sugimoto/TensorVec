#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: No input file provided. Please specify a CUDA file to compile."
    exit 1
fi
nvcc --std c++17 --expt-relaxed-constexpr -I/workspace/lib/include -o a $1

./a

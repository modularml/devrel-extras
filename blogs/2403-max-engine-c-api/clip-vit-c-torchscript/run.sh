#!/bin/bash

set -e

MAX_PKG_DIR="$(modular config max.path)"
export MAX_PKG_DIR

CURRENT_DIR=$(dirname "$0")
MODEL_PATH="models/clip_vit.torchscript"

# download and convert clip-vit model to torchscript
python3 clip_vit_torchscript.py

# inputs
TEXT="a photo of a cat,a photo of a dog"
echo "text input: $TEXT"
URL="http://images.cocodataset.org/val2017/000000039769.jpg"
echo "image url $URL"

# if have imagemagick installed
# wget $URL && display "000000039769.jpg"

# preprocess inputs
python3 preprocess.py --text "$TEXT"  --url "$URL"

# Build
cmake -B build -S "$CURRENT_DIR"
cmake --build build

# Compile and run the model
./build/clip-vit "$MODEL_PATH"

# check for memory leak with valgrind
# valgrind --leak-check=full ./build/clip-vit "$MODEL_PATH"

# Post process
python3 postprocess.py

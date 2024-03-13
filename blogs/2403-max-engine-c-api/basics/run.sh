set -e

# set MAX path
MAX_PKG_DIR="$(modular config max.path)"
export MAX_PKG_DIR

CURRENT_DIR=$(dirname "$0")

# Build the example
cmake -B build -S "$CURRENT_DIR"
cmake --build build

# Run example
./build/basics

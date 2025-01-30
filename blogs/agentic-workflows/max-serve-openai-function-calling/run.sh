#!/bin/bash

PROGRAM=$1
declare -a SERVICE_PIDS
CLEANUP_DONE=0

cleanup() {
    if [ $CLEANUP_DONE -eq 1 ]; then
        return
    fi
    CLEANUP_DONE=1

    echo -e "\nShutting down services..."
    for pid in "${SERVICE_PIDS[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            kill -SIGTERM $pid
            sleep 2 # Give time for graceful shutdown
        fi
    done

    # Kill any remaining process on port 8000
    local PIDS=$(lsof -t -i:8000)
    if [ ! -z "$PIDS" ]; then
        kill -9 $PIDS 2>/dev/null || true
    fi

    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        echo "Cleaning up GPU processes..."
        nvidia-smi pmon -c 1 | grep python | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
    fi
}

setup_max() {
    if [ -d "max" ]; then
        cd max || exit 1
        git fetch
        git checkout 5d7caad6d463d29ef911df7a48b22d4cf615d284
        cd .. || exit 1
    else
        git clone https://github.com/modular/max || exit 1
        cd max || exit 1
        git checkout 5d7caad6d463d29ef911df7a48b22d4cf615d284 || exit 1
        cd .. || exit 1
    fi
}

trap cleanup SIGINT SIGTERM

setup_max
cd max/pipelines/python || exit 1

# Check for GPU and set command accordingly
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    echo "GPU detected, using GPU mode..."
    magic run serve --huggingface-repo-id=modularai/llama-3.1 --use-gpu --max-length=2048 &
else
    echo "No GPU detected, using CPU mode..."
    magic run serve --huggingface-repo-id=modularai/llama-3.1 --max-length=2048 &
fi

SERVICE_PIDS+=($!)
cd ../../.. || exit 1

echo "Waiting for MAX server to be ready..."
while ! curl -s "http://0.0.0.0:8000/v1/health" >/dev/null; do sleep 2; done

echo "MAX server is ready!"

# Run the program based on type
if [ "$PROGRAM" = "app.py" ]; then
    uvicorn app:app --port 8001 --host 0.0.0.0 --reload
else
    python "$PROGRAM"
fi

cleanup

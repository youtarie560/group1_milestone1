#!/bin/bash

echo "Building docker image..."
docker build -t ift6758-serving -f Dockerfile.serving .

# to adjust build.sh for part 6:
docker build -t ift6758-streamlit -f Dockerfile.streamlit .
#!/bin/bash
echo "Running Streamlit container..."
docker run -p 8501:8501 \
  -e BACKEND_URL="http://host.docker.internal:8080" \
  ift6758-streamlit

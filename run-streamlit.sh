#!/bin/bash
echo "Running Streamlit container..."

# Note: We use 'host.docker.internal' to allow this container to talk
# to the Serving container running on localhost:8080 on your machine.
# otherwise try host.docker.internal
# otherwise try serving
docker run -p 8501:8501 \
  -e BACKEND_URL="http://host.docker.internal:8080" \
  ift6758-streamlit
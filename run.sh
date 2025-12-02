#!/bin/bash
echo "Running docker container..."
docker run -p 8080:8080 -e API_KEY=${API_KEY} ift6758-serving

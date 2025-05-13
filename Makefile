.PHONY: env format clean lint

# Environment name - change as needed
ENV_NAME := beer_pong_env

# Create and configure conda environment
env:
	pip install -r requirements.txt
	curl -o pose_landmarker.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
	
# Format code using black and isort
format:
	black .
	isort .

# Run ruff with auto-fix capabilities
lint:
	ruff check --fix src
	ruff format src

# Clean up Python cache files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete

# Default target
all: format lint 
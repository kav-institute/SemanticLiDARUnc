# Semantic LiDAR training tool 

Tool for semantic lidar model training.

## Development environment:

### VS-Code:
The project is designed to be delevoped within vs-code IDE using remote container development.

## Usage:
### 1. Using docker compose
In docker-compse.yaml all parameters are defined.
```bash
# Add user to environment
sh setup.sh

# Build the image from scratch using Dockerfile, can be skipped if image already exists or is loaded from docker registry
docker-compose build --no-cache

# Start the container
docker-compose up -d

# Stop the container
docker compose down
```



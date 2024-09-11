# Use an official Python runtime as a parent image
FROM python:3.12-slim AS builder

# Set environment variables
# Default number of workers
ENV PYTHONUNBUFFERED=1
ENV UVICORN_WORKERS=4

# Set the working directory in the container
WORKDIR /app

# Install essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy the requirements file to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 5001 to the outside world
EXPOSE 5001

# Command to run your application with Uvicorn using environment variable for workers
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 5001 --workers ${UVICORN_WORKERS}"]


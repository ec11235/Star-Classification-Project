# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Create an output directory if it doesn't already exist
RUN mkdir -p /app/output

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run the tuning script to generate the best_params.json file
RUN python -m model.tuning

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run the visualization script when the container launches
CMD ["python", "-m", "visualization.visualizations"]




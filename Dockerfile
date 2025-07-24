# Challenge 1B: Multi-Collection PDF Analysis Docker Image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY challenge1b_processor.py .

# Create directory structure for collections
RUN mkdir -p /app/Collection_1/PDFs \
    && mkdir -p /app/Collection_2/PDFs \
    && mkdir -p /app/Collection_3/PDFs

# Copy input configurations
COPY Collection\ 1/challenge1b_input.json /app/Collection_1/
COPY Collection\ 2/challenge1b_input.json /app/Collection_2/
COPY Collection\ 3/challenge1b_input.json /app/Collection_3/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the processor
CMD ["python", "challenge1b_processor.py"]

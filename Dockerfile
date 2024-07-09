FROM continuumio/miniconda3

WORKDIR /app

# Install dependencies
COPY requirements.yaml .
RUN conda env create -f requirements.yaml
SHELL ["conda", "run", "-n", "fastapi-recommendation-system", "/bin/bash", "-c"]

# Install additional packages
RUN pip install uvicorn

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application with uvicorn
CMD ["conda", "run", "--no-capture-output", "-n", "fastapi-recommendation-system", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

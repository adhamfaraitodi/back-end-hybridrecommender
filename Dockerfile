FROM continuumio/miniconda3
WORKDIR /app
COPY requirements.yaml .
RUN conda env create -f requirements.yaml
SHELL ["conda", "run", "-n", "fastapi-recommendation-system", "/bin/bash", "-c"]
RUN pip install uvicorn
COPY . .
EXPOSE 8000
CMD ["conda", "run", "--no-capture-output", "-n", "fastapi-recommendation-system", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

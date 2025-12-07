FROM python:3.13-slim-bullseye

# note: install system dependencies
RUN apt-get update 
RUN apt-get install openjdk-17-jdk --yes

WORKDIR /bitcoinheist-app

# note: install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# note: make project importable
ENV PYTHONPATH=/bitcoinheist-app

# note: project code
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY tests/ tests/
COPY models/.placeholder models/.placeholder

# note: include data directory with the raw CSV
COPY data/ data/

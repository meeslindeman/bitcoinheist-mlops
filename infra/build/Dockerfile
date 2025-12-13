FROM python:3.13-slim-bullseye

# note: install system dependencies
RUN apt-get update && \
# note: curl is needed for healthchecks
  apt-get install -y curl openjdk-17-jdk

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

EXPOSE 5001

#note: execute the Python API app (running with Flask)
CMD ["flask", "--app", "src.app.main_api", "run",  "--host", "0.0.0.0", "--port", "5001"]

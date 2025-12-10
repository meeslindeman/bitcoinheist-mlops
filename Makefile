.PHONY: test format build build-nocache up up-app down clean logs-app test-integration

# note: unit tests and coverage for src/ only (local tests)
test:
	coverage run -m pytest -v tests/unit -p no:warnings .
	coverage report --rcfile=.coveragerc

format:
	ruff format . --line-length 120

build:
	docker build . -t bitcoinheist-app -f infra/build/Dockerfile

build-nocache:
	docker build --no-cache . -t bitcoinheist-app -f infra/build/Dockerfile

# note: bring up full stack 
up: build
	docker compose --file infra/docker-compose.yaml down
	docker compose --file infra/docker-compose.yaml up -d mlflow airflow prometheus pushgateway grafana app

down:
	docker compose --file infra/docker-compose.yaml down

clean:
	docker compose --file infra/docker-compose.yaml down -v

logs-app:
	docker compose --file infra/docker-compose.yaml logs -f app

# note: integration tests run inside the app container
test-integration:
	docker compose --file infra/docker-compose.yaml exec app pytest -v tests/integration -p no:warnings



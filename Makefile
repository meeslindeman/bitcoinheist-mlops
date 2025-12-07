.PHONY: test format build build-nocache up up-app down clean logs-app

test:
	coverage run -m pytest -v -p no:warnings .
	coverage report --rcfile=.coveragerc

format:
	ruff format . --line-length 120

build:
	docker build . -t bitcoinheist-app -f infra/build/Dockerfile

build-nocache:
	docker build --no-cache . -t bitcoinheist-app -f infra/build/Dockerfile

up: build
	docker compose --file infra/docker-compose.yaml down
	docker compose --file infra/docker-compose.yaml up -d mlflow airflow prometheus grafana

up-app:
	docker compose --file infra/docker-compose.yaml up -d app

down:
	docker compose --file infra/docker-compose.yaml down

clean:
	docker compose --file infra/docker-compose.yaml down -v

logs-app:
	docker compose --file infra/docker-compose.yaml logs -f app

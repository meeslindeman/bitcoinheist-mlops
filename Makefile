.PHONY: test build build-nocache up drift test-integration down clean enter-app logs-app 

# note: unit tests and coverage for src/ only (local tests)
test:
	coverage run -m pytest -v tests/unit -p no:warnings .
	coverage report --rcfile=.coveragerc

build:
	docker build . -t bitcoinheist-app -f infra/build/Dockerfile

build-nocache:
	docker build --no-cache . -t bitcoinheist-app -f infra/build/Dockerfile

# note: bring up full stack 
up: build
	docker compose --file infra/docker-compose.yaml down
	docker compose --file infra/docker-compose.yaml up -d mlflow airflow prometheus pushgateway grafana app

drift:
	docker compose --file infra/docker-compose.yaml exec app \
		sh -c "python -m src.telemetry.mock_live_data && python -m src.telemetry.drift_calculation"

# note: integration tests run inside the app container
test-integration:
	docker compose --file infra/docker-compose.yaml exec app pytest -v tests/integration -p no:warnings

down:
	docker compose --file infra/docker-compose.yaml down

# note: usefull commands
clean:
	docker compose --file infra/docker-compose.yaml down -v

logs-app:
	docker compose --file infra/docker-compose.yaml logs -f app

enter-app:
	docker compose --file infra/docker-compose.yaml exec app sh




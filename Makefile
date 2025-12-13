.PHONY: test-integration test-unit build build-nocache up drift up down build-nocache restart clean logs-app enter-app	 

build:
	docker build . -t bitcoinheist-app -f infra/build/Dockerfile

build-nocache:
	docker build --no-cache . -t bitcoinheist-app -f infra/build/Dockerfile

# note: bring up full stack 
up:
	docker compose -f infra/docker-compose.yaml up -d

# note: bring down full stack 
down:
	docker compose -f infra/docker-compose.yaml down

restart:
	docker compose -f infra/docker-compose.yaml restart

drift:
	docker compose --file infra/docker-compose.yaml exec app \
		sh -c "python -m src.telemetry.mock_live_data && python -m src.telemetry.drift_calculation"

test-unit:
	docker compose --file infra/docker-compose.yaml exec app sh -c 'coverage run -m pytest -v tests/unit -p no:warnings . && coverage report --rcfile=.coveragerc'

test-integration:
	docker compose --file infra/docker-compose.yaml exec app sh -c 'pytest -v tests/integration -p no:warnings'

# note: usefull commands
clean:
	docker compose --file infra/docker-compose.yaml down -v

logs-app:
	docker compose --file infra/docker-compose.yaml logs -f app

enter-app:
	docker compose --file infra/docker-compose.yaml exec app sh


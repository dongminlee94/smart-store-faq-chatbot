setup:
	pip install -r requirements.txt

setup-dev:
	make setup
	pip install -r requirements-dev.txt
	pre-commit install

format:
	black . --line-length 110
	isort . --profile black

lint:
	flake8 . --max-line-length 110 --extend-ignore E203

check:
	make format
	make lint

chat:
	docker compose up -d --build

chat-end:
	docker compose down -v

restart:
	make chat-end
	make chat

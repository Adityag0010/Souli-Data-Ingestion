.PHONY: install fmt lint run

install:
	pip install -r requirements.txt

fmt:
	python -m pip install ruff
	ruff format .

lint:
	python -m pip install ruff
	ruff check .

run:
	souli --help

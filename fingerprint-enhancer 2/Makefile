.PHONY: install run api test methods

install:
	python -m pip install -U pip
	pip install -r requirements.txt

methods:
	python -m app.cli methods

run:
	python -m app.cli enhance --input sample.jpg --method "Gabor Filter" --output out.bmp

api:
	uvicorn app.main:app --reload --host 0.0.0.0 --port $${PORT:-8000}

test:
	pytest -q

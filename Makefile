.PHONY: install run test docker
install:
	python -m pip install -r requirements.txt
run:
	uvicorn app.main:app --reload

test:
	pytest -q

docker:
	docker compose up --build

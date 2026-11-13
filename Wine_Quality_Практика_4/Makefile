PYTHON ?= python
IMAGE ?= wine-quality-service
VERSION ?= 1.0.0
.PHONY: install train test serve build run
install:
	$(PYTHON) -m pip install -r requirements.txt
train:
	$(PYTHON) train.py
# Demo test artifacts generated in a separate temporary dir by fixtures
test:
	$(PYTHON) -m pytest -q
serve:
	$(PYTHON) -m uvicorn app.main:app --host 127.0.0.1 --port 8077
build:
	docker build -t $(IMAGE):$(VERSION) .
run:
	docker run --rm -p 8077:8077 $(IMAGE):$(VERSION)

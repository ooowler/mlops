.PHONY: build data run api mlflow mlflow-ui predict-api train-rf train-logreg train-all seed-mlflow dvc-add dvc-push check-fast smoke-train sample-pipeline eda-report dvc-where dvc-status dvc-dag dvc-pull-host dvc-push-host dvc-demo
DC = docker-compose
PY ?= .venv/bin/python
DVCENV = env PATH="$(CURDIR)/.venv/bin:$$PATH"
MLFLOW_UI_PORT ?= 5000
PREDICT_API_PORT ?= 8000
SERVE_MODEL ?= logreg

check-fast:
	$(PY) -m compileall -q src scripts
	PYTHONPATH=src $(PY) -m pytest tests/ -q --tb=line

smoke-train:
	test -f data/processed/train.csv || (echo "нет train.csv — сначала preprocess или dvc repro" && exit 1)
	PYTHONPATH=src env -u MLFLOW_TRACKING_URI $(PY) scripts/train.py --model rf --max-rows 10000

sample-pipeline:
	$(PY) scripts/make_sample_raw.py 15000
	PYTHONPATH=src $(PY) scripts/load_data.py
	PYTHONPATH=src $(PY) scripts/preprocess.py
	rm -rf .mlflow_sample && mkdir -p .mlflow_sample
	PYTHONPATH=src MLFLOW_TRACKING_URI=file://$(CURDIR)/.mlflow_sample $(PY) scripts/train.py --model rf --max-rows 10000
	PYTHONPATH=src MLFLOW_TRACKING_URI=file://$(CURDIR)/.mlflow_sample $(PY) scripts/train.py --model logreg --max-rows 10000

build:
	$(DC) build

data:
	$(DVCENV) $(PY) -m dvc repro

dvc-where:
	@echo "remote local url из .dvc/config: $(CURDIR)/dvc_store"
	@ls -la dvc_store 2>/dev/null || echo "(папка появится после make dvc-push-host)"

dvc-status:
	$(DVCENV) dvc status

dvc-dag:
	$(DVCENV) dvc dag

dvc-pull-host:
	$(DVCENV) dvc pull

dvc-push-host:
	$(DVCENV) dvc push

dvc-demo: dvc-where dvc-status dvc-dag
	@echo "--- дальше: make dvc-push-host (в remote) / dvc-pull-host (с remote) ---"

dvc-add: build
	$(DC) run --rm -v "$$(pwd):/work" -w /work -e HOME=/work app dvc add data/raw/creditcard.csv

dvc-push: build
	$(DC) run --rm -v "$$(pwd):/work" -w /work -e HOME=/work app dvc push

run: build data
	$(DC) --profile mlflow up -d mlflow
	sleep 20
	$(DC) run --rm -e MLFLOW_TRACKING_URI=http://mlflow:5000 app python scripts/train.py --model rf
	$(DC) run --rm -e MLFLOW_TRACKING_URI=http://mlflow:5000 app python scripts/train.py --model logreg

api:
	$(DC) --profile api up api

mlflow:
	$(DC) --profile mlflow up -d mlflow

mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri file://$(CURDIR)/mlflow_data --host 127.0.0.1 --port $(MLFLOW_UI_PORT)

predict-api:
	PYTHONPATH=src MODEL_DIR=$(CURDIR)/models SERVE_MODEL=$(SERVE_MODEL) $(PY) -m uvicorn fraud_detection.api:app --host 127.0.0.1 --port $(PREDICT_API_PORT)

train-rf:
	PYTHONPATH=src $(PY) scripts/train.py --model rf

train-logreg:
	PYTHONPATH=src $(PY) scripts/train.py --model logreg

train-all: train-rf train-logreg

eda-report:
	PYTHONPATH=src $(PY) scripts/eda_metrics_report.py

seed-mlflow: build
	$(DC) --profile mlflow run --rm seed
	$(DC) --profile mlflow up -d mlflow

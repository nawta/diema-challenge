.PHONY: help install test lint prepare-data build-splits cv-baseline cv-exp infer-test make-submission check-submission summarize

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in editable mode with dev dependencies
	pip install -e ".[dev]"

test:  ## Run all tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ -v --cov=diema --cov=utils --cov-report=term-missing

lint:  ## Run ruff linter
	ruff check .

lint-fix:  ## Run ruff linter with auto-fix
	ruff check --fix .

# --- Data Pipeline ---

prepare-data:  ## raw BVH -> processed NPZ
	python tools/prepare_dataset.py

build-splits:  ## Generate LPO splits
	python tools/build_splits.py

# --- Training ---

cv-baseline:  ## Run exp001 10-fold LPO baseline
	python tools/run_cv.py --experiment exp001_benchmark_repro --num-folds 10

cv-exp:  ## Run CV for specific experiment (EXP=exp002_xxx)
	python tools/run_cv.py --experiment $(EXP) --num-folds 10

# --- Inference & Submission ---

infer-test:  ## Test inference with best checkpoints (EXP=exp001_benchmark_repro)
	python tools/infer_test.py --experiment $(EXP)

make-submission:  ## Convert predictions to submission format (EXP=exp001_benchmark_repro)
	python tools/make_submission.py --experiment $(EXP)

check-submission:  ## Validate submission file (FILE=output/submissions/xxx.csv)
	python tools/check_submission.py --file $(FILE)

summarize:  ## Summarize CV results to markdown
	python tools/summarize_cv.py

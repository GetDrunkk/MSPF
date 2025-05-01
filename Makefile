# Makefile for FMTS interpolation tasks
ENV_DIR := .venv
PYTHON_VERSION := 3.8.10

.PHONY: setup clean activate run train evaluate

setup:
	rm -rf $(ENV_DIR)
	@command -v pyenv >/dev/null 2>&1 || { echo '❌ pyenv not found. Please install pyenv.'; exit 1; }
	@if ! pyenv versions --bare | grep -qx "$(PYTHON_VERSION)"; then \
		echo "⬇️ Installing Python $(PYTHON_VERSION)..."; \
		pyenv install $(PYTHON_VERSION); \
	fi
	@echo "$(PYTHON_VERSION)" > .python-version
	@rm -rf $(ENV_DIR)
	@pyenv exec python -m venv $(ENV_DIR)

	@$(ENV_DIR)/bin/pip install --upgrade pip setuptools wheel

	@$(ENV_DIR)/bin/pip install -r requirements.txt || echo "⚠️ Please provide a requirements.txt file."
	@echo "✅ Setup complete. Activate with: source $(ENV_DIR)/bin/activate"


run:
	bash scripts/run_interpolation.sh

train:
	@$(ENV_DIR)/bin/python train.py

evaluate:
	@$(ENV_DIR)/bin/python evaluate.py

clean:
	rm -rf $(ENV_DIR)

debug:
	@export results_folder=./Checkpoints_debug && \
	PYTHONPATH=. .venv/bin/python main.py \
		--train \
		--config_file Config/etth.yaml \
		--name debug_run \
		--long_len 24 \
		--missing_ratio 0.3

sample:
	@echo "🚀 Running inference and plotting..."
	@$(ENV_DIR)/bin/python sample_and_plot.py

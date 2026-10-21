PYTHON ?= python3

.PHONY: all install clean-data features model plots

all: install clean-data features model plots

install:
	$(PYTHON) -m pip install -r requirements.txt

clean-data:
	$(PYTHON) src/clean.py

features:
	$(PYTHON) src/features.py

model:
	$(PYTHON) src/model.py

plots:
	$(PYTHON) src/plots.py

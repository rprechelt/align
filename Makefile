.PHONY: prospect init test mypy

init:
	pip install -r requirements.txt

prospect:
	prospector -F --strictness=high

test:
	PYTHONPATH=`pwd` pytest --cov=align tests

mypy:
	PYTHONPATH=`pwd` mypy align --ignore-missing-imports

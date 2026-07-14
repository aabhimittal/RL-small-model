.PHONY: install test quickstart train evaluate demo clean

install:
	pip install -r requirements.txt

test:
	PYTHONPATH=. pytest tests/ -q

quickstart:
	python examples/quickstart.py

train:
	python scripts/train.py --steps 300 --out runs/demo

evaluate:
	python scripts/evaluate.py --ckpt runs/demo/policy.pkl

demo:
	python scripts/demo.py --ckpt runs/demo/policy.pkl

clean:
	rm -rf runs __pycache__ */__pycache__ .pytest_cache

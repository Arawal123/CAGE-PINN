.PHONY: install test lint smoke literature novelty verify freeze report paper-check

install:
	python -m pip install -e ".[dev,stats]"

test:
	pytest -q

lint:
	ruff check .

smoke:
	python -m cage_pinn.cli study smoke --steps 8

literature:
	python -m cage_pinn.cli literature build

novelty:
	python -m cage_pinn.cli novelty audit

verify:
	python -m cage_pinn.cli benchmark verify-references

freeze:
	python -m cage_pinn.cli study freeze

report:
	python -m cage_pinn.cli report build

paper-check:
	python scripts/analysis/check_claims.py

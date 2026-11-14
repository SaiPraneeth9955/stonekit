test:
	pytest -rs

lint:
	ruff check src tests scripts

# stonekit app is implemented in Stage 11.
app:
	stonekit app

.PHONY: test lint app

PACKAGES := eval-harness judge-cal

.PHONY: install test lint example

install:
	uv sync

test:
	@for p in $(PACKAGES); do (cd packages/$$p && uv run pytest -q) || exit 1; done
	uv run pytest -q tests/integration

lint:
	uv run ruff check .

example:
	uv run python examples/end_to_end.py

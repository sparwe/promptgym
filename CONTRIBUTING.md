# Contributing

To add a new agent:

1. Create a file under `promptgym/agents/` implementing `act()` and optionally `current_best()`.
2. Register the class in `promptgym/agents/__init__.py`.
3. Add unit tests for your agent's basic behaviour.
4. Run `ruff check .` and `pytest -q` before submitting a pull request.


# OptiFlux Backend

Python 3.12 backend with FastAPI and Poetry.

## Setup

1. Install dependencies:
   ```bash
   poetry install
   ```

2. Run the API:
   ```bash
   poetry run uvicorn app.main:app --reload
   ```

3. Run tests:
   ```bash
   poetry run pytest
   ```

## Development

- Code formatting: `poetry run black .`
- Import sorting: `poetry run isort .`
- Linting: `poetry run flake8 .`
- Type checking: `poetry run mypy .`
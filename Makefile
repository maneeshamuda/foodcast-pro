.PHONY: install train test lint api dashboard docker
install:   ; pip install -r requirements-dev.txt
train:     ; python -m fdf.train
test:      ; pytest -q
lint:      ; ruff check .
api:       ; uvicorn api.main:app --reload --port 8000
dashboard: ; streamlit run dashboard/app.py
docker:    ; docker compose up --build

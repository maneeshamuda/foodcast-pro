# FoodCast Pro 3.0
Run SETUP_WINDOWS.bat with Python 3.12 and Node.js installed, then RUN_WINDOWS.bat. Open http://localhost:8000 after the server starts. Stop older servers using port 8000 first. API docs: /docs. Original Streamlit app remains available separately.
Business mode: sample model forecasts, CSV/XLSX upload, recent-observation average with a 15% stock buffer for uploaded data, downloadable plans. Technical mode: documented backtest metrics and architecture. Uploaded data is not used to retrain LightGBM. Meal names are unavailable in the sample dataset; IDs are retained. Uploaded files are processed in memory; no accounts or persistent database. Model results are historical, not guarantees. The original README describes the underlying ML pipeline.
Frontend build requires npm registry access; build and end-to-end runtime have not been verified in the packaging environment. Run python -m pytest -q after activating the environment to verify existing tests.

## Polished release
Adds historical overview charts, top meal IDs, final-fold actual/predicted chart, champion highlighting, Excel example, drag-and-drop uploads and plain-language preparation advice. For an existing 3.0 installation, stop the server, copy the updated files over the existing project (keep .venv), run UPDATE_WINDOWS.bat and restart. Fresh installations use SETUP_WINDOWS.bat. New backend endpoint functions were smoke-tested; frontend build remains to be verified locally.

## Redesigned edition
Public landing page, original vector food illustration, light workspace, dark workspace toggle, and direct technology page entry. Existing forecast and upload workflows are retained. The illustration is decorative; sample meal IDs remain unchanged. Run UPDATE_WINDOWS.bat after copying into an existing installation, or SETUP_WINDOWS.bat for a fresh installation. Build and smoke-test locally before deployment.

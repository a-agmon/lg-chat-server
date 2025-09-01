# Web UI (React, zero‑install)

A minimal React web UI (served statically) to chat with the FastAPI + LangGraph server. It uses CDN React + Babel so you can run it without `npm install`.

## How to use

- Start the API server (from project root):
  - `uvicorn app.main:app --reload`
- Open: `http://localhost:8000/web/`
- Enter a main complaint and click Start Chat. Then type replies.

Notes:
- The UI defaults to `window.location.origin` for API calls. If you are hosting the API at a different origin, set the "API Base" field accordingly (e.g., `http://localhost:8001`).
- If you serve the UI from `/web` on the same origin (default), no CORS configuration is required.

## Endpoints used

- `POST /chat/start` → `{ session_id, done, message }`
- `POST /chat/{session_id}/message` → `{ session_id, done, message }`
- `GET /chat/{session_id}/state` (linked in API docs for debugging)

## Customize

- Edit `webapp/style.css` to tweak styles.
- The app is a single-file React component inside `webapp/index.html` for simplicity.

## Optional: building a full React app

If you prefer a full toolchain (Vite/CRA), you can scaffold and build separately, then output to this folder or mount a different static directory in `app/main.py`. For this demo, the CDN approach avoids local package installs.

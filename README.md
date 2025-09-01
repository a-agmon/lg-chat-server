# Medical Intake Chatbot (LangGraph + FastAPI)

A minimal, production-ready-ish REST service that runs a 3-node LangGraph chatbot for a clinical intake flow:

1) Accept a main complaint from the patient.
2) Use the LLM to select the most relevant schema record from a JSON knowledge base (id + complaint synonyms + guidance).
3) Manage a conversation loop (one question per request) where the LLM develops its own questions based on the record's "relevant_information" until it has enough info, then produce a summary.

The LLM receives a strict instruction to either ask the next question or return a final JSON with `done=true`, `collected_information`, and a `summary` to signal conversation end. The API surfaces the aggregated info as `collected_fields`.

## Recent Improvements

- **Type Safety**: Fixed checkpointer API usage with proper type annotations (`BaseCheckpointSaver`, `RunnableConfig`)
- **Code Refactoring**: Extracted reusable helper functions (`get_session_checkpoint`, `get_session_messages`) to reduce duplication
- **Cleaner API**: Removed OpenAPI tags to reduce noise in the documentation
- **Better Error Handling**: Consistent checkpoint access pattern with proper exception handling

## Quickstart

Prereqs: Python 3.11+, an OpenAI API key.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then edit with your OPENAI_API_KEY
uvicorn app.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Swagger / ReDoc:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

## Web UI (React Demo)

A minimal, zero-install React web app is bundled and served statically from the API at `/web`.

Run and open:

```bash
uvicorn app.main:app --reload
# then visit:
open http://localhost:8000/web/
```

How to operate:

- Enter the main complaint (e.g., "Headache since yesterday") and click "Start Chat".
- The assistant asks one question at a time.
- Type your reply in the input box and click "Send".
- Continue until `done=true` and the final summary is shown in the chat.

Notes:

- The UI calls the API at `window.location.origin` by default. If your API is on a different origin/port, change the "API Base" field in the header (e.g., `http://localhost:8001`).
- When served from the same origin (`/web`), CORS is not required. If you choose to host the UI elsewhere, configure CORS in FastAPI accordingly.

Implementation:

- Static files live under `webapp/` and are mounted in `app/main.py` at `/web` using `StaticFiles`.
- The UI is a single-file React component loaded via CDN (no `npm install` needed). Styles are in `webapp/style.css`.

## API

- POST `/chat/start` — start a session and get the first question (or summary if the initial message already satisfies requirements).
- POST `/chat/{session_id}/message` — send the patient’s reply and receive the next question or final summary.
- GET `/chat/{session_id}/state` — debugging endpoint to inspect the stored graph state.

### Start a chat (Step 1)

```bash
curl -X POST http://localhost:8000/chat/start \
  -H 'Content-Type: application/json' \
  -d '{
    "complaint": "I have a bad headache since yesterday"
  }'
```

Response:

```json
{
  "session_id": "<generated uuid>",
  "done": false,
  "message": "How severe is the pain (0-10)?",
  "collected_fields": null,
  "summary": null,
  "next_question": "How severe is the pain (0-10)?",
  "selected_record_id": "headache_migraine",
  "relevant_information": "onset and duration, location, severity, character (throbbing), associated symptoms (nausea, photophobia, phonophobia, aura), triggers, relieving factors, previous episodes, red flags (fever, stiff neck, neuro deficits)",
  "relevant_tests": "Usually none; consider imaging if red flags"
}
```

### Continue the chat (Step 2+)

```bash
curl -X POST http://localhost:8000/chat/abc123/message \
  -H 'Content-Type: application/json' \
  -d '{"content": "It is around 7 out of 10"}'
```

Repeat the previous call (Step 2) with the user's new content until the response contains `"done": true`. When `done` is true, `message` contains the final summary and `next_question` is null.

Tip: You can perform the same flow from Swagger UI at `/docs` — open the `chat` tag, try `POST /chat/start`, then repeatedly call `POST /chat/{session_id}/message` using the same `session_id`.

### Example end-to-end (single terminal)

```bash
# 1) Start (server generates session_id)
curl -s -X POST http://localhost:8000/chat/start \
  -H 'Content-Type: application/json' \
  -d '{"complaint":"I have a bad headache since yesterday"}' | tee /tmp/start.json | jq .

# 2) Continue with your answer (replace with the question you got above)
SID=$(jq -r .session_id /tmp/start.json)
curl -s -X POST http://localhost:8000/chat/$SID/message \
  -H 'Content-Type: application/json' \
  -d '{"content":"About 7 out of 10"}' | jq .

# 3) Repeat step 2 with new answers until done=true
```

## Knowledge Base Schema

`data/complaints.json` is a list of records with this shape:

```json
{
  "id": "chest_pain_cardiac",
  "main_complaint": ["chest pain", "chest discomfort", "heart pain"],
  "relevant_information": "onset, duration, character, radiation…",
  "relevant_tests": "ECG, troponins, chest X-ray"
}
```

During `/chat/start`, the service injects all records' `id` and `main_complaint` synonyms to the LLM, which selects the most relevant `id` for the session. The chosen record's `relevant_information` is then used to guide the interview.

## Implementation Notes

- **LangGraph nodes**:
  - `retrieve_kb`: Uses the LLM to select a schema record from the catalog (by `id` and `main_complaint`), then stores `kb_record` with `relevant_information` and `relevant_tests`.
  - `interview`: Calls the LLM with the complaint, selected record, and full message history; the LLM generates its own questions until it decides `done=true` and returns `collected_information` and a `summary`. No partial "collected" state is injected; the model relies on the conversation history.
- **Checkpointing**: Uses `MemorySaver` for per-session state (`thread_id=session_id`). Swap for a persistent checkpointer for horizontal scaling.
  - Helper functions `get_session_checkpoint()` and `get_session_messages()` provide clean access to checkpoint data
  - Proper type annotations with `BaseCheckpointSaver` and `RunnableConfig` ensure type safety
- **Model**: Configurable via `OPENAI_MODEL` (default `gpt-4o-mini`).
- **Config**: See `.env.example` or environment variables.
- **Code organization**: 
  - Refactored checkpoint access into reusable helper functions to reduce code duplication
  - Removed OpenAPI tags for cleaner API documentation
  - Proper type hints throughout for better IDE support and type checking

## Production Considerations

- Replace `MemorySaver` with a persistent checkpointer (e.g., Redis/SQL) for multi-instance deployment.
- Add authentication, rate limiting, and request timeouts.
- Validate and sanitize user input; enforce maximum message length.
- Add structured logging and observability (trace IDs per `session_id`).
- Consider using structured output tooling for stricter JSON guarantees.
- Testing: For manual verification use Swagger UI or cURL examples above. For automated tests you can stub the LLM layer or point to a mock, and use a persistent checkpointer for deterministic state.

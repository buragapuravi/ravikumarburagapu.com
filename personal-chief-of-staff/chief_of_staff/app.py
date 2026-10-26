"""FastAPI web interface and JSON API."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from chief_of_staff.service import demo_status, run_task

app = FastAPI(
    title="Personal Chief of Staff",
    description="A local-first, read-only MCP assistant for calendar, inbox, tasks, and meeting prep.",
    version="0.1.0",
)


class TaskRequest(BaseModel):
    task: str = Field(min_length=1, max_length=4000)


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", **demo_status()}


@app.post("/api/run")
async def run(request: TaskRequest) -> dict:
    try:
        return await run_task(request.task)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Personal Chief of Staff</title>
  <style>
    :root { color-scheme: light dark; font: 16px/1.6 system-ui, sans-serif; }
    body { max-width: 820px; margin: 4rem auto; padding: 0 1.25rem; }
    h1 { line-height: 1.15; }
    .notice { padding: 1rem; border: 1px solid #8887; border-radius: .75rem; }
    textarea { box-sizing: border-box; width: 100%; min-height: 110px; padding: .8rem; font: inherit; }
    button { margin: .75rem 0; padding: .65rem 1rem; font: inherit; cursor: pointer; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; padding: 1rem; border-radius: .5rem; background: #8882; }
    .actions { display: flex; flex-wrap: wrap; gap: .5rem; margin: 1rem 0; }
    .actions button { margin: 0; }
  </style>
</head>
<body>
  <p>LOCAL-FIRST · HUMAN-CONTROLLED · MCP-POWERED</p>
  <h1>Personal Chief of Staff</h1>
  <p>Ask for a daily brief, email triage, meeting preparation, or scheduling suggestions.</p>
  <p class="notice"><strong>Read-only by design:</strong> this assistant never sends email or changes your calendar. Demo mode uses synthetic sample data and makes no external calls.</p>
  <div class="actions">
    <button type="button" data-task="Give me my daily briefing">Daily briefing</button>
    <button type="button" data-task="Triage my inbox and draft replies">Email triage</button>
    <button type="button" data-task="Prepare me for my next meeting">Meeting prep</button>
    <button type="button" data-task="Find candidate times for a 30-minute meeting">Find meeting times</button>
  </div>
  <form id="task-form">
    <label for="task">What do you need?</label>
    <textarea id="task" maxlength="4000" required>Give me my daily briefing</textarea>
    <button type="submit">Prepare</button>
  </form>
  <p id="status" role="status" aria-live="polite"></p>
  <pre id="result" hidden></pre>
  <script>
    const form = document.querySelector("#task-form");
    const task = document.querySelector("#task");
    const status = document.querySelector("#status");
    const output = document.querySelector("#result");
    document.querySelectorAll("[data-task]").forEach(button => {
      button.addEventListener("click", () => { task.value = button.dataset.task; form.requestSubmit(); });
    });
    form.addEventListener("submit", async event => {
      event.preventDefault();
      status.textContent = "Preparing your response…";
      output.hidden = true;
      try {
        const response = await fetch("/api/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ task: task.value })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Request failed");
        output.textContent = data.response;
        output.hidden = false;
        status.textContent = "Agent: " + data.workflow + " · Mode: " + data.mode;
      } catch (error) {
        status.textContent = error.message;
      }
    });
  </script>
</body>
</html>"""

# Personal Chief of Staff

A local-first, multi-agent assistant for daily briefings, email triage, meeting prep, and calendar scheduling suggestions. It uses a LangGraph supervisor, specialist agents, and user-configured MCP servers.

**Writes are deliberately disabled.** The assistant uses only read-only MCP tools, produces email replies as unsent draft text, and suggests meeting slots without creating or modifying calendar events. Demo mode is synthetic and makes no network calls.

## Architecture

```text
                  ┌─────────────────────────┐
   CLI / browser ─▶│ FastAPI service         │
                  │ local UI · JSON API     │
                  └────────────┬────────────┘
                               │
                   ┌───────────▼───────────┐
                   │ LangGraph supervisor │
                   │ intent → specialist  │
                   └─┬────┬──────┬──────┬─┘
                     │    │      │      │
                 Scheduler Email  Meeting Daily
                           triage prep   briefing
                     │    │      │      │
                   read-only MCP tool calls only
                     │    │      │      │
             Calendar  Gmail  Notion/Todoist  Slack/search
```

The supervisor routes a request to one specialist. Each specialist only receives relevant tools that advertise the MCP `readOnlyHint` annotation or have a recognized read-only name (`get`, `list`, `search`, `find`, `read`, `fetch`, `query`, `lookup`, or `retrieve`). Unclassified tools are excluded. The graph never executes write tools.

## Quick start: local demo

Requires Python 3.11+.

```sh
cd personal-chief-of-staff
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# No API key or MCP server required; synthetic demo data only.
chief-of-staff run "Give me my daily briefing"
chief-of-staff run "Triage my inbox and draft replies" --format json

# Local browser UI and JSON API at http://127.0.0.1:8787
chief-of-staff web
```

The web page has one-click demos for the four agents. The API exposes `GET /api/health` and `POST /api/run` with `{"task":"..."}`. Requests and results are not saved to disk.

## Connect trusted MCP servers

1. Review the command and package for each MCP server yourself. MCP servers run as local child processes and can access whatever credentials and permissions you grant them. Do not run an untrusted server.
2. Copy `mcp-servers.example.json` to `mcp-servers.json`. Replace the placeholder commands with servers you trust and configure only read scopes where the provider supports scopes.
3. Export credentials in your shell; config uses `${ENV_VAR}` references so secrets need not be stored in the JSON file:

   ```sh
   export OPENAI_API_KEY="..."
   export CALENDAR_ACCESS_TOKEN="..."
   export GMAIL_ACCESS_TOKEN="..."
   export TASKS_ACCESS_TOKEN="..."
   export CHIEF_OF_STAFF_DEMO=false
   export CHIEF_OF_STAFF_MCP_CONFIG="$PWD/mcp-servers.json"
   ```

4. Restart `chief-of-staff web` or run the CLI again. The live model defaults to `gpt-4o-mini`; set `CHIEF_OF_STAFF_MODEL` to choose another compatible model. Configure the provider's supported base URL in your environment/model setup if needed.

Provider integrations are supplied by MCP servers you configure; this repository does not claim or bundle provider OAuth clients. Tools without read-only annotations are allowed only when their tool name has a recognized read verb and no write verb. Review server tool names and permissions before connecting accounts. Tool descriptions and retrieved content are sent to the configured model provider; avoid sending information your provider is not authorized to process.

### Example runs

**Daily briefing:** “Give me my daily briefing.” The daily agent summarizes retrieved calendar events, important inbox items, and tasks; it separates facts from suggestions.

**Email triage:** “Triage my inbox and draft replies.” The email agent ranks messages and can suggest reply text. It cannot send, archive, mark read, or otherwise change email.

**Meeting prep:** “Prepare me for my next meeting.” The prep agent uses relevant read-only calendar, email, task, and search tools when available and flags missing context.

**Scheduling:** “Find candidate times for a 30-minute meeting.” The scheduler can identify openings from retrieved availability but never creates or moves an event. Specify timezone and duration when they are not clear.

In default demo mode these examples use labeled synthetic data. In live mode they use only the MCP servers you configured.

## Configuration and safety

| Setting | Default | Purpose |
| --- | --- | --- |
| `CHIEF_OF_STAFF_DEMO` | `true` | Use deterministic synthetic responses; set to `false` for live model/MCP calls |
| `OPENAI_API_KEY` | unset | Required for live model calls |
| `CHIEF_OF_STAFF_MODEL` | `gpt-4o-mini` | Model for live supervisor and specialists |
| `CHIEF_OF_STAFF_MCP_CONFIG` | unset | Path to your local stdio MCP server JSON |

- The web server binds to `127.0.0.1` only. The CLI refuses a non-loopback host.
- Start with provider read-only scopes. There are no write tools in the graph.
- Do not commit `.env`, tokens, OAuth credentials, private prompts, or `mcp-servers.json`.
- This is a personal assistant starter, not a managed multi-user service: no authentication, persistence, or audit store is included.
- The agent may make mistakes. Verify meeting availability and summaries before acting.

## Development

```sh
pip install -e ".[dev]"
pytest
```

Contributions should include tests and document MCP permissions, data flow, and any change to the read-only defaults. Never enable external writes implicitly.

## License

MIT. See [LICENSE](LICENSE).

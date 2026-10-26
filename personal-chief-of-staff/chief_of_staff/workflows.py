"""Task routing and sample data for the local, no-credentials demo."""

from typing import Literal

Workflow = Literal["scheduler", "email_triager", "meeting_prep", "daily_briefing"]

DEMO_DATA = {
    "calendar": [
        {"time": "09:00", "title": "Platform reliability review", "attendees": ["Alex", "Priya"]},
        {"time": "11:00", "title": "1:1 with Morgan", "attendees": ["Morgan"]},
        {"time": "14:00", "title": "Inference capacity planning", "attendees": ["Sam", "Jordan"]},
    ],
    "emails": [
        {
            "from": "alex@example.test",
            "subject": "Reliability review: decision needed",
            "snippet": "Can we confirm the rollout owner before today's review?",
            "priority": "high",
        },
        {
            "from": "updates@example.test",
            "subject": "Weekly platform digest",
            "snippet": "Your weekly platform metrics digest is ready.",
            "priority": "low",
        },
    ],
    "tasks": [
        {"title": "Review inference capacity proposal", "due": "today", "priority": "high"},
        {"title": "Send reliability review notes", "due": "tomorrow", "priority": "medium"},
    ],
}

SYSTEM_PROMPTS: dict[str, str] = {
    "scheduler": (
        "You are the Scheduler. Use read-only calendar tools to identify availability. "
        "Suggest candidate times and explain conflicts. Never create, move, or cancel events. "
        "Ask for a timezone or duration if needed; do not infer either."
    ),
    "email_triager": (
        "You are the Email Triager. Summarize and prioritize messages using read-only email tools. "
        "You may draft suggested reply text, clearly labeled as a draft. Never send, archive, "
        "delete, label, or mark messages read."
    ),
    "meeting_prep": (
        "You are the Meeting Prep agent. Prepare a concise briefing using read-only calendar, "
        "email, task, and search tools. Separate retrieved facts from suggestions and call out "
        "missing context. Never change external systems."
    ),
    "daily_briefing": (
        "You are the Daily Briefing agent. Prepare a concise, actionable daily briefing using "
        "read-only calendar, email, task, and search tools. Separate facts from suggestions. "
        "Never modify or send anything."
    ),
}


def choose_demo_workflow(task: str) -> Workflow:
    normalized = task.casefold()
    if any(word in normalized for word in ("email", "inbox", "reply", "triage")):
        return "email_triager"
    if any(
        word in normalized
        for word in (
            "schedule",
            "availability",
            "calendar",
            "time slot",
            "candidate time",
            "find time",
            "meeting time",
        )
    ):
        return "scheduler"
    if any(
        word in normalized
        for word in ("meeting prep", "prepare for", "prepare me for", "brief me", "before my")
    ):
        return "meeting_prep"
    return "daily_briefing"


def run_demo(task: str, workflow: Workflow | None = None) -> dict[str, str]:
    """Return a deterministic demo briefing without calling external services."""
    selected = workflow or choose_demo_workflow(task)
    if selected == "scheduler":
        response = (
            "Demo calendar availability (sample data only):\n"
            "- 10:00–11:00 AM: candidate opening\n"
            "- 12:00–2:00 PM: candidate opening\n"
            "These are illustrative suggestions, not a live calendar check. No event was created."
        )
    elif selected == "email_triager":
        response = (
            "Demo inbox triage (sample messages only):\n"
            "1. HIGH — Reliability review: decision needed. Confirm the rollout owner.\n"
            "   Suggested draft: “I’ll confirm the rollout owner at today’s review.”\n"
            "2. LOW — Weekly platform digest. Review when convenient.\n"
            "No message was sent or changed."
        )
    elif selected == "meeting_prep":
        response = (
            "Demo meeting brief (sample data only): Platform reliability review at 09:00.\n"
            "Attendees: Alex, Priya. Open item: confirm the rollout owner.\n"
            "Suggested preparation: bring rollout status and current reliability risks.\n"
            "This brief is illustrative; no live calendar or inbox was accessed."
        )
    else:
        response = (
            "Demo daily briefing (sample data only)\n\n"
            "Calendar\n"
            "- 09:00 — Platform reliability review (Alex, Priya)\n"
            "- 11:00 — 1:1 with Morgan\n"
            "- 14:00 — Inference capacity planning (Sam, Jordan)\n\n"
            "Priority\n"
            "- Review inference capacity proposal (due today)\n"
            "- Reliability review needs a rollout owner decision\n\n"
            "Inbox\n"
            "- High: confirm the rollout owner before the reliability review\n"
            "- Low: weekly platform metrics digest\n\n"
            "Sample data only. No calendar or email account was accessed."
        )
    return {"workflow": selected, "response": response, "mode": "demo"}

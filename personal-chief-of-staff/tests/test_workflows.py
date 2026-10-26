import unittest

from chief_of_staff.workflows import choose_demo_workflow, run_demo


class WorkflowTests(unittest.TestCase):
    def test_routes_common_requests(self) -> None:
        self.assertEqual(choose_demo_workflow("Triage my inbox"), "email_triager")
        self.assertEqual(choose_demo_workflow("Find calendar availability"), "scheduler")
        self.assertEqual(choose_demo_workflow("Find candidate times for a meeting"), "scheduler")
        self.assertEqual(choose_demo_workflow("Prepare me for my next meeting"), "meeting_prep")
        self.assertEqual(choose_demo_workflow("Give me the morning update"), "daily_briefing")

    def test_email_demo_never_sends(self) -> None:
        result = run_demo("triage my inbox")
        self.assertEqual(result["mode"], "demo")
        self.assertIn("No message was sent", result["response"])


if __name__ == "__main__":
    unittest.main()

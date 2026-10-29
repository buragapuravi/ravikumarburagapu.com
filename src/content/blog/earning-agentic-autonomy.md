---
title: "Earning Agentic Autonomy: An Eval Architecture for AI Agents"
description: "How replayable traces, component-level attribution, planted failures, and error budgets can decide how much freedom an agent gets."
pubDate: 2026-10-22
---

In my last article I wrote that autonomy should be earned, and then left it there. This is my attempt at the mechanics, drawn from platform work at previous organizations and my own experiments. It's a design I'm still testing, so push on it.

The core idea is that evals should set how much freedom an agent gets, not just report on it.

![The agent assurance loop: runtime, trace ledger, replay bench, grader mesh, autonomy governor, execution gate, and canary injector](/images/fig1_agent_assurance_loop.png)

## 1. Log decisions

Each step writes one immutable event:

```
run_id, step_id, parent_step
input_hash, context_refs[]
decision: { tool, params, stated_reason }
tool_response_ref      # content-addressed
policy_version, tier, cost_usd, latency_ms
```

Content-addressed tool responses make a run replayable exactly. I keep full payloads for failures, a sample of successes, and metadata for the rest.

## 2. Measure failure as a rate

Replay a recorded run with tool responses frozen, N times. Report the pass rate with a Wilson interval, and track all-k-succeed rather than at-least-one. A single failed run says little, because agent failures vary between runs.

## 3. Attribute by intervention

Swap one component at a time for a known-good version (gold context, correct route, previous prompt, different model) and compute

`A_i = P(pass | swap_i) − P(pass | baseline)`

The largest significant `A_i` is the suspect. If a swap needs a tool result that was never recorded, mark the replay `unreplayable` instead of inventing one. Then shrink the failing trace by delta debugging: drop halves of the context and keep any reduction that still fails. The smallest survivor becomes a regression test.

## 4. Grade cheapest first, then audit the graders

Order: deterministic checks (schemas, validators, invariants), then diffs against the last known-good run, then model judges for the remainder. To audit a grader, inject known faults into passing runs (a wrong argument, a missing citation, a leaked field) and measure its kill rate, the share of injected faults it catches. A judge with a low kill rate shouldn't influence promotion.

## 5. Test the reviewers

Plant known-bad changes in the human review queue. Catch rate is a Beta posterior updated per canary, and I alert on the posterior mean instead of a single miss. A falling catch rate shows rubber-stamping well before approval times change. I'd chart it next to latency and error rate.

## 6. Tiers as a state machine

![The autonomy governor's four tiers, with fast demotion and slow promotion, and the execution gate's four checks](/images/fig2_autonomy_governor_and_gate.png)

Each workflow version has an SLO and an error budget, and the budget picks one of four tiers: unattended, sampled review, approve writes, read only.

- Demote when burn rate exceeds threshold in both a short and a long window. Requiring both avoids flapping on one noisy hour.
- Promote only after a long clean window, canary catch rate above target, zero gate violations, and a shadow replay of the new version on recorded traffic.
- The tier attaches to a workflow version. A new prompt, model, or toolset starts low.

## 7. Check again at execution

Queued work outlives policy changes, so approvals are signed receipts: action hash, hash of the record the reviewer saw, policy version, expiry, approver.

```python
def execute(action, r):
    if not verify(r):                            return reroute(action)
    if r.policy_version != policy.version:       return reroute(action)
    if r.record_hash != h(read(action.target)):  return reroute(action)
    if now() > r.expires_at:                     return reroute(action)
    return run(action, cred=mint_short_lived(action))  # retries re-mint
```

Fresh credentials per attempt also cover mid-run revocation, so writes, retrieval, and cache hits stop together. I test that case explicitly.

## 8. Cost and escalation

Track cost per accepted outcome. Escalate to a larger model on checkable signals (validator failure, schema violation, retry count, step budget) instead of the model's self-reported confidence. Then score the escalation policy on precision and recall.

## Where it's weak

Frozen-tool replay breaks on stateful tools. Low-volume workflows don't produce enough runs for honest intervals. And judges drift when the underlying model changes, so the instrument I measure with is moving too.

## Open questions

1. If a tool's state can't be fully captured, is exact replay even the right goal? Should we replay for distributional fidelity instead, by resampling state, and what does that do to the intervals we promote on?
2. Planted failures teach reviewers to catch the errors we can imagine, and the worst production failures are the ones we didn't. How do you keep the catch-rate number from giving false comfort?
3. The line between deterministic checks and model judges is really a decision about which uncertainty we accept. How should it move as the cost of a missed failure rises, or as the judge model drifts, without the mesh ending up grading itself in circles?
4. Early-warning signals are cheap only while they stay predictive. When the governor changes a workflow's tier mid-run, how do you tell real degradation from the effects of the governor's own intervention?

Some of these ideas were sharpened by the comments on my last article. Thanks to Timothy Urista, Jacek Bajorski, Muhammad Ali Zaidi, Mark Tsai, Arslan Zaheer, Mohit Sudhakar, and Pavel Grigorev.

 
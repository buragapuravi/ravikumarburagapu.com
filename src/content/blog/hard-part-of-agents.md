---
title: "The Hard Part of Agents Isn't the Model"
description: "Getting a model to do something impressive is easy. Running agents safely, reliably, and at scale is where the engineering starts."
pubDate: 2026-10-04
---

I’ve spent the last several years building the platforms underneath large-scale software systems: developer platforms, GenAI programs in regulated environments, and AI engineering, infrastructure and reliability.

One pattern keeps showing up.

Getting a model to do something impressive is becoming surprisingly easy.

Getting that same system to operate safely, reliably, repeatedly, and at scale is where the real engineering starts.

A demo can come together in a week. The next six months are usually spent dealing with everything around the model: identity, permissions, observability, cost, failures, approvals, state, and the uncomfortable question of what happens when the agent does something you didn’t anticipate.

That’s where I think the hard part of agents really is.

## 1. Agents need an identity of their own

A lot of early agent implementations run with a human’s token or a long-lived service account with broad permissions.

It’s convenient. It’s also a problem waiting to happen.

Once you have hundreds or thousands of agent runs, someone will eventually ask: which agent did this? On whose authority? What was it allowed to do? What did it actually do?

You need a much stronger answer than “it was running under the service account.”

I’d rather see each agent run get its own workload identity, with short-lived credentials minted for the specific task and scoped to what that task actually requires. SPIFFE-style workload identity combined with token exchange is one direction worth considering.

Write access should be the exception, not the default.

And authorization should happen at the tool and resource boundary through policy, using OPA, Cedar, or an equivalent policy engine, rather than relying on instructions embedded in a prompt.

A prompt is an instruction. It is not a security boundary.

## 2. Human approval doesn't scale as a security model

The natural response to agent risk is: “Put a human in the loop.”

That works until it doesn’t.

Five agents opening pull requests is manageable. Two hundred agents creating tickets, modifying infrastructure, updating configurations, and interacting with production systems is a different problem.

At some point, the approval queue becomes the bottleneck.

And when humans are asked to approve hundreds of similar actions every day, you eventually get rubber-stamping. A control that everybody approves without really looking at it isn’t much of a control.

I think the better model is tiered autonomy based on blast radius.

A reversible, low-risk action can run autonomously with good auditability.

An action that changes production infrastructure, modifies customer data, moves money, or crosses a regulatory boundary should have a stronger gate.

The important part is that autonomy should be earned. You should be able to look at an agent or workflow and say: “We trust this workflow to operate autonomously because we have evidence that it behaves reliably under these conditions.”

That requires measurement. Which brings me to the next problem.

## 3. Agents need evals the way services need tests

We learned this lesson with distributed systems a long time ago.

You don’t deploy a service repeatedly without regression testing. I don’t think agents should be treated differently.

Every meaningful change to the model, prompt, retrieval strategy, tool set, or orchestration logic can change behavior. And unlike traditional software, the change isn’t always obvious.

A new model version might improve the average task success rate while breaking one workflow that happens to be critical to your compliance team.

So I’d treat evaluations as part of the engineering lifecycle, not as a one-time exercise before launch. Build a regression suite from real tasks, and run it whenever any of these change:

- the model
- the prompt
- a tool
- the retrieval layer

And measure more than “did it get the answer right?” For production workflows, I’d want metrics such as task success rate, escalation rate, policy violations, latency, and cost per successful task.

There’s also an interesting opportunity to apply an SRE concept here: error budgets. If an agent workflow is consistently failing its reliability target, its level of autonomy should decrease until the underlying problem is understood.

That feels much more practical to me than pretending we can make an inherently probabilistic system deterministic.

## 4. Agent observability has to capture the journey, not just the outcome

Traditional application monitoring can tell you that something was slow or returned an error. That isn’t enough for agents.

If an agent takes seven steps to complete a task and makes the wrong tool selection on step four, I need to know that. What context did it retrieve? What did it see? Which tool did it select? What parameters did it send? What came back? What happened next? Where did the trajectory diverge from what we expected?

This means agent observability has to capture more than metrics and logs. We need traces that represent the trajectory of the task, including model interactions, retrieval, tool calls, decisions, and outcomes. OpenTelemetry’s emerging GenAI conventions are a useful starting point.

But there is another practical problem: payload size. Agent traces can be enormous. If you want to replay a failed execution, you need to retain enough context to reproduce what happened without turning your observability platform into another runaway cost center. That trade-off needs to be designed up front.

Without replay, debugging an agent often becomes guesswork. And incident reviews shouldn’t turn into a debate about what the model “probably” did.

## 5. The deployable unit is changing

For years, we’ve optimized our infrastructure around relatively predictable workloads: services, containers, queues, databases, APIs.

Agents are different. An agent can hold state, acquire credentials, execute code, call multiple external systems, loop unexpectedly, and run for minutes or hours.

That changes the infrastructure boundary.

I expect to see much stronger isolation around individual agent executions, potentially sandboxed environments using technologies such as gVisor or microVMs when the risk warrants it.

I also think model gateways will become increasingly important: not simply as a way to route requests between models, but for things like rate limiting, model selection, caching, policy enforcement, and cost controls. Because cost isn’t theoretical. A badly behaved agent can turn a small task into hundreds of model calls very quickly.

Then there is the tool layer. Whether that ultimately settles around MCP or something else, I think the important idea is broader than the protocol itself: tools need to become governed interfaces. Authentication, authorization, auditing, rate limits, and input/output controls should exist at that boundary.

And then there’s memory and retrieval. This is one area where I think teams can get into serious trouble very quickly. If your RAG system can retrieve information that the user would not otherwise be authorized to access, you’ve created a data-access problem regardless of how good your model is.

Authorization can’t stop at the application layer. It has to follow the data into retrieval.

## 6. What should we build and what should we buy?

My current bet is that a lot of the infrastructure around agents will become commodity: sandboxing, orchestration, model access, routing, basic memory infrastructure. These are areas where I expect platforms and vendors to mature quickly.

The parts I would be much more reluctant to outsource are policy, evaluation, and audit. Those encode something very specific to your organization:

- What are you willing to let an agent do?
- What requires approval?
- What constitutes an acceptable failure rate?
- What data can it access?
- What evidence do you need when something goes wrong?

Those aren’t generic infrastructure questions. They’re engineering, security, business, and regulatory decisions specific to your environment.

## Where I think this is heading

The interesting thing about the current AI cycle is that model capability is improving incredibly quickly. But production adoption isn’t going to be determined by benchmark scores alone. It will be determined by whether we can build systems around these models that people are willing to trust.

Identity. Policy. Observability. Evaluation. Isolation. Cost control. Reliable recovery.

These aren’t new disciplines. They are largely the disciplines we’ve been practicing in platform engineering and SRE for years. The difference is that we’re now applying them to systems that can reason, make decisions, and take actions.

That’s what makes agents exciting. And that’s also what makes them an infrastructure problem as much as an AI problem.

If you’re building agents in production, I’m curious what is hurting first for you: permissions, human review load, cost, debugging, or something else entirely?

---

*Originally published on [LinkedIn](https://www.linkedin.com/pulse/hard-part-agents-isnt-model-ravikumar-buragapu-3nrwc/). The follow-up, [Earning Agentic Autonomy: An Eval Architecture for AI Agents](/blog/earning-agentic-autonomy), covers how to measure what the agent is allowed to do.*
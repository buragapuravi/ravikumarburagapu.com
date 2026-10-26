# Prompt and Context Design

A system instruction defines the assistant's role and response boundaries. A user message contains the question to answer. Retrieved context should be clearly separated from instructions. A citation marker can connect an answer statement to a source passage. An abstention instruction tells the model what to do when evidence is insufficient.

Context ordering can affect which evidence a model uses. A relevant passage should be easy to distinguish from unrelated material. A prompt should avoid including the same passage repeatedly. A concise answer format makes automated review easier. Structured output should be validated before downstream use.

A prompt injection is untrusted text that attempts to change model behavior. Retrieved documents should be treated as data rather than instructions. Tool permissions should be enforced outside the language model. A model response should not itself grant access to a protected resource. Sensitive content should only be sent to an approved model endpoint.

A generation temperature controls randomness in token selection. A deterministic evaluation sets a fixed seed when supported. A model version should be recorded with experiment results. A context window sets an upper bound on input and output tokens. Truncation can silently remove evidence if the prompt exceeds that limit.

A grounded answer uses facts supported by the provided passages. A citation should be checked against the exact retrieved text. A refusal is appropriate when the corpus lacks sufficient evidence. A comparison question may require evidence from multiple documents. An answer should distinguish direct evidence from an inference.

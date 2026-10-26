# Answer Evaluation

Exact match gives full credit only when a normalized answer matches its reference. Token overlap can provide a rough partial-coverage signal. An LLM judge estimates whether an answer is correct and supported by evidence. Human review is needed to validate judge reliability on a domain task. A judge prompt should include the question, reference, answer, and source evidence.

Retrieval recall and answer accuracy measure different stages of a RAG pipeline. High retrieval recall does not guarantee a grounded answer. An answer can be correct by chance even when retrieval misses its evidence. Citation correctness checks whether cited passages support the associated claims. Unsupported claims should be counted separately from stylistic issues.

A benchmark question should have a clearly identified evidence passage. Reference answers should be concise and unambiguous. Questions should cover both common and difficult cases. A benchmark should disclose how examples were sampled. A public dataset should retain its license and attribution.

Latency measurement should state whether it includes embedding, retrieval, generation, and network time. Warm-up requests can reduce one-time startup effects. Repeated trials help expose timing variation. A percentile latency summarizes the slow tail of measured requests. Hardware and software versions are part of a reproducible benchmark report.

Cost per query depends on the model's input and output token prices. An extractive baseline does not incur hosted model API cost. Token pricing changes over time and should be supplied as benchmark configuration. Retrieval compute and storage costs are separate from model token charges. A zero configured price means unknown or unpriced, not free production service.

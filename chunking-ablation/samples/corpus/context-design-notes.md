# Context Design Notes

A retrieval pipeline selects passages before a response model reads them. Passage boundaries affect how much neighboring explanation is available. A short result can miss a qualification, while a long result can contain many unrelated details. Repeating text between results can use space that might otherwise carry distinct evidence.

Prompt construction should preserve the question and provide source material in a predictable format. A model can be asked to abstain when the source does not support an answer. Its response still needs review, because fluent wording is not proof of correctness.

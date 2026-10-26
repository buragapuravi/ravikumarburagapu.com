# Data Handling and Governance

Data minimization limits collection to information needed for a stated purpose. Data retention defines how long stored information is kept. A deletion workflow should cover both source files and derived indexes. Access control determines which identities may read or change a dataset. An audit log records sensitive administrative actions.

An approved data source has a documented owner and permitted use. A public URL does not automatically grant permission to republish its contents. A corpus manifest can record origin, license, and ingestion date. A data steward reviews changes to sensitive collections. A separate development fixture can avoid exposing production records.

Encryption at rest protects stored data on persistent media. Transport encryption protects data sent across a network. A secret manager stores credentials outside application source code. A least-privilege role receives only the permissions required for its task. Credential rotation replaces keys on a defined schedule.

A privacy review identifies personal information and its processing purpose. A threat model documents likely misuse and mitigation boundaries. A retention policy should have an actionable deletion process. A restricted corpus should not be copied into a public demo. A model provider's data terms should be reviewed before sending documents.

A backup supports recovery after data loss. A restore test verifies that a backup can be used. An access review checks whether permissions remain appropriate. A change record links an update to its reason and approver. A governance control should be testable and assigned to an accountable owner.

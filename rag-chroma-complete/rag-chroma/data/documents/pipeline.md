# Atlas Kitchen — Telemetry pipeline
This is fictional documentation for the RAG demonstration.

The MQTT consumer validates each telemetry message with a Pydantic schema. Invalid messages go to a dead-letter queue with a validation reason. Valid messages are normalized to UTC and written into SQL Server in batches of 500 rows or every five seconds, whichever occurs first.

The table is named device_telemetry. Its unique key is the pair device_id and timestamp_utc. Database operations use parameterized statements and transactions. Transient connection failures are retried with exponential backoff.

Missing temperature measurements remain NULL in the raw table. Interpolation is performed only in a separate feature-building stage, so raw measurements remain unchanged. Daily reconciliation compares input message counts, rejected message counts, duplicates, and inserted rows.

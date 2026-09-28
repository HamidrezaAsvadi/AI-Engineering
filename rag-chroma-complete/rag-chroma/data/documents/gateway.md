# Atlas Kitchen — Gateway connection guide
This is fictional documentation for the RAG demonstration, not a real product manual.

The Atlas gateway reads appliance telemetry over Modbus TCP. It publishes messages to an MQTT broker using TLS on port 8883. Each message contains device_id, timestamp_utc, temperature_c, setpoint_c, and heater_on.

When the network connection is unavailable, the gateway buffers telemetry locally for up to 24 hours. After reconnection it replays buffered messages in timestamp order. The ingestion service deduplicates records using device_id and timestamp_utc.

The gateway sends a heartbeat every 30 seconds. Three missing heartbeats trigger an offline alert. Operators should check network connectivity and gateway power before restarting the gateway.

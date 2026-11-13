# Notebook connection startup contract

Before implementation for the repeated pre-worker #1585 dispatch failure:

- Preserve official CLI 0.7.4, shared installed files, credentials, registry,
  allocation ownership and every scientific/dependency/adapter byte. Change only
  the initial WebSocket connection wait inside a separate process-local launcher.
- Reproduce a connection that opens after the original wait expires. The
  original behavior must return disconnected; the launcher must wait for the
  same connection and allow the caller to continue exactly once, without retry.
- A connection that never opens must fail explicitly within the new bound;
  preserve original errors and restore the normal channel shutdown timeout.
  Repeated installation, unexpected client source/version and invalid bounds
  must fail, rather than silently applying a different compatibility patch.
- Use lightweight CPU tests with a delayed connection event and the original
  upstream channel implementation separately. Existing campaign/adapter tests
  do not exercise the host CLI WebSocket handshake, so they are reused as
  previously qualified evidence, with focused new transport tests only.
- Target at most ten CPU cases, under one second each for simulated tests; one
  actual upstream delayed-handshake test may take twelve seconds. A foreground
  constant-only real Notebook probe, including cleanup, must finish in thirty
  seconds and retain the same registered kernel. No model/sample RNG or GPU
  workload is involved.

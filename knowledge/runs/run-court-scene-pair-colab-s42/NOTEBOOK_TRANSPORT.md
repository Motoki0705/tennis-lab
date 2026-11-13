# Host Notebook connection startup

The original client waits ten seconds for its WebSocket to open, then returns
without checking readiness. Two recovery submissions reported connection loss
before their sockets opened, after approximately ten and twenty-five seconds.
No remote worker had started. The owned VM and idle kernel remained healthy;
subsequent constant-only probes succeeded with the original CLI.

`notebook_client.py` is a separate host launcher around the unchanged official
CLI. It verifies the qualified installed versions and source bytes, then extends
only the initial channel connection wait to sixty seconds. It checks readiness
explicitly and closes only the client on timeout. Normal shutdown and execution
timeouts are retained. It neither retries a submission nor restarts a kernel,
allocates a VM, changes credentials or writes installed package files.

Invoke the launcher using the qualified host CLI interpreter through a dedicated
executable supplied to the current main helper's `--colab-bin`. Record the
launcher commit, executable bytes and installed dependency inventory separately
from both the scientific source and the closed-evaluation adapter. The parent
owns the long native transport; the operator registers the single campaign only
after the actual empty-service acknowledgment.

The contracts in [NOTEBOOK_TEST_PLAN.md](NOTEBOOK_TEST_PLAN.md) are checked by
`test_notebook_client.py`, a delayed-handshake check using the installed upstream
channel class, and a thirty-second foreground constant-only Notebook probe.
The scientific campaign, adapter, source, environment and evaluation checks stay
unchanged. This is a scoped host transport repair for the recovery.

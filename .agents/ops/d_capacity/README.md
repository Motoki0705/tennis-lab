# Physical D: capacity watchdog

The local training queue, dispatcher and common agent runner check Windows D:
at admission. The shared [capacity module](../lib/d_capacity.py) verifies that
`/mnt/d` is the D: volume root mounted via DrvFs and reads available bytes as
`statvfs("/mnt/d").f_bavail * f_frsize`. A start is allowed at or above the
threshold. An invalid mount, configuration or measurement refuses admission.
The checks apply to new starts. Running workloads continue.

The default is 20 GiB. To change it across the host, create the readable
`/etc/tennis-lab-d-capacity.toml` with a positive integer:

```toml
threshold_gib = 20
```

The local checks and watchdog read the file on each check. For an isolated
check, set `TENNIS_D_CAPACITY_CONFIG` to the path of another TOML file.

## Watchdog and CI queue

`tennis-lab-d-capacity.service` runs under the main checkout owner's user
manager. It samples every 30 seconds and atomically publishes
`/run/tennis-lab-d-capacity/status.json` with available bytes, threshold,
decision and Unix measurement time in nanoseconds. The queue service reads that
file while its mount namespace hides `/mnt/d`. A measurement error, missing or
invalid status, a future measurement, or a sample older than 90 seconds blocks
new admissions. The queue retains the FIFO head and checks again until a fresh
allowed decision arrives.

The watchdog asks Windows PowerShell to show a toast on the first low sample,
on a transition to low, and on recovery. Toasts use Windows Terminal's
registered notification identity. Windows interop and Windows Terminal
notifications must be enabled. The watchdog records each transition before
sending the toast, so a restart or an uncertain send result does not repeat it.
Measurement and toast errors go to the user journal. [Windows toast API](https://learn.microsoft.com/en-us/windows/win32/shell/quickstart-sending-desktop-toast)

## Deployment

From the updated main checkout, follow the [CI queue update
procedure](../../../scripts/github_actions/README.md#物理dの容量保護を導入する).
The installer creates a tmpfiles rule for the shared state directory. Its
owner is the main checkout owner, and the CI user has read access. After
installation, check the service and status.

```bash
spin ops status d-capacity
spin ops logs d-capacity --lines 20
cat /run/tennis-lab-d-capacity/status.json
```

Use these commands to verify the local verdict and the cached verdict the CI
queue reads.

```bash
python3 -B .agents/ops/lib/d_capacity.py
python3 -B .agents/ops/lib/d_capacity.py --state-file /run/tennis-lab-d-capacity/status.json
```

A refusal prints its reason and exits with status 75.

Other operations can call `lib.windows_toast.send_windows_toast(title, message)`
with `.agents/ops` on the Python import path. The helper raises `ToastError` when
PowerShell fails or times out.

"""Send a Windows toast from WSL through the installed PowerShell."""

from __future__ import annotations

import base64
import subprocess
from xml.sax.saxutils import escape

POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
WINDOWS_TERMINAL_APP_ID = "Microsoft.WindowsTerminal_8wekyb3d8bbwe!App"


class ToastError(RuntimeError):
    pass


def send_windows_toast(title: str, message: str) -> None:
    """Submit one toast under Windows Terminal's registered notification identity."""
    toast_xml = (
        '<toast><visual><binding template="ToastGeneric">'
        f"<text>{escape(title)}</text><text>{escape(message)}</text>"
        "</binding></visual></toast>"
    )
    payload = base64.b64encode(toast_xml.encode("utf-8")).decode("ascii")
    command = f"""
$ErrorActionPreference = 'Stop'
$null = [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime]
$null = [Windows.UI.Notifications.ToastNotification, Windows.UI.Notifications, ContentType=WindowsRuntime]
$xml = [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType=WindowsRuntime]::new()
$xml.LoadXml([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{payload}')))
$notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('{WINDOWS_TERMINAL_APP_ID}')
if ($notifier.Setting -ne 'Enabled') {{ throw "Windows Terminal notifications unavailable ($($notifier.Setting))" }}
$notifier.Show([Windows.UI.Notifications.ToastNotification]::new($xml))
"""
    encoded = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
    try:
        result = subprocess.run(
            [
                POWERSHELL,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-EncodedCommand",
                encoded,
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ToastError(f"cannot run Windows PowerShell: {exc}") from exc
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ToastError(f"Windows PowerShell exited {result.returncode}: {detail}")

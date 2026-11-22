"""Send a Windows toast from WSL through the installed PowerShell."""

from __future__ import annotations

import base64
import subprocess
from xml.sax.saxutils import escape

POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
WINDOWS_TERMINAL_APP_ID = "Microsoft.WindowsTerminal_8wekyb3d8bbwe!App"
PRE_SUBMISSION_EXIT = 10


class ToastError(RuntimeError):
    """Failure while preparing or submitting the Windows toast."""


class ToastNotSent(ToastError):
    """PowerShell could not reach the Windows toast submission call."""


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
try {{
    $null = [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime]
    $null = [Windows.UI.Notifications.ToastNotification, Windows.UI.Notifications, ContentType=WindowsRuntime]
    $xml = [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType=WindowsRuntime]::new()
    $xml.LoadXml([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{payload}')))
    $notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('{WINDOWS_TERMINAL_APP_ID}')
    if ($notifier.Setting -ne 'Enabled') {{ throw "Windows Terminal notifications unavailable ($($notifier.Setting))" }}
    $toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
}} catch {{
    [Console]::Error.WriteLine($_.ToString())
    exit {PRE_SUBMISSION_EXIT}
}}
$notifier.Show($toast)
"""
    encoded = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
    try:
        process = subprocess.Popen(
            [
                POWERSHELL,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-EncodedCommand",
                encoded,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise ToastNotSent(f"cannot start Windows PowerShell: {exc}") from exc
    with process:
        try:
            stdout, stderr = process.communicate(timeout=15)
        except (OSError, subprocess.TimeoutExpired) as exc:
            process.kill()
            raise ToastError(f"cannot confirm the Windows toast: {exc}") from exc
    if process.returncode:
        detail = stderr.strip() or stdout.strip()
        error = (
            ToastNotSent if process.returncode == PRE_SUBMISSION_EXIT else ToastError
        )
        raise error(f"Windows PowerShell exited {process.returncode}: {detail}")

"""Conservative terminal causes supported by confirmed scores and V4 labels."""

from __future__ import annotations

from dataclasses import dataclass

from src.tennis_scene.analytics.ingest.score_adapter import ScoredClip
from src.tennis_scene.analytics.ingest.v4_reader import SERVE_SUBTYPES, V4Clip, V4Event
from src.tennis_scene.analytics.metrics import TerminalCause


@dataclass(frozen=True)
class TerminalEvidence:
    cause: TerminalCause
    event_ids: tuple[str, ...] = ()
    missing_reason: str | None = None


def _certain_link(clip: V4Clip, shot: V4Event, outcome: V4Event) -> bool:
    return (
        shot.status == outcome.status == "certain"
        and shot.frame < outcome.frame
        and bool(clip.coverage[shot.frame : outcome.frame + 1].all())
    )


def _double_fault(clip: V4Clip, events: list[V4Event]) -> bool:
    # Each observed attempt has its own certain serve and certain outcome.
    # A replay repeats the attempt and never counts toward the two faults.
    if len(events) % 2:
        return False
    faults = 0
    server_side = events[0].court_side
    for serve, outcome in zip(events[::2], events[1::2], strict=True):
        if (
            serve.subtype not in SERVE_SUBTYPES
            or server_side not in (-1, 1)
            or serve.court_side != server_side
            or outcome.subtype not in ("net_fault", "service_let")
            or faults == 2
            or not _certain_link(clip, serve, outcome)
        ):
            return False
        faults += outcome.subtype == "net_fault"
    return faults == 2


def derive_terminal_cause(scored: ScoredClip) -> TerminalEvidence:
    score = scored.confirmed
    if score is None:
        return TerminalEvidence("unknown", missing_reason="score_not_confirmed")
    events = [event for event in scored.v4.events if event.in_point]
    if len(events) < 2 or events[-1].subtype != "net_fault":
        return TerminalEvidence("unknown", missing_reason="no_terminal_net_fault_label")

    shot, net = events[-2:]
    if shot.kind != "shot" or not _certain_link(scored.v4, shot, net):
        return TerminalEvidence(
            "unknown", missing_reason="terminal_contact_not_certain"
        )
    if shot.subtype in SERVE_SUBTYPES:
        if score.server != score.winner and _double_fault(scored.v4, events):
            return TerminalEvidence(
                "double_fault", tuple(event.source_event_id for event in events)
            )
        return TerminalEvidence(
            "unknown", missing_reason="two_service_faults_not_established"
        )

    serve = None
    for event in events[:-2]:
        if event.subtype in SERVE_SUBTYPES:
            serve = event
        elif event.subtype in ("service_let", "net_fault"):
            serve = None
    if (
        serve is None
        or serve.status != "certain"
        or serve.frame >= shot.frame
        or serve.court_side not in (-1, 1)
        or shot.court_side not in (-1, 1)
    ):
        return TerminalEvidence(
            "unknown", missing_reason="terminal_hitter_team_unresolved"
        )

    # Scores identify the serving team. Both V4 sides are local to the clip.
    winner_lost_contact = (serve.court_side == shot.court_side) == (
        score.server == score.winner
    )
    if winner_lost_contact:
        return TerminalEvidence(
            "unknown", missing_reason="terminal_contact_conflicts_with_winner"
        )
    return TerminalEvidence(
        "net_error", (serve.source_event_id, shot.source_event_id, net.source_event_id)
    )

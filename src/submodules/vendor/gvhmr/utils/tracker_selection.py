from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence

import numpy as np


def _gather_available_ids(track_history: Sequence[Sequence[dict]]) -> list[int]:
    ids = set()
    for frame in track_history:
        for det in frame:
            if "id" in det:
                ids.add(int(det["id"]))
    return sorted(ids)


def _load_video_frames(video_path: str):
    from src.utils.video.reader import read_video_rgb

    try:
        return read_video_rgb(video_path)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[Tracker UI] Failed to load video frames: {exc}", file=sys.stderr)
        return None


def _select_ids_via_cli(available_ids: Sequence[int], default: Iterable[int]) -> list[int]:
    default_list = list(default)
    prompt = (
        "Enter track id(s) separated by comma.\n"
        f"Available ids: {available_ids}\n"
        f"Press Enter to use default {default_list}: "
    )
    raw = input(prompt)
    if not raw.strip():
        return default_list
    selected = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            print(f"[Tracker UI] Ignoring invalid id '{token}'.", file=sys.stderr)
            continue
        if value not in available_ids:
            print(f"[Tracker UI] Id {value} not in available list.", file=sys.stderr)
            continue
        selected.append(value)
    if not selected:
        print("[Tracker UI] No valid ids selected. Using defaults.", file=sys.stderr)
        return default_list
    return selected


def select_track_ids(track_history: Sequence[Sequence[dict]], video_path: str, suggested_ids: Iterable[int]) -> list[int]:
    """Launch an interactive matplotlib UI to inspect tracks and return chosen ids."""

    available_ids = _gather_available_ids(track_history)
    if not available_ids:
        raise RuntimeError("No track ids detected in the provided track history.")

    suggested_list = list(suggested_ids)
    if not suggested_list:
        suggested_list = available_ids[:1]

    frames = _load_video_frames(str(video_path))
    ui_failed = False

    if frames is not None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib import patches
            from matplotlib.widgets import Slider

            num_frames = min(len(frames), len(track_history))
            frame_index = 0

            fig, ax = plt.subplots(figsize=(10, 6))
            plt.subplots_adjust(bottom=0.18)
            ax.set_title("Track preview - slide to change frame")
            img_artist = ax.imshow(frames[frame_index])
            artists = []

            def draw(idx: int):
                nonlocal artists
                idx = int(np.clip(idx, 0, num_frames - 1))
                img_artist.set_data(frames[idx])
                for artist in artists:
                    artist.remove()
                artists = []
                ax.set_xlabel(f"Frame {idx + 1}/{num_frames}")
                for det in track_history[idx]:
                    try:
                        x1, y1, x2, y2 = det["bbx_xyxy"]
                    except KeyError:
                        continue
                    rect = patches.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        edgecolor="red",
                        facecolor="none",
                    )
                    text = ax.text(
                        x1,
                        max(y1 - 5, 0),
                        f"id:{int(det['id'])}",
                        color="yellow",
                        fontsize=10,
                        bbox={"facecolor": "black", "alpha": 0.5, "pad": 2},
                    )
                    artists.extend([rect, text])
                    ax.add_patch(rect)
                fig.canvas.draw_idle()

            draw(frame_index)

            slider_ax = fig.add_axes([0.12, 0.07, 0.76, 0.04])
            slider = Slider(
                ax=slider_ax,
                label="Frame",
                valmin=0,
                valmax=num_frames - 1,
                valinit=frame_index,
                valfmt="%0.0f",
            )

            def on_slider(val):
                draw(int(val))

            slider.on_changed(on_slider)

            def on_key(event):
                nonlocal frame_index
                if event.key in {"left", "a"}:
                    frame_index = max(frame_index - 1, 0)
                elif event.key in {"right", "d"}:
                    frame_index = min(frame_index + 1, num_frames - 1)
                else:
                    return
                slider.set_val(frame_index)

            fig.canvas.mpl_connect("key_press_event", on_key)
            plt.show()
        except Exception as exc:  # pragma: no cover - UI fallback
            ui_failed = True
            print(f"[Tracker UI] Matplotlib UI failed ({exc}). Falling back to CLI selection.", file=sys.stderr)
    else:
        ui_failed = True

    if ui_failed:
        print("[Tracker UI] Unable to open interactive preview. Falling back to CLI selection.", file=sys.stderr)

    return _select_ids_via_cli(available_ids, suggested_list)

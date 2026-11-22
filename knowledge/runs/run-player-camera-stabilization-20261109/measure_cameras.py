"""Measure saved C2-derived cameras without bridging missing observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
from scipy.spatial.transform import Rotation


def runs(valid):
    edges = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    return list(zip(edges[::2], edges[1::2], strict=True))


def stats(chunks):
    values = np.concatenate([x for x in chunks if len(x)])
    norms = np.linalg.norm(values, axis=-1) if values.ndim == 2 else np.abs(values)
    return {
        "count": len(values),
        "rms": float(np.sqrt(np.mean(norms**2))),
        "p95": float(np.percentile(norms, 95)),
        "max": float(norms.max()),
    }


def angular_rates(rotations, valid, fps):
    return [
        np.rad2deg(
            Rotation.from_matrix(
                rotations[start + 1 : stop]
                @ rotations[start : stop - 1].transpose(0, 2, 1)
            ).as_rotvec()
        )
        * fps
        for start, stop in runs(valid)
        if stop - start >= 2
    ]


def spectrum(chunks, fps):
    # Same 240-sample windows as planning: skip short runs explicitly.
    eligible = [x for x in chunks if len(x) >= 240]
    if not eligible:
        return None
    densities, weights = [], []
    for chunk in eligible:
        frequency, density = welch(
            chunk,
            fs=fps,
            nperseg=240,
            noverlap=120,
            nfft=256,
            window="hann",
            detrend="linear",
            scaling="density",
            axis=0,
        )
        densities.append(density)
        weights.append(1 + (len(chunk) - 240) // 120)
    power = np.average(densities, axis=0, weights=weights).sum(axis=1)
    power *= frequency[1] - frequency[0]
    return {
        "eligible_samples": sum(len(x) for x in eligible),
        "windows": int(sum(weights)),
        "frequency_hz": frequency.tolist(),
        "power_per_bin": power.tolist(),
        "power_below_1hz": float(power[frequency < 1].sum()),
        "power_3hz_and_above": float(power[frequency >= 3].sum()),
    }


def metrics(transforms, valid, fps):
    rotation = transforms[:, :3, :3]
    omega = angular_rates(rotation, valid, fps)
    alpha = [np.diff(x, axis=0) * fps for x in omega if len(x) >= 2]
    jerk = [np.diff(x, axis=0) * fps for x in alpha if len(x) >= 2]
    translation = [
        np.diff(transforms[start:stop, :3, 3], axis=0) * fps
        for start, stop in runs(valid)
        if stop - start >= 2
    ]
    return {
        "angular_velocity_deg_s": stats(omega),
        "angular_acceleration_deg_s2": stats(alpha),
        "angular_jerk_deg_s3": stats(jerk),
        "position_velocity_m_s": stats(translation),
        "angular_velocity_spectrum": spectrum(omega, fps),
    }


def view_angles(rotations):
    forward = rotations[:, :, 2]
    up = -rotations[:, :, 1]
    yaw = np.arctan2(forward[:, 1], forward[:, 0])
    pitch = np.arctan2(forward[:, 2], np.linalg.norm(forward[:, :2], axis=1))
    right = np.stack([np.sin(yaw), -np.cos(yaw), np.zeros(len(yaw))], axis=1)
    level_up = np.cross(right, forward)
    roll = np.arctan2((up * right).sum(axis=1), (up * level_up).sum(axis=1))
    return np.stack([np.unwrap(yaw), pitch, np.unwrap(roll)], axis=1)


def lag(raw, target, valid, fps):
    # Diagnostic: <=1Hz rates, trim 0.5s edges, search +/-18 frames.
    filt = butter(3, 1, fs=fps, output="sos")
    pairs = []
    for start, stop in runs(valid):
        if stop - start < 180:
            continue
        pairs.append(
            tuple(
                sosfiltfilt(
                    filt, np.diff(view_angles(r[start:stop]), axis=0) * fps, axis=0
                )[30:-30]
                for r in (raw, target)
            )
        )
    result = []
    for axis in range(3):
        scores = []
        for offset in range(-18, 19):
            left = np.concatenate([a[18:-18, axis] for a, _ in pairs])
            right = np.concatenate(
                [b[18 + offset : len(b) - 18 + offset, axis] for _, b in pairs]
            )
            scores.append(float(np.corrcoef(left, right)[0, 1]))
        best = int(np.argmax(scores)) - 18
        result.append({"frames": best, "milliseconds": 1000 * best / fps})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((args.raw / "manifest.json").read_text())
    inputs = json.loads((args.inputs / "manifest.json").read_text())
    fps = source["fps"]
    report = {
        "schema": "player_view_stabilization_metrics_v1",
        "fps": fps,
        "angular_difference_basis": "fixed court coordinates",
        "lag_method": "<=1Hz yaw/pitch/roll angular-rate correlation; +/-18 frames; positive means delayed",
        "spectral_gaps": "skip runs shorter than 240 derivative samples; never join across gaps",
        "players": {},
    }
    for player in (0, 1):
        with np.load(args.raw / f"player_{player}.npz") as archive:
            raw = {key: archive[key] for key in archive.files}
        valid = raw["camera_valid"]
        transform = raw["camera_to_court"]
        entry = {
            "valid_frames": int(valid.sum()),
            "valid_runs": [[int(a), int(b)] for a, b in runs(valid)],
            "raw": metrics(transform, valid, fps),
            "variants": {},
        }
        for variant in inputs["packages"]:
            name = variant["name"]
            with np.load(
                args.inputs / "cameras" / name / f"player_{player}.npz"
            ) as derived:
                result = derived["camera_to_court"]
                for key in raw:
                    if key != "camera_to_court":
                        np.testing.assert_array_equal(derived[key], raw[key])
            np.testing.assert_array_equal(result[:, :3, 3], transform[:, :3, 3])
            if variant["sigma_seconds"] == 0:
                np.testing.assert_array_equal(result, transform)
            deviation = np.rad2deg(
                Rotation.from_matrix(
                    result[valid, :3, :3] @ transform[valid, :3, :3].transpose(0, 2, 1)
                ).magnitude()
            )
            entry["variants"][name] = {
                "sigma_seconds": variant["sigma_seconds"],
                "full": metrics(result, valid, fps),
                "first_240_frames": metrics(result[:240], valid[:240], fps),
                "correction_degrees": stats([deviation]),
                "lag_yaw_pitch_roll": lag(
                    transform[:, :3, :3], result[:, :3, :3], valid, fps
                ),
                "positions_masks_reasons_and_timeline_unchanged": True,
            }
        report["players"][str(player)] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

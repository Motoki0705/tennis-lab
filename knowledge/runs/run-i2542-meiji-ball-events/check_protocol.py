"""Check the one-to-one scoring and exclusion rules without benchmark assets."""

from itertools import combinations_with_replacement, permutations

import torch
from evaluate import pair_frames, report, score_events
from predict import net_probability_window


def brute_force(left, right, tolerance):
    costs = {
        (min(len(left), len(right)) - len(valid), sum(valid))
        for i in permutations(left, min(len(left), len(right)))
        for j in permutations(right, min(len(left), len(right)))
        for valid in [
            [abs(a - b) for a, b in zip(i, j, strict=True) if abs(a - b) <= tolerance]
        ]
    }
    return min(costs)


def check_optimum():
    frame_lists = [
        list(frames)
        for count in range(4)
        for frames in combinations_with_replacement(range(5), count)
    ]
    for left in frame_lists:
        for right in frame_lists:
            for tolerance in range(3):
                pairs = pair_frames(left, right, tolerance)
                assert (
                    len({a for a, _ in pairs})
                    == len({b for _, b in pairs})
                    == len(pairs)
                )
                distances = [abs(left[a] - right[b]) for a, b in pairs]
                assert all(distance <= tolerance for distance in distances)
                assert (
                    min(len(left), len(right)) - len(pairs),
                    sum(distances),
                ) == brute_force(left, right, tolerance)


def check_exclusions():
    labels = [
        {"id": f"e{frame}", "frame": frame, "status": status, "in_point": frame < 10}
        for frame, status in [
            (5, "certain"),
            (12, "probable"),
            (20, "uncertain"),
            (30, "certain"),
        ]
    ]
    predictions = [{"frame": frame} for frame in (4, 7, 12, 19, 21, 30, 31)]
    rows = score_events(labels, predictions, {30, 31}, 2)
    result = report([rows], 60)
    assert (result["tp"], result["fp"], result["fn"]) == (2, 2, 0)
    assert result["median_signed_error_frames"] == -0.5
    assert result["mean_abs_error_frames"] == 0.5
    assert result["excluded"] == {
        "uncertain_labels": 1,
        "masked_labels": 1,
        "masked_predictions": 2,
        "uncertain_predictions": 1,
    }
    assert result["by_point_context"] == {
        "in_point": {"matched": 1, "missed": 0},
        "out_of_point": {"matched": 1, "missed": 0},
    }
    empty = report([], 60)
    assert all(
        empty[key] is None
        for key in ("precision", "recall", "f1", "mean_abs_error_frames")
    )


def check_net_window():
    probabilities = torch.zeros(13, 3)
    probabilities[4, 0] = 0.75
    probabilities[10, 1] = 0.625
    probabilities[[0, 1, 11, 12], 2] = torch.tensor([0.875, 0.5, 0.5, 1.0])
    middle = net_probability_window(probabilities, 6)
    assert (middle["start_frame"], middle["end_frame"]) == (1, 11)
    assert middle["maxima"] == {
        "shot": {"frame": 4, "probability": 0.75},
        "bounce": {"frame": 10, "probability": 0.625},
        "net": {"frame": 1, "probability": 0.5},
    }
    assert middle["probabilities_by_frame"][-1] == [0, 0, 0.5]
    start = net_probability_window(probabilities, 0)
    assert (start["start_frame"], start["end_frame"]) == (0, 5)
    assert start["maxima"]["net"] == {"frame": 0, "probability": 0.875}
    end = net_probability_window(probabilities, 12)
    assert (end["start_frame"], end["end_frame"]) == (7, 12)
    assert end["maxima"]["net"] == {"frame": 12, "probability": 1.0}


if __name__ == "__main__":
    check_optimum()
    check_exclusions()
    check_net_window()
    print("Verified matching, exclusions, metric denominators and probability windows")

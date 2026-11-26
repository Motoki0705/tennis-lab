"""Check the one-to-one scoring and exclusion rules without benchmark assets."""

from itertools import combinations_with_replacement, permutations

from evaluate import pair_frames, report, score_events


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


if __name__ == "__main__":
    check_optimum()
    check_exclusions()
    print("Matching objective, one-to-one exclusions and metric denominators verified")

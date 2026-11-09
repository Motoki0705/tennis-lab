"""Prepare a real exported store bundle for the two offline browser scenarios."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from tests.support.agents_ops.explain_html import (
    candidate_for,
    schema,
    store_api,
    validation_for,
)
from tests.support.agents_ops.explain_html_visuals import make_visual_bundle, visuals


def main(root: Path) -> None:
    document, payloads = make_visual_bundle()
    later = document.figures[1].model_copy(update={"figure_id": "later-error"})
    document = visuals.render_document(
        document.model_copy(
            update={
                "figures": [*document.figures, later],
                "html": document.html.replace(
                    "</aside>", '<figure data-figure-id="later-error"></figure></aside>'
                ),
            }
        )
    )
    visuals.validate_visuals(document, payloads)
    store = store_api.RevisionStore(root / "store")
    candidate = candidate_for(document)
    first = store.import_initial(candidate, payloads, validation_for(candidate))
    # The SVG is a valid asset reference and digest. The visual policy must still
    # reject its content before publication, retaining the complete current view.
    attack = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 7 3"><script>window.PWNED=1</script></svg>'
    body = document.model_dump(mode="json")
    body["assets"][0]["sha256"] = hashlib.sha256(attack).hexdigest()
    bad = candidate_for(
        schema.parse(schema.Document, json.dumps(body)),
        before=first,
        request_id="unsafe-svg",
        changes=[{"content_id": "geometry", "kind": "updated"}],
        evidence_changes=[],
    )
    staged = store.stage(bad, {"court": attack})
    try:
        visuals.validate_visuals(
            store.staged_candidate("ray-geometry", staged.stage_id).document,
            {
                asset.asset_id: store.staged_asset_bytes(
                    "ray-geometry", staged.stage_id, asset.asset_id
                )
                for asset in bad.document.assets
            },
        )
    except schema.ContractError as exc:
        rejected = exc.error.code
    else:
        raise AssertionError("Unsafe SVG passed visual validation")
    assert store.current("ray-geometry").revision_id == first.revision_id
    snapshot = store.snapshot("ray-geometry", first.revision_id)
    result = {
        "index": str(root / "store/exports" / snapshot.snapshot_id / "index.html"),
        "rejected": rejected,
        "revision": first.revision_id,
        "current": store.current("ray-geometry").revision_id,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())

"""Identity documents and atomic JSON receipts of the component pipeline.

``json_value`` inlines every value (arrays as lists) for identities and
receipts; component outputs are stored by ``storage.codec``, which moves
arrays into checksummed ``.npy`` files instead.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.utils.io import json_value as json_value
from src.utils.io import write_json_atomic as write_json_atomic


def document_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()

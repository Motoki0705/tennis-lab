"""External literature identities shared by Papers and citation exploration."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from urllib.parse import unquote, urlsplit, urlunsplit

NAMESPACES = ("doi", "arxiv", "openalex", "url")


def normalize_identifier(namespace: str, value: str) -> str:
    if namespace not in NAMESPACES or not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid external ID: {namespace}:{value}")
    value = value.strip()
    if namespace == "doi":
        value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.I)
        value = re.sub(r"^doi:\s*", "", value, flags=re.I)
        value = unquote(value).lower()
        valid = bool(re.fullmatch(r"10\.[0-9]{4,9}/[^\s?#]+", value))
    elif namespace == "arxiv":
        value = re.sub(r"^https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/", "", value, flags=re.I)
        value = re.sub(r"^arxiv:\s*", "", value, flags=re.I).lower()
        value = re.sub(r"\.pdf$", "", value)
        value = re.sub(r"v[1-9][0-9]*$", "", value)
        valid = bool(re.fullmatch(r"(?:[0-9]{4}\.[0-9]{4,5}|[a-z][a-z0-9.-]*/[0-9]{7})", value))
    elif namespace == "openalex":
        value = re.sub(r"^https?://(?:api\.)?openalex\.org/(?:works/)?", "", value, flags=re.I).upper()
        valid = bool(re.fullmatch(r"W[1-9][0-9]*", value))
    else:
        try:
            url = urlsplit(value)
            if url.scheme.lower() not in {"http", "https"} or not url.hostname or url.username or url.password or re.search(r"\s", value):
                raise ValueError("expected an HTTP(S) URL without credentials or whitespace")
            value = urlunsplit((url.scheme.lower(), url.netloc.lower(), url.path or "/", url.query, ""))
        except ValueError as exc:
            raise ValueError(f"invalid url external ID: {value}") from exc
        valid = True
    if not valid:
        raise ValueError(f"invalid {namespace} external ID: {value}")
    return value


def normalize_external_ids(values: Mapping[str, str]) -> dict[str, str]:
    return {namespace: normalize_identifier(namespace, value) for namespace, value in values.items()}


def identity_keys(values: Mapping[str, str]) -> set[str]:
    return {f"{namespace}:{value}" for namespace, value in normalize_external_ids(values).items()}


def initial_identity_key(values: Mapping[str, str]) -> str:
    normalized = normalize_external_ids(values)
    for namespace in NAMESPACES:
        if namespace in normalized:
            return f"{namespace}:{normalized[namespace]}"
    raise ValueError("a citation document needs at least one external ID")


def citation_id(identity_key: str) -> str:
    return "cite-" + hashlib.sha256(identity_key.encode("utf-8")).hexdigest()[:16]


def external_id_index(entries: Iterable[tuple[str, Mapping[str, str]]]) -> dict[str, str]:
    """Reject aliases that connect multiple documents instead of choosing one."""
    index: dict[str, str] = {}
    seen: set[str] = set()
    for document_id, values in entries:
        if document_id in seen:
            raise ValueError(f"duplicate document id: {document_id}")
        seen.add(document_id)
        for key in identity_keys(values):
            if key in index and index[key] != document_id:
                raise ValueError(f"duplicate literature: {key} belongs to {index[key]} and {document_id}")
            index[key] = document_id
    return index


def resolve_identity(values: Mapping[str, str], index: Mapping[str, str]) -> str | None:
    matches = {index[key] for key in identity_keys(values) if key in index}
    if len(matches) > 1:
        raise ValueError(f"ambiguous literature aliases: {sorted(matches)}; consolidate documents and rewrite references")
    return next(iter(matches), None)

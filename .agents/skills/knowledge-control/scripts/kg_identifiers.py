"""External literature identities shared by Papers and citation exploration."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from urllib.parse import unquote, urlsplit, urlunsplit

from kg_schema import http_url

NAMESPACES = ("doi", "arxiv", "openalex", "url")


def normalize_identifier(namespace: str, value: str) -> str:
    if namespace not in NAMESPACES or not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid external ID: {namespace}:{value}")
    value = value.strip()
    if namespace == "doi":
        value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.I | re.ASCII)
        # Keep Unicode whitespace outside the ASCII-only prefix comparison.
        value = re.sub(r"^(?ai:doi:)\s*", "", value)
        value = unquote(value).lower()
        valid = bool(re.fullmatch(r"10\.[0-9]{4,9}/[^\s?#]+", value))
    elif namespace == "arxiv":
        value = re.sub(r"^https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/", "", value, flags=re.I | re.ASCII)
        value = re.sub(r"^(?ai:arxiv:)\s*", "", value).lower()
        value = re.sub(r"\.pdf$", "", value)
        value = re.sub(r"v[1-9][0-9]*$", "", value)
        valid = bool(re.fullmatch(r"(?:[0-9]{4}\.[0-9]{4,5}|[a-z][a-z0-9.-]*/[0-9]{7})", value))
    elif namespace == "openalex":
        value = re.sub(r"^https?://(?:api\.)?openalex\.org/(?:works/)?", "", value, flags=re.I | re.ASCII).upper()
        valid = bool(re.fullmatch(r"W[1-9][0-9]*", value))
    else:
        try:
            if not http_url(value):
                raise ValueError("expected an absolute HTTP(S) URL without repairs, credentials or whitespace")
            url = urlsplit(value)
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
    """Reject duplicate documents and inconsistent aliases in the same store."""
    index: dict[str, str] = {}
    seen: set[str] = set()
    for document_id, values in entries:
        if document_id in seen:
            raise ValueError(f"duplicate document id: {document_id}")
        seen.add(document_id)
        match = resolve_identity(values, index)
        if match is not None:
            raise ValueError(f"duplicate literature: external IDs belong to {match} and {document_id}")
        for key in identity_keys(values):
            index[key] = document_id
    return index


def resolve_identity(values: Mapping[str, str], index: Mapping[str, str]) -> str | None:
    """Match an alias only when every shared namespace agrees after normalization."""
    normalized = normalize_external_ids(values)
    matches = {index[key] for key in identity_keys(normalized) if key in index}
    if len(matches) > 1:
        raise ValueError(f"ambiguous literature aliases: {sorted(matches)}; consolidate documents and rewrite references")
    if not matches:
        return None
    match = next(iter(matches))
    for key, owner in index.items():
        if owner != match:
            continue
        namespace, _, known = key.partition(":")
        if namespace in normalized and normalized[namespace] != known:
            raise ValueError(f"conflicting external IDs for {match}: {namespace}={known} versus {normalized[namespace]}; resolve the conflict before joining or merging aliases")
    return match

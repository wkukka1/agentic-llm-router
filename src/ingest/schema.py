"""The one record format the collection server stores: anonymous routing features, no text.

Strict on purpose. An unknown key is refused (so a client that starts sending a prompt,
a path or an id is refused rather than archived), every string is a short token or a
model name (a free-text field cannot hide a sentence), and every number has a range.
:func:`validate_record` returns the record rebuilt from known fields only.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping

SCHEMA = 1
#: the embedding: this many int8 numbers (a 384-d vector projected to 64 and rounded)
EMBEDDING_DIMS = 64

_NAME = re.compile(r"^[A-Za-z0-9_.:/\[\]+\-]{1,80}$")   # a model id or a version: no spaces
_TOKEN = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")          # a category, a role, an error type
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _name(v):
    return isinstance(v, str) and bool(_NAME.match(v))


def _token(v):
    return isinstance(v, str) and bool(_TOKEN.match(v))


def _day(v):
    return isinstance(v, str) and bool(_DAY.match(v))


def _count(v):
    return isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 10**10


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 10**9


def _status(v):
    return isinstance(v, int) and not isinstance(v, bool) and 100 <= v <= 599


def _bool(v):
    return isinstance(v, bool)


def _version(v):
    return v == SCHEMA and isinstance(v, int) and not isinstance(v, bool)


def _power_of_two(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 1 and v & (v - 1) == 0 and v <= 2**30


def _embedding(v):
    return (isinstance(v, list) and len(v) == EMBEDDING_DIMS
            and all(isinstance(x, int) and not isinstance(x, bool) and -127 <= x <= 127 for x in v))


#: field -> (check, may be null). Exactly these keys, all present.
FIELDS: dict[str, tuple[Callable, bool]] = {
    "v": (_version, False),
    "day": (_day, False),
    "gateway": (_name, False),
    "requested_model": (_name, True),
    "routed_model": (_name, True),
    "decision": (_token, False),
    "subagent_role": (_token, True),
    "correction": (_bool, False),
    "thread": (_token, True),
    "status": (_status, True),
    "error_type": (_token, True),
    "latency_s": (_number, True),
    "input_tokens": (_count, True),
    "output_tokens": (_count, True),
    "cache_read_input_tokens": (_count, True),
    "cache_creation_input_tokens": (_count, True),
    "cost_usd": (_number, True),
    "cost_if_requested_usd": (_number, True),
    "prompt_chars": (_power_of_two, True),
    "embedding_model": (_name, False),
    "embedding": (_embedding, True),
}


def validate_record(record: object) -> dict:
    """The record rebuilt from its known fields, or ``ValueError`` naming the field (never its value)."""
    if not isinstance(record, Mapping):
        raise ValueError("a record must be a json object")
    unknown = set(record) - set(FIELDS)
    if unknown:
        raise ValueError(f"unknown field(s): {', '.join(sorted(map(str, unknown))[:3])}")
    out = {}
    for name, (check, nullable) in FIELDS.items():
        if name not in record:
            raise ValueError(f"missing field: {name}")
        value = record[name]
        if value is None:
            if not nullable:
                raise ValueError(f"field may not be null: {name}")
        elif not check(value):
            raise ValueError(f"invalid field: {name}")
        out[name] = value
    return out

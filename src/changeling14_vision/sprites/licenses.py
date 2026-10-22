"""Which sprite licenses may go into training.

Only freely redistributable sheets are used, and every one is credited.
Non-commercial, custom, missing and unknown licenses are excluded.
"""

from __future__ import annotations

ALLOWED = {
    "CC-BY-SA-3.0",
    "CC-BY-SA-4.0",
    "CC-BY-3.0",
    "CC-BY-4.0",
    "CC0-1.0",
}

_ALIASES = {
    "CC0": "CC0-1.0",
    "CC-0": "CC0-1.0",
    "CC0-1": "CC0-1.0",
}


def normalize(license_id: object) -> str | None:
    if not isinstance(license_id, str) or not license_id.strip():
        return None
    value = license_id.strip().upper().replace(" ", "-").replace("_", "-")
    return _ALIASES.get(value, value)


def exclusion_reason(license_id: object) -> str | None:
    """None if the sheet may be used, else a short reason."""
    norm = normalize(license_id)
    if norm is None:
        return "missing license"
    if "-NC" in norm:
        return "non-commercial license"
    if "CUSTOM" in norm:
        return "custom license"
    if norm not in ALLOWED:
        return f"unrecognised license {license_id!r}"
    return None

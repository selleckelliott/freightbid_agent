"""The single PII redaction chokepoint for real-world broker ingestion (Phase 7.1).

Real broker records carry contact PII - email, phone, postal address, a person's name. None
of it is decision-relevant (the engine never reads it) and none of it should ever be persisted
or exported. Every external broker passes through :func:`redact_contact` exactly once, on the
way to a :class:`~application.ingestion.real_broker_schema.BrokerReference`, and the raw values
are dropped on the floor here.

Two redaction modes, by field sensitivity:

* **email / phone -> pseudonymous token.** A deterministic, non-reversible
  ``sha256(salt + normalized_value)`` digest (truncated). It links the *same* contact across
  records (useful for audit dedup) without ever revealing the address/number. The salt is a
  constant - this is pseudonymization, not secret-grade hashing, and is documented as such.
* **address / contact name -> dropped to a boolean.** We keep only *whether* one was present.

:func:`contains_pii` / :func:`assert_no_raw_pii` are the test+guard side: they scan an
about-to-be-exported payload for anything shaped like a raw email or phone number and refuse it,
so a future code path cannot quietly leak PII past this module.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Optional

# Constant, non-secret salt: makes tokens deterministic (same contact -> same token across
# runs/records) so audit can dedup. NOT a security control - PII is dropped, not protected.
PII_SALT = "freightbid.ingestion.v1"

# Shapes used to *detect* raw PII that should never reach an export (defense-in-depth).
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?:\+?\d[\s.\-]?){9,}\d")  # >=10 digits w/ common separators


def _norm(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def hash_token(value: Any, *, prefix: str = "h") -> Optional[str]:
    """Deterministic, non-reversible pseudonym for ``value`` (``None`` for empty)."""
    norm = _norm(value)
    if norm is None:
        return None
    digest = hashlib.sha256(f"{PII_SALT}:{norm.lower()}".encode("utf-8")).hexdigest()
    return f"{prefix}:{digest[:12]}"


def _digits(value: Any) -> Optional[str]:
    norm = _norm(value)
    if norm is None:
        return None
    digits = re.sub(r"\D", "", norm)
    return digits or None


@dataclass(frozen=True)
class RedactedContact:
    """A broker's contact info reduced to presence flags + non-reversible tokens.

    Carries **no** raw email, phone, address, or name - only whether each was present and a
    pseudonymous token for the linkable fields.
    """

    has_email: bool
    has_phone: bool
    has_address: bool
    has_contact_name: bool
    email_token: Optional[str]
    phone_token: Optional[str]

    def as_dict(self) -> dict:
        return {
            "has_email": self.has_email,
            "has_phone": self.has_phone,
            "has_address": self.has_address,
            "has_contact_name": self.has_contact_name,
            "email_token": self.email_token,
            "phone_token": self.phone_token,
        }


def redact_contact(
    *,
    email: Any = None,
    phone: Any = None,
    address: Any = None,
    contact_name: Any = None,
) -> RedactedContact:
    """Reduce raw contact PII to a :class:`RedactedContact`. The only place raw PII is handled."""
    email_norm = _norm(email)
    phone_digits = _digits(phone)
    return RedactedContact(
        has_email=email_norm is not None,
        has_phone=phone_digits is not None,
        has_address=_norm(address) is not None,
        has_contact_name=_norm(contact_name) is not None,
        email_token=hash_token(email_norm, prefix="email") if email_norm else None,
        phone_token=hash_token(phone_digits, prefix="phone") if phone_digits else None,
    )


def contains_pii(value: Any) -> bool:
    """True if ``value`` (recursively, for dict/list) holds anything shaped like a raw email/phone.

    Token strings (``email:...`` / ``phone:...``) are pseudonyms and do not match.
    """
    if isinstance(value, dict):
        return any(contains_pii(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(contains_pii(v) for v in value)
    if isinstance(value, str):
        if value.startswith(("email:", "phone:", "h:")):
            return False
        return bool(_EMAIL_RE.search(value) or _PHONE_RE.search(value))
    return False


def assert_no_raw_pii(payload: Any) -> bool:
    """Guard: raise ``ValueError`` if ``payload`` carries raw email/phone-shaped PII. Returns True."""
    if contains_pii(payload):
        raise ValueError("payload contains raw PII (email/phone-shaped value) past the redaction boundary")
    return True

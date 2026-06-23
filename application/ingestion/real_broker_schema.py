"""Phase 7.1 real-world **broker** contract: validate an external broker record and reduce it to
a redaction-safe :class:`BrokerReference`.

A real broker record mixes two very different kinds of field:

* **decision-relevant, non-PII** - credit bucket, advertised days-to-pay, bonded, quick-pay,
  account age, regulatory MC/DOT numbers. These are kept (a future phase may surface them on an
  audit record).
* **contact PII** - a person's name, email, phone, postal address. These are *never* kept raw:
  :func:`application.ingestion.redaction.redact_contact` turns them into presence flags +
  pseudonymous tokens at the single redaction chokepoint.

Per the Phase 7 decision, a :class:`BrokerReference` is carried as **reference data** (e.g. onto
a decision record/export); it is **not** fed into the winnability/payment ML feature builders.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, field_validator

from application.ingestion.errors import FieldError, RowValidationError
from application.ingestion.real_load_schema import _pydantic_errors_to_fields, clean_number
from application.ingestion.redaction import RedactedContact, redact_contact

_TRUE = {"true", "t", "yes", "y", "1"}
_FALSE = {"false", "f", "no", "n", "0"}


def _parse_bool(v: Any) -> Optional[bool]:
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return None
    if isinstance(v, bool):
        return v
    text = str(v).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"unparseable boolean '{v}'")


def _norm_credit(v: Any) -> Optional[str]:
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return None
    text = str(v).strip()
    return text.upper() if len(text) <= 2 and text.isalpha() else text


class RawExternalBroker(BaseModel):
    """A permissive, alias-aware view of one external broker record (carries contact PII)."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore", str_strip_whitespace=True)

    broker_id: str = Field(validation_alias=AliasChoices("broker_id", "id", "broker"))
    name: Optional[str] = Field(default=None, validation_alias=AliasChoices("name", "broker_name", "company"))

    # Contact PII - redacted on the way to BrokerReference, never kept raw.
    contact_name: Optional[str] = Field(default=None, validation_alias=AliasChoices("contact_name", "contact", "rep"))
    contact_email: Optional[str] = Field(default=None, validation_alias=AliasChoices("contact_email", "email"))
    contact_phone: Optional[str] = Field(default=None, validation_alias=AliasChoices("contact_phone", "phone", "telephone"))
    address: Optional[str] = Field(default=None, validation_alias=AliasChoices("address", "street", "mailing_address"))

    # Public regulatory identifiers (not PII).
    mc_number: Optional[str] = Field(default=None, validation_alias=AliasChoices("mc_number", "mc", "motor_carrier"))
    dot_number: Optional[str] = Field(default=None, validation_alias=AliasChoices("dot_number", "dot", "usdot"))

    # Decision-relevant, non-PII.
    credit_bucket: Optional[str] = Field(default=None, validation_alias=AliasChoices("credit_bucket", "credit", "credit_rating", "credit_score"))
    days_to_pay: Optional[float] = Field(default=None, validation_alias=AliasChoices("days_to_pay", "dtp", "pay_days", "terms_days"))
    bonded: Optional[bool] = Field(default=None, validation_alias=AliasChoices("bonded", "is_bonded"))
    quick_pay_available: Optional[bool] = Field(default=None, validation_alias=AliasChoices("quick_pay_available", "quick_pay", "quickpay"))
    age_days: Optional[float] = Field(default=None, validation_alias=AliasChoices("age_days", "broker_age_days", "account_age_days"))

    _v_number = field_validator("days_to_pay", "age_days", mode="before")(clean_number)
    _v_bool = field_validator("bonded", "quick_pay_available", mode="before")(_parse_bool)
    _v_credit = field_validator("credit_bucket", mode="before")(_norm_credit)


@dataclass(frozen=True)
class BrokerReference:
    """A redaction-safe broker record: decision fields + presence-flagged, tokenized contact.

    Carries no raw email/phone/address/contact name - only the :class:`RedactedContact` tokens.
    """

    broker_id: str
    name: Optional[str]
    mc_number: Optional[str]
    dot_number: Optional[str]
    credit_bucket: Optional[str]
    days_to_pay: Optional[float]
    bonded: Optional[bool]
    quick_pay_available: Optional[bool]
    age_days: Optional[int]
    contact: RedactedContact

    def as_dict(self) -> dict:
        return {
            "broker_id": self.broker_id,
            "name": self.name,
            "mc_number": self.mc_number,
            "dot_number": self.dot_number,
            "credit_bucket": self.credit_bucket,
            "days_to_pay": self.days_to_pay,
            "bonded": self.bonded,
            "quick_pay_available": self.quick_pay_available,
            "age_days": self.age_days,
            "contact": self.contact.as_dict(),
        }


def to_broker_reference(raw: RawExternalBroker) -> BrokerReference:
    """Reduce a validated :class:`RawExternalBroker` to a redacted :class:`BrokerReference`."""
    contact = redact_contact(
        email=raw.contact_email,
        phone=raw.contact_phone,
        address=raw.address,
        contact_name=raw.contact_name,
    )
    return BrokerReference(
        broker_id=raw.broker_id,
        name=raw.name,
        mc_number=raw.mc_number,
        dot_number=raw.dot_number,
        credit_bucket=raw.credit_bucket,
        days_to_pay=raw.days_to_pay,
        bonded=raw.bonded,
        quick_pay_available=raw.quick_pay_available,
        age_days=int(raw.age_days) if raw.age_days is not None else None,
        contact=contact,
    )


def broker_reference_from_mapping(mapping: Any, row_index: int = 0) -> BrokerReference:
    """Validate + redact one raw broker mapping into a :class:`BrokerReference`.

    Raises :class:`RowValidationError` (``row_index`` stamped) for any problem - never a raw
    ``pydantic.ValidationError``, and never echoing PII into the error (identifier is broker_id).
    """
    try:
        raw = RawExternalBroker.model_validate(mapping)
    except ValidationError as exc:
        ident = None
        if isinstance(mapping, dict):
            ident = mapping.get("broker_id") or mapping.get("id")
            ident = str(ident) if ident is not None else None
        fields: List[FieldError] = _pydantic_errors_to_fields(exc, RawExternalBroker)
        raise RowValidationError(row_index, ident, fields) from exc
    return to_broker_reference(raw)

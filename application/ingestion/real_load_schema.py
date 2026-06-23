"""Phase 7.1 real-world **load** contract: validate messy external load-board rows and map them
INTO the existing :class:`domain.models.load.Load`.

This is an **anti-corruption layer**. The internal ``Load`` (and the engine on top of it) is not
touched; instead a permissive :class:`RawExternalLoad` absorbs the variety of a real feed -
aliased column names, money strings (``"$1,450"``), comma/unit numbers (``"22,000 lb"``), full
state names, equipment codes (``"V"`` / ``"FD"``), single pickup/delivery dates instead of
windows, per-mile *or* total rate - and :func:`to_domain_load` normalizes it to the exact field
set the engine already expects. Anything the engine genuinely needs but the feed cannot supply
(coordinates, miles, a rate, a pickup time) is a **structured error**, never a guess - we do not
add a geocoder or any new service in Phase 7.

``inference``/engine code keeps consuming plain ``Load`` objects; only this module knows the feed
dialect.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Type

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, field_validator

from application.ingestion.errors import (
    CODE_MISSING,
    CODE_TYPE,
    CODE_VALUE,
    FieldError,
    RowValidationError,
)
from domain.models.load import Load

# --------------------------------------------------------------------------- #
# Normalization tables
# --------------------------------------------------------------------------- #
EQUIPMENT_ALIASES = {
    "v": "Dry Van", "van": "Dry Van", "dv": "Dry Van", "dry van": "Dry Van", "dryvan": "Dry Van",
    "r": "Reefer", "reefer": "Reefer", "refrigerated": "Reefer", "reef": "Reefer",
    "f": "Flatbed", "fd": "Flatbed", "flat": "Flatbed", "flatbed": "Flatbed",
    "sd": "Step Deck", "step deck": "Step Deck", "stepdeck": "Step Deck", "step-deck": "Step Deck",
    "hs": "Hotshot", "hotshot": "Hotshot", "hot shot": "Hotshot",
    "po": "Power Only", "power only": "Power Only", "power-only": "Power Only",
}

_US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA",
    "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT",
    "VA", "WA", "WV", "WI", "WY", "DC",
}
_STATE_NAME_TO_CODE = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH",
    "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN",
    "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC", "washington dc": "DC", "washington d.c.": "DC",
}

_DT_FALLBACK_FORMATS = ("%m/%d/%Y %H:%M", "%m/%d/%Y", "%Y/%m/%d %H:%M", "%Y/%m/%d")


# --------------------------------------------------------------------------- #
# Coercion helpers (raise ValueError on failure -> pydantic reports the field)
# --------------------------------------------------------------------------- #
def _blank_to_none(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    return v


def clean_money(v: Any) -> Optional[float]:
    v = _blank_to_none(v)
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    cleaned = re.sub(r"[,$\s]", "", str(v))
    cleaned = re.sub(r"(usd|/mi|permile|permi)$", "", cleaned, flags=re.IGNORECASE)
    return float(cleaned)


def clean_number(v: Any) -> Optional[float]:
    v = _blank_to_none(v)
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    cleaned = re.sub(r"[,\s]", "", str(v))
    cleaned = re.sub(r"(lbs?|kg|mi|miles)$", "", cleaned, flags=re.IGNORECASE)
    return float(cleaned)


def normalize_state(v: Any) -> Optional[str]:
    v = _blank_to_none(v)
    if v is None:
        return None
    text = str(v).strip()
    if len(text) == 2 and text.isalpha():
        code = text.upper()
        if code not in _US_STATES:
            raise ValueError(f"unknown US state code '{text}'")
        return code
    code = _STATE_NAME_TO_CODE.get(text.lower())
    if code is None:
        raise ValueError(f"unrecognized state '{text}'")
    return code


def normalize_equipment(v: Any) -> Optional[str]:
    v = _blank_to_none(v)
    if v is None:
        return None
    text = str(v).strip()
    return EQUIPMENT_ALIASES.get(text.lower(), text.title())


def parse_datetime(v: Any) -> Optional[datetime]:
    """Parse an external timestamp to a timezone-aware UTC datetime (``None`` for blank)."""
    v = _blank_to_none(v)
    if v is None:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    text = str(v).strip()
    iso = text[:-1] + "+00:00" if text.endswith("Z") else text
    parsed: Optional[datetime] = None
    try:
        parsed = datetime.fromisoformat(iso)
    except ValueError:
        for fmt in _DT_FALLBACK_FORMATS:
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
    if parsed is None:
        raise ValueError(f"unparseable datetime '{text}'")
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# The raw external load contract
# --------------------------------------------------------------------------- #
class RawExternalLoad(BaseModel):
    """A permissive, alias-aware view of one external load-board row.

    Field types/coercion are enforced here; cross-field domain rules (a rate must be derivable,
    a pickup time must exist, windows must be ordered) live in :func:`to_domain_load`.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore", str_strip_whitespace=True)

    load_id: int = Field(validation_alias=AliasChoices("load_id", "id", "posting_id", "load_number"))
    reference: Optional[str] = Field(default=None, validation_alias=AliasChoices("reference", "ref", "external_id"))
    equipment_type: str = Field(validation_alias=AliasChoices("equipment_type", "equipment", "equip", "trailer_type", "equipment_code"))
    weight: float = Field(validation_alias=AliasChoices("weight", "weight_lbs", "weight_lb", "gross_weight"))

    origin_city: str = Field(validation_alias=AliasChoices("origin_city", "origin", "pickup_city"))
    origin_state: str = Field(validation_alias=AliasChoices("origin_state", "origin_st", "pickup_state"))
    origin_latitude: float = Field(validation_alias=AliasChoices("origin_latitude", "origin_lat", "orig_lat"))
    origin_longitude: float = Field(validation_alias=AliasChoices("origin_longitude", "origin_lon", "origin_lng", "orig_lon", "orig_lng"))

    destination_city: str = Field(validation_alias=AliasChoices("destination_city", "destination", "dest_city", "delivery_city"))
    destination_state: str = Field(validation_alias=AliasChoices("destination_state", "dest_state", "dest_st", "delivery_state"))
    destination_latitude: float = Field(validation_alias=AliasChoices("destination_latitude", "destination_lat", "dest_lat"))
    destination_longitude: float = Field(validation_alias=AliasChoices("destination_longitude", "destination_lon", "destination_lng", "dest_lon", "dest_lng"))

    miles: float = Field(validation_alias=AliasChoices("miles", "trip_miles", "distance", "loaded_miles"))
    total_rate: Optional[float] = Field(default=None, validation_alias=AliasChoices("total_rate", "rate", "offer", "amount", "line_haul"))
    rate_per_mile: Optional[float] = Field(default=None, validation_alias=AliasChoices("rate_per_mile", "rpm", "rate_mile"))

    posted_at: Optional[datetime] = Field(default=None, validation_alias=AliasChoices("posted_at", "created_at", "posted", "posted_date", "post_date"))
    pickup_window_start: Optional[datetime] = Field(default=None, validation_alias=AliasChoices("pickup_window_start", "pickup_start", "pickup_from"))
    pickup_window_end: Optional[datetime] = Field(default=None, validation_alias=AliasChoices("pickup_window_end", "pickup_end", "pickup_to"))
    pickup_date: Optional[datetime] = Field(default=None, validation_alias=AliasChoices("pickup_date", "pickup", "pickup_time"))
    delivery_window_start: Optional[datetime] = Field(default=None, validation_alias=AliasChoices("delivery_window_start", "delivery_start", "delivery_from"))
    delivery_window_end: Optional[datetime] = Field(default=None, validation_alias=AliasChoices("delivery_window_end", "delivery_end", "delivery_to"))
    delivery_date: Optional[datetime] = Field(default=None, validation_alias=AliasChoices("delivery_date", "delivery", "delivery_time", "dropoff"))

    broker_id: Optional[str] = Field(default=None, validation_alias=AliasChoices("broker_id", "broker", "mc", "broker_ref"))

    _v_money = field_validator("total_rate", "rate_per_mile", mode="before")(clean_money)
    _v_number = field_validator("weight", "miles", "origin_latitude", "origin_longitude",
                                "destination_latitude", "destination_longitude", mode="before")(clean_number)
    _v_state = field_validator("origin_state", "destination_state", mode="before")(normalize_state)
    _v_equipment = field_validator("equipment_type", mode="before")(normalize_equipment)
    _v_dt = field_validator("posted_at", "pickup_window_start", "pickup_window_end", "pickup_date",
                            "delivery_window_start", "delivery_window_end", "delivery_date",
                            mode="before")(parse_datetime)

    @field_validator("reference", "broker_id", mode="before")
    @classmethod
    def _v_blank_str(cls, v: Any) -> Any:
        return _blank_to_none(v)


# --------------------------------------------------------------------------- #
# Map a validated raw load into the domain model (cross-field rules here)
# --------------------------------------------------------------------------- #
def to_domain_load(raw: RawExternalLoad) -> Load:
    """Map a validated :class:`RawExternalLoad` to a domain :class:`Load`.

    Resolves the rate (total, else per-mile x miles), derives pickup/delivery windows from a
    single date when no explicit window is given, and defaults ``created_at`` to the posting time
    (or pickup start). Raises :class:`RowValidationError` aggregating every cross-field problem.
    """
    errors: List[FieldError] = []
    identifier = raw.reference or str(raw.load_id)

    if raw.miles <= 0:
        errors.append(FieldError("miles", CODE_VALUE, f"miles must be > 0 (got {raw.miles})"))
    if raw.weight <= 0:
        errors.append(FieldError("weight", CODE_VALUE, f"weight must be > 0 (got {raw.weight})"))

    total_rate: Optional[float] = None
    if raw.total_rate is not None:
        total_rate = raw.total_rate
    elif raw.rate_per_mile is not None and raw.miles > 0:
        total_rate = round(raw.rate_per_mile * raw.miles, 2)
    else:
        errors.append(FieldError("total_rate", CODE_MISSING,
                                 "provide total_rate, or rate_per_mile with positive miles"))
    if total_rate is not None and total_rate <= 0:
        errors.append(FieldError("total_rate", CODE_VALUE, f"rate must be > 0 (got {total_rate})"))

    pickup_start = raw.pickup_window_start or raw.pickup_date
    pickup_end = raw.pickup_window_end or raw.pickup_date or pickup_start
    if pickup_start is None:
        errors.append(FieldError("pickup_window_start", CODE_MISSING,
                                 "provide pickup_window_start/end or a single pickup_date"))
    delivery_start = raw.delivery_window_start or raw.delivery_date
    delivery_end = raw.delivery_window_end or raw.delivery_date or delivery_start
    if delivery_start is None:
        errors.append(FieldError("delivery_window_start", CODE_MISSING,
                                 "provide delivery_window_start/end or a single delivery_date"))

    if pickup_start and pickup_end and pickup_end < pickup_start:
        errors.append(FieldError("pickup_window_end", CODE_VALUE, "pickup window end precedes start"))
    if delivery_start and delivery_end and delivery_end < delivery_start:
        errors.append(FieldError("delivery_window_end", CODE_VALUE, "delivery window end precedes start"))
    if pickup_start and delivery_end and delivery_end < pickup_start:
        errors.append(FieldError("delivery_window_end", CODE_VALUE, "delivery precedes pickup"))

    if errors:
        raise RowValidationError(row_index=-1, identifier=identifier, errors=errors)

    created_at = raw.posted_at or pickup_start
    return Load(
        load_id=raw.load_id,
        weight=raw.weight,
        created_at=created_at,
        origin_city=raw.origin_city,
        origin_state=raw.origin_state,
        origin_latitude=raw.origin_latitude,
        origin_longitude=raw.origin_longitude,
        destination_city=raw.destination_city,
        destination_state=raw.destination_state,
        destination_latitude=raw.destination_latitude,
        destination_longitude=raw.destination_longitude,
        pickup_window_start=pickup_start,
        pickup_window_end=pickup_end,
        delivery_window_start=delivery_start,
        delivery_window_end=delivery_end,
        miles=raw.miles,
        total_rate=total_rate,
        equipment_type=raw.equipment_type,
    )


@lru_cache(maxsize=None)
def _alias_to_field_map(model: Type[BaseModel]) -> Dict[str, str]:
    """Map every accepted input alias (and the field name itself) back to the canonical field name,
    so a validation report names the contract field regardless of which alias the feed used."""
    mapping: Dict[str, str] = {}
    for name, info in model.model_fields.items():
        mapping[name] = name
        alias = info.validation_alias
        choices = getattr(alias, "choices", None)
        if choices:
            for choice in choices:
                if isinstance(choice, str):
                    mapping[choice] = name
        elif isinstance(alias, str):
            mapping[alias] = name
    return mapping


def _pydantic_errors_to_fields(exc: ValidationError, model: Optional[Type[BaseModel]] = None) -> List[FieldError]:
    amap = _alias_to_field_map(model) if model is not None else {}
    fields: List[FieldError] = []
    for err in exc.errors():
        loc = err.get("loc", ())
        raw_field = ".".join(str(p) for p in loc) if loc else "__root__"
        field = amap.get(str(loc[0]), raw_field) if loc else raw_field
        etype = str(err.get("type", ""))
        given = err.get("input", None)
        if "missing" in etype or given in (None, ""):
            code = CODE_MISSING
        elif etype.endswith("_type") or etype.endswith("_parsing"):
            code = CODE_TYPE
        else:
            code = CODE_VALUE
        fields.append(FieldError(field, code, err.get("msg", "invalid value")))
    return fields


def domain_load_from_mapping(mapping: Any, row_index: int = 0) -> Load:
    """Validate + map one raw mapping to a domain :class:`Load`.

    Raises :class:`RowValidationError` (with ``row_index`` stamped) for any type or cross-field
    problem - never a raw ``pydantic.ValidationError``.
    """
    try:
        raw = RawExternalLoad.model_validate(mapping)
    except ValidationError as exc:
        ident = None
        if isinstance(mapping, dict):
            ident = mapping.get("reference") or mapping.get("load_id") or mapping.get("id")
            ident = str(ident) if ident is not None else None
        raise RowValidationError(row_index, ident, _pydantic_errors_to_fields(exc, RawExternalLoad)) from exc
    try:
        return to_domain_load(raw)
    except RowValidationError as exc:
        raise RowValidationError(row_index, exc.identifier, exc.errors) from exc

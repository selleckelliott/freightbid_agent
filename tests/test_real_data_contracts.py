"""Phase 7.1 - real-world data contracts (ingestion anti-corruption layer).

Covers: messy external rows mapping into the domain ``Load``; CSV and JSON producing equivalent
objects; reject-row-not-batch structured errors; PII redaction invariants; datetime/state/equipment
normalization; rate resolution; structural failures; determinism; and a regression guard that the
internal ``Load`` / existing API ingress is untouched.
"""
import json
from dataclasses import asdict, fields
from datetime import timezone
from pathlib import Path

import pytest

from application.ingestion import (
    BrokerReference,
    RowValidationError,
    SchemaContractError,
    broker_reference_from_mapping,
    domain_load_from_mapping,
    import_brokers,
    import_loads,
    redact_contact,
    validate_brokers,
    validate_loads,
)
from application.ingestion.import_contract import read_rows_from_text
from application.ingestion.real_load_schema import parse_datetime, to_domain_load, RawExternalLoad
from application.ingestion.redaction import assert_no_raw_pii, contains_pii, hash_token
from domain.models.load import Load

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "sample_data" / "external"


# --------------------------------------------------------------------------- #
# Mapping a messy external row into the domain model
# --------------------------------------------------------------------------- #
def test_messy_row_maps_into_domain_load():
    row = {
        "id": "101", "equipment": "V", "weight_lbs": "22,000 lb",
        "origin": "Dallas", "origin_state": "Texas", "origin_lat": "32.7767", "origin_lng": "-96.7970",
        "dest_city": "Houston", "dest_state": "TX", "dest_lat": "29.7604", "dest_lng": "-95.3698",
        "trip_miles": "240", "rate_per_mile": "$3.54",
        "pickup": "2026-05-27T18:00:00Z", "delivery": "2026-05-28T06:00:00Z", "broker": "BRK-9",
    }
    load = domain_load_from_mapping(row, 0)
    assert isinstance(load, Load)
    assert load.load_id == 101
    assert load.equipment_type == "Dry Van"
    assert load.origin_state == "TX" and load.destination_state == "TX"
    assert load.weight == 22000.0
    assert load.miles == 240.0
    assert load.total_rate == pytest.approx(3.54 * 240, rel=1e-6)
    assert load.pickup_window_start.tzinfo is not None
    assert load.created_at == load.pickup_window_start  # defaults to pickup when no posted_at


def test_total_rate_takes_precedence_over_per_mile():
    base = {
        "load_id": 1, "equipment": "Van", "weight": 1000,
        "origin": "A", "origin_state": "TX", "origin_lat": 0, "origin_lng": 0,
        "dest_city": "B", "dest_state": "TX", "dest_lat": 1, "dest_lng": 1,
        "miles": 100, "total_rate": 500, "rate_per_mile": 9.99,
        "pickup_date": "2026-06-01T08:00:00Z", "delivery_date": "2026-06-02T08:00:00Z",
    }
    load = domain_load_from_mapping(base, 0)
    assert load.total_rate == 500.0  # not 9.99 * 100


def test_single_date_fills_window_start_and_end():
    row = {
        "load_id": 2, "equipment": "Reefer", "weight": 1000,
        "origin": "A", "origin_state": "TX", "origin_lat": 0, "origin_lng": 0,
        "dest_city": "B", "dest_state": "TX", "dest_lat": 1, "dest_lng": 1,
        "miles": 100, "total_rate": 500,
        "pickup_date": "2026-06-01T08:00:00Z", "delivery_date": "2026-06-02T08:00:00Z",
    }
    load = domain_load_from_mapping(row, 0)
    assert load.pickup_window_start == load.pickup_window_end
    assert load.delivery_window_start == load.delivery_window_end


# --------------------------------------------------------------------------- #
# CSV / JSON files and their equivalence
# --------------------------------------------------------------------------- #
def test_import_loads_csv_all_valid():
    result = import_loads(EXT / "loads.csv")
    assert result.total_rows == 4
    assert result.accepted == 4
    assert result.rejected == 0
    by_id = {l.load_id: l for l in result.loads}
    assert by_id[5001].equipment_type == "Flatbed"  # FD -> Flatbed
    assert by_id[5001].origin_state == "TX"          # Texas -> TX
    assert by_id[5001].weight == 9800.0              # "9,800" -> 9800
    assert by_id[5001].total_rate == 1050.0          # "$1,050" -> 1050


def test_import_loads_json_wrapper_and_per_mile():
    result = import_loads(EXT / "loads.json")
    assert result.accepted == 2 and result.rejected == 0
    by_id = {l.load_id: l for l in result.loads}
    # 6001 used rate_per_mile 2.65 * 212 miles
    assert by_id[6001].total_rate == pytest.approx(2.65 * 212, rel=1e-6)
    assert by_id[6001].equipment_type == "Dry Van"


def test_csv_and_json_dialects_produce_equal_domain_load():
    csv_row = {
        "posting_id": "7001", "equipment_code": "V", "weight_lbs": "22,000",
        "origin": "Dallas", "origin_state": "Texas", "origin_lat": "32.7767", "origin_lng": "-96.7970",
        "dest_city": "Houston", "dest_state": "TX", "dest_lat": "29.7604", "dest_lng": "-95.3698",
        "trip_miles": "240", "rate": "$850",
        "pickup": "2026-06-01T18:00:00Z", "delivery": "2026-06-02T06:00:00Z",
    }
    json_row = {
        "load_id": 7001, "equipment": "Dry Van", "gross_weight": 22000,
        "origin_city": "Dallas", "origin_state": "TX",
        "origin_latitude": 32.7767, "origin_longitude": -96.7970,
        "destination_city": "Houston", "destination_state": "TX",
        "destination_latitude": 29.7604, "destination_longitude": -95.3698,
        "miles": 240, "total_rate": 850,
        "pickup_date": "2026-06-01T18:00:00Z", "delivery_date": "2026-06-02T06:00:00Z",
    }
    assert domain_load_from_mapping(csv_row, 0) == domain_load_from_mapping(json_row, 0)


# --------------------------------------------------------------------------- #
# Reject-row-not-batch + structured errors
# --------------------------------------------------------------------------- #
def test_malformed_csv_rejects_rows_not_batch():
    result = import_loads(EXT / "loads_malformed.csv")
    assert result.total_rows == 3
    assert result.accepted == 1            # the one valid row survives
    assert result.rejected == 2
    assert result.loads[0].load_id == 5101
    report = result.error_report()
    assert all("row_index" in r and "errors" in r for r in report)
    # the row missing coordinates flags missing lat/lon
    codes = {(e["field"], e["code"]) for r in report for e in r["errors"]}
    assert ("origin_latitude", "missing") in codes
    # the bad row (non-numeric id, weight 0, no rate, unknown state) yields multiple field errors
    assert any(len(r["errors"]) >= 2 for r in report)


def test_missing_rate_is_structured_error_not_crash():
    row = {
        "load_id": 9, "equipment": "Van", "weight": 1000,
        "origin": "A", "origin_state": "TX", "origin_lat": 0, "origin_lng": 0,
        "dest_city": "B", "dest_state": "TX", "dest_lat": 1, "dest_lng": 1,
        "miles": 100, "pickup_date": "2026-06-01T08:00:00Z", "delivery_date": "2026-06-02T08:00:00Z",
    }
    with pytest.raises(RowValidationError) as exc:
        domain_load_from_mapping(row, 5)
    err = exc.value
    assert err.row_index == 5
    assert any(e.field == "total_rate" and e.code == "missing" for e in err.errors)


def test_unknown_state_is_value_error():
    row = {
        "load_id": 10, "equipment": "Van", "weight": 1000,
        "origin": "A", "origin_state": "Nowhere", "origin_lat": 0, "origin_lng": 0,
        "dest_city": "B", "dest_state": "TX", "dest_lat": 1, "dest_lng": 1,
        "miles": 100, "total_rate": 500,
        "pickup_date": "2026-06-01T08:00:00Z", "delivery_date": "2026-06-02T08:00:00Z",
    }
    with pytest.raises(RowValidationError) as exc:
        domain_load_from_mapping(row, 0)
    assert any(e.field == "origin_state" for e in exc.value.errors)


# --------------------------------------------------------------------------- #
# Datetime / equipment normalization
# --------------------------------------------------------------------------- #
def test_parse_datetime_normalizes_to_utc():
    assert parse_datetime("2026-06-01T08:00:00Z").tzinfo == timezone.utc
    assert parse_datetime("2026-06-01").tzinfo == timezone.utc      # date-only
    assert parse_datetime("06/01/2026 08:00").tzinfo == timezone.utc  # fallback format
    assert parse_datetime("") is None


def test_unknown_equipment_passes_through_titlecased():
    raw = RawExternalLoad.model_validate({
        "load_id": 1, "equipment": "conestoga", "weight": 1, "miles": 1,
        "origin": "A", "origin_state": "TX", "origin_lat": 0, "origin_lng": 0,
        "dest_city": "B", "dest_state": "TX", "dest_lat": 0, "dest_lng": 0,
        "total_rate": 1, "pickup_date": "2026-06-01", "delivery_date": "2026-06-02",
    })
    assert raw.equipment_type == "Conestoga"


# --------------------------------------------------------------------------- #
# Broker schema + PII redaction
# --------------------------------------------------------------------------- #
def test_broker_reference_redacts_pii():
    raw = {
        "broker_id": "BRK-9", "company": "Acme Freight", "email": "Jane.Doe@acme.com",
        "phone": "(555) 123-4567", "address": "123 Main St", "contact": "Jane Doe",
        "credit": "b", "days_to_pay": "30", "bonded": "yes", "quick_pay": "n", "age_days": "1200",
    }
    ref = broker_reference_from_mapping(raw, 0)
    assert isinstance(ref, BrokerReference)
    assert ref.credit_bucket == "B" and ref.days_to_pay == 30.0
    assert ref.bonded is True and ref.quick_pay_available is False and ref.age_days == 1200
    d = ref.as_dict()
    assert d["contact"]["has_email"] and d["contact"]["has_phone"]
    assert d["contact"]["has_address"] and d["contact"]["has_contact_name"]
    # no raw PII survives anywhere in the exported dict
    blob = json.dumps(d)
    assert "Jane.Doe@acme.com" not in blob and "Jane Doe" not in blob
    assert "123 Main St" not in blob and "5551234567" not in blob and "123-4567" not in blob
    assert not contains_pii(d)
    assert_no_raw_pii(d)


def test_import_brokers_file_no_pii_leak():
    result = import_brokers(EXT / "brokers.csv")
    assert result.accepted == 3 and result.rejected == 0
    assert set(result.by_id()) == {"BRK-100", "BRK-101", "BRK-102"}
    for ref in result.brokers:
        assert not contains_pii(ref.as_dict())


def test_redaction_tokens_are_deterministic_and_irreversible():
    a = redact_contact(email="a@b.com", phone="555-000-1111")
    b = redact_contact(email="A@B.COM", phone="(555) 000-1111")
    assert a.email_token == b.email_token       # case/format-insensitive, stable
    assert a.phone_token == b.phone_token
    assert "a@b.com" not in (a.email_token or "")
    assert hash_token("a@b.com") == hash_token("a@b.com")


def test_contains_pii_detects_raw_but_not_tokens():
    assert contains_pii({"x": "reach me at foo@bar.com"})
    assert contains_pii({"x": "call 555-123-4567 today"})
    assert not contains_pii({"email_token": "email:abcdef012345", "has_email": True})


# --------------------------------------------------------------------------- #
# Structural failures
# --------------------------------------------------------------------------- #
def test_json_not_a_list_raises_schema_contract_error():
    with pytest.raises(SchemaContractError):
        read_rows_from_text('{"foo": 1}', "json")


def test_csv_without_header_raises_schema_contract_error():
    with pytest.raises(SchemaContractError):
        read_rows_from_text("", "csv")


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_import_is_deterministic():
    r1 = import_loads(EXT / "loads.csv")
    r2 = import_loads(EXT / "loads.csv")
    assert [asdict(l) for l in r1.loads] == [asdict(l) for l in r2.loads]
    assert r1.error_report() == r2.error_report()

    b1 = import_brokers(EXT / "brokers.csv")
    b2 = import_brokers(EXT / "brokers.csv")
    assert [x.as_dict() for x in b1.brokers] == [x.as_dict() for x in b2.brokers]


# --------------------------------------------------------------------------- #
# Regression: the internal domain model + existing API ingress are untouched
# --------------------------------------------------------------------------- #
def test_domain_load_field_set_unchanged():
    # Phase 7.1 must not add fields to the engine's Load (anti-corruption, not model change).
    assert {f.name for f in fields(Load)} == {
        "load_id", "weight", "created_at",
        "origin_city", "origin_state", "origin_latitude", "origin_longitude",
        "destination_city", "destination_state", "destination_latitude", "destination_longitude",
        "pickup_window_start", "pickup_window_end", "delivery_window_start", "delivery_window_end",
        "miles", "total_rate", "equipment_type",
    }


def test_contract_output_is_compatible_with_existing_api_dto():
    # A Load produced by the new contract round-trips through the existing API LoadDTO/mappers
    # unchanged -> the live /loads ingress path is not disturbed.
    from adapters.inbound.api.mappers import load_from_dto, load_to_dto

    load = domain_load_from_mapping({
        "load_id": 8001, "equipment": "Dry Van", "weight": 20000,
        "origin": "Dallas", "origin_state": "TX", "origin_lat": 32.7767, "origin_lng": -96.7970,
        "dest_city": "Houston", "dest_state": "TX", "dest_lat": 29.7604, "dest_lng": -95.3698,
        "miles": 240, "total_rate": 850,
        "pickup_date": "2026-06-01T18:00:00Z", "delivery_date": "2026-06-02T06:00:00Z",
    }, 0)
    dto = load_to_dto(load)
    assert load_from_dto(dto) == load


def test_existing_sample_loads_still_ingest_unchanged():
    # The original committed sample feed still parses via the existing API DTO exactly as before.
    from adapters.inbound.api.schemas import IngestRequest

    payload = json.loads((ROOT / "sample_data" / "loads.json").read_text(encoding="utf-8"))
    req = IngestRequest(**payload)
    assert len(req.loads) == 4

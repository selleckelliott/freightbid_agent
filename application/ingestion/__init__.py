"""Phase 7.1 real-world data contracts: validate external load-board + broker feeds and map them
into the existing domain objects.

This package is an **anti-corruption layer**. Nothing here changes the internal
:class:`domain.models.load.Load` or the engine; it only absorbs the messiness of a real feed
(aliased columns, money/number strings, full state names, equipment codes, single dates instead
of windows, per-mile or total rates) and either produces a clean domain object or a structured,
PII-free error. Broker contact PII is redacted at a single chokepoint into a
:class:`BrokerReference` carried as reference data - never fed to the ML feature builders.
"""
from application.ingestion.errors import (
    FieldError,
    IngestionError,
    RowValidationError,
    SchemaContractError,
)
from application.ingestion.import_contract import (
    BrokerResult,
    FeedImport,
    IngestResult,
    import_brokers,
    import_feed,
    import_loads,
    read_rows_from_file,
    read_rows_from_text,
    validate_brokers,
    validate_loads,
)
from application.ingestion.real_broker_schema import (
    BrokerReference,
    RawExternalBroker,
    broker_reference_from_mapping,
    to_broker_reference,
)
from application.ingestion.real_load_schema import (
    RawExternalLoad,
    domain_load_from_mapping,
    to_domain_load,
)
from application.ingestion.redaction import RedactedContact, redact_contact

__all__ = [
    "FieldError",
    "IngestionError",
    "RowValidationError",
    "SchemaContractError",
    "IngestResult",
    "BrokerResult",
    "FeedImport",
    "import_loads",
    "import_brokers",
    "import_feed",
    "read_rows_from_file",
    "read_rows_from_text",
    "validate_loads",
    "validate_brokers",
    "RawExternalLoad",
    "to_domain_load",
    "domain_load_from_mapping",
    "RawExternalBroker",
    "BrokerReference",
    "to_broker_reference",
    "broker_reference_from_mapping",
    "RedactedContact",
    "redact_contact",
]

import json
from pathlib import Path

import pytest

from adapters.inbound.api.container import build_container
from adapters.inbound.api.mappers import load_from_dto, truck_from_dto
from adapters.inbound.api.schemas import LoadDTO, TruckStateDTO

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def container():
    return build_container(ROOT / "config")


@pytest.fixture
def sample_loads():
    payload = json.loads((ROOT / "sample_data" / "loads.json").read_text())
    return [load_from_dto(LoadDTO(**l)) for l in payload["loads"]]


@pytest.fixture
def sample_truck():
    payload = json.loads((ROOT / "sample_data" / "truck.json").read_text())
    return truck_from_dto(TruckStateDTO(**payload))

from dataclasses import asdict

from domain.models.load import Load
from domain.models.truck_state import TruckState

from .schemas import LoadDTO, TruckStateDTO


def load_from_dto(dto: LoadDTO) -> Load:
    return Load(**dto.model_dump())


def truck_from_dto(dto: TruckStateDTO) -> TruckState:
    return TruckState(**dto.model_dump())


def load_to_dto(load: Load) -> LoadDTO:
    return LoadDTO(**asdict(load))

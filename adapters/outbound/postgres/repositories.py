from datetime import datetime
from typing import Iterable, List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from domain.models.load import Load
from domain.models.truck_state import TruckState
from ports.load_repository import LoadRepositoryPort
from ports.truck_repository import TruckRepositoryPort


class Base(DeclarativeBase):
    pass


class LoadRow(Base):
    __tablename__ = "loads"
    load_id = Column(Integer, primary_key=True)
    weight = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    origin_city = Column(String, nullable=False)
    origin_state = Column(String, nullable=False)
    origin_latitude = Column(Float, nullable=False)
    origin_longitude = Column(Float, nullable=False)
    destination_city = Column(String, nullable=False)
    destination_state = Column(String, nullable=False)
    destination_latitude = Column(Float, nullable=False)
    destination_longitude = Column(Float, nullable=False)
    pickup_window_start = Column(DateTime(timezone=True), nullable=False)
    pickup_window_end = Column(DateTime(timezone=True), nullable=False)
    delivery_window_start = Column(DateTime(timezone=True), nullable=False)
    delivery_window_end = Column(DateTime(timezone=True), nullable=False)
    miles = Column(Float, nullable=False)
    total_rate = Column(Float, nullable=False)
    equipment_type = Column(String, nullable=False)


class TruckRow(Base):
    __tablename__ = "trucks"
    truck_id = Column(Integer, primary_key=True)
    current_city = Column(String, nullable=False)
    current_state = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    available_at = Column(DateTime(timezone=True), nullable=False)
    trailer_type = Column(String, nullable=False)
    max_load_capacity = Column(Float, nullable=False)
    current_load_id = Column(Integer, nullable=True)
    home_city = Column(String, nullable=False)
    home_state = Column(String, nullable=False)
    remaining_capacity = Column(Float, nullable=False)
    driver_hours_left = Column(Float, nullable=False)
    speed = Column(Float, nullable=False)
    heading = Column(Float, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)


def make_engine(url: str):
    return create_engine(url, future=True)


def init_schema(engine) -> None:
    Base.metadata.create_all(engine)


def make_session_factory(engine):
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def _row_to_load(r: LoadRow) -> Load:
    return Load(
        load_id=r.load_id,
        weight=r.weight,
        created_at=r.created_at,
        origin_city=r.origin_city,
        origin_state=r.origin_state,
        origin_latitude=r.origin_latitude,
        origin_longitude=r.origin_longitude,
        destination_city=r.destination_city,
        destination_state=r.destination_state,
        destination_latitude=r.destination_latitude,
        destination_longitude=r.destination_longitude,
        pickup_window_start=r.pickup_window_start,
        pickup_window_end=r.pickup_window_end,
        delivery_window_start=r.delivery_window_start,
        delivery_window_end=r.delivery_window_end,
        miles=r.miles,
        total_rate=r.total_rate,
        equipment_type=r.equipment_type,
    )


def _load_to_row(l: Load) -> LoadRow:
    return LoadRow(
        load_id=l.load_id,
        weight=l.weight,
        created_at=l.created_at,
        origin_city=l.origin_city,
        origin_state=l.origin_state,
        origin_latitude=l.origin_latitude,
        origin_longitude=l.origin_longitude,
        destination_city=l.destination_city,
        destination_state=l.destination_state,
        destination_latitude=l.destination_latitude,
        destination_longitude=l.destination_longitude,
        pickup_window_start=l.pickup_window_start,
        pickup_window_end=l.pickup_window_end,
        delivery_window_start=l.delivery_window_start,
        delivery_window_end=l.delivery_window_end,
        miles=l.miles,
        total_rate=l.total_rate,
        equipment_type=l.equipment_type,
    )


def _row_to_truck(r: TruckRow) -> TruckState:
    return TruckState(
        truck_id=r.truck_id,
        current_city=r.current_city,
        current_state=r.current_state,
        latitude=r.latitude,
        longitude=r.longitude,
        available_at=r.available_at,
        trailer_type=r.trailer_type,
        max_load_capacity=r.max_load_capacity,
        current_load_id=r.current_load_id,
        home_city=r.home_city,
        home_state=r.home_state,
        remaining_capacity=r.remaining_capacity,
        driver_hours_left=r.driver_hours_left,
        speed=r.speed,
        heading=r.heading,
        timestamp=r.timestamp,
    )


def _truck_to_row(t: TruckState) -> TruckRow:
    return TruckRow(
        truck_id=t.truck_id,
        current_city=t.current_city,
        current_state=t.current_state,
        latitude=t.latitude,
        longitude=t.longitude,
        available_at=t.available_at,
        trailer_type=t.trailer_type,
        max_load_capacity=t.max_load_capacity,
        current_load_id=t.current_load_id,
        home_city=t.home_city,
        home_state=t.home_state,
        remaining_capacity=t.remaining_capacity,
        driver_hours_left=t.driver_hours_left,
        speed=t.speed,
        heading=t.heading,
        timestamp=t.timestamp,
    )


class PostgresLoadRepository(LoadRepositoryPort):
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def add_many(self, loads: Iterable[Load]) -> List[Load]:
        session: Session = self._session_factory()
        try:
            stored: List[Load] = []
            for l in loads:
                session.merge(_load_to_row(l))
                stored.append(l)
            session.commit()
            return stored
        finally:
            session.close()

    def get(self, load_id: int) -> Optional[Load]:
        session: Session = self._session_factory()
        try:
            row = session.get(LoadRow, load_id)
            return _row_to_load(row) if row else None
        finally:
            session.close()

    def list_all(self) -> List[Load]:
        session: Session = self._session_factory()
        try:
            rows = session.query(LoadRow).all()
            return [_row_to_load(r) for r in rows]
        finally:
            session.close()

    def clear(self) -> None:
        session: Session = self._session_factory()
        try:
            session.query(LoadRow).delete()
            session.commit()
        finally:
            session.close()


class PostgresTruckRepository(TruckRepositoryPort):
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def upsert(self, truck: TruckState) -> TruckState:
        session: Session = self._session_factory()
        try:
            session.merge(_truck_to_row(truck))
            session.commit()
            return truck
        finally:
            session.close()

    def get(self, truck_id: int) -> Optional[TruckState]:
        session: Session = self._session_factory()
        try:
            row = session.get(TruckRow, truck_id)
            return _row_to_truck(row) if row else None
        finally:
            session.close()

    def list_all(self) -> List[TruckState]:
        session: Session = self._session_factory()
        try:
            rows = session.query(TruckRow).all()
            return [_row_to_truck(r) for r in rows]
        finally:
            session.close()

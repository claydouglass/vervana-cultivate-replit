from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Integer, String, Float, DateTime
from sqlalchemy.orm import Mapped, mapped_column

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

class EnvironmentalData(db.Model):
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[DateTime] = mapped_column(DateTime, nullable=False)
    temperature: Mapped[float] = mapped_column(Float, nullable=False)
    humidity: Mapped[float] = mapped_column(Float, nullable=False)
    co2_level: Mapped[float] = mapped_column(Float, nullable=False)
    light_intensity: Mapped[float] = mapped_column(Float, nullable=False)

class BatchData(db.Model):
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[str] = mapped_column(String(50), nullable=False)
    harvest_date: Mapped[DateTime] = mapped_column(DateTime, nullable=False)
    thc_level: Mapped[float] = mapped_column(Float, nullable=False)
    terpene_profile: Mapped[str] = mapped_column(String(500), nullable=False)
    yield_amount: Mapped[float] = mapped_column(Float, nullable=True)

class NutrientData(db.Model):
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[DateTime] = mapped_column(DateTime, nullable=False)
    nitrogen_level: Mapped[float] = mapped_column(Float, nullable=False)
    phosphorus_level: Mapped[float] = mapped_column(Float, nullable=False)
    potassium_level: Mapped[float] = mapped_column(Float, nullable=False)

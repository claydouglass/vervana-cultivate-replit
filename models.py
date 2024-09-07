from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy import Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from typing import Optional, List

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
    vpd: Mapped[float] = mapped_column(Float, nullable=False)  # Add this line

class BatchData(db.Model):
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_number: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    batch_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    
    # Growth stages (all nullable)
    veg_week_1_2_start: Mapped[Optional[DateTime]] = mapped_column(DateTime, nullable=True)
    veg_week_1_2_end: Mapped[Optional[DateTime]] = mapped_column(DateTime, nullable=True)
    veg_week_3_start: Mapped[Optional[DateTime]] = mapped_column(DateTime, nullable=True)
    veg_week_3_end: Mapped[Optional[DateTime]] = mapped_column(DateTime, nullable=True)
    flower_week_1_3_start: Mapped[Optional[DateTime]] = mapped_column(DateTime, nullable=True)
    flower_week_1_3_end: Mapped[Optional[DateTime]] = mapped_column(DateTime, nullable=True)
    flower_week_4_6_5_start: Mapped[Optional[DateTime]] = mapped_column(DateTime, nullable=True)
    flower_week_4_6_5_end: Mapped[Optional[DateTime]] = mapped_column(DateTime, nullable=True)
    flower_week_6_5_8_5_start: Mapped[Optional[DateTime]] = mapped_column(DateTime, nullable=True)
    flower_week_6_5_8_5_end: Mapped[Optional[DateTime]] = mapped_column(DateTime, nullable=True)
    flower_week_8_5_plus_start: Mapped[Optional[DateTime]] = mapped_column(DateTime, nullable=True)
    flower_week_8_5_plus_end: Mapped[Optional[DateTime]] = mapped_column(DateTime, nullable=True)
    
    # Processing phases
    harvest_date: Mapped[Optional[DateTime]] = mapped_column(DateTime, nullable=True)
    drying_start: Mapped[Optional[DateTime]] = mapped_column(DateTime, nullable=True)
    drying_end: Mapped[Optional[DateTime]] = mapped_column(DateTime, nullable=True)
    curing_start: Mapped[Optional[DateTime]] = mapped_column(DateTime, nullable=True)
    curing_end: Mapped[Optional[DateTime]] = mapped_column(DateTime, nullable=True)
    
    # Other batch data
    yield_amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    thc_level: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    terpene_profile: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    cultivation_tasks: Mapped[List["CultivationSchedule"]] = relationship("CultivationSchedule", back_populates="batch")
    processing_tasks: Mapped[List["ProcessingSchedule"]] = relationship("ProcessingSchedule", back_populates="batch")

    def __repr__(self):
        return f'<BatchData {self.batch_id}>'

class NutrientData(db.Model):
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[DateTime] = mapped_column(DateTime, nullable=False)
    nitrogen_level: Mapped[float] = mapped_column(Float, nullable=False)
    phosphorus_level: Mapped[float] = mapped_column(Float, nullable=False)
    potassium_level: Mapped[float] = mapped_column(Float, nullable=False)

class CultivationSchedule(db.Model):
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[str] = mapped_column(String(50), ForeignKey('batch_data.batch_id'), nullable=False)
    day: Mapped[int] = mapped_column(Integer, nullable=False)
    task: Mapped[str] = mapped_column(String(100), nullable=False)
    completed: Mapped[bool] = mapped_column(Boolean, default=False)
    completion_date: Mapped[Optional[DateTime]] = mapped_column(DateTime, nullable=True)

    batch: Mapped["BatchData"] = relationship("BatchData", back_populates="cultivation_tasks")

class ProcessingSchedule(db.Model):
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[str] = mapped_column(String(50), ForeignKey('batch_data.batch_id'), nullable=False)
    day: Mapped[int] = mapped_column(Integer, nullable=False)
    task: Mapped[str] = mapped_column(String(100), nullable=False)
    completed: Mapped[bool] = mapped_column(Boolean, default=False)
    completion_date: Mapped[Optional[DateTime]] = mapped_column(DateTime, nullable=True)

    batch: Mapped["BatchData"] = relationship("BatchData", back_populates="processing_tasks")

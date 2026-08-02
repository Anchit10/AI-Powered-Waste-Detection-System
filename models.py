from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime, Boolean, Text
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime
import os as _os

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)
    role = Column(String)  # "citizen", "admin", "authority"
    name = Column(String, nullable=True)
    
    # Authority specific fields
    organization_name = Column(String, nullable=True)
    organization_type = Column(String, nullable=True)
    assigned_locality = Column(String, nullable=True)
    phone = Column(String, nullable=True)

    reports = relationship("WasteReport", back_populates="reporter", foreign_keys="WasteReport.reporter_id")
    assigned_reports = relationship("WasteReport", back_populates="assigned_authority", foreign_keys="WasteReport.authority_id")


class WasteReport(Base):
    __tablename__ = "waste_reports"

    id = Column(Integer, primary_key=True, index=True)
    lat = Column(Float, index=True)
    lng = Column(Float, index=True)
    image_path = Column(String, nullable=True)
    waste_type = Column(String)
    severity = Column(String)
    confidence = Column(Float)
    locality = Column(String, nullable=True)
    all_detected_objects = Column(Text, nullable=True)

    # Report lifecycle
    status = Column(String, default="Reported")  # Reported → Taken Up → Being Solved → Solved
    
    # Timestamps per stage
    reported_at = Column(DateTime, default=datetime.utcnow)
    taken_up_at = Column(DateTime, nullable=True)
    being_solved_at = Column(DateTime, nullable=True)
    solved_at = Column(DateTime, nullable=True)
    
    # Notes
    admin_notes = Column(Text, nullable=True)
    resolution_notes = Column(Text, nullable=True)
    after_image_path = Column(String, nullable=True)

    # User verification of resolution
    user_verified = Column(Boolean, default=False)
    verified_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    verification_comment = Column(Text, nullable=True)

    # Relationships
    reporter_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    reporter = relationship("User", back_populates="reports", foreign_keys=[reporter_id])
    
    authority_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    assigned_authority = relationship("User", back_populates="assigned_reports", foreign_keys=[authority_id])


# Database setup
_db_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "waste_app.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{_db_path}"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

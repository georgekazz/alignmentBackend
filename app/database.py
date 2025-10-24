from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Σύνδεση με τη βάση MySQL
DATABASE_URL = "mysql+pymysql://root:@localhost/alignmedb"

# Δημιουργία engine και session
engine = create_engine(DATABASE_URL, echo=True, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Βασική βάση για τα models
Base = declarative_base()

# Dependency για FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

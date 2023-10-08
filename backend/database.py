from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

URL = "postgresql://valfrank:sFldP3uQeK9f@ep-wispy-cake-47463361.us-east-2.aws.neon.tech/neondb"

engine = create_engine(URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

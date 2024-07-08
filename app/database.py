from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path('..') / '.env'
load_dotenv(dotenv_path=env_path)

DATABASE_URL = os.getenv("DATABASE_URL", "NOT_SET")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Recipe(Base):
    __tablename__ = 'recipe'
    recipe_id = Column(Integer, primary_key=True, index=True)
    recipe_name = Column(String, index=True)
    image_url = Column(String)
    ingredients = Column(String)
    cooking_directions = Column(String)

class UserRecipe(Base):
    __tablename__ = 'user_recipe'
    user_id = Column(Integer, primary_key=True, index=True)
    recipe_id = Column(Integer, primary_key=True, index=True)
    rating = Column(Float)
    dateLastModified = Column(DateTime)

# Create all tables
Base.metadata.create_all(bind=engine)

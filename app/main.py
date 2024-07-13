from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session,Mapped
from sqlalchemy import String
from datetime import datetime, timedelta
from typing import List, Optional
from surprise import Dataset, Reader, SVD, accuracy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from surprise.model_selection import train_test_split
import pandas as pd
import numpy as np
import re
import secrets
import uvicorn
import random

# Importing SessionLocal from database.py
from .database import SessionLocal, Recipe, UserRecipe, User

# Secret key
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 token URL
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()

# CORS settings
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# Dependency to get the session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserInDB(BaseModel):
    username: str
    hashed_password: str

class UserCreate(BaseModel):
    username: str
    password: str

class UserUpdate(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None

class RecipeModel(BaseModel):
    recipe_id: int
    recipe_name: str
    image_url: str
    ingredients: Optional[str] = None
    cooking_directions: Optional[str] = None

# Helper functions for authentication
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(db, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == token_data.username).first()
    if user is None:
        raise credentials_exception
    return user

# Load and preprocess data from database
def load_data(db: Session):
    recipes = db.query(Recipe).all()
    user_recipes = db.query(UserRecipe).all()
    
    content_df = pd.DataFrame([{
        'recipe_id': recipe.recipe_id,
        'recipe_name': recipe.recipe_name,
        'image_url': recipe.image_url,
        'ingredients': recipe.ingredients,
        'cooking_directions': recipe.cooking_directions
    } for recipe in recipes])
    
    ratings_df = pd.DataFrame([{
        'user_id': ur.user_id,
        'recipe_id': ur.recipe_id,
        'rating': ur.rating,
        'dateLastModified': ur.dateLastModified
    } for ur in user_recipes])
    
    # Preprocessing
    content_df['recipe_name'] = content_df['recipe_name'].str.lower().str.strip()
    content_df['recipe_name'] = content_df['recipe_name'].apply(lambda name: re.sub(r'^clone of ', '', name))
    content_df['recipe_name'] = content_df['recipe_name'].apply(lambda name: re.sub(r'\(.*?\)', '', name).strip())
    content_df['ingredients'] = content_df['ingredients'].str.lower().str.strip()
    content_df['ingredients'] = content_df['ingredients'].apply(lambda ingredients: re.sub(r'\s*\^\s*', ' ', ingredients).strip())
    content_df['ingredients'] = content_df['ingredients'].str.split()
    content_df['content'] = content_df.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
    
    return content_df, ratings_df

@app.on_event("startup")
def startup():
    global content_df, tfidf_vectorizer, content_matrix, content_distance, algo, trainset, testset
    db = SessionLocal()
    content_df, ratings_df = load_data(db)
    
    # Content-based filtering
    tfidf_vectorizer = TfidfVectorizer()
    content_matrix = tfidf_vectorizer.fit_transform(content_df['content'])
    content_distance = euclidean_distances(content_matrix)
    
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'recipe_id', 'rating']], reader)
    
    # Train-test split and model training
    trainset, testset = train_test_split(data, test_size=.25)
    algo = SVD()
    algo.fit(trainset)

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users/", response_model=UserCreate)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    db_user = User(username=user.username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/recipes/{recipe_id}", response_model=RecipeModel)
def get_recipe(recipe_id: int, db: Session = Depends(get_db)):
    recipe = db.query(Recipe).filter(Recipe.recipe_id == recipe_id).first()
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")
    return RecipeModel(
        recipe_id=recipe.recipe_id,
        recipe_name=recipe.recipe_name,
        image_url=recipe.image_url,
        ingredients=recipe.ingredients,
        cooking_directions=recipe.cooking_directions
    )

@app.get("/random/", response_model=List[RecipeModel])
def get_random_recipes(db: Session = Depends(get_db)):
    recipes = db.query(Recipe).all()
    random_recipes = random.sample(recipes, min(len(recipes), 8))
    return [
        RecipeModel(
            recipe_id=recipe.recipe_id,
            recipe_name=recipe.recipe_name,
            image_url=recipe.image_url,
            ingredients=recipe.ingredients,
            cooking_directions=recipe.cooking_directions
        )
        for recipe in random_recipes
    ]

@app.get("/recommendations/", response_model=List[RecipeModel])
async def get_hybrid_recommendations(recipe_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    user_id = current_user.user_id
    top_n = 8

    content_based_recommendations = set(get_content_based_recommendations(recipe_id, top_n))
    collaborative_filtering_predictions = get_collaborative_filtering_recommendations(user_id, top_n)

    # Moderated weight scores
    cf_weight = 0.75  # Collaborative
    cb_weight = 0.25  # Content-based

    hybrid_scores = {}
    for pred in collaborative_filtering_predictions:
        hybrid_scores[pred.iid] = cf_weight * pred.est

    for recipe_id in content_based_recommendations:
        if recipe_id not in hybrid_scores:
            hybrid_scores[recipe_id] = cb_weight * 5.0

    # Sort by hybrid score
    sorted_recommendations = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    hybrid_recommendations = [recipe_id for recipe_id, _ in sorted_recommendations][:top_n]

    hybrid_recipes = content_df[content_df['recipe_id'].isin(hybrid_recommendations)].drop_duplicates(subset=['recipe_id'])
    
    # Adjusting ingredients field to ensure it's a string
    hybrid_recipes['ingredients'] = hybrid_recipes['ingredients'].apply(lambda ingredients: ' '.join(ingredients))
    
    hybrid_recipes = hybrid_recipes.to_dict('records')

    return [RecipeModel(**recipe) for recipe in hybrid_recipes]

def get_content_based_recommendations(recipe_id: int, top_n: int = 7):
    idx = content_df[content_df['recipe_id'] == recipe_id].index[0]
    distances = content_distance[idx]
    content_based_indices = np.argsort(distances)[1:top_n + 1]
    return content_df.iloc[content_based_indices]['recipe_id'].tolist()

def get_collaborative_filtering_recommendations(user_id: int, top_n: int = 7):
    testset_user = [(user_id, recipe.recipe_id, 0) for recipe in content_df.itertuples()]
    predictions = algo.test(testset_user)
    predictions.sort(key=lambda x: x.est, reverse=True)
    return predictions[:top_n]

# Endpoint to update user information (username and/or password)
@app.put("/users/{user_id}", response_model=UserInDB)
async def update_user(user_id: int, user_update: UserUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_user = db.query(User).filter(User.user_id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update username if provided
    if user_update.username:
        db_user.username = user_update.username
    
    # Update password if provided
    if user_update.password:
        db_user.hashed_password = get_password_hash(user_update.password)
    
    db.commit()
    db.refresh(db_user)
    return db_user

# Endpoint to update user password only
@app.put("/users/{user_id}/password", response_model=UserInDB)
async def update_user_password(user_id: int, new_password: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_user = db.query(User).filter(User.user_id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update password
    db_user.hashed_password = get_password_hash(new_password)
    
    db.commit()
    db.refresh(db_user)
    return db_user

# Main entry point for the Uvicorn server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

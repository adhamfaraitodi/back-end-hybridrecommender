# app/main.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
import re
from surprise import Dataset, Reader, SVD, accuracy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from surprise.model_selection import train_test_split
import uvicorn

# Importing SessionLocal from database.py
from .database import SessionLocal, Recipe, UserRecipe

app = FastAPI()

# Dependency to get the session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Models
class RecipeModel(BaseModel):
    recipe_id: int
    recipe_name: str
    image_url: str
    ingredients: str
    cooking_directions: str

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

@app.get("/recommendations/", response_model=List[RecipeModel])
def get_hybrid_recommendations(user_id: int, recipe_id: int, top_n: int):
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
    hybrid_recipes = hybrid_recipes[['recipe_id', 'recipe_name', 'image_url']]
    
    return [RecipeModel(recipe_id=row['recipe_id'], recipe_name=row['recipe_name'], image_url=row['image_url']) for _, row in hybrid_recipes.iterrows()]

def get_content_based_recommendations(recipe_id, top_n):
    index = content_df[content_df['recipe_id'] == recipe_id].index[0]
    distance_scores = content_distance[index]
    similar_indices = distance_scores.argsort()[:top_n + 1]
    recommendations = content_df.loc[similar_indices, 'recipe_id'].values
    return recommendations

def get_collaborative_filtering_recommendations(user_id, top_n):
    testset = trainset.build_anti_testset()
    testset = filter(lambda x: x[0] == user_id, testset)
    predictions = algo.test(list(testset))
    predictions.sort(key=lambda x: x.est, reverse=True)
    return predictions[:top_n]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

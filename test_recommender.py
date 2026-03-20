#!/usr/bin/env python
"""Script de prueba para diagnosticar problemas con el recomendador"""
import json
import pandas as pd
from pathlib import Path
import sys
import traceback

# Agregar backend al path
sys.path.insert(0, str(Path(__file__).parent))

from Backend.content_recommender import ContentBasedRecommender


def test_recommender():
    print("=" * 60)
    print("🧪 TEST DEL RECOMENDADOR")
    print("=" * 60)

    # Cargar movies
    data_path = Path(__file__).parent / "Data" / "Clean_data"
    movies_path = data_path / "enhanced_movies.json"

    print(f"\n📁 Buscando películas en: {movies_path}")
    if not movies_path.exists():
        print(f"❌ Archivo no encontrado: {movies_path}")
        return

    with open(movies_path, "r", encoding="utf-8") as f:
        movies_data = json.load(f)

    print(f"✅ Se cargaron {len(movies_data)} películas")

    # Convertir a DataFrame
    movies_df = pd.DataFrame(movies_data)
    print(f"📊 DataFrame shape: {movies_df.shape}")
    print(f"📋 Columnas: {list(movies_df.columns)[:10]}...")

    # Preparar movies_df
    if "id" in movies_df.columns:
        movies_df = movies_df.rename(columns={"id": "movieId"})

    required_cols = [
        "generos",
        "overview",
        "tagline",
        "keywords",
        "tags",
        "cast",
        "crew",
        "director",
        "title",
    ]
    for col in required_cols:
        if col not in movies_df.columns:
            print(f"   ⚠️ Agregando columna vacía: {col}")
            movies_df[col] = ""

    # Fillna
    text_cols = ["overview", "tagline", "keywords", "tags", "cast", "crew", "director"]
    for col in text_cols:
        movies_df[col] = movies_df[col].fillna("").astype(str)

    print(f"\n📝 Preparando recomendador...")

    # intentar cargar ratings y user_registry para replicar exacto
    ratings_df = None
    ratings_path = data_path / "train_ratings.json"
    if ratings_path.exists():
        try:
            with open(ratings_path, "r", encoding="utf-8") as f:
                ratings_data = json.load(f)
                if ratings_data:
                    ratings_df = pd.DataFrame(ratings_data)
                    print(f"📊 Ratings cargados: {ratings_df.shape}")
        except Exception as e:
            print(f"⚠️ No se pudieron cargar ratings: {e}")

    user_registry = None
    registry_path = Path(__file__).parent / "Data" / "user_registry.json"
    if registry_path.exists():
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                user_registry = json.load(f)
                print(f"📁 User registry cargado: {len(user_registry)} usuarios")
        except Exception as e:
            print(f"⚠️ No se pudo cargar user registry: {e}")

    try:
        recommender = ContentBasedRecommender(
            categorical_columns=["genres"],
            feature_text_columns=["overview", "keywords", "tags"],
            profile_strategy="weighted",
            min_reduced_genres=5,
            max_reduced_genres=8,
            w_genre=0.60,
            w_text=0.25,
            w_quality=0.10,
            w_popularity=0.05,
        )
        # end of recommender initialization

        print(f"✅ Recomendador creado")
        print(f"\n🔧 Llamando a fit()...")

        recommender.fit(movies_df, ratings=ratings_df, user_registry=user_registry)

        print(f"✅ fit() completado exitosamente")
        print(f"\n📈 Matriz de características:")
        print(f"   - Shape: {recommender.item_feature_matrix.shape}")
        print(f"   - Tipo: {type(recommender.item_feature_matrix)}")
        print(f"   - Features: {len(recommender.feature_names_)}")

        print(f"\n✨ ¡Recomendador inicializado exitosamente!")
        return True

    except Exception as e:
        print(f"\n❌ ERROR:")
        print(f"   Tipo: {type(e).__name__}")
        print(f"   Mensaje: {str(e)}")
        print(f"\n📍 Traceback completo:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_recommender()
    sys.exit(0 if success else 1)

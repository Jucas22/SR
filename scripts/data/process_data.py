import pandas as pd
import json
from pathlib import Path
import numpy as np
import random


def load_data(file_path):
    """Carga los archivos CSV con la configuración correcta"""
    print("Cargando datos...")

    data = pd.read_csv(file_path, sep=";", decimal=",", encoding="utf-8")

    print(f"Datos cargados: {len(data)}")

    return data


def save_json_data(data, filename, indent=2):
    """Guarda datos en formato JSON"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    print(f"Guardado: {filename}")


def clean_ratings(ratings_df, movies_df):
    """Elimina ratings de películas que no existen en movies_df"""
    print("Limpiando ratings...")

    # IDs de películas disponibles
    peliculas_ids = set(movies_df["id"])

    # Filtrar ratings para mantener solo los que corresponden a películas válidas
    cleaned_ratings = ratings_df[ratings_df["movieId"].isin(peliculas_ids)]

    print(
        f"Ratings originales: {len(ratings_df)}, Ratings limpios: {len(cleaned_ratings)}"
    )

    return cleaned_ratings


def split_train_test_ratings(ratings_clean, test_size=0.3, random_state=42):
    """Divide los ratings en conjuntos de train y test por usuario"""
    print(
        f"\nDividiendo ratings en train ({int((1-test_size)*100)}%) y test ({int(test_size*100)}%)..."
    )

    # Establecer semilla para reproducibilidad
    random.seed(random_state)
    np.random.seed(random_state)

    train_ratings = []
    test_ratings = []

    # Agrupar por usuario
    users_ratings = ratings_clean.groupby("userId")

    users_processed = 0
    users_with_single_rating = 0

    for user_id, user_ratings in users_ratings:
        user_ratings = user_ratings.copy()

        if len(user_ratings) == 1:
            # Si el usuario tiene solo 1 rating, va a train
            train_ratings.append(user_ratings)
            users_with_single_rating += 1
        else:
            # Dividir ratings del usuario aleatoriamente
            user_ratings = user_ratings.sample(
                frac=1, random_state=random_state + user_id
            ).reset_index(drop=True)

            # Calcular índice de división
            split_idx = int(len(user_ratings) * (1 - test_size))

            train_user = user_ratings.iloc[:split_idx]
            test_user = user_ratings.iloc[split_idx:]

            train_ratings.append(train_user)
            test_ratings.append(test_user)

        users_processed += 1

    # Concatenar todos los ratings
    train_df = (
        pd.concat(train_ratings, ignore_index=True) if train_ratings else pd.DataFrame()
    )
    test_df = (
        pd.concat(test_ratings, ignore_index=True) if test_ratings else pd.DataFrame()
    )

    print(f"Usuarios procesados: {users_processed}")
    print(
        f"Usuarios con un solo rating (asignados a train): {users_with_single_rating}"
    )
    print(f"Ratings de entrenamiento: {len(train_df)}")
    print(f"Ratings de test: {len(test_df)}")
    print(f"Total: {len(train_df) + len(test_df)} (original: {len(ratings_clean)})")

    # Verificar que no perdimos datos
    assert len(train_df) + len(test_df) == len(
        ratings_clean
    ), "Se perdieron ratings en la división"

    return train_df, test_df


def process_keywords(keywords_df):
    """Procesa las keywords y las organiza por película"""
    print("\nProcesando keywords...")

    keywords_by_movie = {}

    for _, row in keywords_df.iterrows():
        movie_id = int(row["id"])

        # Extraer todas las keywords no vacías
        # Las keywords están en las columnas desde la posición 2 en adelante (después de id y contador)
        keywords = []
        for i in range(2, len(row)):  # Empezar desde la columna 2
            keyword = row.iloc[i]
            if pd.notna(keyword) and str(keyword).strip() != "":
                keywords.append(str(keyword).strip())

        if keywords:  # Solo agregar si hay keywords
            keywords_by_movie[movie_id] = keywords

    print(f"Películas con keywords: {len(keywords_by_movie)}")
    return keywords_by_movie


def manage_ratings(movies_df, ratings_df):
    # Limpiar ratings de peliculas no existentes
    cleaned_ratings_df = clean_ratings(ratings_df, movies_df)
    # Guardar ratings limpios en JSON
    ratings_json = cleaned_ratings_df.to_dict(orient="records")
    save_json_data(ratings_json, "Data/Clean_data/cleaned_ratings.json")

    # Dividir ratings en entrenamiento y prueba
    train_df, test_df = split_train_test_ratings(cleaned_ratings_df)

    # Guardar los conjuntos de entrenamiento y prueba en JSON
    train_json = train_df.to_dict(orient="records")
    test_json = test_df.to_dict(orient="records")

    save_json_data(train_json, "Data/Clean_data/train_ratings.json")
    save_json_data(test_json, "Data/Clean_data/test_ratings.json")


def manage_movies(movies_df, links_df, keywords_by_movie):
    """Procesa el DataFrame de películas para crear una estructura más limpia"""
    print("\nProcesando datos de películas...")

    # Crear lista de películas estructuradas
    peliculas_list = []

    for _, row in movies_df.iterrows():
        movie_id = int(row["id"])

        # Extraer géneros (eliminar columnas vacías de géneros)
        generos = []
        for col in movies_df.columns:
            if col.startswith("id_genero") and pd.notna(row[col]) and row[col] != "":
                generos.append(int(row[col]))

        pelicula = {
            "id": movie_id,
            "imdb_id": row["imdb_id"],
            "titulo": row["titulo"],
            "poster_path": row["poster_path"] if pd.notna(row["poster_path"]) else None,
            "puntuacion_media": (
                float(row["puntuacion_media"])
                if pd.notna(row["puntuacion_media"])
                else None
            ),
            "votos": int(row["votos"]) if pd.notna(row["votos"]) else 0,
            "generos": generos,
        }

        # Agregar keywords si existen
        if keywords_by_movie and movie_id in keywords_by_movie:
            pelicula["keywords"] = keywords_by_movie[movie_id]
        else:
            pelicula["keywords"] = []

        peliculas_list.append(pelicula)

    save_json_data(peliculas_list, "Data/Clean_data/cleaned_movies.json")


def main():

    # Cargar datos
    movies_df = load_data("Data/Raw_data/peliculas.csv")
    links_df = load_data("Data/Raw_data/links.csv")
    keywords_df = load_data("Data/Raw_data/keywords.csv")
    ratings_df = load_data("Data/Raw_data/ratings_small.csv")

    # Gestionar ratings (limpieza y división)
    manage_ratings(movies_df, ratings_df)

    keywords_by_movie = process_keywords(keywords_df)
    # Gestionar peliculas (fusionar en una sola estructura JSON)
    manage_movies(movies_df, links_df, keywords_by_movie)


if __name__ == "__main__":
    main()

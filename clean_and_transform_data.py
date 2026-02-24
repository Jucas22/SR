#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para limpiar y transformar datos de películas y ratings
Elimina ratings huérfanos y convierte a formato JSON estructurado
"""

import pandas as pd
import json
from pathlib import Path
import numpy as np
import random


def load_data():
    """Carga los archivos CSV con la configuración correcta"""
    print("Cargando datos...")

    # Cargar películas
    peliculas_df = pd.read_csv(
        "Data_set/peliculas.csv", sep=";", decimal=",", encoding="utf-8"
    )

    # Cargar ratings
    ratings_df = pd.read_csv(
        "Data_set/ratings_small.csv", sep=";", decimal=",", encoding="utf-8"
    )

    keywords_df = pd.read_csv(
        "Data_set/keywords.csv", sep=";", decimal=",", encoding="utf-8"
    )

    print(f"Películas cargadas: {len(peliculas_df)}")
    print(f"Ratings cargados: {len(ratings_df)}")
    print(f"Keywords cargados: {len(keywords_df)}")

    return peliculas_df, ratings_df, keywords_df


def clean_ratings(peliculas_df, ratings_df):
    """Limpia los ratings eliminando referencias a películas inexistentes"""
    print("\nAnalizando integridad de los datos...")

    # IDs de películas disponibles
    peliculas_ids = set(peliculas_df["id"])

    # IDs de películas referenciados en ratings
    ratings_movie_ids = set(ratings_df["movieId"])

    # Encontrar ratings huérfanos
    orphaned_ratings = ratings_movie_ids - peliculas_ids
    valid_ratings = ratings_movie_ids & peliculas_ids

    print(f"IDs de películas existentes: {len(peliculas_ids)}")
    print(f"IDs únicos en ratings: {len(ratings_movie_ids)}")
    print(f"Ratings con película válida: {len(valid_ratings)}")
    print(f"Ratings huérfanos encontrados: {len(orphaned_ratings)}")

    if orphaned_ratings:
        print(f"Ejemplos de IDs huérfanos: {list(orphaned_ratings)[:10]}")

    # Filtrar ratings válidos
    ratings_clean = ratings_df[ratings_df["movieId"].isin(peliculas_ids)].copy()

    ratings_eliminados = len(ratings_df) - len(ratings_clean)
    print(f"Ratings eliminados: {ratings_eliminados}")
    print(f"Ratings finales: {len(ratings_clean)}")

    return ratings_clean


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


def process_movies(peliculas_df, keywords_by_movie=None):
    """Procesa el DataFrame de películas para crear una estructura más limpia"""
    print("\nProcesando datos de películas...")

    # Crear lista de películas estructuradas
    peliculas_list = []

    for _, row in peliculas_df.iterrows():
        movie_id = int(row["id"])

        # Extraer géneros (eliminar columnas vacías de géneros)
        generos = []
        for col in peliculas_df.columns:
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

    return peliculas_list


def process_ratings(ratings_clean):
    """Procesa los ratings limpios agrupándolos por película y usuario"""
    print("\nProcesando ratings...")

    # Convertir a estructura más eficiente
    ratings_by_movie = {}
    ratings_by_user = {}

    for _, row in ratings_clean.iterrows():
        movie_id = int(row["movieId"])
        user_id = int(row["userId"])
        rating = float(row["rating"])

        # Por película
        if movie_id not in ratings_by_movie:
            ratings_by_movie[movie_id] = []
        ratings_by_movie[movie_id].append({"userId": user_id, "rating": rating})

        # Por usuario
        if user_id not in ratings_by_user:
            ratings_by_user[user_id] = []
        ratings_by_user[user_id].append({"movieId": movie_id, "rating": rating})

    # Calcular estadísticas por película
    movie_stats = {}
    for movie_id, ratings in ratings_by_movie.items():
        ratings_values = [r["rating"] for r in ratings]
        movie_stats[movie_id] = {
            "num_ratings": len(ratings_values),
            "avg_rating": round(np.mean(ratings_values), 2),
            "min_rating": min(ratings_values),
            "max_rating": max(ratings_values),
        }

    return ratings_by_movie, ratings_by_user, movie_stats


def create_comprehensive_dataset(peliculas_list, ratings_by_movie, movie_stats):
    """Crea un dataset integrado con películas y sus ratings"""
    print("\nIntegrando datos...")

    comprehensive_data = []

    for pelicula in peliculas_list:
        movie_id = pelicula["id"]

        # Agregar estadísticas de ratings si existen
        if movie_id in movie_stats:
            pelicula["rating_stats"] = movie_stats[movie_id]
            pelicula["user_ratings"] = ratings_by_movie[movie_id]
        else:
            pelicula["rating_stats"] = {
                "num_ratings": 0,
                "avg_rating": None,
                "min_rating": None,
                "max_rating": None,
            }
            pelicula["user_ratings"] = []

        comprehensive_data.append(pelicula)

    return comprehensive_data


def save_json_data(data, filename, indent=2):
    """Guarda datos en formato JSON"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    print(f"Guardado: {filename}")


def main():
    """Función principal"""
    print("=== LIMPIEZA Y TRANSFORMACIÓN DE DATOS ===\n")

    try:
        # 1. Cargar datos
        peliculas_df, ratings_df, keywords_df = load_data()

        # 2. Limpiar ratings
        ratings_clean = clean_ratings(peliculas_df, ratings_df)

        # 3. Dividir ratings en train/test
        train_ratings, test_ratings = split_train_test_ratings(ratings_clean)

        # 4. Procesar datos
        keywords_by_movie = process_keywords(keywords_df)
        peliculas_list = process_movies(peliculas_df, keywords_by_movie)

        # Procesar ratings de entrenamiento y completos
        train_ratings_by_movie, train_ratings_by_user, train_movie_stats = (
            process_ratings(train_ratings)
        )
        ratings_by_movie, ratings_by_user, movie_stats = process_ratings(
            ratings_clean
        )  # Completos para referencia

        # 5. Crear datasets integrados
        comprehensive_data = create_comprehensive_dataset(
            peliculas_list, ratings_by_movie, movie_stats
        )

        # Dataset solo con datos de entrenamiento (para entrenamiento del modelo)
        train_comprehensive_data = create_comprehensive_dataset(
            peliculas_list, train_ratings_by_movie, train_movie_stats
        )

        # 6. Guardar resultados
        print("\n=== GUARDANDO RESULTADOS ===")

        # Dataset completo (más pesado pero completo)
        save_json_data(comprehensive_data, "Data_set/peliculas_con_ratings.json")

        # Dataset solo con datos de entrenamiento
        save_json_data(
            train_comprehensive_data, "Data_set/peliculas_con_ratings_train.json"
        )

        # Solo películas (más ligero)
        save_json_data(peliculas_list, "Data_set/peliculas_clean.json")

        # Ratings completos
        save_json_data(ratings_by_movie, "Data_set/ratings_por_pelicula.json")
        save_json_data(ratings_by_user, "Data_set/ratings_por_usuario.json")

        # Ratings de entrenamiento
        save_json_data(
            train_ratings_by_movie, "Data_set/train_ratings_por_pelicula.json"
        )
        save_json_data(train_ratings_by_user, "Data_set/train_ratings_por_usuario.json")

        # Keywords por película
        save_json_data(keywords_by_movie, "Data_set/keywords_por_pelicula.json")

        # Estadísticas de películas (completas y entrenamiento)
        save_json_data(movie_stats, "Data_set/estadisticas_peliculas.json")
        save_json_data(train_movie_stats, "Data_set/train_estadisticas_peliculas.json")

        # Guardar también los CSVs limpios de ratings
        ratings_clean.to_csv(
            "Data_set/ratings_clean.csv",
            sep=";",
            decimal=",",
            index=False,
            encoding="utf-8",
        )
        print("Guardado: Data_set/ratings_clean.csv")

        # Guardar train y test CSVs
        train_ratings.to_csv(
            "Data_set/train_ratings.csv",
            sep=";",
            decimal=",",
            index=False,
            encoding="utf-8",
        )
        print("Guardado: Data_set/train_ratings.csv")

        test_ratings.to_csv(
            "Data_set/test_ratings.csv",
            sep=";",
            decimal=",",
            index=False,
            encoding="utf-8",
        )
        print("Guardado: Data_set/test_ratings.csv")

        # 6. Reporte final
        print("\n=== REPORTE FINAL ===")
        print(f"Películas procesadas: {len(peliculas_list)}")
        print(f"Ratings válidos totales: {len(ratings_clean)}")
        print(
            f"Ratings de entrenamiento: {len(train_ratings)} ({len(train_ratings)/len(ratings_clean)*100:.1f}%)"
        )
        print(
            f"Ratings de test: {len(test_ratings)} ({len(test_ratings)/len(ratings_clean)*100:.1f}%)"
        )
        print(f"Usuarios únicos: {len(ratings_by_user)}")
        print(f"Usuarios en entrenamiento: {len(train_ratings_by_user)}")
        print(
            f"Películas con ratings: {len([p for p in peliculas_list if p.get('rating_stats', {}).get('num_ratings', 0) > 0])}"
        )
        print(
            f"Películas sin ratings: {len([p for p in peliculas_list if p.get('rating_stats', {}).get('num_ratings', 0) == 0])}"
        )
        print(
            f"Películas con keywords: {len([p for p in peliculas_list if len(p.get('keywords', [])) > 0])}"
        )
        print(
            f"Películas sin keywords: {len([p for p in peliculas_list if len(p.get('keywords', [])) == 0])}"
        )

        print("\n✅ Proceso completado exitosamente!")
        print("\nArchivos generados:")
        print("\n📊 DATASETS COMPLETOS:")
        print("- peliculas_con_ratings.json (dataset completo)")
        print("- peliculas_clean.json (solo películas)")
        print("- ratings_por_pelicula.json (todos los ratings por película)")
        print("- ratings_por_usuario.json (todos los ratings por usuario)")
        print("- ratings_clean.csv (todos los ratings en CSV)")
        print("\n🎯 DATASETS DE ENTRENAMIENTO:")
        print(
            "- peliculas_con_ratings_train.json (dataset solo con datos de entrenamiento)"
        )
        print(
            "- train_ratings_por_pelicula.json (ratings de entrenamiento por película)"
        )
        print("- train_ratings_por_usuario.json (ratings de entrenamiento por usuario)")
        print("- train_ratings.csv (ratings de entrenamiento en CSV)")
        print(
            "- train_estadisticas_peliculas.json (estadísticas basadas solo en entrenamiento)"
        )
        print("\n🧪 DATASET DE TEST:")
        print("- test_ratings.csv (ratings para evaluación del modelo)")
        print("\n🏷️ OTROS:")
        print("- keywords_por_pelicula.json (keywords por película)")
        print("- estadisticas_peliculas.json (estadísticas completas)")

    except Exception as e:
        print(f"❌ Error durante el proceso: {e}")
        raise


if __name__ == "__main__":
    main()

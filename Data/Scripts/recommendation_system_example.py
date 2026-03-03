#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejemplo de cómo usar los datos train/test para sistema de recomendación
"""

import pandas as pd
import json
import numpy as np
from collections import defaultdict


class SimpleRecommendationSystem:
    """Sistema de recomendación básico usando solo datos de entrenamiento"""

    def __init__(self):
        self.user_ratings = {}
        self.movie_ratings = {}
        self.user_means = {}
        self.movie_means = {}
        self.all_movies = set()
        self.all_users = set()

    def load_train_data(self):
        """Carga SOLO los datos de entrenamiento"""
        print("🎯 Cargando datos de ENTRENAMIENTO...")

        # Cargar ratings de entrenamiento
        train_df = pd.read_csv(
            "Data_set/train_ratings.csv", sep=";", decimal=",", encoding="utf-8"
        )

        # Cargar información de películas
        with open("Data_set/peliculas_clean.json", "r", encoding="utf-8") as f:
            self.movies_info = {movie["id"]: movie for movie in json.load(f)}

        print(f"Ratings de entrenamiento cargados: {len(train_df)}")
        print(f"Información de películas cargada: {len(self.movies_info)}")

        # Organizar datos por usuario y por película
        for _, row in train_df.iterrows():
            user_id = int(row["userId"])
            movie_id = int(row["movieId"])
            rating = float(row["rating"])

            # Por usuario
            if user_id not in self.user_ratings:
                self.user_ratings[user_id] = {}
            self.user_ratings[user_id][movie_id] = rating

            # Por película
            if movie_id not in self.movie_ratings:
                self.movie_ratings[movie_id] = {}
            self.movie_ratings[movie_id][user_id] = rating

            self.all_users.add(user_id)
            self.all_movies.add(movie_id)

        # Calcular medias
        self._calculate_means()

        print(f"Usuarios únicos en entrenamiento: {len(self.all_users)}")
        print(f"Películas únicas en entrenamiento: {len(self.all_movies)}")

    def _calculate_means(self):
        """Calcula medias de usuarios y películas"""
        # Media por usuario
        for user_id, ratings in self.user_ratings.items():
            self.user_means[user_id] = np.mean(list(ratings.values()))

        # Media por película
        for movie_id, ratings in self.movie_ratings.items():
            self.movie_means[movie_id] = np.mean(list(ratings.values()))

    def predict_rating(self, user_id, movie_id):
        """Predice rating para un usuario y película específica"""

        # Si no conocemos al usuario o la película, usar medias globales
        if user_id not in self.user_ratings:
            if movie_id in self.movie_means:
                return self.movie_means[movie_id]
            return 3.0  # Rating neutro

        if movie_id not in self.movie_ratings:
            return self.user_means.get(user_id, 3.0)

        # Si el usuario ya calificó esta película en train, no debería pasar
        if movie_id in self.user_ratings[user_id]:
            return self.user_ratings[user_id][movie_id]

        # Predicción simple: media del usuario
        return self.user_means[user_id]

    def recommend_movies(self, user_id, n_recommendations=10):
        """Genera recomendaciones para un usuario específico"""

        if user_id not in self.user_ratings:
            print(f"Usuario {user_id} no encontrado en datos de entrenamiento")
            return []

        # Películas que el usuario NO ha visto en entrenamiento
        seen_movies = set(self.user_ratings[user_id].keys())
        unseen_movies = self.all_movies - seen_movies

        # Predecir ratings para películas no vistas
        predictions = []
        for movie_id in unseen_movies:
            predicted_rating = self.predict_rating(user_id, movie_id)
            movie_title = self.movies_info.get(movie_id, {}).get(
                "titulo", f"Película {movie_id}"
            )

            predictions.append(
                {
                    "movie_id": movie_id,
                    "predicted_rating": predicted_rating,
                    "title": movie_title,
                }
            )

        # Ordenar por rating predicho (mayor a menor)
        predictions.sort(key=lambda x: x["predicted_rating"], reverse=True)

        return predictions[:n_recommendations]

    def evaluate_on_test(self):
        """Evalúa el modelo usando los datos de test"""
        print("\\n🧪 Evaluando modelo con datos de TEST...")

        # Cargar datos de test
        test_df = pd.read_csv(
            "Data_set/test_ratings.csv", sep=";", decimal=",", encoding="utf-8"
        )

        predictions = []
        actual_ratings = []

        for _, row in test_df.iterrows():
            user_id = int(row["userId"])
            movie_id = int(row["movieId"])
            actual_rating = float(row["rating"])

            predicted_rating = self.predict_rating(user_id, movie_id)

            predictions.append(predicted_rating)
            actual_ratings.append(actual_rating)

        # Calcular métricas
        predictions = np.array(predictions)
        actual_ratings = np.array(actual_ratings)

        mae = np.mean(np.abs(predictions - actual_ratings))
        rmse = np.sqrt(np.mean((predictions - actual_ratings) ** 2))

        print(f"Evaluación en {len(test_df)} ratings de test:")
        print(f"MAE (Error Absoluto Medio): {mae:.3f}")
        print(f"RMSE (Raíz del Error Cuadrático Medio): {rmse:.3f}")

        return mae, rmse


def main():
    """Ejemplo de uso del sistema de recomendación"""
    print("=== SISTEMA DE RECOMENDACIÓN - EJEMPLO DE USO ===\\n")

    # 1. Crear y entrenar el modelo
    recommender = SimpleRecommendationSystem()
    recommender.load_train_data()

    # 2. Hacer recomendaciones para algunos usuarios
    print("\\n📋 GENERANDO RECOMENDACIONES:")

    # Tomar algunos usuarios de ejemplo
    example_users = [1, 5, 10, 20, 50]

    for user_id in example_users:
        print(f"\\n👤 Recomendaciones para Usuario {user_id}:")
        recommendations = recommender.recommend_movies(user_id, n_recommendations=5)

        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(
                    f"  {i}. {rec['title']} (Rating predicho: {rec['predicted_rating']:.2f})"
                )
        else:
            print("  No se pudieron generar recomendaciones")

    # 3. Evaluar el modelo
    mae, rmse = recommender.evaluate_on_test()

    print("\\n" + "=" * 60)
    print("✅ RESUMEN:")
    print(f"- Modelo entrenado SOLO con datos de train")
    print(f"- Evaluado con datos de test independientes")
    print(f"- MAE: {mae:.3f} (menor es mejor)")
    print(f"- RMSE: {rmse:.3f} (menor es mejor)")
    print("\\n💡 Ahora puedes implementar algoritmos más sofisticados!")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import json
import math
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler


class ColaborativeRecommender:

    FILE_PATH_PREFERENCE_MATRIX = "Backend/Colaborativo/preference_matrix.npz"

    FILE_PATH_RATINGS = "Data/Clean_data/train_ratings.json"
    MIN_COMMON_PREFS = 10  # umbral para decidir intersección vs unión
    FAVORABLE_RATING_THRESHOLD = 3.5  # rating mínimo para considerar "favorable"

    def __init__(self, user_registry_path: str, movie_data_path: str):
        self.user_registry_path = user_registry_path
        self.movie_data_path = movie_data_path
        self.user_registry = self._load_user_registry()
        self.movie_data = self._load_movie_data()
        self.genre_map = self._build_genre_map()
        self.user_id_to_idx = {}  # user_id (str) -> row index
        self.idx_to_user_id = {}  # row index -> user_id (str)
        self.preference_matrix = None
        self.ratings_by_user = self._load_ratings()

    def get_preference_matrix(self):
        """
        Obtiene la matriz de preferencias usuario-género. Si ya ha sido calculada, la devuelve.
        Si no, la construye a partir de los datos de entrenamiento y la guarda para uso futuro.
        """
        if self.preference_matrix is None:
            if os.path.exists(self.FILE_PATH_PREFERENCE_MATRIX):
                self.preference_matrix = sparse.load_npz(
                    self.FILE_PATH_PREFERENCE_MATRIX
                )
                self._rebuild_user_mappings()
            else:
                self._build_preference_matrix()
        return self.preference_matrix

    def _rebuild_user_mappings(self):
        """Reconstruye los mapeos user_id <-> índice cuando la matriz se carga de disco."""
        idx = 0
        for uid, user in self.user_registry["users"].items():
            if user.get("status") != "active":
                continue
            self.user_id_to_idx[uid] = idx
            self.idx_to_user_id[idx] = uid
            idx += 1

    def _build_preference_matrix(self):
        """
        Construye una matriz de preferencias usuario-género a partir del genre_vector
        de cada usuario. Cada fila es un usuario activo, cada columna un género.
        Valores en rango [0, 100] donde 0 = sin interés.
        """
        genre_names = sorted(self.genre_map.keys(), key=lambda g: self.genre_map[g])
        print(f"Géneros encontrados: {genre_names}")
        n_genres = len(genre_names)
        users = self.user_registry["users"]

        # Construir mapeos user_id <-> índice de fila
        self.user_id_to_idx = {}
        self.idx_to_user_id = {}
        rows = []
        idx = 0
        for uid, user in users.items():
            if user.get("status") != "active":
                continue
            genre_vector = user["preferences"].get("genre_vector", {})
            row = np.array(
                [genre_vector.get(g, 0.0) for g in genre_names], dtype=np.float64
            )
            rows.append(row)
            self.user_id_to_idx[uid] = idx
            self.idx_to_user_id[idx] = uid
            idx += 1

        matrix = np.vstack(rows)  # shape: (n_users, n_genres)
        self.preference_matrix = sparse.csr_matrix(matrix)
        sparse.save_npz(self.FILE_PATH_PREFERENCE_MATRIX, self.preference_matrix)
        print(
            f"Matriz de preferencias construida: {self.preference_matrix.shape} "
            f"({idx} usuarios × {n_genres} géneros)"
        )

    def precompute_all_neighbors(
        self, top_n: int = 40
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Precalcula los vecinos más similares para todos los usuarios activos.
        Almacena los vecinos en el user_registry y lo persiste a disco.
        Retorna dict {user_id: [(neighbor_id, similarity), ...]}.
        """
        self.get_preference_matrix()
        all_neighbors = {}
        user_ids = list(self.user_id_to_idx.keys())
        total = len(user_ids)

        for i, uid in enumerate(user_ids):
            neighbors = self._find_neighbors(uid, top_n=top_n)
            all_neighbors[uid] = neighbors
            # Guardar en el registro del usuario
            self.user_registry["users"][uid]["neighbors"] = [
                {"user_id": nid, "similarity": round(sim, 6)}
                for nid, sim, _ in neighbors
            ]
            if (i + 1) % 100 == 0:
                print(f"Vecinos calculados: {i + 1}/{total}")

        self._save_user_registry()
        print(f"Vecinos precalculados para {total} usuarios (top_n={top_n})")
        return all_neighbors

    def recommend(self, user_id: str, top_n: int = 10) -> List[Tuple[int, float]]:
        """
        Genera recomendaciones para el usuario dado.
        Pipeline: vecinos → ítems candidatos → predicción ponderada → ranking.
        Retorna lista de tuplas (movie_id, predicted_rating) ordenada desc.
        """
        user_id = str(user_id)
        # Obtener vecinos (del registro si están precalculados, sino calcular)
        user_data = self.user_registry["users"].get(user_id)
        if user_data is None:
            return []

        if "neighbors" in user_data and user_data["neighbors"]:
            neighbors = [
                (n["user_id"], n["similarity"]) for n in user_data["neighbors"]
            ]
        else:
            self.get_preference_matrix()
            neighbors = self._find_neighbors(user_id)

        if not neighbors:
            return []

        # Obtener candidatos y predecir
        candidates = self._get_candidate_items(user_id, neighbors)
        if not candidates:
            return []

        predictions = self._predict_ratings(user_id, candidates, neighbors)

        # Ordenar por rating predicho descendente
        ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    def explain_recommendation(self, user_id: str, movie_id: int) -> Dict[str, Any]:
        """
        Genera una explicación para la recomendación de un ítem a un usuario.
        Retorna dict con vecinos contribuyentes, sus ratings y pesos.
        """
        user_id = str(user_id)
        user_data = self.user_registry["users"].get(user_id)
        if user_data is None:
            return {"error": "Usuario no encontrado"}

        if "neighbors" in user_data and user_data["neighbors"]:
            neighbors = [
                (n["user_id"], n["similarity"]) for n in user_data["neighbors"]
            ]
        else:
            self.get_preference_matrix()
            neighbors = self._find_neighbors(user_id)

        contributors = []
        for neighbor_id, sim, _ in neighbors:
            nid_int = int(neighbor_id)
            neighbor_ratings = self.ratings_by_user.get(nid_int, {})
            if movie_id in neighbor_ratings:
                contributors.append(
                    {
                        "neighbor_id": neighbor_id,
                        "similarity": round(sim, 4),
                        "rating": neighbor_ratings[movie_id],
                        "weight": round(sim, 4),
                    }
                )

        if not contributors:
            return {
                "movie_id": movie_id,
                "explanation": "Ningún vecino ha visto esta película",
            }

        total_weight = sum(abs(c["weight"]) for c in contributors)
        predicted = (
            sum(c["weight"] * c["rating"] for c in contributors) / total_weight
            if total_weight > 0
            else 0.0
        )

        return {
            "movie_id": movie_id,
            "predicted_rating": round(predicted, 3),
            "num_contributors": len(contributors),
            "contributors": contributors,
        }

    def _pearson_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calcula la similitud de Pearson entre dos vectores de preferencias.
        Aplica estrategia híbrida intersección/unión:
        - Si hay >MIN_COMMON_PREFS preferencias comunes no nulas → intersección
        - Si no → unión (0 = sin interés, no dato faltante)
        """
        if len(vec_a) != len(vec_b):
            raise ValueError("Los vectores deben tener la misma longitud")

        # Determinar dimensiones comunes no nulas
        nonzero_a = vec_a != 0
        nonzero_b = vec_b != 0
        common_nonzero = nonzero_a & nonzero_b
        n_common = np.sum(common_nonzero)

        if n_common >= self.MIN_COMMON_PREFS:
            # Intersección: solo dimensiones donde ambos tienen valor
            a = vec_a[common_nonzero]
            b = vec_b[common_nonzero]
        else:
            # Unión: todas las dimensiones (0 = no interés)
            a = vec_a
            b = vec_b

        if len(a) < 2:
            return 0.0

        mean_a = np.mean(a)
        mean_b = np.mean(b)

        diff_a = a - mean_a
        diff_b = b - mean_b

        numerator = np.sum(diff_a * diff_b)
        denominator = math.sqrt(np.sum(diff_a**2)) * math.sqrt(np.sum(diff_b**2))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _find_neighbors(self, user_id: str, top_n: int = 40) -> List[Tuple[str, float]]:
        """
        Encuentra los top_n vecinos más similares al usuario usando Pearson
        sobre la matriz de preferencias (genre_vector).
        Retorna lista de tuplas (neighbor_id, similarity) ordenada desc.
        """
        if user_id not in self.user_id_to_idx:
            return []

        user_idx = self.user_id_to_idx[user_id]
        user_vec = self.preference_matrix[user_idx].toarray().flatten()

        similarities = []
        for other_uid, other_idx in self.user_id_to_idx.items():
            if other_uid == user_id:
                continue
            other_vec = self.preference_matrix[other_idx].toarray().flatten()
            sim = self._pearson_similarity(user_vec, other_vec)
            if sim > 0:  # Solo vecinos con correlación positiva
                similarities.append((other_uid, sim, other_vec))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]

    def _build_genre_map(self) -> Dict[str, int]:
        """
        Construye un mapa de géneros a índices a partir del genre_vector del primer usuario.
        Retorna un diccionario {genre_name: genre_index} con orden estable.
        """
        first_user = next(iter(self.user_registry["users"].values()))
        genre_names = sorted(first_user["preferences"]["genre_vector"].keys())
        return {genre: idx for idx, genre in enumerate(genre_names)}

    def _load_user_registry(self) -> Dict[str, Any]:
        with open(self.user_registry_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_movie_data(self) -> List[Dict[str, Any]]:
        with open(self.movie_data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_ratings(self) -> Dict[int, Dict[int, float]]:
        """
        Carga train_ratings.json y organiza como {userId: {movieId: rating}}.
        """
        path = self.FILE_PATH_RATINGS
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        ratings_by_user: Dict[int, Dict[int, float]] = {}
        for entry in raw:
            uid = entry["userId"]
            mid = entry["movieId"]
            r = entry["rating"]
            if uid not in ratings_by_user:
                ratings_by_user[uid] = {}
            ratings_by_user[uid][mid] = r
        return ratings_by_user

    def _save_user_registry(self):
        """Persiste el user_registry actual a disco."""
        with open(self.user_registry_path, "w", encoding="utf-8") as f:
            json.dump(self.user_registry, f, ensure_ascii=False, indent=2)

    def _get_candidate_items(
        self, user_id: str, neighbors: List[Tuple[str, float]]
    ) -> List[int]:
        """
        Obtiene películas candidatas: ítems puntuados favorablemente por los vecinos
        que el usuario actual no ha visto.
        """
        user_data = self.user_registry["users"][user_id]
        seen = set(user_data["preferences"].get("watched_movies", []))

        candidates = set()
        for neighbor_id, _sim, _ in neighbors:
            nid_int = int(neighbor_id)
            neighbor_ratings = self.ratings_by_user.get(nid_int, {})
            for movie_id, rating in neighbor_ratings.items():
                if rating >= self.FAVORABLE_RATING_THRESHOLD and movie_id not in seen:
                    candidates.add(movie_id)

        return list(candidates)

    def _predict_ratings(
        self,
        user_id: str,
        candidate_items: List[int],
        neighbors: List[Tuple[str, float]],
    ) -> Dict[int, float]:
        """
        Predice ratings para los candidatos usando la fórmula ponderada:
        r(u,i) = Σ sim(u,u') * r(u',i) / Σ |sim(u,u')|
        Solo se suman vecinos que hayan visto el ítem.
        """
        predictions = {}
        for movie_id in candidate_items:
            numerator = 0.0
            denominator = 0.0
            for neighbor_id, sim, _ in neighbors:
                nid_int = int(neighbor_id)
                neighbor_ratings = self.ratings_by_user.get(nid_int, {})
                if movie_id in neighbor_ratings:
                    numerator += sim * neighbor_ratings[movie_id]
                    denominator += abs(sim)
            if denominator > 0:
                predictions[movie_id] = numerator / denominator
        return predictions

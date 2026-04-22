from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy import sparse

from project_paths import CLEAN_DATA_DIR, COLLABORATIVE_MATRIX_PATH, RAW_DATA_DIR


class ColaborativeRecommender:
    FILE_PATH_PREFERENCE_MATRIX = COLLABORATIVE_MATRIX_PATH
    FILE_PATH_RATINGS = CLEAN_DATA_DIR / "train_ratings.json"

    MIN_COMMON_PREFS = 10
    FAVORABLE_RATING_THRESHOLD = 3.5
    NEGATIVE_RATING_THRESHOLD = 2.5
    MIN_RECOMMENDATION_CONTRIBUTORS = 2
    CONFIDENCE_NEIGHBOR_TARGET = 3

    def __init__(
        self,
        user_registry_path: str,
        movie_data_path: str,
        min_recommendation_contributors: int = MIN_RECOMMENDATION_CONTRIBUTORS,
        confidence_neighbor_target: int = CONFIDENCE_NEIGHBOR_TARGET,
    ):
        self.user_registry_path = user_registry_path
        self.movie_data_path = movie_data_path
        self.min_recommendation_contributors = max(
            1, int(min_recommendation_contributors)
        )
        self.confidence_neighbor_target = max(1, int(confidence_neighbor_target))

        self.user_registry = self._load_user_registry()
        self.movie_data = self._load_movie_data()
        self.genre_name_lookup = self._load_genre_name_lookup()
        self.movie_genres_by_id = self._build_movie_genre_lookup()
        self.genre_map = self._build_genre_map()
        self.user_id_to_idx: Dict[str, int] = {}
        self.idx_to_user_id: Dict[int, str] = {}
        self.preference_matrix: Optional[sparse.csr_matrix] = None
        self.ratings_by_user = self._load_ratings()

    def get_preference_matrix(self):
        """
        Obtiene la matriz de preferencias usuario-genero. Si ya ha sido calculada, la devuelve.
        Si no, la construye a partir de los datos de entrenamiento y la guarda para uso futuro.
        """
        profiles_repaired = self._ensure_user_genre_vectors()
        if profiles_repaired:
            self.genre_map = self._build_genre_map()

        if self.preference_matrix is None:
            if self.FILE_PATH_PREFERENCE_MATRIX.exists() and not profiles_repaired:
                self.preference_matrix = sparse.load_npz(
                    str(self.FILE_PATH_PREFERENCE_MATRIX)
                )
                self._rebuild_user_mappings()
                if not self._is_cached_matrix_compatible():
                    print(
                        "Matriz de preferencias desactualizada detectada. "
                        "Reconstruyendo desde user_registry..."
                    )
                    self._build_preference_matrix()
            else:
                self._build_preference_matrix()
        return self.preference_matrix

    def refresh_preference_matrix(self):
        """
        Fuerza la reconstruccion de la matriz usuario-genero desde disco.

        Este metodo recarga el `user_registry`, recompone el mapa de generos y vuelve
        a generar la matriz cacheada para reflejar altas, bajas o cambios recientes
        en las preferencias de los usuarios.
        """
        self.user_registry = self._load_user_registry()
        self.movie_data = self._load_movie_data()
        self.movie_genres_by_id = self._build_movie_genre_lookup()
        self._ensure_user_genre_vectors()
        self.genre_map = self._build_genre_map()
        self.preference_matrix = None
        self._build_preference_matrix()
        return self.preference_matrix

    def _is_cached_matrix_compatible(self) -> bool:
        """Valida que la matriz cacheada sigue alineada con usuarios activos y generos."""
        if self.preference_matrix is None:
            return False

        active_users = sum(
            1
            for user in self.user_registry["users"].values()
            if user.get("status") == "active"
        )
        expected_genres = len(self.genre_map)
        registry_path = Path(self.user_registry_path)
        cache_mtime = (
            self.FILE_PATH_PREFERENCE_MATRIX.stat().st_mtime
            if self.FILE_PATH_PREFERENCE_MATRIX.exists()
            else 0
        )
        registry_mtime = registry_path.stat().st_mtime if registry_path.exists() else 0

        return (
            self.preference_matrix.shape[0] == active_users
            and self.preference_matrix.shape[1] == expected_genres
            and len(self.user_id_to_idx) == active_users
            and cache_mtime >= registry_mtime
        )

    def _rebuild_user_mappings(self):
        """Reconstruye los mapeos user_id <-> indice cuando la matriz se carga de disco."""
        self.user_id_to_idx = {}
        self.idx_to_user_id = {}
        idx = 0
        for uid, user in self.user_registry["users"].items():
            if user.get("status") != "active":
                continue
            self.user_id_to_idx[uid] = idx
            self.idx_to_user_id[idx] = uid
            idx += 1

    def _build_preference_matrix(self):
        """
        Construye una matriz de preferencias usuario-genero a partir del genre_vector
        de cada usuario. Cada fila es un usuario activo, cada columna un genero.
        Valores en rango [0, 100] donde 0 = sin interes.
        """
        genre_names = sorted(self.genre_map.keys(), key=lambda genre: self.genre_map[genre])
        n_genres = len(genre_names)
        users = self.user_registry["users"]

        self.user_id_to_idx = {}
        self.idx_to_user_id = {}
        rows: List[np.ndarray] = []
        idx = 0

        for uid, user in users.items():
            if user.get("status") != "active":
                continue

            genre_vector = user["preferences"].get("genre_vector", {})
            row = np.array(
                [genre_vector.get(genre, 0.0) for genre in genre_names],
                dtype=np.float64,
            )
            rows.append(row)
            self.user_id_to_idx[uid] = idx
            self.idx_to_user_id[idx] = uid
            idx += 1

        matrix = np.vstack(rows)
        self.preference_matrix = sparse.csr_matrix(matrix)
        sparse.save_npz(str(self.FILE_PATH_PREFERENCE_MATRIX), self.preference_matrix)
        print(
            f"Matriz de preferencias construida: {self.preference_matrix.shape} "
            f"({idx} usuarios x {n_genres} generos)"
        )

    def precompute_all_neighbors(
        self, top_n: int = 40
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Precalcula los vecinos mas similares para todos los usuarios activos.
        Almacena los vecinos en el user_registry y lo persiste a disco.
        """
        self.get_preference_matrix()
        all_neighbors: Dict[str, List[Tuple[str, float]]] = {}
        user_ids = list(self.user_id_to_idx.keys())
        total = len(user_ids)

        for i, uid in enumerate(user_ids):
            neighbors = self._find_neighbors(uid, top_n=top_n)
            all_neighbors[uid] = neighbors
            self.user_registry["users"][uid]["neighbors"] = [
                {"user_id": neighbor_id, "similarity": round(similarity, 6)}
                for neighbor_id, similarity, _ in neighbors
            ]
            if (i + 1) % 100 == 0:
                print(f"Vecinos calculados: {i + 1}/{total}")

        self._save_user_registry()
        print(f"Vecinos precalculados para {total} usuarios (top_n={top_n})")
        return all_neighbors

    def recommend(
        self, user_id: str, top_n: int = 10, return_details: bool = False
    ) -> List[Tuple[int, float]]:
        """
        Genera recomendaciones para el usuario dado.

        La prediccion devuelve dos conceptos distintos:
        - predicted_rating: rating esperado en escala 0-5
        - confidence: confianza en [0, 1] segun similitud media y numero de vecinos

        El ranking prioriza primero la confianza y el soporte para evitar que una sola
        coincidencia con rating 5.0 domine la lista.
        """
        user_id = str(user_id)
        user_data = self.user_registry["users"].get(user_id)
        if user_data is None:
            return []

        if "neighbors" in user_data and user_data["neighbors"]:
            raw_neighbors = [
                (neighbor["user_id"], neighbor["similarity"])
                for neighbor in user_data["neighbors"]
            ]
        else:
            self.get_preference_matrix()
            raw_neighbors = self._find_neighbors(user_id)

        neighbors = self._normalize_neighbors(raw_neighbors)
        if not neighbors:
            return []

        candidates = self._get_candidate_items(user_id, neighbors)
        if not candidates:
            return []

        predictions = self._predict_ratings(candidate_items=candidates, neighbors=neighbors)
        ranked = self._rank_predictions(predictions, top_n=top_n)

        if return_details:
            return ranked

        return [
            (int(prediction["movie_id"]), float(prediction["predicted_rating"]))
            for prediction in ranked
        ]

    def explain_recommendation(self, user_id: str, movie_id: int) -> Dict[str, Any]:
        """
        Genera una explicacion para la recomendacion de un item a un usuario.
        Retorna metricas de prediccion, confianza y vecinos contribuyentes.
        """
        user_id = str(user_id)
        user_data = self.user_registry["users"].get(user_id)
        if user_data is None:
            return {"error": "Usuario no encontrado"}

        if "neighbors" in user_data and user_data["neighbors"]:
            raw_neighbors = [
                (neighbor["user_id"], neighbor["similarity"])
                for neighbor in user_data["neighbors"]
            ]
        else:
            self.get_preference_matrix()
            raw_neighbors = self._find_neighbors(user_id)

        neighbors = self._normalize_neighbors(raw_neighbors)
        contributors = self._get_contributors(movie_id, neighbors)
        if not contributors:
            return {
                "movie_id": movie_id,
                "explanation": "Ningun vecino ha visto esta pelicula",
            }

        prediction = self._build_prediction(movie_id, contributors)
        rounded_contributors = [
            {
                "neighbor_id": contributor["neighbor_id"],
                "similarity": round(contributor["similarity"], 4),
                "rating": contributor["rating"],
                "weight": round(contributor["weight"], 4),
            }
            for contributor in contributors
        ]

        return {
            "movie_id": movie_id,
            "predicted_rating": round(prediction["predicted_rating"], 3),
            "confidence": round(prediction["confidence"], 3),
            "mean_similarity": round(prediction["mean_similarity"], 3),
            "support_ratio": round(prediction["support_ratio"], 3),
            "ranking_score": round(prediction["ranking_score"], 3),
            "num_contributors": prediction["num_contributors"],
            "positive_contributors": prediction["positive_contributors"],
            "neutral_contributors": prediction["neutral_contributors"],
            "negative_contributors": prediction["negative_contributors"],
            "average_contributor_rating": round(
                prediction["average_contributor_rating"], 3
            ),
            "contributors": rounded_contributors,
        }

    def _normalize_neighbors(
        self, neighbors: Iterable[Tuple[Any, ...]]
    ) -> List[Tuple[str, float, Optional[np.ndarray]]]:
        """Normaliza vecinos a tuplas (neighbor_id, similarity, other_vec)."""
        normalized_neighbors: List[Tuple[str, float, Optional[np.ndarray]]] = []
        for neighbor in neighbors:
            if len(neighbor) == 3:
                neighbor_id, similarity, other_vec = neighbor
            elif len(neighbor) == 2:
                neighbor_id, similarity = neighbor
                other_vec = None
            else:
                continue

            normalized_neighbors.append(
                (str(neighbor_id), float(similarity), other_vec)
            )

        return normalized_neighbors

    def _pearson_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calcula la similitud de Pearson entre dos vectores de preferencias.
        Usa interseccion si ambos comparten suficientes generos no nulos.
        En caso contrario usa la union completa.
        """
        if len(vec_a) != len(vec_b):
            raise ValueError("Los vectores deben tener la misma longitud")

        nonzero_a = vec_a != 0
        nonzero_b = vec_b != 0
        common_nonzero = nonzero_a & nonzero_b
        n_common = np.sum(common_nonzero)

        if n_common >= self.MIN_COMMON_PREFS:
            a = vec_a[common_nonzero]
            b = vec_b[common_nonzero]
        else:
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

        return float(numerator / denominator)

    def _find_neighbors(
        self, user_id: str, top_n: int = 40
    ) -> List[Tuple[str, float, np.ndarray]]:
        """
        Encuentra los top_n vecinos mas similares al usuario usando Pearson
        sobre la matriz de preferencias.
        """
        if user_id not in self.user_id_to_idx:
            return []

        assert self.preference_matrix is not None
        user_idx = self.user_id_to_idx[user_id]
        user_vec = self.preference_matrix[user_idx].toarray().flatten()

        similarities: List[Tuple[str, float, np.ndarray]] = []
        for other_uid, other_idx in self.user_id_to_idx.items():
            if other_uid == user_id:
                continue
            other_vec = self.preference_matrix[other_idx].toarray().flatten()
            sim = self._pearson_similarity(user_vec, other_vec)
            if sim > 0:
                similarities.append((other_uid, sim, other_vec))

        similarities.sort(key=lambda item: item[1], reverse=True)
        return similarities[:top_n]

    def _build_genre_map(self) -> Dict[str, int]:
        """Construye un mapa de generos a indices con orden estable."""
        genre_names = set()

        for user in self.user_registry["users"].values():
            genre_vector = user.get("preferences", {}).get("genre_vector", {})
            if isinstance(genre_vector, dict):
                genre_names.update(
                    str(genre).strip() for genre in genre_vector.keys() if str(genre).strip()
                )

        if not genre_names:
            genre_names.update(
                genre for genre in self.genre_name_lookup.values() if str(genre).strip()
            )

        genre_names = sorted(genre_names)
        return {genre: idx for idx, genre in enumerate(genre_names)}

    def _load_genre_name_lookup(self) -> Dict[int, str]:
        """Carga el mapeo de ids de genero a nombres en espanol."""
        genre_mapping_path = RAW_DATA_DIR / "generos.csv"
        if not genre_mapping_path.exists():
            return {}

        lookup: Dict[int, str] = {}
        with genre_mapping_path.open("r", encoding="utf-8") as file:
            reader = csv.DictReader(file, delimiter=";")
            for row in reader:
                try:
                    genre_id = int(row["IdDataset"])
                except Exception:
                    continue
                genre_name = str(row.get("GeneroSP", "")).strip()
                if genre_name:
                    lookup[genre_id] = genre_name
        return lookup

    def _build_movie_genre_lookup(self) -> Dict[int, List[str]]:
        """Construye un indice movie_id -> nombres de genero."""
        movie_genres: Dict[int, List[str]] = {}

        for movie in self.movie_data:
            try:
                movie_id = int(movie.get("id"))
            except Exception:
                continue

            raw_genres = movie.get("generos", [])
            genres: List[str] = []
            if isinstance(raw_genres, list):
                for raw_genre in raw_genres:
                    if isinstance(raw_genre, str):
                        genre_name = raw_genre.strip()
                    else:
                        genre_name = self.genre_name_lookup.get(int(raw_genre), "").strip()
                    if genre_name:
                        genres.append(genre_name)

            movie_genres[movie_id] = genres

        return movie_genres

    def _ensure_user_genre_vectors(self) -> bool:
        """
        Rellena genre_vector para usuarios que aun no lo tienen usando favoritos y watched_movies.
        """
        changed = False

        for user in self.user_registry["users"].values():
            if user.get("status") != "active":
                continue

            preferences = user.setdefault("preferences", {})
            genre_vector = preferences.get("genre_vector", {})
            if self._has_nonzero_genre_vector(genre_vector):
                continue

            derived_vector = self._derive_genre_vector_from_registry(user)
            if not derived_vector:
                continue

            preferences["genre_vector"] = derived_vector
            user.pop("neighbors", None)
            changed = True

        if changed:
            self._save_user_registry()

        return changed

    def _has_nonzero_genre_vector(self, genre_vector: Any) -> bool:
        if not isinstance(genre_vector, dict):
            return False

        for value in genre_vector.values():
            try:
                if float(value) > 0:
                    return True
            except Exception:
                continue
        return False

    def _derive_genre_vector_from_registry(self, user: Dict[str, Any]) -> Dict[str, float]:
        """Deriva un vector de generos a partir de favoritos y peliculas vistas."""
        preferences = user.get("preferences", {})
        favorite_genres = preferences.get("favorite_genres", [])
        watched_movies = preferences.get("watched_movies", [])

        scores: Dict[str, float] = {}

        for favorite in favorite_genres:
            genre_name = str(favorite).strip()
            if genre_name:
                scores[genre_name] = scores.get(genre_name, 0.0) + 3.0

        for movie_id in watched_movies:
            try:
                normalized_movie_id = int(movie_id)
            except Exception:
                continue

            for genre_name in self.movie_genres_by_id.get(normalized_movie_id, []):
                scores[genre_name] = scores.get(genre_name, 0.0) + 1.0

        if not scores:
            return {}

        max_score = max(scores.values())
        if max_score <= 0:
            return {}

        return {
            genre_name: round((score / max_score) * 100.0, 2)
            for genre_name, score in scores.items()
            if score > 0
        }

    def _load_user_registry(self) -> Dict[str, Any]:
        with open(self.user_registry_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _load_movie_data(self) -> List[Dict[str, Any]]:
        with open(self.movie_data_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _load_ratings(self) -> Dict[int, Dict[int, float]]:
        """Carga train_ratings.json y organiza como {userId: {movieId: rating}}."""
        path = self.FILE_PATH_RATINGS
        if not path.exists():
            return {}

        with path.open("r", encoding="utf-8") as file:
            raw = json.load(file)

        ratings_by_user: Dict[int, Dict[int, float]] = {}
        for entry in raw:
            user_id = entry["userId"]
            movie_id = entry["movieId"]
            rating = entry["rating"]
            ratings_by_user.setdefault(user_id, {})[movie_id] = rating

        return ratings_by_user

    def _save_user_registry(self):
        """Persiste el user_registry actual a disco."""
        with open(self.user_registry_path, "w", encoding="utf-8") as file:
            json.dump(self.user_registry, file, ensure_ascii=False, indent=2)

    def _get_candidate_items(
        self,
        user_id: str,
        neighbors: List[Tuple[str, float, Optional[np.ndarray]]],
    ) -> List[int]:
        """
        Obtiene peliculas candidatas: items puntuados favorablemente por los vecinos
        que el usuario actual no ha visto.
        """
        user_data = self.user_registry["users"][user_id]
        seen = set(user_data["preferences"].get("watched_movies", []))

        candidates = set()
        for neighbor_id, _similarity, _ in neighbors:
            neighbor_ratings = self.ratings_by_user.get(int(neighbor_id), {})
            for movie_id, rating in neighbor_ratings.items():
                if rating >= self.FAVORABLE_RATING_THRESHOLD and movie_id not in seen:
                    candidates.add(movie_id)

        return list(candidates)

    def _predict_ratings(
        self,
        candidate_items: List[int],
        neighbors: List[Tuple[str, float, Optional[np.ndarray]]],
    ) -> Dict[int, Dict[str, Any]]:
        """
        Predice ratings para los candidatos usando media ponderada por similitud.
        Devuelve tambien metricas de confianza y soporte.
        """
        predictions: Dict[int, Dict[str, Any]] = {}

        for movie_id in candidate_items:
            contributors = self._get_contributors(movie_id, neighbors)
            if not contributors:
                continue
            predictions[movie_id] = self._build_prediction(movie_id, contributors)

        return predictions

    def _get_contributors(
        self,
        movie_id: int,
        neighbors: List[Tuple[str, float, Optional[np.ndarray]]],
    ) -> List[Dict[str, float]]:
        """Obtiene los vecinos que realmente contribuyen a una pelicula concreta."""
        contributors: List[Dict[str, float]] = []

        for neighbor_id, similarity, _ in neighbors:
            neighbor_ratings = self.ratings_by_user.get(int(neighbor_id), {})
            if movie_id not in neighbor_ratings:
                continue
            contributors.append(
                {
                    "neighbor_id": neighbor_id,
                    "similarity": float(similarity),
                    "rating": float(neighbor_ratings[movie_id]),
                    "weight": float(similarity),
                }
            )

        return contributors

    def _build_prediction(
        self,
        movie_id: int,
        contributors: List[Dict[str, float]],
    ) -> Dict[str, Any]:
        """Construye rating predicho, confianza y score de ranking para un candidato."""
        numerator = sum(
            contributor["weight"] * contributor["rating"]
            for contributor in contributors
        )
        denominator = sum(abs(contributor["weight"]) for contributor in contributors)
        predicted_rating = numerator / denominator if denominator > 0 else 0.0

        num_contributors = len(contributors)
        mean_similarity = (
            sum(abs(contributor["similarity"]) for contributor in contributors)
            / num_contributors
            if num_contributors > 0
            else 0.0
        )
        support_ratio = min(
            1.0, num_contributors / float(self.confidence_neighbor_target)
        )
        confidence = mean_similarity * support_ratio
        ranking_score = predicted_rating * confidence
        average_contributor_rating = (
            sum(contributor["rating"] for contributor in contributors) / num_contributors
            if num_contributors > 0
            else 0.0
        )

        positive_contributors = sum(
            1
            for contributor in contributors
            if contributor["rating"] >= self.FAVORABLE_RATING_THRESHOLD
        )
        negative_contributors = sum(
            1
            for contributor in contributors
            if contributor["rating"] <= self.NEGATIVE_RATING_THRESHOLD
        )
        neutral_contributors = max(
            0, num_contributors - positive_contributors - negative_contributors
        )

        return {
            "movie_id": int(movie_id),
            "predicted_rating": float(predicted_rating),
            "num_contributors": int(num_contributors),
            "mean_similarity": float(mean_similarity),
            "support_ratio": float(support_ratio),
            "confidence": float(confidence),
            "ranking_score": float(ranking_score),
            "average_contributor_rating": float(average_contributor_rating),
            "positive_contributors": int(positive_contributors),
            "neutral_contributors": int(neutral_contributors),
            "negative_contributors": int(negative_contributors),
        }

    def _rank_predictions(
        self,
        predictions: Dict[int, Dict[str, Any]],
        top_n: int,
    ) -> List[Dict[str, Any]]:
        """
        Ordena las predicciones priorizando:
        1) recomendaciones con suficiente soporte
        2) score ponderado por confianza
        3) confianza y rating predicho
        """
        prediction_list = list(predictions.values())
        if not prediction_list:
            return []

        strong_predictions = [
            prediction
            for prediction in prediction_list
            if prediction["num_contributors"] >= self.min_recommendation_contributors
        ]
        weak_predictions = [
            prediction
            for prediction in prediction_list
            if prediction["num_contributors"] < self.min_recommendation_contributors
        ]

        def prediction_sort_key(prediction: Dict[str, Any]) -> Tuple[float, ...]:
            return (
                float(prediction["ranking_score"]),
                float(prediction["confidence"]),
                float(prediction["predicted_rating"]),
                float(prediction["mean_similarity"]),
                float(prediction["num_contributors"]),
            )

        strong_predictions.sort(key=prediction_sort_key, reverse=True)
        weak_predictions.sort(key=prediction_sort_key, reverse=True)

        return (strong_predictions + weak_predictions)[:top_n]

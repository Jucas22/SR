from __future__ import annotations

"""
Sistema recomendador basado en contenido para películas.

Mejoras incluidas respecto a la versión original:
- Usa explícitamente el genre_vector del user_registry como base del perfil.
- Reduce el vector de 20 géneros a un vector de 5-8 géneros relevantes.
- Separa la similitud por géneros y por texto (overview, keywords, tags...).
- Aplica una fórmula de recomendación interpretable:
    score = w_genre * sim_genre
          + w_text * sim_text
          + w_quality * quality
          + w_popularity * popularity
- Tolera esquemas parcialmente distintos.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import json
import math
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer


PathLike = Union[str, bytes]
DataLike = Union[pd.DataFrame, PathLike]


@dataclass
class RecommendationResult:
    movie_id: int
    title: str
    score: float
    genre_score: float
    text_score: float
    quality_score: float
    popularity_score: float
    reasons: List[str]


class ContentBasedRecommender:
    """
    Recomendador basado en contenido para películas.

    Idea:
    1) Construye features de ítems con:
       - géneros (multi-label)
       - texto libre (overview, keywords, tags...)
    2) Construye el perfil del usuario a partir de:
       - genre_vector del user_registry (20 géneros -> 5-8 géneros)
       - ratings y/o histórico visto
    3) Recomienda según:
       score = w_genre*sim_genre + w_text*sim_text + w_quality*quality + w_popularity*popularity
    """

    def __init__(
        self,
        feature_text_columns: Optional[Sequence[str]] = None,
        categorical_columns: Optional[Sequence[str]] = None,
        rating_threshold_like: float = 3.5,
        profile_strategy: str = "weighted",
        use_registry_history_for_cold_start: bool = True,
        recency_weighting: bool = True,
        genre_profile_weight: float = 0.70,
        history_profile_weight: float = 0.30,
        w_genre: float = 0.60,
        w_text: float = 0.25,
        w_quality: float = 0.10,
        w_popularity: float = 0.05,
        diversity_penalty: float = 0.15,
        min_df: int = 1,
        max_tfidf_features: int = 15000,
        stop_words: Optional[str] = "english",
        lowercase: bool = True,
        ngram_range: Tuple[int, int] = (1, 2),
        min_reduced_genres: int = 5,
        max_reduced_genres: int = 8,
        min_genre_preference_threshold: float = 10.0,
        relative_genre_threshold: float = 0.25,
    ) -> None:
        self.feature_text_columns = list(feature_text_columns or [
            "overview",
            "tagline",
            "keywords",
            "tags",
            "cast",
            "crew",
            "director",
        ])
        self.categorical_columns = list(categorical_columns or ["generos"])

        self.rating_threshold_like = rating_threshold_like
        self.profile_strategy = profile_strategy
        self.use_registry_history_for_cold_start = use_registry_history_for_cold_start
        self.recency_weighting = recency_weighting

        self.genre_profile_weight = genre_profile_weight
        self.history_profile_weight = history_profile_weight

        self.w_genre = w_genre
        self.w_text = w_text
        self.w_quality = w_quality
        self.w_popularity = w_popularity

        self.diversity_penalty = diversity_penalty

        self.min_df = min_df
        self.max_tfidf_features = max_tfidf_features
        self.stop_words = stop_words
        self.lowercase = lowercase
        self.ngram_range = ngram_range

        self.min_reduced_genres = min_reduced_genres
        self.max_reduced_genres = max_reduced_genres
        self.min_genre_preference_threshold = min_genre_preference_threshold
        self.relative_genre_threshold = relative_genre_threshold

        self.movies: Optional[pd.DataFrame] = None
        self.ratings: Optional[pd.DataFrame] = None
        self.user_registry: Dict[str, Any] = {}

        self.movie_id_col = "movieId"
        self.user_id_col = "userId"

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.mlb_map: Dict[str, MultiLabelBinarizer] = {}
        self.item_feature_matrix: Optional[sparse.csr_matrix] = None
        self.feature_names_: List[str] = []
        self.movie_id_to_idx: Dict[int, int] = {}
        self.idx_to_movie_id: Dict[int, int] = {}
        self.global_user_profiles: Dict[int, np.ndarray] = {}

        self.genre_columns_: List[str] = []
        self.genre_feature_idx_: List[int] = []
        self.text_feature_idx_: List[int] = []

        self.quality_signal_: Optional[np.ndarray] = None
        self.popularity_signal_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def fit(
        self,
        movies: DataLike,
        ratings: Optional[DataLike] = None,
        user_registry: Optional[Union[Dict[str, Any], PathLike]] = None,
    ) -> "ContentBasedRecommender":
        """Carga datos, construye características de ítems y perfiles."""
        self.movies = self._load_table(movies)
        if self.movies is None or self.movies.empty:
            raise ValueError("El DataFrame de películas no puede estar vacío.")

        self.ratings = self._load_table(ratings) if ratings is not None else None
        self.user_registry = self._load_registry(user_registry) if user_registry is not None else {}

        self.movies = self._prepare_movies(self.movies)
        if self.ratings is not None and not self.ratings.empty:
            self.ratings = self._prepare_ratings(self.ratings)

        self._build_item_features()
        self._build_global_user_profiles()
        return self

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        candidate_movie_ids: Optional[Sequence[int]] = None,
        exclude_seen: bool = True,
        return_df: bool = True,
    ) -> Union[pd.DataFrame, List[RecommendationResult]]:
        """Genera recomendaciones para un usuario."""
        self._check_is_fitted()

        profile = self.build_user_profile(user_id)
        if profile is None:
            empty = pd.DataFrame(columns=[
                "movie_id", "title", "score", "genre_score", "text_score",
                "quality_score", "popularity_score", "reasons"
            ])
            return empty if return_df else []

        genre_scores = np.zeros(len(self.movies), dtype=float)
        text_scores = np.zeros(len(self.movies), dtype=float)

        if self.genre_feature_idx_:
            genre_profile = profile[self.genre_feature_idx_].reshape(1, -1)
            genre_matrix = self.item_feature_matrix[:, self.genre_feature_idx_]
            if np.linalg.norm(genre_profile) > 0:
                genre_scores = cosine_similarity(genre_profile, genre_matrix).ravel()

        if self.text_feature_idx_:
            text_profile = profile[self.text_feature_idx_].reshape(1, -1)
            text_matrix = self.item_feature_matrix[:, self.text_feature_idx_]
            if np.linalg.norm(text_profile) > 0:
                text_scores = cosine_similarity(text_profile, text_matrix).ravel()

        quality_scores = self.quality_signal_ if self.quality_signal_ is not None else np.zeros(len(self.movies))
        popularity_scores = self.popularity_signal_ if self.popularity_signal_ is not None else np.zeros(len(self.movies))

        scores = (
            self.w_genre * genre_scores
            + self.w_text * text_scores
            + self.w_quality * quality_scores
            + self.w_popularity * popularity_scores
        )

        seen = self.get_seen_movies(user_id)
        candidate_mask = np.ones(len(self.movies), dtype=bool)

        if exclude_seen and seen:
            seen_idx = [self.movie_id_to_idx[mid] for mid in seen if mid in self.movie_id_to_idx]
            if seen_idx:
                candidate_mask[seen_idx] = False

        if candidate_movie_ids is not None:
            allowed = set(int(x) for x in candidate_movie_ids)
            candidate_mask &= self.movies[self.movie_id_col].isin(allowed).to_numpy()

        valid_indices = np.where(candidate_mask)[0]
        if len(valid_indices) == 0:
            empty = pd.DataFrame(columns=[
                "movie_id", "title", "score", "genre_score", "text_score",
                "quality_score", "popularity_score", "reasons"
            ])
            return empty if return_df else []

        ranked = sorted(valid_indices, key=lambda idx: scores[idx], reverse=True)
        selected: List[int] = []

        for idx in ranked:
            if len(selected) >= top_k:
                break

            if not selected:
                selected.append(idx)
                continue

            too_similar = False
            for prev_idx in selected:
                sim = cosine_similarity(
                    self.item_feature_matrix[idx],
                    self.item_feature_matrix[prev_idx],
                )[0, 0]
                if sim >= (1.0 - self.diversity_penalty):
                    too_similar = True
                    break

            if not too_similar:
                selected.append(idx)

        if len(selected) < top_k:
            for idx in ranked:
                if idx not in selected:
                    selected.append(idx)
                if len(selected) >= top_k:
                    break

        rows: List[RecommendationResult] = []
        for idx in selected[:top_k]:
            movie_row = self.movies.iloc[idx]
            movie_id = int(movie_row[self.movie_id_col])

            rows.append(
                RecommendationResult(
                    movie_id=movie_id,
                    title=str(movie_row.get("title", movie_id)),
                    score=float(scores[idx]),
                    genre_score=float(genre_scores[idx]),
                    text_score=float(text_scores[idx]),
                    quality_score=float(quality_scores[idx]),
                    popularity_score=float(popularity_scores[idx]),
                    reasons=self.explain_recommendation(user_id, movie_id),
                )
            )

        if return_df:
            return pd.DataFrame([r.__dict__ for r in rows])
        return rows

    def build_user_profile(self, user_id: int) -> Optional[np.ndarray]:
        """
        Construye el perfil del usuario combinando:
        1) perfil de géneros reducido (20 -> 5-8)
        2) perfil histórico (ratings / watched_movies)
        """
        self._check_is_fitted()

        if user_id in self.global_user_profiles:
            return self.global_user_profiles[user_id]

        genre_profile = self._build_user_genre_profile(user_id)
        history_profile = self._build_user_history_profile(user_id)

        if genre_profile is not None and history_profile is not None:
            profile = (
                self.genre_profile_weight * genre_profile
                + self.history_profile_weight * history_profile
            )
        elif genre_profile is not None:
            profile = genre_profile
        elif history_profile is not None:
            profile = history_profile
        else:
            if self.item_feature_matrix is not None:
                mean_profile = self.item_feature_matrix.mean(axis=0).A1
                norm = np.linalg.norm(mean_profile)
                if norm > 0:
                    profile = mean_profile / norm
                else:
                    return None
            else:
                return None

        norm = np.linalg.norm(profile)
        if norm <= 1e-12:
            return None
        return profile / norm

    def update_user_profile(
        self,
        user_id: int,
        movie_id: int,
        rating: float,
        timestamp: Optional[Union[int, float]] = None,
        refit_cache: bool = True,
    ) -> None:
        """Actualiza dinámicamente el perfil del usuario con una nueva interacción."""
        self._check_is_fitted()

        if movie_id not in self.movie_id_to_idx:
            raise ValueError(f"La película {movie_id} no existe en movies.")

        new_row = pd.DataFrame([{
            self.user_id_col: int(user_id),
            self.movie_id_col: int(movie_id),
            "rating": float(rating),
            "timestamp": timestamp,
        }])

        if self.ratings is None:
            self.ratings = new_row.copy()
        else:
            self.ratings = pd.concat([self.ratings, new_row], ignore_index=True)

        if refit_cache:
            profile = self._build_single_user_profile_from_ratings(user_id)
            registry_genre_profile = self._build_user_genre_profile(user_id)

            if profile is not None and registry_genre_profile is not None:
                profile = (
                    self.genre_profile_weight * registry_genre_profile
                    + self.history_profile_weight * profile
                )
                norm = np.linalg.norm(profile)
                if norm > 0:
                    profile = profile / norm
            elif profile is None:
                profile = registry_genre_profile

            if profile is not None:
                self.global_user_profiles[user_id] = profile

    def explain_recommendation(self, user_id: int, movie_id: int, top_n_reasons: int = 5) -> List[str]:
        """Explicación simple basada en géneros, overview y similitud con visto."""
        self._check_is_fitted()

        if movie_id not in self.movie_id_to_idx:
            return ["Película no encontrada en el catálogo."]

        reasons: List[str] = []
        seen = self.get_seen_movies(user_id)
        target_idx = self.movie_id_to_idx[movie_id]
        target_row = self.movies.iloc[target_idx]

        reduced_genres = self.get_reduced_user_genre_preferences(user_id)
        target_genres = self._safe_split_multi_value(target_row.get("generos", []))
        target_genres_norm = [self._canonicalize_genre(g) for g in target_genres]

        common_genres = [g for g in reduced_genres.keys() if g in target_genres_norm]
        if common_genres:
            reasons.append(
                f"Coincide con tus géneros principales: {', '.join(common_genres[:3])}."
            )

        target_tags = self._safe_split_multi_value(target_row.get("tags", []))
        if target_tags:
            reasons.append(
                f"Incluye etiquetas relevantes: {', '.join([str(x) for x in target_tags[:3]])}."
            )

        if "overview" in self.movies.columns and pd.notna(target_row.get("overview")):
            overview = str(target_row.get("overview")).strip()
            if overview:
                snippet = overview[:140] + ("..." if len(overview) > 140 else "")
                reasons.append(f"Su contenido encaja con tu perfil temático: {snippet}")

        similar_seen_title = None
        best_sim = -1.0
        for seen_id in seen:
            if seen_id not in self.movie_id_to_idx:
                continue
            seen_idx = self.movie_id_to_idx[seen_id]
            sim = cosine_similarity(
                self.item_feature_matrix[target_idx],
                self.item_feature_matrix[seen_idx]
            )[0, 0]
            if sim > best_sim:
                best_sim = sim
                similar_seen_title = self.movies.iloc[seen_idx].get("title")

        if similar_seen_title is not None:
            reasons.append(f"Es parecida a una película que ya viste: {similar_seen_title}.")

        quality_parts = []
        for col in ["vote_average", "imdb_rating", "tmdb_score", "popularity"]:
            if col in self.movies.columns and pd.notna(target_row.get(col)):
                quality_parts.append(f"{col}={target_row.get(col)}")
        if quality_parts:
            reasons.append("También destaca por calidad/popularidad: " + ", ".join(quality_parts[:2]) + ".")

        if not reasons:
            reasons.append("Alta similitud de contenido con tu perfil.")

        return reasons[:top_n_reasons]

    def get_seen_movies(self, user_id: int) -> set:
        """Películas vistas/puntuadas por el usuario."""
        seen = set()

        if self.ratings is not None and not self.ratings.empty:
            user_r = self.ratings[self.ratings[self.user_id_col] == user_id]
            if not user_r.empty:
                seen.update(user_r[self.movie_id_col].astype(int).tolist())

        registry_users = self.user_registry.get("users", {}) if isinstance(self.user_registry, dict) else {}
        user_entry = registry_users.get(str(user_id)) or registry_users.get(user_id)
        if user_entry:
            watched = user_entry.get("preferences", {}).get("watched_movies", [])
            seen.update(int(x) for x in watched)

        return seen

    def get_reduced_user_genre_preferences(self, user_id: int) -> Dict[str, float]:
        """Devuelve el vector reducido de géneros ya normalizado."""
        genre_vector = self._get_registry_genre_vector(user_id)
        return self._reduce_genre_vector(genre_vector)

    # ------------------------------------------------------------------
    # Preparación de datos
    # ------------------------------------------------------------------
    def _load_table(self, data: Optional[DataLike]) -> Optional[pd.DataFrame]:
        if data is None:
            return None
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, (str, bytes)):
            path = data.decode() if isinstance(data, bytes) else data
            if path.endswith(".csv"):
                return pd.read_csv(path)
            if path.endswith(".parquet"):
                return pd.read_parquet(path)
            if path.endswith(".json"):
                return pd.read_json(path)
        raise ValueError("Formato no soportado. Usa DataFrame, CSV, Parquet o JSON.")

    def _load_registry(self, registry: Optional[Union[Dict[str, Any], PathLike]]) -> Dict[str, Any]:
        if registry is None:
            return {}
        if isinstance(registry, dict):
            return registry
        with open(registry, "r", encoding="utf-8") as f:
            return json.load(f)

    def _prepare_movies(self, movies: pd.DataFrame) -> pd.DataFrame:
        movies = movies.copy()
        movies.columns = [str(c).strip() for c in movies.columns]

        if "movieId" in movies.columns:
            self.movie_id_col = "movieId"
        elif "id" in movies.columns:
            movies = movies.rename(columns={"id": "movieId"})
            self.movie_id_col = "movieId"
        else:
            raise ValueError("movies debe contener una columna 'movieId' o 'id'.")

        if "title" not in movies.columns:
            movies["title"] = movies[self.movie_id_col].astype(str)

        for col in set(self.feature_text_columns + self.categorical_columns):
            if col not in movies.columns:
                movies[col] = ""

        movies = movies.drop_duplicates(subset=[self.movie_id_col]).reset_index(drop=True)
        movies[self.movie_id_col] = movies[self.movie_id_col].astype(int)

        for col in self.categorical_columns + ["tags", "keywords", "cast", "crew", "director"]:
            if col in movies.columns:
                movies[col] = movies[col].apply(self._normalize_multi_value_field)

        if "generos" in movies.columns:
            movies["generos"] = movies["generos"].apply(
                lambda vals: [self._canonicalize_genre(v) for v in self._safe_split_multi_value(vals)]
            )

        return movies

    def _prepare_ratings(self, ratings: pd.DataFrame) -> pd.DataFrame:
        ratings = ratings.copy()
        ratings.columns = [str(c).strip() for c in ratings.columns]

        if "userId" not in ratings.columns:
            if "user_id" in ratings.columns:
                ratings = ratings.rename(columns={"user_id": "userId"})
            else:
                return pd.DataFrame()

        if "movieId" not in ratings.columns:
            if "movie_id" in ratings.columns:
                ratings = ratings.rename(columns={"movie_id": "movieId"})
            elif "id" in ratings.columns:
                ratings = ratings.rename(columns={"id": "movieId"})
            else:
                return pd.DataFrame()

        if "rating" not in ratings.columns:
            return pd.DataFrame()

        ratings[self.user_id_col] = pd.to_numeric(ratings[self.user_id_col], errors="coerce")
        ratings[self.movie_id_col] = pd.to_numeric(ratings[self.movie_id_col], errors="coerce")
        ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")

        ratings = ratings.dropna(subset=[self.user_id_col, self.movie_id_col, "rating"]).reset_index(drop=True)
        ratings[self.user_id_col] = ratings[self.user_id_col].astype(int)
        ratings[self.movie_id_col] = ratings[self.movie_id_col].astype(int)

        if "timestamp" in ratings.columns:
            ratings["timestamp"] = pd.to_numeric(ratings["timestamp"], errors="coerce")

        return ratings

    # ------------------------------------------------------------------
    # Construcción del espacio de ítems
    # ------------------------------------------------------------------
    def _build_item_features(self) -> None:
        assert self.movies is not None

        matrices = []
        feature_names: List[str] = []
        self.genre_feature_idx_ = []
        self.text_feature_idx_ = []

        for col in self.categorical_columns:
            if col not in self.movies.columns:
                continue

            values = self.movies[col].apply(self._safe_split_multi_value)
            mlb = MultiLabelBinarizer()
            encoded = mlb.fit_transform(values)
            self.mlb_map[col] = mlb

            if encoded.size > 0 and len(mlb.classes_) > 0:
                start = len(feature_names)
                matrices.append(sparse.csr_matrix(encoded.astype(float)))
                names = [f"{col}::{self._canonicalize_genre(cls) if col == 'generos' else str(cls).lower()}"
                         for cls in mlb.classes_]
                feature_names.extend(names)
                end = len(feature_names)

                if col == "generos":
                    self.genre_columns_ = names
                    self.genre_feature_idx_ = list(range(start, end))

        text_corpus = self.movies.apply(self._build_text_document, axis=1).tolist()
        if any(len(doc.strip()) > 0 for doc in text_corpus):
            self.vectorizer = TfidfVectorizer(
                min_df=self.min_df,
                max_features=self.max_tfidf_features,
                stop_words=self.stop_words,
                lowercase=self.lowercase,
                ngram_range=self.ngram_range,
            )
            tfidf = self.vectorizer.fit_transform(text_corpus)
            if tfidf.shape[1] > 0:
                start = len(feature_names)
                matrices.append(tfidf)
                feature_names.extend([
                    f"tfidf::{t}" for t in self.vectorizer.get_feature_names_out().tolist()
                ])
                end = len(feature_names)
                self.text_feature_idx_ = list(range(start, end))

        if not matrices:
            self.item_feature_matrix = sparse.identity(len(self.movies), format="csr")
            self.feature_names_ = [f"movie_{i}" for i in range(len(self.movies))]
        else:
            self.item_feature_matrix = sparse.hstack(matrices).tocsr()
            self.feature_names_ = feature_names

        self.movie_id_to_idx = {
            int(movie_id): idx for idx, movie_id in enumerate(self.movies[self.movie_id_col].tolist())
        }
        self.idx_to_movie_id = {idx: movie_id for movie_id, idx in self.movie_id_to_idx.items()}

        self.quality_signal_ = self._build_quality_signal()
        self.popularity_signal_ = self._build_popularity_signal()

    def _build_text_document(self, row: pd.Series) -> str:
        parts: List[str] = []

        for col in self.feature_text_columns:
            if col not in row.index:
                continue
            value = row.get(col)
            if isinstance(value, list):
                value = " ".join(str(x) for x in value)
            parts.append(str(value) if value is not None else "")

        for col in self.categorical_columns:
            value = row.get(col, [])
            if isinstance(value, list):
                parts.extend([str(v) for v in value])
            else:
                parts.append(str(value))

        text = " ".join(parts)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _build_quality_signal(self) -> Optional[np.ndarray]:
        try:
            assert self.movies is not None

            candidate_cols = ["vote_average", "imdb_rating", "tmdb_score", "mean_rating", "avg_rating"]
            available = [c for c in candidate_cols if c in self.movies.columns]

            if not available:
                if self.ratings is not None and not self.ratings.empty:
                    agg = self.ratings.groupby(self.movie_id_col)["rating"].mean().rename("mean_rating")
                    tmp = self.movies[[self.movie_id_col]].merge(agg, on=self.movie_id_col, how="left")
                    values = tmp["mean_rating"].fillna(tmp["mean_rating"].mean()).fillna(0.5)
                    return self._minmax(values.to_numpy(dtype=float))
                return None

            values = self.movies[available].apply(pd.to_numeric, errors="coerce")
            values = values.fillna(values.mean()).fillna(0.5)
            mean_score = values.mean(axis=1)
            return self._minmax(mean_score.to_numpy(dtype=float))
        except Exception:
            return None

    def _build_popularity_signal(self) -> Optional[np.ndarray]:
        try:
            assert self.movies is not None

            if "popularity" in self.movies.columns:
                values = pd.to_numeric(self.movies["popularity"], errors="coerce").fillna(0)
                return self._minmax(values.to_numpy(dtype=float))

            if self.ratings is not None and not self.ratings.empty:
                counts = self.ratings.groupby(self.movie_id_col).size().rename("n_ratings")
                tmp = self.movies[[self.movie_id_col]].merge(counts, on=self.movie_id_col, how="left")
                values = tmp["n_ratings"].fillna(0).to_numpy(dtype=float)
                return self._minmax(values)

            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Construcción del perfil del usuario
    # ------------------------------------------------------------------
    def _build_global_user_profiles(self) -> None:
        self.global_user_profiles = {}

        user_ids = set()
        if self.ratings is not None and not self.ratings.empty:
            user_ids.update(self.ratings[self.user_id_col].dropna().astype(int).unique().tolist())

        registry_users = self.user_registry.get("users", {}) if isinstance(self.user_registry, dict) else {}
        for uid in registry_users.keys():
            try:
                user_ids.add(int(uid))
            except Exception:
                pass

        for user_id in user_ids:
            profile = self.build_user_profile(user_id)
            if profile is not None:
                self.global_user_profiles[user_id] = profile

    def _build_user_genre_profile(self, user_id: int) -> Optional[np.ndarray]:
        """
        Crea un perfil sobre el espacio completo de features, pero rellenando
        únicamente la parte de géneros con el vector reducido y normalizado.
        """
        reduced = self.get_reduced_user_genre_preferences(user_id)
        if not reduced or self.item_feature_matrix is None:
            return None

        profile = np.zeros(self.item_feature_matrix.shape[1], dtype=float)
        feature_index = {name: i for i, name in enumerate(self.feature_names_)}

        found_any = False
        for genre, weight in reduced.items():
            key = f"generos::{self._canonicalize_genre(genre)}"
            if key in feature_index:
                profile[feature_index[key]] = float(weight)
                found_any = True

        if not found_any:
            return None

        norm = np.linalg.norm(profile)
        if norm <= 1e-12:
            return None
        return profile / norm

    def _build_user_history_profile(self, user_id: int) -> Optional[np.ndarray]:
        """
        Construye perfil a partir de ratings.
        Si no hay ratings, usa watched_movies como fallback.
        """
        profile_from_ratings = self._build_single_user_profile_from_ratings(user_id)
        if profile_from_ratings is not None:
            return profile_from_ratings

        seen = self.get_seen_movies(user_id)
        if self.use_registry_history_for_cold_start and seen:
            item_vectors = []
            weights = []

            for movie_id in seen:
                if movie_id not in self.movie_id_to_idx:
                    continue
                idx = self.movie_id_to_idx[movie_id]
                item_vectors.append(self.item_feature_matrix[idx])
                weights.append(1.0)

            if item_vectors:
                return self._weighted_average_vectors(item_vectors, weights)

        return None

    def _build_single_user_profile_from_ratings(self, user_id: int) -> Optional[np.ndarray]:
        if self.ratings is None or self.ratings.empty:
            return None

        user_ratings = self.ratings[self.ratings[self.user_id_col] == user_id].copy()
        if user_ratings.empty:
            return None

        user_ratings = user_ratings[user_ratings[self.movie_id_col].isin(self.movie_id_to_idx)]
        if user_ratings.empty:
            return None

        item_vectors = []
        weights = []
        centered = user_ratings["rating"] - user_ratings["rating"].mean()

        timestamps = None
        if self.recency_weighting and "timestamp" in user_ratings.columns:
            timestamps = pd.to_numeric(user_ratings["timestamp"], errors="coerce")

        for row_idx, (_, row) in enumerate(user_ratings.iterrows()):
            movie_id = int(row[self.movie_id_col])
            idx = self.movie_id_to_idx[movie_id]

            if self.profile_strategy == "binary":
                weight = 1.0 if float(row["rating"]) >= self.rating_threshold_like else -0.25
            elif self.profile_strategy == "liked_only":
                if float(row["rating"]) < self.rating_threshold_like:
                    continue
                weight = max(0.1, float(row["rating"]) - self.rating_threshold_like + 0.1)
            else:
                weight = float(centered.iloc[row_idx])
                if abs(weight) < 1e-6:
                    weight = float(row["rating"]) - self.rating_threshold_like

            if timestamps is not None and timestamps.notna().any() and pd.notna(row.get("timestamp")):
                rank = timestamps.rank(pct=True).iloc[row_idx]
                weight *= (0.75 + 0.5 * float(rank))

            if not np.isnan(weight) and not np.isinf(weight):
                item_vectors.append(self.item_feature_matrix[idx])
                weights.append(weight)

        if not item_vectors or not weights:
            return None

        return self._weighted_average_vectors(item_vectors, weights)

    def _weighted_average_vectors(
        self,
        vectors: Sequence[sparse.csr_matrix],
        weights: Sequence[float]
    ) -> np.ndarray:
        dense_vectors = np.vstack([v.toarray().ravel() for v in vectors])
        w = np.array(weights, dtype=float)
        w = np.nan_to_num(w, nan=0.0)

        if np.allclose(w, 0):
            w = np.ones_like(w)

        if np.any(w < 0):
            shifted = w - np.min(w) + 1e-6
            if np.allclose(shifted, 0):
                shifted = np.ones_like(shifted)
            w = shifted

        profile = np.average(dense_vectors, axis=0, weights=w)
        norm = np.linalg.norm(profile)

        if norm <= 1e-12:
            ones = np.ones_like(profile, dtype=float)
            return ones / np.linalg.norm(ones)

        return profile / norm

    # ------------------------------------------------------------------
    # Reducción del vector de 20 géneros a 5-8 géneros
    # ------------------------------------------------------------------
    def _get_registry_favorite_genres(self, user_id: int) -> List[str]:
        registry_users = self.user_registry.get("users", {}) if isinstance(self.user_registry, dict) else {}
        user_entry = registry_users.get(str(user_id)) or registry_users.get(user_id)
        if not user_entry:
            return []
        genres = user_entry.get("preferences", {}).get("favorite_genres", [])
        return [self._canonicalize_genre(str(g).strip()) for g in genres if str(g).strip()]

    def _get_registry_genre_vector(self, user_id: int) -> Dict[str, float]:
        registry_users = self.user_registry.get("users", {}) if isinstance(self.user_registry, dict) else {}
        user_entry = registry_users.get(str(user_id)) or registry_users.get(user_id)
        if not user_entry:
            return {}

        genre_vector = user_entry.get("preferences", {}).get("genre_vector", {})
        if not isinstance(genre_vector, dict):
            return {}

        cleaned: Dict[str, float] = {}
        for k, v in genre_vector.items():
            try:
                g = self._canonicalize_genre(str(k).strip())
                val = float(v)
                cleaned[g] = val
            except Exception:
                continue
        return cleaned

    def _reduce_genre_vector(self, genre_vector: Dict[str, float]) -> Dict[str, float]:
        """
        Reduce el vector original de 20 géneros a 5-8 géneros:
        1) elimina géneros con 0 o muy bajos
        2) aplica umbral absoluto/relativo
        3) si quedan pocos, completa hasta min_reduced_genres con top global
        4) si quedan muchos, recorta a max_reduced_genres
        5) normaliza para que sumen 1
        """
        if not genre_vector:
            return {}

        cleaned = {
            self._canonicalize_genre(g): float(v)
            for g, v in genre_vector.items()
            if pd.notna(v) and float(v) > 0
        }
        if not cleaned:
            return {}

        vmax = max(cleaned.values())
        threshold = max(self.min_genre_preference_threshold, self.relative_genre_threshold * vmax)

        filtered = {g: v for g, v in cleaned.items() if v >= threshold}
        sorted_all = sorted(cleaned.items(), key=lambda x: x[1], reverse=True)

        if len(filtered) < self.min_reduced_genres:
            filtered = dict(sorted_all[:self.min_reduced_genres])
        else:
            filtered = dict(sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:self.max_reduced_genres])

        total = sum(filtered.values())
        if total <= 0:
            return {}

        return {g: v / total for g, v in filtered.items()}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_multi_value_field(self, value: Any) -> List[str]:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return []
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        if isinstance(value, (set, tuple)):
            return [str(x).strip() for x in value if str(x).strip()]
        if isinstance(value, dict):
            return [str(k).strip() for k, v in value.items() if v]

        text = str(value).strip()
        if not text:
            return []

        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]

        separators = ["|", ",", ";"]
        regex = "|".join(re.escape(sep) for sep in separators)
        parts = [p.strip() for p in re.split(regex, text) if p.strip()]
        return parts if parts else [text]

    def _safe_split_multi_value(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(x) for x in value if str(x).strip()]
        return self._normalize_multi_value_field(value)

    def _canonicalize_genre(self, genre: str) -> str:
        """
        Normaliza nombres de géneros para evitar problemas de mayúsculas,
        tildes rotas y pequeñas inconsistencias.
        """
        if genre is None:
            return ""

        g = str(genre).strip().lower()

        replacements = {
            "animaci¢n": "animacion",
            "animación": "animacion",
            "fantas¡a": "fantasia",
            "fantasía": "fantasia",
            "historica": "historica",
            "histórica": "historica",
            "ciencia ficcion": "ciencia ficcion",
            "ciencia ficción": "ciencia ficcion",
            "pelicula de tv": "pelicula de tv",
            "foreign": "extranjera",
            "thriller": "suspense",
            "war": "belica",
            "music": "musical",
            "tv movie": "pelicula de tv",
            "mystery": "misterio",
            "science fiction": "ciencia ficcion",
            "family": "familia",
        }

        g = replacements.get(g, g)
        g = unicodedata.normalize("NFKD", g).encode("ascii", "ignore").decode("utf-8")
        g = re.sub(r"\s+", " ", g).strip()
        return g

    def _minmax(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return values
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return np.zeros_like(values)
        if vmax <= vmin:
            return np.full_like(values, 0.5, dtype=float)
        return (values - vmin) / (vmax - vmin)

    def _check_is_fitted(self) -> None:
        if self.movies is None or self.item_feature_matrix is None:
            raise ValueError("El recomendador no está entrenado. Llama antes a fit().")


if __name__ == "__main__":
    recommender = ContentBasedRecommender(
        categorical_columns=["generos"],
        feature_text_columns=["overview", "keywords", "tags"],
        profile_strategy="weighted",
        min_reduced_genres=5,
        max_reduced_genres=8,
        w_genre=0.60,
        w_text=0.25,
        w_quality=0.10,
        w_popularity=0.05,
    )

    # Ejemplo:
    # recommender.fit(
    #     movies="movies.csv",
    #     ratings="ratings.csv",
    #     user_registry="user_registry.json",
    # )
    # print(recommender.recommend(user_id=1, top_k=10))

    print("Módulo listo. Importa ContentBasedRecommender desde este archivo.")
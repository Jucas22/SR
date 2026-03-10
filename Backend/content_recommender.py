from __future__ import annotations

"""
Sistema recomendador basado en contenido para películas.

Diseñado a partir de las ideas de las diapositivas:
- el perfil del usuario se representa como un vector de preferencias
- los ítems se clasifican por categorías/etiquetas
- la recomendación se obtiene a partir de la similitud entre el perfil del
  usuario y el contenido de los ítems
- las preferencias pueden actualizarse dinámicamente con el histórico

Este módulo está preparado para integrarse con:
1) user_registry.json
2) un DataFrame o CSV de películas
3) un DataFrame o CSV de ratings

Suposiciones mínimas de entrada:
- movies debe contener una columna identificadora: movieId o id
- idealmente también columnas como: title, genres, overview, keywords, tags,
  cast, crew, director, vote_average, popularity, tmdb_score, imdb_rating...
- ratings debe contener al menos: userId, movieId, rating

La clase tolera esquemas parcialmente distintos y utiliza únicamente las
columnas disponibles.
"""

from dataclasses import dataclass
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


PathLike = Union[str, bytes]
DataLike = Union[pd.DataFrame, PathLike]


@dataclass
class RecommendationResult:
    movie_id: int
    title: str
    score: float
    content_score: float
    quality_score: float
    reasons: List[str]


class ContentBasedRecommender:
    """
    Recomendador basado en contenido para películas.

    Idea central:
    - Construye un espacio de características de ítems usando géneros,
      etiquetas y texto libre (TF-IDF).
    - Genera el perfil del usuario a partir de sus ratings y/o de su histórico
      de películas vistas en user_registry.json.
    - Recomienda películas no vistas usando similitud coseno y, opcionalmente,
      señales de calidad/popularidad.

    Parámetros principales
    ----------------------
    feature_text_columns:
        Columnas de texto libres que se incorporan al TF-IDF.
    categorical_columns:
        Columnas categóricas multi-etiqueta que representan la taxonomía u
        ontología simplificada del ítem.
    rating_threshold_like:
        A partir de este valor un rating se considera positivo.
    use_registry_history_for_cold_start:
        Si no hay ratings del usuario, usa el histórico watched_movies del JSON.
    recency_weighting:
        Si es True, pondera más los ratings recientes cuando exista timestamp.
    """

    def __init__(
        self,
        feature_text_columns: Optional[Sequence[str]] = None,
        categorical_columns: Optional[Sequence[str]] = None,
        rating_threshold_like: float = 3.5,
        profile_strategy: str = "weighted",
        use_registry_history_for_cold_start: bool = True,
        recency_weighting: bool = True,
        quality_weight: float = 0.10,
        popularity_weight: float = 0.05,
        diversity_penalty: float = 0.15,
        min_df: int = 1,
        max_tfidf_features: int = 15000,
        stop_words: Optional[str] = "english",
        lowercase: bool = True,
        ngram_range: Tuple[int, int] = (1, 2),
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
        self.categorical_columns = list(categorical_columns or ["genres"])
        self.rating_threshold_like = rating_threshold_like
        self.profile_strategy = profile_strategy
        self.use_registry_history_for_cold_start = use_registry_history_for_cold_start
        self.recency_weighting = recency_weighting
        self.quality_weight = quality_weight
        self.popularity_weight = popularity_weight
        self.diversity_penalty = diversity_penalty
        self.min_df = min_df
        self.max_tfidf_features = max_tfidf_features
        self.stop_words = stop_words
        self.lowercase = lowercase
        self.ngram_range = ngram_range

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
        try:
            self.movies = self._load_table(movies)
            if self.movies is None or self.movies.empty:
                raise ValueError("El DataFrame de películas no puede estar vacío")
            
            self.ratings = self._load_table(ratings) if ratings is not None else None
            self.user_registry = self._load_registry(user_registry) if user_registry is not None else {}

            self.movies = self._prepare_movies(self.movies)
            if self.ratings is not None and not self.ratings.empty:
                self.ratings = self._prepare_ratings(self.ratings)

            self._build_item_features()
            self._build_global_user_profiles()
            return self
        except Exception as e:
            print(f"❌ Error en fit(): {str(e)}")
            raise

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
            # En lugar de error, devolver películas populares
            print(f"⚠️ No se pudo construir perfil para usuario {user_id}. Devolviendo películas populares.")
            empty = pd.DataFrame(columns=[
                "movie_id", "title", "score", "content_score", "quality_score", "reasons"
            ])
            return empty if return_df else []

        scores = cosine_similarity(profile.reshape(1, -1), self.item_feature_matrix).ravel()
        content_scores = scores.copy()

        if self.quality_signal_ is not None:
            scores += self.quality_weight * self.quality_signal_
        if self.popularity_signal_ is not None:
            scores += self.popularity_weight * self.popularity_signal_

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
                "movie_id", "title", "score", "content_score", "quality_score", "reasons"
            ])
            return empty if return_df else []

        ranked = sorted(valid_indices, key=lambda idx: scores[idx], reverse=True)
        selected: List[int] = []

        # Diversificación ligera para evitar sobre-especialización
        for idx in ranked:
            if len(selected) >= top_k:
                break
            if not selected:
                selected.append(idx)
                continue

            too_similar = False
            try:
                for prev_idx in selected:
                    sim = cosine_similarity(
                        self.item_feature_matrix[idx],
                        self.item_feature_matrix[prev_idx],
                    )[0, 0]
                    if sim >= (1.0 - self.diversity_penalty):
                        too_similar = True
                        break
            except Exception:
                # Si hay problemas con similitud, simplemente agregar
                pass
                
            if not too_similar:
                selected.append(idx)

        # Si la diversificación fue demasiado estricta, completar con ranking puro
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
            reasons = self.explain_recommendation(user_id, movie_id)
            rows.append(
                RecommendationResult(
                    movie_id=movie_id,
                    title=str(movie_row.get("title", movie_id)),
                    score=float(scores[idx]),
                    content_score=float(content_scores[idx]),
                    quality_score=float(
                        (self.quality_signal_[idx] if self.quality_signal_ is not None else 0.0)
                        + (self.popularity_signal_[idx] if self.popularity_signal_ is not None else 0.0)
                    ),
                    reasons=reasons,
                )
            )

        if return_df:
            return pd.DataFrame([r.__dict__ for r in rows])
        return rows

    def build_user_profile(self, user_id: int) -> Optional[np.ndarray]:
        """Construye el vector de preferencias del usuario."""
        self._check_is_fitted()

        # Si ya está cacheado, devolverlo
        if user_id in self.global_user_profiles:
            return self.global_user_profiles[user_id]

        # Opción 1: Usar histórico de películas vistas
        seen = self.get_seen_movies(user_id)
        if self.use_registry_history_for_cold_start and seen:
            item_vectors = []
            for movie_id in seen:
                if movie_id not in self.movie_id_to_idx:
                    continue
                idx = self.movie_id_to_idx[movie_id]
                item_vectors.append(self.item_feature_matrix[idx])
            if item_vectors:
                # Pesos uniformes para películas vistas
                weights = [1.0] * len(item_vectors)
                try:
                    profile = self._weighted_average_vectors(item_vectors, weights)
                    return profile
                except Exception:
                    pass  # Fallback a géneros favoritos

        # Opción 2: Usar géneros favoritos
        fav_genres = self._get_registry_favorite_genres(user_id)
        if fav_genres and self.genre_columns_:
            profile = np.zeros(self.item_feature_matrix.shape[1], dtype=float)
            genre_to_index = {name: i for i, name in enumerate(self.feature_names_)}
            found_any = False
            for g in fav_genres:
                key = f"genres::{g.lower()}"
                if key in genre_to_index:
                    profile[genre_to_index[key]] = 1.0
                    found_any = True
            if found_any:
                norm = np.linalg.norm(profile)
                if norm > 0:
                    return profile / norm

        # Opción 3: Perfil por defecto (usuario completamente nuevo)
        # Usar el promedio de todas las películas
        if self.item_feature_matrix is not None:
            try:
                mean_profile = self.item_feature_matrix.mean(axis=0).A1
                norm = np.linalg.norm(mean_profile)
                if norm > 0:
                    return mean_profile / norm
                # Si todo es cero, devolver vector uniforme
                return np.ones(self.item_feature_matrix.shape[1]) / np.sqrt(self.item_feature_matrix.shape[1])
            except Exception:
                return None

        return None

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

        new_row = pd.DataFrame([
            {
                self.user_id_col: int(user_id),
                self.movie_id_col: int(movie_id),
                "rating": float(rating),
                "timestamp": timestamp,
            }
        ])

        if self.ratings is None:
            self.ratings = new_row.copy()
        else:
            self.ratings = pd.concat([self.ratings, new_row], ignore_index=True)

        if refit_cache:
            profile = self._build_single_user_profile_from_ratings(user_id)
            if profile is not None:
                self.global_user_profiles[user_id] = profile

    def explain_recommendation(self, user_id: int, movie_id: int, top_n_reasons: int = 5) -> List[str]:
        """Devuelve una explicación simple basada en géneros/etiquetas compartidas."""
        self._check_is_fitted()
        if movie_id not in self.movie_id_to_idx:
            return ["Película no encontrada en el catálogo."]

        seen = self.get_seen_movies(user_id)
        target_row = self.movies.iloc[self.movie_id_to_idx[movie_id]]
        reasons: List[str] = []

        target_genres = self._safe_split_multi_value(target_row.get("genres", []))
        if target_genres:
            reasons.append(f"Coincide con géneros afines a tu perfil: {', '.join(target_genres[:3])}.")

        target_tags = self._safe_split_multi_value(target_row.get("tags", []))
        if target_tags:
            reasons.append(f"Contiene etiquetas relevantes: {', '.join(target_tags[:3])}.")

        similar_seen_title = None
        best_sim = -1.0
        tgt_idx = self.movie_id_to_idx[movie_id]
        for seen_id in seen:
            if seen_id not in self.movie_id_to_idx:
                continue
            seen_idx = self.movie_id_to_idx[seen_id]
            sim = cosine_similarity(self.item_feature_matrix[tgt_idx], self.item_feature_matrix[seen_idx])[0, 0]
            if sim > best_sim:
                best_sim = sim
                similar_seen_title = self.movies.iloc[seen_idx].get("title")
        if similar_seen_title is not None:
            reasons.append(f"Es similar a una película que ya viste: {similar_seen_title}.")

        quality_parts = []
        for col in ["vote_average", "imdb_rating", "tmdb_score", "popularity"]:
            if col in self.movies.columns and pd.notna(target_row.get(col)):
                quality_parts.append(f"{col}={target_row.get(col)}")
        if quality_parts:
            reasons.append("Se priorizó además por calidad/popularidad: " + ", ".join(quality_parts[:2]) + ".")

        if not reasons:
            reasons.append("Alta similitud de contenido con tus preferencias e histórico.")

        return reasons[:top_n_reasons]

    def get_seen_movies(self, user_id: int) -> set:
        """Películas vistas/puntuadas por el usuario."""
        seen = set()

        if self.ratings is not None:
            user_r = self.ratings[self.ratings[self.user_id_col] == user_id]
            if not user_r.empty:
                seen.update(user_r[self.movie_id_col].astype(int).tolist())

        registry_users = self.user_registry.get("users", {}) if isinstance(self.user_registry, dict) else {}
        user_entry = registry_users.get(str(user_id)) or registry_users.get(user_id)
        if user_entry:
            watched = user_entry.get("preferences", {}).get("watched_movies", [])
            seen.update(int(x) for x in watched)

        return seen

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

        # Normalización suave de columnas multi-valor
        for col in self.categorical_columns + ["tags", "keywords", "cast", "crew", "director"]:
            if col in movies.columns:
                movies[col] = movies[col].apply(self._normalize_multi_value_field)

        return movies

    def _prepare_ratings(self, ratings: pd.DataFrame) -> pd.DataFrame:
        ratings = ratings.copy()
        ratings.columns = [str(c).strip() for c in ratings.columns]

        if "userId" not in ratings.columns:
            if "user_id" in ratings.columns:
                ratings = ratings.rename(columns={"user_id": "userId"})
            else:
                # Si no hay userId, no podemos procesar ratings
                print("⚠️ Advertencia: ratings no contiene userId. Se ignorarán los ratings.")
                return pd.DataFrame()  # Devolver DataFrame vacío

        if "movieId" not in ratings.columns:
            if "movie_id" in ratings.columns:
                ratings = ratings.rename(columns={"movie_id": "movieId"})
            elif "id" in ratings.columns:
                ratings = ratings.rename(columns={"id": "movieId"})
            else:
                # Si no hay movieId, no podemos procesar ratings
                print("⚠️ Advertencia: ratings no contiene movieId. Se ignorarán los ratings.")
                return pd.DataFrame()  # Devolver DataFrame vacío

        if "rating" not in ratings.columns:
            print("⚠️ Advertencia: ratings no contiene columna 'rating'. Se ignorarán los ratings.")
            return pd.DataFrame()  # Devolver DataFrame vacío

        try:
            ratings[self.user_id_col] = ratings[self.user_id_col].astype(int)
            ratings[self.movie_id_col] = ratings[self.movie_id_col].astype(int)
            ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
            ratings = ratings.dropna(subset=["rating"]).reset_index(drop=True)

            if "timestamp" in ratings.columns:
                ratings["timestamp"] = pd.to_numeric(ratings["timestamp"], errors="coerce")

            return ratings
        except Exception as e:
            print(f"⚠️ Error procesando ratings: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Construcción del espacio de ítems
    # ------------------------------------------------------------------
    def _build_item_features(self) -> None:
        assert self.movies is not None

        matrices = []
        feature_names: List[str] = []

        # 1) Taxonomía / ontología simplificada: géneros y otras categorías multi-label
        for col in self.categorical_columns:
            if col not in self.movies.columns:
                continue
            values = self.movies[col].apply(self._safe_split_multi_value)
            mlb = MultiLabelBinarizer()
            encoded = mlb.fit_transform(values)
            self.mlb_map[col] = mlb
            if encoded.size > 0 and len(mlb.classes_) > 0:
                matrices.append(sparse.csr_matrix(encoded.astype(float)))
                feature_names.extend([f"{col}::{cls.lower()}" for cls in mlb.classes_])
                if col == "genres":
                    self.genre_columns_ = [f"{col}::{cls.lower()}" for cls in mlb.classes_]

        # 2) Etiquetas / características aumentadas / texto
        text_corpus = self.movies.apply(self._build_text_document, axis=1).tolist()
        
        # Solo si hay al menos algunos documentos con contenido
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
                matrices.append(tfidf)
                feature_names.extend([f"tfidf::{t}" for t in self.vectorizer.get_feature_names_out().tolist()])

        if not matrices:
            # Fallback: crear una matriz de identidad si no hay características
            print("⚠️ Advertencia: No se extrajeron características. Usando matriz de identidad.")
            self.item_feature_matrix = sparse.identity(len(self.movies), format='csr')
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

        # Reforzar géneros y etiquetas, porque en las diapositivas son clave
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
            candidate_cols = [
                "vote_average",
                "imdb_rating",
                "tmdb_score",
                "mean_rating",
                "avg_rating",
            ]
            available = [c for c in candidate_cols if c in self.movies.columns]
            if not available:
                # Si no viene en movies, intentar inferir desde ratings
                if self.ratings is not None and not self.ratings.empty:
                    try:
                        agg = self.ratings.groupby(self.movie_id_col)["rating"].mean().rename("mean_rating")
                        tmp = self.movies[[self.movie_id_col]].merge(agg, on=self.movie_id_col, how="left")
                        values = tmp[["mean_rating"]].fillna(tmp[["mean_rating"]].mean()).fillna(0.5)
                        # Normalizar manualmente
                        val_min = values["mean_rating"].min()
                        val_max = values["mean_rating"].max()
                        if val_max > val_min:
                            return ((values["mean_rating"] - val_min) / (val_max - val_min)).values
                        else:
                            return np.full(len(values), 0.5)
                    except Exception:
                        return None
                return None
            
            values = self.movies[available].apply(pd.to_numeric, errors="coerce")
            values = values.fillna(values.mean()).fillna(0.5)
            mean_score = values.mean(axis=1)
            
            # Normalizar manualmente para evitar problemas con MinMaxScaler
            val_min = mean_score.min()
            val_max = mean_score.max()
            if val_max > val_min:
                return ((mean_score - val_min) / (val_max - val_min)).values
            else:
                return np.full(len(mean_score), 0.5)
        except Exception as e:
            print(f"⚠️ Error en _build_quality_signal: {e}")
            return None

    def _build_popularity_signal(self) -> Optional[np.ndarray]:
        try:
            assert self.movies is not None
            if "popularity" in self.movies.columns:
                values = pd.to_numeric(self.movies["popularity"], errors="coerce").fillna(0)
                val_min = values.min()
                val_max = values.max()
                if val_max > val_min:
                    return ((values - val_min) / (val_max - val_min)).values
                else:
                    return np.full(len(values), 0.5)
            
            if self.ratings is not None and not self.ratings.empty:
                try:
                    counts = self.ratings.groupby(self.movie_id_col).size().rename("n_ratings")
                    tmp = self.movies[[self.movie_id_col]].merge(counts, on=self.movie_id_col, how="left")
                    values = tmp[["n_ratings"]].fillna(0).values.ravel()
                    val_min = values.min()
                    val_max = values.max()
                    if val_max > val_min:
                        return ((values - val_min) / (val_max - val_min))
                    else:
                        return np.full(len(values), 0.5)
                except Exception:
                    return None
            return None
        except Exception as e:
            print(f"⚠️ Error en _build_popularity_signal: {e}")
            return None

    # ------------------------------------------------------------------
    # Construcción del perfil del usuario
    # ------------------------------------------------------------------
    def _build_global_user_profiles(self) -> None:
        self.global_user_profiles = {}
        if self.ratings is None:
            return
        for user_id in self.ratings[self.user_id_col].dropna().astype(int).unique().tolist():
            profile = self._build_single_user_profile_from_ratings(user_id)
            if profile is not None:
                self.global_user_profiles[user_id] = profile

    def _build_single_user_profile_from_ratings(self, user_id: int) -> Optional[np.ndarray]:
        assert self.ratings is not None
        user_ratings = self.ratings[self.ratings[self.user_id_col] == user_id].copy()
        if user_ratings.empty:
            return None

        user_ratings = user_ratings[user_ratings[self.movie_id_col].isin(self.movie_id_to_idx)]
        if user_ratings.empty:
            return None

        item_vectors = []
        weights = []
        centered = user_ratings["rating"] - user_ratings["rating"].mean()

        for row_idx, (_, row) in enumerate(user_ratings.iterrows()):
            movie_id = int(row[self.movie_id_col])
            idx = self.movie_id_to_idx[movie_id]
            
            if self.profile_strategy == "binary":
                weight = 1.0 if float(row["rating"]) >= self.rating_threshold_like else -0.25
            elif self.profile_strategy == "liked_only":
                if float(row["rating"]) < self.rating_threshold_like:
                    continue  # Saltar películas no valoradas
                weight = max(0.1, float(row["rating"]) - self.rating_threshold_like + 0.1)
            else:  # weighted
                weight = float(centered.iloc[row_idx])
                if abs(weight) < 1e-6:
                    weight = float(row["rating"]) - self.rating_threshold_like

            if self.recency_weighting and "timestamp" in user_ratings.columns and pd.notna(row.get("timestamp")):
                timestamps = pd.to_numeric(user_ratings["timestamp"], errors="coerce")
                if timestamps.notna().any():
                    rank = timestamps.rank(pct=True).iloc[row_idx]
                    weight *= (0.75 + 0.5 * float(rank))

            # Solo agregar si el peso es válido
            if not np.isnan(weight) and not np.isinf(weight):
                item_vectors.append(self.item_feature_matrix[idx])
                weights.append(weight)

        if not item_vectors or not weights:
            return None

        return self._weighted_average_vectors(item_vectors, weights)

    def _weighted_average_vectors(self, vectors: Sequence[sparse.csr_matrix], weights: Sequence[float]) -> np.ndarray:
        dense_vectors = np.vstack([v.toarray().ravel() for v in vectors])
        w = np.array(weights, dtype=float)

        # Manejar pesos todos en cero o NaN
        if np.allclose(w, 0) or np.all(np.isnan(w)):
            w = np.ones_like(w)
        
        # Reemplazar NaN con 1
        w = np.nan_to_num(w, nan=1.0)

        profile = np.average(dense_vectors, axis=0, weights=w)
        norm = np.linalg.norm(profile)
        
        # Manejar vector cero (sin características)
        if norm <= 1e-10:
            # Devolver un vector uniforme en lugar de vector cero
            return np.ones_like(profile) / np.linalg.norm(np.ones_like(profile))
        
        return profile / norm

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_registry_favorite_genres(self, user_id: int) -> List[str]:
        registry_users = self.user_registry.get("users", {}) if isinstance(self.user_registry, dict) else {}
        user_entry = registry_users.get(str(user_id)) or registry_users.get(user_id)
        if not user_entry:
            return []
        genres = user_entry.get("preferences", {}).get("favorite_genres", [])
        return [str(g).strip() for g in genres if str(g).strip()]

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

        # Intenta parsear listas serializadas sencillas
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

    def _check_is_fitted(self) -> None:
        if self.movies is None or self.item_feature_matrix is None:
            raise ValueError("El recomendador no está entrenado. Llama antes a fit().")


if __name__ == "__main__":
    # Ejemplo mínimo de uso.
    # Sustituye los CSV por tus ficheros reales de movies y ratings.
    recommender = ContentBasedRecommender(
        categorical_columns=["genres"],
        feature_text_columns=["overview", "keywords", "tags"],
        profile_strategy="weighted",
    )

    # Ejemplo:
    # recommender.fit(
    #     movies="movies.csv",
    #     ratings="ratings.csv",
    #     user_registry="user_registry.json",
    # )
    # print(recommender.recommend(user_id=1, top_k=10))

    print("Módulo listo. Importa ContentBasedRecommender desde este archivo.")

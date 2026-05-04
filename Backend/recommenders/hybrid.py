from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class HybridRecommendationResult:
    movie_id: int
    title: str
    score: float
    alpha: float
    beta: float
    content_score: float
    collaborative_score: float
    content_contribution: float
    collaborative_contribution: float
    content_signal: float
    collaborative_signal: float
    appears_in_content: bool
    appears_in_collaborative: bool
    agreement_count: int
    reasons: List[str]


class HybridRecommender:
    """
    Recomendador híbrido que combina contenido y colaborativo.

    La mezcla sigue tres ideas principales:
    1. Obtener dos listas base: contenido y colaborativo.
    2. Normalizar las puntuaciones a una escala común [0, 1].
    3. Combinar los ratios con pesos dinámicos alpha/beta calculados
       para cada usuario y cada petición.
    """

    DEFAULT_CANDIDATE_POOL = 100
    MAX_NEIGHBORS_REFERENCE = 40
    CONTENT_RATING_TARGET = 15
    CONTENT_WATCH_TARGET = 20

    def __init__(self, content_model, collaborative_model):
        self.content_model = content_model
        self.collaborative_model = collaborative_model

    def recommend(
        self,
        user_id: Union[int, str],
        top_k: int = 10,
        candidate_pool_size: Optional[int] = None,
        return_details: bool = False,
    ) -> Union[List[Tuple[int, float]], List[Dict[str, Any]]]:
        candidate_pool_size = max(
            int(candidate_pool_size or 0),
            int(top_k) * 5,
            self.DEFAULT_CANDIDATE_POOL,
        )

        content_candidates = self._get_content_candidates(user_id, candidate_pool_size)
        collaborative_candidates = self._get_collaborative_candidates(
            user_id,
            candidate_pool_size,
        )

        content_signal, content_diagnostics = self._build_content_signal(
            user_id=user_id,
            candidates=content_candidates,
        )
        collaborative_signal, collaborative_diagnostics = (
            self._build_collaborative_signal(
                user_id=user_id,
                candidates=collaborative_candidates,
            )
        )

        alpha, beta = self._normalize_blend_weights(
            content_signal=content_signal,
            collaborative_signal=collaborative_signal,
        )

        if alpha <= 0 and beta <= 0:
            return []

        content_by_movie = {
            int(row["movie_id"]): row.to_dict()
            for _, row in content_candidates.iterrows()
        }
        collaborative_by_movie = {
            int(row["movie_id"]): row
            for row in collaborative_candidates
        }

        all_items = sorted(set(content_by_movie.keys()).union(collaborative_by_movie.keys()))
        ranked_items: List[HybridRecommendationResult] = []

        for movie_id in all_items:
            content_row = content_by_movie.get(movie_id)
            collaborative_row = collaborative_by_movie.get(movie_id)

            content_score = self._normalize_content_ratio(content_row)
            collaborative_score = self._normalize_collaborative_ratio(collaborative_row)

            content_contribution = alpha * content_score
            collaborative_contribution = beta * collaborative_score
            hybrid_score = content_contribution + collaborative_contribution

            if hybrid_score <= 0:
                continue

            ranked_items.append(
                HybridRecommendationResult(
                    movie_id=int(movie_id),
                    title=self._resolve_title(content_row, collaborative_row, movie_id),
                    score=float(hybrid_score),
                    alpha=float(alpha),
                    beta=float(beta),
                    content_score=float(content_score),
                    collaborative_score=float(collaborative_score),
                    content_contribution=float(content_contribution),
                    collaborative_contribution=float(collaborative_contribution),
                    content_signal=float(content_signal),
                    collaborative_signal=float(collaborative_signal),
                    appears_in_content=content_row is not None,
                    appears_in_collaborative=collaborative_row is not None,
                    agreement_count=int(content_row is not None) + int(collaborative_row is not None),
                    reasons=self._build_hybrid_reasons(
                        movie_id=movie_id,
                        alpha=alpha,
                        beta=beta,
                        content_row=content_row,
                        collaborative_row=collaborative_row,
                        content_diagnostics=content_diagnostics,
                        collaborative_diagnostics=collaborative_diagnostics,
                    ),
                )
            )

        ranked_items.sort(
            key=lambda item: (
                item.score,
                item.agreement_count,
                item.collaborative_score,
                item.content_score,
            ),
            reverse=True,
        )

        top_items = ranked_items[:top_k]
        if return_details:
            return [asdict(item) for item in top_items]

        return [(item.movie_id, item.score) for item in top_items]

    def explain_recommendation(
        self,
        user_id: Union[int, str],
        movie_id: Union[int, str],
        top_n_reasons: int = 4,
    ) -> List[str]:
        """Devuelve la explicación híbrida para una película concreta."""
        movie_id = int(movie_id)
        recommendations = self.recommend(
            user_id=user_id,
            top_k=max(20, int(top_n_reasons) * 5),
            candidate_pool_size=self.DEFAULT_CANDIDATE_POOL,
            return_details=True,
        )

        for recommendation in recommendations:
            if int(recommendation["movie_id"]) == movie_id:
                return list(recommendation.get("reasons", []))[:top_n_reasons]

        return ["No se encontro una explicacion hibrida para esta pelicula."]

    def _get_content_candidates(
        self,
        user_id: Union[int, str],
        candidate_pool_size: int,
    ) -> pd.DataFrame:
        recommendations = self.content_model.recommend(
            user_id=int(user_id),
            top_k=int(candidate_pool_size),
            return_df=True,
        )
        if recommendations is None or recommendations.empty:
            return pd.DataFrame()
        return recommendations.copy()

    def _get_collaborative_candidates(
        self,
        user_id: Union[int, str],
        candidate_pool_size: int,
    ) -> List[Dict[str, Any]]:
        recommendations = self.collaborative_model.recommend(
            user_id=str(user_id),
            top_n=int(candidate_pool_size),
            return_details=True,
        )
        return recommendations or []

    def _build_content_signal(
        self,
        user_id: Union[int, str],
        candidates: pd.DataFrame,
    ) -> Tuple[float, Dict[str, float]]:
        user_id_int = int(user_id)
        reduced_genres = self.content_model.get_reduced_user_genre_preferences(user_id_int)

        favorite_genres = []
        if hasattr(self.content_model, "_get_registry_favorite_genres"):
            favorite_genres = self.content_model._get_registry_favorite_genres(user_id_int)

        genre_count = max(len(reduced_genres), len(favorite_genres))
        min_reduced_genres = max(
            1,
            int(getattr(self.content_model, "min_reduced_genres", 5)),
        )
        genre_signal = min(1.0, genre_count / float(min_reduced_genres))

        ratings_count = 0
        ratings_df = getattr(self.content_model, "ratings", None)
        user_id_col = getattr(self.content_model, "user_id_col", "userId")
        if ratings_df is not None and not ratings_df.empty and user_id_col in ratings_df.columns:
            ratings_count = int((ratings_df[user_id_col] == user_id_int).sum())

        ratings_signal = min(1.0, ratings_count / float(self.CONTENT_RATING_TARGET))

        watched_count = 0
        if hasattr(self.content_model, "get_seen_movies"):
            watched_count = len(self.content_model.get_seen_movies(user_id_int))
        watched_signal = min(1.0, watched_count / float(self.CONTENT_WATCH_TARGET))

        candidate_signal = 0.0
        if candidates is not None and not candidates.empty:
            candidate_signal = min(1.0, len(candidates) / float(self.DEFAULT_CANDIDATE_POOL))

        has_content_information = any(
            metric > 0
            for metric in (genre_signal, ratings_signal, watched_signal)
        )
        if not has_content_information or candidates is None or candidates.empty:
            return 0.0, {
                "genre_signal": float(genre_signal),
                "ratings_signal": float(ratings_signal),
                "watched_signal": float(watched_signal),
                "candidate_signal": float(candidate_signal),
                "ratings_count": float(ratings_count),
                "watched_count": float(watched_count),
                "genre_count": float(genre_count),
            }

        history_signal = ratings_signal if ratings_count > 0 else (0.6 * watched_signal)
        content_signal = (
            0.45 * genre_signal
            + 0.35 * history_signal
            + 0.20 * candidate_signal
        )

        return float(min(1.0, content_signal)), {
            "genre_signal": float(genre_signal),
            "ratings_signal": float(ratings_signal),
            "watched_signal": float(watched_signal),
            "candidate_signal": float(candidate_signal),
            "ratings_count": float(ratings_count),
            "watched_count": float(watched_count),
            "genre_count": float(genre_count),
        }

    def _build_collaborative_signal(
        self,
        user_id: Union[int, str],
        candidates: Sequence[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, float]]:
        neighbors = self._get_neighbors(user_id)
        if not neighbors or not candidates:
            return 0.0, {
                "neighbor_count": float(len(neighbors)),
                "neighbor_count_signal": 0.0 if not neighbors else min(
                    1.0,
                    len(neighbors) / float(self.MAX_NEIGHBORS_REFERENCE),
                ),
                "mean_similarity": float(
                    np.mean([float(neighbor[1]) for neighbor in neighbors]) if neighbors else 0.0
                ),
                "candidate_signal": 0.0 if not candidates else min(
                    1.0,
                    len(candidates) / float(self.DEFAULT_CANDIDATE_POOL),
                ),
            }

        neighbor_count = len(neighbors)
        mean_similarity = float(np.mean([float(neighbor[1]) for neighbor in neighbors]))
        neighbor_count_signal = min(
            1.0,
            neighbor_count / float(self.MAX_NEIGHBORS_REFERENCE),
        )
        candidate_signal = min(1.0, len(candidates) / float(self.DEFAULT_CANDIDATE_POOL))

        collaborative_signal = (
            0.55 * mean_similarity
            + 0.30 * neighbor_count_signal
            + 0.15 * candidate_signal
        )

        return float(min(1.0, collaborative_signal)), {
            "neighbor_count": float(neighbor_count),
            "neighbor_count_signal": float(neighbor_count_signal),
            "mean_similarity": float(mean_similarity),
            "candidate_signal": float(candidate_signal),
        }

    def _get_neighbors(self, user_id: Union[int, str]):
        normalized_user_id = str(user_id)
        user_data = self.collaborative_model.user_registry["users"].get(normalized_user_id)
        if user_data is None:
            return []

        if "neighbors" in user_data and user_data["neighbors"]:
            raw_neighbors = [
                (neighbor["user_id"], neighbor["similarity"])
                for neighbor in user_data["neighbors"]
            ]
        else:
            self.collaborative_model.get_preference_matrix()
            raw_neighbors = self.collaborative_model._find_neighbors(normalized_user_id)

        if hasattr(self.collaborative_model, "_normalize_neighbors"):
            return self.collaborative_model._normalize_neighbors(raw_neighbors)
        return raw_neighbors

    def _normalize_blend_weights(
        self,
        content_signal: float,
        collaborative_signal: float,
    ) -> Tuple[float, float]:
        content_signal = max(0.0, float(content_signal))
        collaborative_signal = max(0.0, float(collaborative_signal))
        if content_signal <= 1e-12 and collaborative_signal <= 1e-12:
            return 0.0, 0.0
        # Trato equilibrado entre ambos modelos cuando los dos aportan señal.
        if content_signal > 1e-12 and collaborative_signal > 1e-12:
            return 0.5, 0.5
        if content_signal > 1e-12:
            return 1.0, 0.0
        return 0.0, 1.0

    def _normalize_content_ratio(self, content_row: Optional[Dict[str, Any]]) -> float:
        if not content_row:
            return 0.0
        score = pd.to_numeric(pd.Series([content_row.get("score", 0.0)]), errors="coerce").iloc[0]
        if pd.isna(score):
            return 0.0
        return float(np.clip(score, 0.0, 1.0))

    def _normalize_collaborative_ratio(
        self,
        collaborative_row: Optional[Dict[str, Any]],
    ) -> float:
        if not collaborative_row:
            return 0.0

        ranking_score = pd.to_numeric(
            pd.Series([collaborative_row.get("ranking_score")]),
            errors="coerce",
        ).iloc[0]
        if pd.notna(ranking_score):
            return float(np.clip(float(ranking_score) / 5.0, 0.0, 1.0))

        predicted_rating = pd.to_numeric(
            pd.Series([collaborative_row.get("predicted_rating")]),
            errors="coerce",
        ).iloc[0]
        confidence = pd.to_numeric(
            pd.Series([collaborative_row.get("confidence", 0.0)]),
            errors="coerce",
        ).iloc[0]
        if pd.isna(predicted_rating):
            return 0.0

        if pd.isna(confidence):
            confidence = 0.0

        return float(
            np.clip((float(predicted_rating) / 5.0) * float(confidence), 0.0, 1.0)
        )

    def _resolve_title(
        self,
        content_row: Optional[Dict[str, Any]],
        collaborative_row: Optional[Dict[str, Any]],
        movie_id: int,
    ) -> str:
        if content_row:
            title = content_row.get("title")
            if title:
                return str(title)

        content_movies = getattr(self.content_model, "movies", None)
        content_movie_id_col = getattr(self.content_model, "movie_id_col", "movieId")
        if content_movies is not None and not content_movies.empty and content_movie_id_col in content_movies.columns:
            matching_rows = content_movies[content_movies[content_movie_id_col] == int(movie_id)]
            if not matching_rows.empty and hasattr(self.content_model, "_resolve_movie_title"):
                return str(self.content_model._resolve_movie_title(matching_rows.iloc[0]))

        collaborative_movies = getattr(self.collaborative_model, "movie_data", None)
        if isinstance(collaborative_movies, list):
            for movie_data in collaborative_movies:
                try:
                    candidate_id = int(movie_data.get("id") or movie_data.get("movieId"))
                except Exception:
                    continue
                if candidate_id != int(movie_id):
                    continue
                for key in ("titulo", "title", "original_title", "name"):
                    value = movie_data.get(key)
                    if value:
                        return str(value)
                break

        if collaborative_row and collaborative_row.get("title"):
            return str(collaborative_row["title"])

        return str(movie_id)

    def _build_hybrid_reasons(
        self,
        movie_id: int,
        alpha: float,
        beta: float,
        content_row: Optional[Dict[str, Any]],
        collaborative_row: Optional[Dict[str, Any]],
        content_diagnostics: Dict[str, float],
        collaborative_diagnostics: Dict[str, float],
    ) -> List[str]:
        reasons: List[str] = []

        in_content = content_row is not None
        in_collaborative = collaborative_row is not None

        if in_content and in_collaborative:
            reasons.append(
                "Esta pelicula aparece en ambos recomendadores, asi que hay consenso entre contenido y colaborativo."
            )
            reasons.append(
                "Se recomienda por una mezcla equilibrada entre afinidad de contenido y evidencia de vecinos."
            )
        elif in_content:
            reasons.append(
                "Esta pelicula encaja por su similitud de genero, tematica e historia con lo que sueles ver."
            )
            reasons.append("La recomendacion se apoya principalmente en tu perfil de contenido.")
        elif in_collaborative:
            reasons.append(
                "Esta pelicula destaca porque usuarios vecinos con gustos parecidos la valoraron bien."
            )
            reasons.append("La recomendacion se apoya principalmente en usuarios vecinos.")

        if content_row:
            content_reasons = content_row.get("reasons", [])
            if isinstance(content_reasons, list):
                for reason in content_reasons[:1]:
                    if reason:
                        reasons.append(reason)

        if collaborative_row:
            num_contributors = int(collaborative_row.get("num_contributors", 0))
            if num_contributors > 0:
                reasons.append(
                    f"{num_contributors} vecinos contribuyen a esta recomendacion."
                )

        genre_count = int(content_diagnostics.get("genre_count", 0))
        ratings_count = int(content_diagnostics.get("ratings_count", 0))
        neighbor_count = int(collaborative_diagnostics.get("neighbor_count", 0))
        mean_similarity = float(collaborative_diagnostics.get("mean_similarity", 0.0))
        similarity_label = "alta" if mean_similarity >= 0.70 else ("media" if mean_similarity >= 0.45 else "baja")
        reasons.append(
            "Se ha tenido en cuenta la informacion disponible del usuario: "
            f"{genre_count} generos relevantes, {ratings_count} ratings historicos, "
            f"{neighbor_count} vecinos y una similitud media {similarity_label}."
        )

        if not reasons:
            reasons.append(
                "La recomendacion hibrida combina cercania por contenido y evidencia de usuarios similares."
            )

        unique_reasons: List[str] = []
        seen = set()
        for reason in reasons:
            normalized = reason.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique_reasons.append(normalized)

        return unique_reasons

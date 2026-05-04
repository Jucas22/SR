import json
import traceback

import pandas as pd
import streamlit as st

from Backend.recommenders import (
    ColaborativeRecommender,
    ContentBasedRecommender,
    HybridRecommender,
)
from Backend.group_recommender import GroupRecommender
from Backend.services import UserRegistryManager
from Frontend.services.data_manager import DataManager as LegacyDataManager
from project_paths import CLEAN_DATA_DIR, POSTERS_DIR, RAW_DATA_DIR, USER_REGISTRY_PATH


class DataManager(LegacyDataManager):
    """Facade limpia sobre el gestor legado, con rutas centralizadas."""

    def __init__(self):
        self.data_path = CLEAN_DATA_DIR
        self.raw_data_path = RAW_DATA_DIR
        self.posters_path = POSTERS_DIR
        self.base_poster_path = "https://image.tmdb.org/t/p/original"

        self.movies = self._load_movies()
        self.genre_mapping = self._load_genre_mapping()
        self.available_genres = self._build_available_genres()
        self.movie_search_index = self._build_movie_search_index()

        self.content_recommender = None
        self.collaborative_recommender = None
        self.hybrid_recommender = None
        self.group_recommender = None
        self.recommender = None
        self.current_recommender_type = "content"
        self.user_registry_manager = UserRegistryManager()

    def _initialize_colaborative_recommender(self):
        """Alias legado para mantener compatibilidad."""
        return self._initialize_collaborative_recommender()

    def _initialize_recommender(self):
        """Inicializar y entrenar el recomendador basado en contenido."""
        try:
            recommender = ContentBasedRecommender(
                categorical_columns=["generos"],
                feature_text_columns=[
                    "overview",
                    "tagline",
                    "keywords",
                    "tags",
                    "cast",
                    "crew",
                    "director",
                ],
                rating_threshold_like=3.5,
                profile_strategy="weighted",
                use_registry_history_for_cold_start=True,
                recency_weighting=False,
                w_quality=0.05,
                w_popularity=0.02,
                diversity_penalty=0.10,
                min_df=1,
                max_tfidf_features=500,
            )

            movies_list = list(self.movies.values())
            if not movies_list:
                st.warning(
                    "No hay peliculas cargadas. El recomendador no se puede inicializar."
                )
                return None

            movies_df = pd.DataFrame(movies_list)

            if "id" in movies_df.columns:
                movies_df = movies_df.rename(columns={"id": "movieId"})
            elif "movieId" not in movies_df.columns:
                movies_df["movieId"] = range(len(movies_df))

            movies_df["movieId"] = movies_df["movieId"].astype(int)

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
                    movies_df[col] = ""

            text_cols = [
                "overview",
                "tagline",
                "keywords",
                "tags",
                "cast",
                "crew",
                "director",
            ]
            for col in text_cols:
                movies_df[col] = movies_df[col].fillna("").astype(str)

            if "generos" in movies_df.columns:
                movies_df["generos"] = movies_df["generos"].apply(
                    lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x])
                )

            ratings_df = None
            ratings_path = self.data_path / "train_ratings.json"
            if ratings_path.exists():
                try:
                    with ratings_path.open("r", encoding="utf-8") as f:
                        ratings_data = json.load(f)
                    if ratings_data:
                        ratings_df = pd.DataFrame(ratings_data)
                        if not ratings_df.empty:
                            if "userId" in ratings_df.columns:
                                ratings_df["userId"] = ratings_df["userId"].astype(int)
                            if "movieId" in ratings_df.columns:
                                ratings_df["movieId"] = ratings_df["movieId"].astype(int)
                            if "rating" in ratings_df.columns:
                                ratings_df["rating"] = pd.to_numeric(
                                    ratings_df["rating"],
                                    errors="coerce",
                                )
                            ratings_df = ratings_df.dropna(subset=["rating"])
                except Exception as exc:
                    st.info(
                        f"Info: No se pudieron cargar ratings ({str(exc)[:50]})"
                    )
                    ratings_df = None

            user_registry = None
            if USER_REGISTRY_PATH.exists():
                try:
                    with USER_REGISTRY_PATH.open("r", encoding="utf-8") as f:
                        user_registry = json.load(f)
                except Exception as exc:
                    st.info(
                        f"Info: No se pudo cargar user_registry ({str(exc)[:50]})"
                    )
                    user_registry = None

            recommender.fit(movies_df, ratings=ratings_df, user_registry=user_registry)
            st.success("Recomendador inicializado correctamente")
            return recommender

        except Exception as exc:
            st.error(f"Error: {str(exc)[:100]}")
            traceback.print_exc()
            return None

    def _initialize_collaborative_recommender(self):
        """Inicializar y entrenar el recomendador colaborativo."""
        try:
            user_registry_path = USER_REGISTRY_PATH
            movie_data_path = self.data_path / "enhanced_movies.json"

            if not user_registry_path.exists() or not movie_data_path.exists():
                st.warning(
                    "Archivos necesarios no encontrados para recomendador colaborativo"
                )
                return None

            recommender = ColaborativeRecommender(
                user_registry_path=str(user_registry_path),
                movie_data_path=str(movie_data_path),
            )
            recommender.get_preference_matrix()
            st.success("Recomendador colaborativo inicializado correctamente")
            return recommender
        except Exception as exc:
            st.error(f"Error: {str(exc)[:100]}")
            traceback.print_exc()
            return None

    def _initialize_hybrid_recommender(self):
        """Inicializar el recomendador hibrido."""
        try:
            if self.content_recommender is None:
                self.content_recommender = self._initialize_recommender()
            if self.collaborative_recommender is None:
                self.collaborative_recommender = (
                    self._initialize_collaborative_recommender()
                )

            if (
                self.content_recommender is None
                or self.collaborative_recommender is None
            ):
                st.warning(
                    "No se pudieron inicializar los recomendadores base para el modo hibrido"
                )
                return None

            recommender = HybridRecommender(
                content_model=self.content_recommender,
                collaborative_model=self.collaborative_recommender,
            )
            st.success("Recomendador hibrido inicializado correctamente")
            return recommender
        except Exception as exc:
            st.error(f"Error: {str(exc)[:100]}")
            traceback.print_exc()
            return None

    def _initialize_group_recommender(self):
        """Inicializar el recomendador de grupo usando el colaborativo como base."""
        try:
            if self.collaborative_recommender is None:
                self.collaborative_recommender = self._initialize_collaborative_recommender()

            if self.collaborative_recommender is None:
                st.warning("No se pudo inicializar el recomendador colaborativo para el modo grupo")
                return None

            recommender = GroupRecommender(self.collaborative_recommender)
            st.success("Recomendador de grupo inicializado correctamente")
            return recommender
        except Exception as exc:
            st.error(f"Error: {str(exc)[:100]}")
            traceback.print_exc()
            return None

    def get_available_genres(self):
        """Devuelve la lista precalculada de generos disponibles."""
        return list(self.available_genres)

    def filter_movies(self, search_term, genre_filter):
        """Filtra peliculas usando un indice ligero precalculado."""
        filtered = {}
        normalized_search = search_term.lower().strip() if search_term else ""
        normalized_genre = genre_filter if genre_filter and genre_filter != "Todos" else None

        for movie_id, movie_data in self.movies.items():
            search_data = self.movie_search_index.get(
                movie_id, {"title": "", "genres": frozenset()}
            )
            if normalized_search and normalized_search not in search_data["title"]:
                continue
            if normalized_genre and normalized_genre not in search_data["genres"]:
                continue
            filtered[movie_id] = movie_data

        return filtered

    def invalidate_recommenders(self):
        """Fuerza la recarga de modelos cuando cambia el perfil del usuario."""
        self.content_recommender = None
        self.collaborative_recommender = None
        self.hybrid_recommender = None
        self.group_recommender = None
        self.recommender = None

    def _build_available_genres(self):
        """Precalcula la lista ordenada de generos disponibles."""
        genres = set()
        for movie_data in self.movies.values():
            if not isinstance(movie_data, dict):
                continue
            genre_ids = movie_data.get("generos", [])
            if not isinstance(genre_ids, list):
                continue
            for gid in genre_ids:
                genres.add(self.genre_mapping.get(gid, f"Genero {gid}"))
        return tuple(sorted(genres))

    def _build_movie_search_index(self):
        """Prepara un indice simple para acelerar la exploracion."""
        index = {}
        for movie_id, movie_data in self.movies.items():
            genre_ids = movie_data.get("generos", [])
            if isinstance(genre_ids, list):
                genre_names = frozenset(
                    self.genre_mapping.get(gid, f"Genero {gid}") for gid in genre_ids
                )
            else:
                genre_names = frozenset()

            index[movie_id] = {
                "title": movie_data.get("titulo", "").lower(),
                "genres": genre_names,
            }

        return index

    def ensure_colaborative_recommender_initialized(self):
        self.ensure_recommender_initialized("collaborative")

    def get_recommendations(
        self, user_id, top_k=20, exclude_seen=True, recommender_type="content"
    ):
        """
        Override del flujo legado para enriquecer las recomendaciones colaborativas
        con rating predicho, confianza y soporte.
        """
        if recommender_type != "collaborative":
            return super().get_recommendations(
                user_id=user_id,
                top_k=top_k,
                exclude_seen=exclude_seen,
                recommender_type=recommender_type,
            )

        self.ensure_recommender_initialized(recommender_type)
        if self.recommender is None:
            return None

        try:
            recommendations = self.recommender.recommend(
                user_id=str(user_id),
                top_n=top_k,
                return_details=True,
            )
            if not recommendations:
                return None

            rows = []
            for recommendation in recommendations:
                confidence = float(recommendation.get("confidence", 0.0))
                predicted_rating = float(recommendation.get("predicted_rating", 0.0))
                mean_similarity = float(recommendation.get("mean_similarity", 0.0))
                num_contributors = int(recommendation.get("num_contributors", 0))
                positive_contributors = int(
                    recommendation.get("positive_contributors", 0)
                )
                neutral_contributors = int(
                    recommendation.get("neutral_contributors", 0)
                )
                negative_contributors = int(
                    recommendation.get("negative_contributors", 0)
                )
                (
                    positive_contributors,
                    neutral_contributors,
                    negative_contributors,
                    breakdown_consistent,
                ) = self._normalize_contributor_breakdown(
                    num_contributors=num_contributors,
                    positive_contributors=positive_contributors,
                    neutral_contributors=neutral_contributors,
                    negative_contributors=negative_contributors,
                )

                rows.append(
                    {
                        "movie_id": int(recommendation["movie_id"]),
                        "score": confidence,
                        "predicted_rating": predicted_rating,
                        "confidence": confidence,
                        "mean_similarity": mean_similarity,
                        "num_contributors": num_contributors,
                        "positive_contributors": positive_contributors,
                        "neutral_contributors": neutral_contributors,
                        "negative_contributors": negative_contributors,
                        "support_ratio": float(
                            recommendation.get("support_ratio", 0.0)
                        ),
                        "ranking_score": float(
                            recommendation.get("ranking_score", 0.0)
                        ),
                        "reasons": self._build_collaborative_reason_lines(
                            confidence=confidence,
                            mean_similarity=mean_similarity,
                            num_contributors=num_contributors,
                            positive_contributors=positive_contributors,
                            neutral_contributors=neutral_contributors,
                            negative_contributors=negative_contributors,
                            breakdown_consistent=breakdown_consistent,
                        ),
                    }
                )

            return pd.DataFrame(rows)

        except Exception as exc:
            st.warning(f"No se pudieron generar recomendaciones para este usuario: {exc}")
            return None

    def _build_collaborative_reason_lines(
        self,
        confidence: float,
        mean_similarity: float,
        num_contributors: int,
        positive_contributors: int,
        neutral_contributors: int,
        negative_contributors: int,
        breakdown_consistent: bool,
    ):
        """Genera una explicacion breve y legible para las tarjetas colaborativas."""
        confidence_label = self._get_confidence_label(confidence)
        if breakdown_consistent:
            if num_contributors == 1:
                audience_line = (
                    "Esta pelicula se te recomienda porque hay 1 usuario con gustos "
                    "parecidos que la ha visto"
                )
            else:
                audience_line = (
                    "Esta pelicula se te recomienda porque hay "
                    f"{num_contributors} usuarios con gustos parecidos que la han visto"
                )

            if positive_contributors == 1:
                positive_line = "y 1 de ellos la ha valorado con una buena puntuacion."
            else:
                positive_line = (
                    f"y {positive_contributors} de ellos la han valorado "
                    "con una buena puntuacion."
                )

            breakdown_line = (
                f"{audience_line}, {positive_line}"
            )
        else:
            breakdown_line = (
                "Esta pelicula se te recomienda porque hay "
                f"{num_contributors} usuarios con gustos parecidos que la han visto. "
                "El desglose detallado de sus valoraciones no estaba disponible."
            )
        return [
            f"Confianza {confidence_label.lower()}: {confidence:.0%}.",
            breakdown_line,
            f"Similitud media de los contribuyentes: {mean_similarity:.0%}.",
        ]

    def _get_confidence_label(self, confidence: float) -> str:
        """Convierte la confianza numerica en una etiqueta legible."""
        if confidence >= 0.70:
            return "Alta"
        if confidence >= 0.45:
            return "Media"
        return "Baja"

    def _normalize_contributor_breakdown(
        self,
        num_contributors: int,
        positive_contributors: int,
        neutral_contributors: int,
        negative_contributors: int,
    ):
        """
        Garantiza que el desglose de valoraciones no muestre combinaciones imposibles.
        Devuelve tambien un flag para indicar si el desglose es confiable.
        """
        total = (
            int(positive_contributors)
            + int(neutral_contributors)
            + int(negative_contributors)
        )
        num_contributors = int(num_contributors)

        if num_contributors <= 0:
            return 0, 0, 0, True

        if total == num_contributors:
            return (
                int(positive_contributors),
                int(neutral_contributors),
                int(negative_contributors),
                True,
            )

        repaired_neutral = max(
            0,
            num_contributors - int(positive_contributors) - int(negative_contributors),
        )
        repaired_total = (
            int(positive_contributors)
            + repaired_neutral
            + int(negative_contributors)
        )

        if repaired_total == num_contributors:
            return (
                int(positive_contributors),
                int(repaired_neutral),
                int(negative_contributors),
                False,
            )

        return 0, num_contributors, 0, False

    def update_user_rating(self, user_id, movie_id, rating):
        """
        Actualiza un rating y descarta modelos cacheados para reflejar el nuevo perfil.
        """
        updated = super().update_user_rating(user_id, movie_id, rating)
        if updated:
            self.invalidate_recommenders()
        return updated

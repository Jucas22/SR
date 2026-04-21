import json
import pandas as pd
from pathlib import Path
import streamlit as st
import sys
import traceback

import requests

# Agregar el directorio backend al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Backend.content_recommender import ContentBasedRecommender
from Backend.Colaborativo.colaborative_recommender import ColaborativeRecommender
from Backend.hybrid_recommender import HybridRecommender
from Backend.Colaborativo.colaborative_recommender import ColaborativeRecommender
from Backend.user_registry_manager import UserRegistryManager


class DataManager:
    """
    Gestor de datos para cargar y manejar películas, géneros, posters y recomendaciones
    """

    def __init__(self):
        """Inicializar el gestor de datos"""
        self.data_path = Path(__file__).parent.parent / "Data" / "Clean_data"
        self.raw_data_path = Path(__file__).parent.parent / "Data" / "Raw_data"
        self.posters_path = Path(__file__).parent.parent / "Data" / "posters"
        self.base_poster_path = "https://image.tmdb.org/t/p/original"

        # Cargar datos
        self.movies = self._load_movies()
        self.genre_mapping = self._load_genre_mapping()

        # Inicializar campos; los recomendadores se crean bajo demanda
        self.content_recommender = None
        self.collaborative_recommender = None
        self.hybrid_recommender = None
        self.recommender = None  # Para compatibilidad hacia atrás
        self.current_recommender_type = "content"  # Por defecto
        self.user_registry_manager = UserRegistryManager()

    def _initialize_colaborative_recommender(self):
        """Inicializar recomendador colaborativo bajo demanda."""
        try:
            user_registry_path = (
                Path(__file__).parent.parent / "Data" / "user_registry.json"
            )
            movie_data_path = self.data_path / "enhanced_movies.json"

            colaborative_recommender = ColaborativeRecommender(
                user_registry_path=str(user_registry_path),
                movie_data_path=str(movie_data_path),
            )
            colaborative_recommender.get_preference_matrix()
            st.success("✅ Recomendador colaborativo inicializado correctamente")
            return colaborative_recommender
        except Exception as e:
            st.error(
                f"❌ Error inicializando recomendador colaborativo: {str(e)[:100]}"
            )
            traceback.print_exc()
            return None

    def _load_movies(self):
        """Cargar datos de películas"""
        try:
            with open(
                self.data_path / "enhanced_movies.json", "r", encoding="utf-8"
            ) as f:
                movies_list = json.load(f)
                # Convertir lista a diccionario usando id como clave
                movies_dict = {str(movie["id"]): movie for movie in movies_list}
                return movies_dict
        except Exception as e:
            st.error(f"Error cargando películas: {e}")
            return {}

    def _load_genre_mapping(self):
        """Cargar mapeo de géneros desde CSV"""
        try:
            df = pd.read_csv(
                self.raw_data_path / "generos.csv", sep=";", encoding="utf-8"
            )
            # Crear mapeo de ID a nombre en español
            genre_map = dict(zip(df["IdDataset"], df["GeneroSP"]))
            return genre_map
        except Exception as e:
            st.error(f"Error cargando géneros: {e}")
            return {}

    def _initialize_recommender(self):
        """Inicializar y entrenar el recomendador basado en contenido"""
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
                recency_weighting=False,  # Desactivar para evitar problemas con pesos
                w_quality=0.05,  # Reducir peso
                w_popularity=0.02,  # Reducir peso
                diversity_penalty=0.10,
                min_df=1,
                max_tfidf_features=500,  # Reducir features
            )

            # Convertir movies dict a DataFrame para el recomendador
            movies_list = list(self.movies.values())
            if not movies_list:
                st.warning(
                    "⚠️ No hay películas cargadas. El recomendador no se puede inicializar."
                )
                return None

            movies_df = pd.DataFrame(movies_list)

            # Asegurarse que las columnas necesarias existen
            if "id" in movies_df.columns:
                movies_df = movies_df.rename(columns={"id": "movieId"})
            elif "movieId" not in movies_df.columns:
                movies_df["movieId"] = range(len(movies_df))

            # Asegurar que movieId es entero
            movies_df["movieId"] = movies_df["movieId"].astype(int)

            # Asegurar que las columnas de características existen (aunque estén vacías)
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

            # Fillna con strings vacíos para evitar problemas
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

            # Asegurar que generos es lista
            if "generos" in movies_df.columns:
                movies_df["generos"] = movies_df["generos"].apply(
                    lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x])
                )

            # Cargar ratings si existen
            ratings_df = None
            ratings_path = self.data_path / "train_ratings.json"
            if ratings_path.exists():
                try:
                    with open(ratings_path, "r", encoding="utf-8") as f:
                        ratings_data = json.load(f)
                        if ratings_data:
                            ratings_df = pd.DataFrame(ratings_data)
                            # Asegurar tipos de datos
                            if not ratings_df.empty:
                                if "userId" in ratings_df.columns:
                                    ratings_df["userId"] = ratings_df["userId"].astype(
                                        int
                                    )
                                if "movieId" in ratings_df.columns:
                                    ratings_df["movieId"] = ratings_df[
                                        "movieId"
                                    ].astype(int)
                                if "rating" in ratings_df.columns:
                                    ratings_df["rating"] = pd.to_numeric(
                                        ratings_df["rating"], errors="coerce"
                                    )
                                # Remover NaN
                                ratings_df = ratings_df.dropna(subset=["rating"])
                except Exception as e:
                    st.info(f"⚠️ Info: No se pudieron cargar ratings ({str(e)[:50]})")
                    ratings_df = None

            # Cargar user_registry
            user_registry_path = (
                Path(__file__).parent.parent / "Data" / "user_registry.json"
            )
            user_registry = None
            if user_registry_path.exists():
                try:
                    with open(user_registry_path, "r", encoding="utf-8") as f:
                        user_registry = json.load(f)
                except Exception as e:
                    st.info(f"⚠️ Info: No se pudo cargar user_registry ({str(e)[:50]})")
                    user_registry = None

            print(f"🔧 Entrenando recomendador con {len(movies_df)} películas...")
            if ratings_df is not None:
                print(f"   - {len(ratings_df)} ratings cargados")

            # Entrenar el recomendador
            recommender.fit(movies_df, ratings=ratings_df, user_registry=user_registry)
            st.success("✅ Recomendador inicializado correctamente")
            return recommender

        except Exception as e:
            error_msg = str(e)
            st.error(f"❌ Error: {error_msg[:100]}")
            print(f"❌ Error completo: {error_msg}")
            import traceback

            traceback.print_exc()
            return None

    def _initialize_collaborative_recommender(self):
        """Inicializar y entrenar el recomendador colaborativo"""
        try:
            user_registry_path = (
                Path(__file__).parent.parent / "Data" / "user_registry.json"
            )
            movie_data_path = (
                Path(__file__).parent.parent
                / "Data"
                / "Clean_data"
                / "enhanced_movies.json"
            )

            if not user_registry_path.exists() or not movie_data_path.exists():
                st.warning(
                    "⚠️ Archivos necesarios no encontrados para recomendador colaborativo"
                )
                return None

            recommender = ColaborativeRecommender(
                user_registry_path=str(user_registry_path),
                movie_data_path=str(movie_data_path),
            )
            st.success("✅ Recomendador colaborativo inicializado correctamente")
            return recommender
        except Exception as e:
            error_msg = str(e)
            st.error(f"❌ Error: {error_msg[:100]}")
            print(f"❌ Error completo: {error_msg}")
            import traceback

            traceback.print_exc()
            return None

    def _initialize_hybrid_recommender(self):
        """Inicializar el recomendador híbrido"""
        try:
            # Asegurar que ambos recomendadores estén inicializados
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
                    "⚠️ No se pudieron inicializar los recomendadores base para el modo híbrido"
                )
                return None

            recommender = HybridRecommender(
                content_model=self.content_recommender,
                collaborative_model=self.collaborative_recommender,
            )
            st.success("✅ Recomendador híbrido inicializado correctamente")
            return recommender
        except Exception as e:
            error_msg = str(e)
            st.error(f"❌ Error: {error_msg[:100]}")
            print(f"❌ Error completo: {error_msg}")
            import traceback

            traceback.print_exc()
            return None

    def get_genre_names(self, genre_ids):
        """Convertir IDs de géneros a nombres"""
        if isinstance(genre_ids, list):
            return [self.genre_mapping.get(gid, f"Género {gid}") for gid in genre_ids]
        return []

    def obtener_datos_pelicula(self, nombre_pelicula):
        """Buscar datos de una película en TMDb (uso opcional)"""
        try:
            base_url = "https://api.themoviedb.org/3/search/movie"
            params = {
                "api_key": st.secrets.get("TMDB_API_KEY", ""),
                "query": nombre_pelicula,
                "language": "es-ES",
            }
            response = requests.get(base_url, params=params, timeout=5)
            datos = response.json()

            if datos.get("results"):
                return datos["results"][0]
        except Exception:
            pass
        return None

    def get_poster_url(self, movie_id, movie_data):
        """Obtener la URL del poster para una película (sin renderizar).
        Devuelve (url, is_placeholder) para que el caller decida cómo mostrarlo.
        """
        try:
            original_id = movie_data.get("imdb_id", movie_id)
            poster_path = movie_data.get("poster_path")
            original_id_str = str(original_id).replace("tt", "").lstrip("0")

            # MÉTODO 1: poster_path online (TMDb)
            if poster_path:
                return f"{self.base_poster_path}{poster_path}", False

            # MÉTODO 2: poster local
            for poster_file in self.posters_path.glob(f"*_{original_id_str}.jpg"):
                return str(poster_file), False

            # Sin poster
            return (
                "https://via.placeholder.com/300x450/1a1a2e/e0e0e0?text=Sin+Poster",
                True,
            )

        except Exception:
            return "https://via.placeholder.com/300x450/1a1a2e/e0e0e0?text=Error", True

    def ensure_recommender_initialized(self, recommender_type="content"):
        """Crea y entrena el recomendador si aún no existe.

        Este método puede ser llamado varias veces de forma segura: si el
        recomendador ya está preparado no hace nada. Está pensado para ser
        invocado desde la interfaz cuando el usuario solicita recomendaciones.

        Args:
            recommender_type: Tipo de recomendador ("content", "collaborative" o "hybrid")
        """
        self.current_recommender_type = recommender_type

        if recommender_type == "content":
            if self.content_recommender is not None:
                self.recommender = self.content_recommender
                return
            with st.spinner("🔧 Inicializando recomendador basado en contenido..."):
                self.content_recommender = self._initialize_recommender()
                self.recommender = self.content_recommender

        elif recommender_type == "collaborative":
            if self.collaborative_recommender is not None:
                self.recommender = self.collaborative_recommender
                return
            with st.spinner("🔧 Inicializando recomendador colaborativo..."):
                self.collaborative_recommender = (
                    self._initialize_collaborative_recommender()
                )
                self.recommender = self.collaborative_recommender

        elif recommender_type == "hybrid":
            if self.hybrid_recommender is not None:
                self.recommender = self.hybrid_recommender
                return
            with st.spinner("🔧 Inicializando recomendador híbrido..."):
                self.hybrid_recommender = self._initialize_hybrid_recommender()
                self.recommender = self.hybrid_recommender

    def render_poster(self, movie_id, movie_data, width=None, use_container=False):
        """Renderizar poster de forma centralizada y consistente.
        Args:
            movie_id: ID de la película
            movie_data: datos de la película
            width: ancho en píxeles (None = usar todo el contenedor)
            use_container: si True, ajusta el ancho al contenedor
        Returns:
            True si se encontró poster real, False si es placeholder
        """
        url, is_placeholder = self.get_poster_url(movie_id, movie_data)
        try:
            if use_container:
                st.image(url, width="stretch")
            elif width:
                st.image(url, width=width)
            else:
                st.image(url, width="stretch")
            return not is_placeholder
        except Exception:
            st.image(
                "https://via.placeholder.com/300x450/1a1a2e/e0e0e0?text=Error",
                width="stretch" if use_container else (width or "content"),
            )
            return False

    # Mantener retrocompatibilidad
    def get_poster_path(self, movie_id, movie_data):
        """Compatibilidad: renderiza poster y devuelve True/False"""
        return self.render_poster(movie_id, movie_data, use_container=False)

    def get_available_genres(self):
        """Obtener géneros disponibles de las películas"""
        try:
            genres = set()
            for movie_data in self.movies.values():
                if isinstance(movie_data, dict) and "generos" in movie_data:
                    genre_ids = movie_data["generos"]
                    if isinstance(genre_ids, list):
                        for gid in genre_ids:
                            genre_name = self.genre_mapping.get(gid, f"Género {gid}")
                            genres.add(genre_name)
            return sorted(list(genres))
        except:
            return []

    def filter_movies(self, search_term, genre_filter):
        """Filtrar películas según criterios"""
        filtered = {}

        for movie_id, movie_data in self.movies.items():
            # Filtro de búsqueda
            if search_term:
                title = movie_data.get("titulo", "").lower()
                if search_term.lower() not in title:
                    continue

            # Filtro de género
            if genre_filter and genre_filter != "Todos":
                genre_ids = movie_data.get("generos", [])
                genre_names = self.get_genre_names(genre_ids)
                if genre_filter not in genre_names:
                    continue

            filtered[movie_id] = movie_data

        return filtered

    def get_movie_count(self):
        """Obtener el número total de películas"""
        return len(self.movies)

    def get_recommendations(
        self, user_id, top_k=10, exclude_seen=True, recommender_type="content"
    ):
        """
        Obtener recomendaciones personalizadas para un usuario

        Args:
            user_id: ID del usuario
            top_k: Número de recomendaciones a devolver
            exclude_seen: Si True, excluye películas ya vistas (solo para basado en contenido)
            recommender_type: Tipo de recomendador ("content", "collaborative" o "hybrid")

        Returns:
            DataFrame con recomendaciones o None si hay error
        """
        # Inicializar el recomendador apropiado
        self.ensure_recommender_initialized(recommender_type)
        if self.recommender is None:
            return None

        try:
            # Diferentes métodos según el tipo
            if recommender_type == "content":
                # El recomendador basado en contenido devuelve DataFrame
                recommendations_df = self.recommender.recommend(
                    user_id=user_id,
                    top_k=top_k,
                    exclude_seen=exclude_seen,
                    return_df=True,
                )
                return recommendations_df

            elif recommender_type == "collaborative":
                # El recomendador colaborativo devuelve lista de tuplas
                recommendations = self.recommender.recommend(
                    user_id=str(user_id), top_n=top_k
                )
                if recommendations:
                    # Convertir a formato DataFrame para compatibilidad
                    df_data = []
                    for movie_id, score in recommendations:
                        # Normalizar score de rating (0-5) a porcentaje (0-1)
                        normalized_score = min(score / 5.0, 1.0) if score > 0 else 0.0
                        df_data.append(
                            {
                                "movie_id": movie_id,
                                "score": normalized_score,
                                "reasons": [],
                            }
                        )
                    return pd.DataFrame(df_data)
                return None

            elif recommender_type == "hybrid":
                # El recomendador híbrido devuelve lista de tuplas
                recommendations = self.recommender.recommend(
                    user_id=user_id, top_k=top_k
                )
                if recommendations:
                    # Convertir a formato DataFrame para compatibilidad
                    df_data = []
                    for movie_id, score in recommendations:
                        # Normalizar score de rating (0-5) a porcentaje (0-1)
                        normalized_score = min(score / 5.0, 1.0) if score > 0 else 0.0
                        df_data.append(
                            {
                                "movie_id": movie_id,
                                "score": normalized_score,
                                "reasons": [],
                            }
                        )
                    return pd.DataFrame(df_data)
                return None
            else:
                st.error(f"Tipo de recomendador no válido: {recommender_type}")
                return None

        except Exception as e:
            st.warning(f"No se pudieron generar recomendaciones para este usuario: {e}")
            return None

    def get_recommendation_explanation(self, user_id, movie_id):
        """
        Obtener explicación de por qué se recomienda una película

        Args:
            user_id: ID del usuario
            movie_id: ID de la película

        Returns:
            Lista de textos explicativos
        """
        if self.recommender is None:
            return []

        try:
            return self.recommender.explain_recommendation(
                user_id, movie_id, top_n_reasons=3
            )
        except Exception:
            return []

    def get_colaborative_recommendations(self, user_id, top_k=10):
        """Obtener recomendaciones del SR colaborativo."""
        self.ensure_colaborative_recommender_initialized()
        if self.colaborative_recommender is None:
            return []

        try:
            recs = self.colaborative_recommender.recommend(str(user_id), top_n=top_k)
            return recs or []
        except Exception as e:
            st.warning(f"No se pudieron generar recomendaciones colaborativas: {e}")
            return []

    def get_colaborative_recommendation_explanation(self, user_id, movie_id):
        """Obtener explicación colaborativa basada en vecinos contribuyentes."""
        self.ensure_colaborative_recommender_initialized()
        if self.colaborative_recommender is None:
            return {}

        try:
            explanation = self.colaborative_recommender.explain_recommendation(
                str(user_id), int(movie_id)
            )
            return explanation if isinstance(explanation, dict) else {}
        except Exception as e:
            st.warning(f"No se pudo obtener la explicación colaborativa: {e}")
            return {}

    def update_user_rating(self, user_id, movie_id, rating):
        """
        Actualizar el rating de un usuario para una película

        Args:
            user_id: ID del usuario
            movie_id: ID de la película
            rating: Calificación (0-5 típicamente)
        """
        try:
            if user_id is None:
                st.error("❌ No hay un usuario autenticado para guardar el rating")
                return False

            # Convertir a tipos correctos
            user_id = int(user_id)
            movie_id = int(movie_id)
            rating = float(rating)

            # Actualizar en el recomendador actual si es compatible
            if self.recommender is not None:
                if self.current_recommender_type == "content":
                    # El recomendador basado en contenido sí tiene update_user_profile
                    try:
                        self.recommender.update_user_profile(user_id, movie_id, rating)
                    except Exception as update_err:
                        st.warning(
                            f"⚠️ El recomendador colaborativo/híbrido no actualiza perfiles en tiempo real"
                        )

            # Actualizar en el registry del usuario (válido para todos)
            user_data = self.user_registry_manager.get_user(user_id)
            watched_movies = (
                user_data.get("preferences", {}).get("watched_movies", [])
                if user_data
                else []
            )
            if not isinstance(watched_movies, list):
                watched_movies = []

            # Asegurar que todos los IDs son enteros
            watched_movies = [int(m) for m in watched_movies]

            # Agregar la película si no está ya
            if movie_id not in watched_movies:
                watched_movies.append(movie_id)

            # Actualizar en el registry del usuario
            ratings_count = len(watched_movies)
            self.user_registry_manager.update_user_activity(
                user_id, new_rating_count=ratings_count, watched_movies=watched_movies
            )
            return True
        except Exception as e:
            error_msg = str(e)
            st.error(f"❌ Error actualizando rating: {error_msg}")
            traceback.print_exc()
            return False

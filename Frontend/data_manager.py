import json
import pandas as pd
from pathlib import Path
import streamlit as st

import requests


class DataManager:
    """
    Gestor de datos para cargar y manejar películas, géneros y posters
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

    def get_genre_names(self, genre_ids):
        """Convertir IDs de géneros a nombres"""
        if isinstance(genre_ids, list):
            return [self.genre_mapping.get(gid, f"Género {gid}") for gid in genre_ids]
        return []

    def obtener_datos_pelicula(self, nombre_pelicula):
        params = {
            "api_key": st.secrets["TMDB_API_KEY"],
            "query": nombre_pelicula,
            "language": "es-ES",  # Para que el título y sinopsis estén en español
        }

        response = requests.get(self.base_poster_path, params=params)
        datos = response.json()

        # Verificamos si hay resultados
        if datos["results"]:
            # Tomamos el primer resultado (el más relevante)
            return datos["results"][0]
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

    def render_poster(self, movie_id, movie_data, width=None, use_container=False):
        """Renderizar poster de forma centralizada y consistente.
        Args:
            movie_id: ID de la película
            movie_data: datos de la película
            width: ancho en píxeles (None = usar todo el contenedor)
            use_container: si True, usa use_container_width
        Returns:
            True si se encontró poster real, False si es placeholder
        """
        url, is_placeholder = self.get_poster_url(movie_id, movie_data)
        try:
            if use_container:
                st.image(url, use_container_width=True)
            elif width:
                st.image(url, width=width)
            else:
                st.image(url, use_container_width=True)
            return not is_placeholder
        except Exception:
            st.image(
                "https://via.placeholder.com/300x450/1a1a2e/e0e0e0?text=Error",
                use_container_width=True if use_container else False,
                width=None if use_container else (width or 300),
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

import streamlit as st
import sys
import json
import pandas as pd
from pathlib import Path
import os

# Agregar el directorio backend al path
sys.path.append(str(Path(__file__).parent.parent / "Backend"))

from user_registry_manager import UserRegistryManager


class NetflixApp:
    """
    Aplicación principal de recomendación de películas con interfaz de login
    """

    def __init__(self):
        """Inicializar la aplicación"""
        self.user_manager = UserRegistryManager()
        self.data_path = Path(__file__).parent.parent / "Data" / "Clean_data"
        self.raw_data_path = Path(__file__).parent.parent / "Data" / "Raw_data"
        self.posters_path = Path(__file__).parent.parent / "Data" / "posters"

        # Cargar datos de películas
        self.movies = self._load_movies()

        # Mapeo de géneros
        self.genre_mapping = self._load_genre_mapping()

    def _load_movies(self):
        """Cargar datos de películas"""
        try:
            with open(
                self.data_path / "cleaned_movies.json", "r", encoding="utf-8"
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

    def _get_genre_names(self, genre_ids):
        """Convertir IDs de géneros a nombres"""
        if isinstance(genre_ids, list):
            return [self.genre_mapping.get(gid, f"Género {gid}") for gid in genre_ids]
        return []

    def _get_poster_path(self, movie_id, movie_data):
        """Buscar el path del poster local para una película"""
        try:
            # Obtener el ID original de la película desde los datos
            original_id = movie_data.get("imdb_id", movie_id)
            # Asegurar que sea string para consistencia
            original_id_str = str(original_id)
            # Remover prefijo "tt" y ceros iniciales
            original_id_str = original_id_str.replace("tt", "").lstrip("0")

            print(
                f"Buscando poster para película ID: {movie_id} (original ID: {original_id_str})"
            )

            # Intentar con el ID original primero
            for poster_file in self.posters_path.glob(f"*_{original_id_str}.jpg"):
                print(f"Poster encontrado con original_id: {poster_file}")
                return str(poster_file)

            print(f"No se encontró poster")
            return None
        except Exception as e:
            st.error(f"Error buscando poster para película {movie_id}: {e}")
            return None

    def login_interface(self):
        """Interfaz de inicio de sesión"""
        st.title("🎬 Sistema de Recomendación de Películas")
        st.markdown("---")

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.subheader("🔑 Iniciar Sesión")

            # Opciones de login
            login_option = st.radio(
                "Selecciona una opción:",
                ["👤 Usuario Existente", "➕ Nuevo Usuario"],
                horizontal=True,
            )

            if login_option == "👤 Usuario Existente":
                self._existing_user_login()
            else:
                self._new_user_registration()

    def _existing_user_login(self):
        """Login para usuarios existentes"""
        st.markdown("### Acceso para usuarios registrados")

        # Obtener lista de usuarios activos
        active_users = self._get_active_users()

        if not active_users:
            st.warning("No hay usuarios registrados. Por favor, crea un nuevo usuario.")
            return

        # Crear opciones de selección
        user_options = {}
        for user_id, user_data in active_users.items():
            if user_data.get("name"):
                display_name = f"ID: {user_id} - {user_data['name']}"
            else:
                display_name = f"Usuario ID: {user_id}"
            user_options[display_name] = user_id

        selected_user_display = st.selectbox(
            "Selecciona tu usuario:", options=list(user_options.keys())
        )

        if st.button("🚀 Iniciar Sesión", type="primary", width="stretch"):
            selected_user_id = user_options[selected_user_display]
            st.session_state.logged_in = True
            st.session_state.user_id = selected_user_id
            st.session_state.user_data = active_users[selected_user_id]
            st.success(f"¡Bienvenido de vuelta!")
            st.rerun()

    def _new_user_registration(self):
        """Registro de nuevo usuario"""
        st.markdown("### Crear nueva cuenta")

        user_name = st.text_input(
            "Nombre de usuario (opcional):", placeholder="Ej: Juan Pérez"
        )

        # Géneros favoritos opcionales
        available_genres = self._get_available_genres()

        if available_genres:
            favorite_genres = st.multiselect(
                "Géneros favoritos (opcional):",
                options=available_genres,
                help="Selecciona tus géneros favoritos para mejores recomendaciones",
            )
        else:
            favorite_genres = []

        if st.button("✨ Crear Usuario", type="primary", width="stretch"):
            try:
                # Crear preferencias del usuario
                user_preferences = {
                    "favorite_genres": favorite_genres,
                    "watched_movies": [],
                }

                # Crear usuario
                new_user_id = self.user_manager.create_new_user(
                    user_preferences=user_preferences,
                    user_name=user_name if user_name.strip() else None,
                )

                # Iniciar sesión automáticamente
                st.session_state.logged_in = True
                st.session_state.user_id = new_user_id
                st.session_state.user_data = self.user_manager.get_user(new_user_id)

                st.success(f"¡Usuario creado con ID: {new_user_id}!")
                st.balloons()
                st.rerun()

            except Exception as e:
                st.error(f"Error al crear usuario: {e}")

    def _get_active_users(self):
        """Obtener usuarios activos"""
        users = self.user_manager.users_data.get("users", {})
        return {k: v for k, v in users.items() if v.get("status") == "active"}

    def _get_available_genres(self):
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

    def main_app(self):
        """Aplicación principal después del login"""
        # Sidebar con info del usuario
        with st.sidebar:
            st.markdown("### 👤 Usuario Actual")
            user_data = st.session_state.user_data

            if user_data.get("name"):
                st.write(f"**Nombre:** {user_data['name']}")
            st.write(f"**ID:** {st.session_state.user_id}")
            st.write(f"**Tipo:** {user_data.get('user_type', 'N/A')}")
            st.write(f"**Ratings:** {user_data.get('total_ratings', 0)}")

            if user_data.get("preferences", {}).get("favorite_genres"):
                st.write("**Géneros favoritos:**")
                for genre in user_data["preferences"]["favorite_genres"]:
                    st.write(f"• {genre}")

            st.markdown("---")

            if st.button("🚪 Cerrar Sesión", width="stretch"):
                # Limpiar sesión
                for key in ["logged_in", "user_id", "user_data"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

        # Contenido principal
        st.title("🎬 Recomendador de Películas")

        # Pestañas principales
        tab1, tab2, tab3 = st.tabs(["🔍 Explorar", "⭐ Mis Ratings", "📊 Estadísticas"])

        with tab1:
            self._movies_browser()

        with tab2:
            self._user_ratings()

        with tab3:
            self._user_statistics()

    def _movies_browser(self):
        """Navegador de películas"""
        st.subheader("Explorar Películas")

        # Filtros
        col1, col2 = st.columns(2)

        with col1:
            search_term = st.text_input(
                "🔍 Buscar película:", placeholder="Nombre de la película..."
            )

        with col2:
            genre_filter = st.selectbox(
                "🎭 Filtrar por género:",
                options=["Todos"] + self._get_available_genres(),
            )

        # Mostrar películas
        filtered_movies = self._filter_movies(search_term, genre_filter)

        if filtered_movies:
            st.write(f"**{len(filtered_movies)} películas encontradas**")

            # Mostrar primeras 10 películas
            count = 0
            for movie_id, movie_data in filtered_movies.items():
                if count >= 10:  # Limitar a 10 para mejor rendimiento
                    break

                with st.expander(
                    f"🎬 {movie_data.get('titulo', 'Sin título')} (ID: {movie_id})"
                ):
                    col1, col2 = st.columns([3, 2])

                    with col1:
                        st.write(
                            f"**Título:** {movie_data.get('titulo', 'Sin título')}"
                        )

                        # Géneros
                        genre_names = self._get_genre_names(
                            movie_data.get("generos", [])
                        )
                        if genre_names:
                            st.write(f"**Géneros:** {', '.join(genre_names)}")

                        # Keywords
                        keywords = movie_data.get("keywords", [])
                        if keywords:
                            st.write(
                                f"**Keywords:** {', '.join(keywords[:5])}"
                            )  # Solo mostrar 5

                        # Puntuación TMDB
                        if "puntuacion_media" in movie_data:
                            st.write(
                                f"**Puntuación TMDB:** {movie_data['puntuacion_media']}/10"
                            )

                        # Rating del usuario
                        user_rating = st.slider(
                            "Tu calificación:",
                            min_value=0.5,
                            max_value=5.0,
                            step=0.5,
                            value=2.5,
                            key=f"rating_{movie_id}",
                        )

                        if st.button(f"💾 Guardar Rating", key=f"save_{movie_id}"):
                            st.success(f"Rating guardado: {user_rating} ⭐")

                    with col2:
                        # Poster de la película
                        local_poster_path = self._get_poster_path(movie_id, movie_data)
                        if local_poster_path:
                            try:
                                st.image(
                                    local_poster_path,
                                    caption=f"Poster: {movie_data.get('titulo', 'Sin título')}",
                                    width="content",
                                )
                            except Exception as e:
                                st.warning(f"Error cargando poster: {e}")
                                st.write(f"**Poster path:** {local_poster_path}")
                        else:
                            st.write("🎬 *Sin poster disponible*")

                        # Estadísticas de la película
                        rating_stats = movie_data.get("rating_stats", {})
                        if rating_stats:
                            st.metric(
                                "Ratings Usuarios", rating_stats.get("num_ratings", 0)
                            )
                            st.metric(
                                "Promedio Usuarios",
                                f"{rating_stats.get('avg_rating', 0):.1f}⭐",
                            )

                count += 1
        else:
            st.info("No se encontraron películas con los filtros seleccionados.")

    def _filter_movies(self, search_term, genre_filter):
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
                genre_names = self._get_genre_names(genre_ids)
                if genre_filter not in genre_names:
                    continue

            filtered[movie_id] = movie_data

        return filtered

    def _user_ratings(self):
        """Sección de ratings del usuario"""
        st.subheader("Mis Calificaciones")

        user_data = st.session_state.user_data
        watched_movies = user_data.get("preferences", {}).get("watched_movies", [])

        if watched_movies:
            st.write(f"Has calificado {len(watched_movies)} películas:")

            for movie_id in watched_movies[:10]:  # Mostrar solo 10
                if str(movie_id) in self.movies:
                    movie = self.movies[str(movie_id)]
                    st.write(f"• {movie.get('titulo', f'Película {movie_id}')}")
        else:
            st.info(
                "Aún no has calificado ninguna película. ¡Ve a la pestaña 'Explorar' para empezar!"
            )

    def _user_statistics(self):
        """Estadísticas del usuario"""
        st.subheader("Mis Estadísticas")

        user_data = st.session_state.user_data

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total de Ratings", user_data.get("total_ratings", 0))

        with col2:
            avg_rating = user_data.get("statistics", {}).get("avg_rating", 0)
            st.metric("Rating Promedio", f"{avg_rating:.1f}⭐")

        with col3:
            st.metric("Tipo de Usuario", user_data.get("user_type", "N/A"))

        # Géneros favoritos
        fav_genres = user_data.get("preferences", {}).get("favorite_genres", [])
        if fav_genres:
            st.subheader("🎭 Tus Géneros Favoritos")
            for genre in fav_genres:
                st.write(f"• {genre}")

        # Información adicional
        st.subheader("📈 Información General")
        st.write(f"**Fecha de registro:** {user_data.get('created_date', 'N/A')}")
        st.write(f"**Última actividad:** {user_data.get('last_activity', 'N/A')}")
        st.write(f"**Total de películas disponibles:** {len(self.movies):,}")

    def run(self):
        """Ejecutar la aplicación"""
        # Configuración de la página
        st.set_page_config(
            page_title="Recomendador de Películas",
            page_icon="🎬",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # CSS personalizado
        st.markdown(
            """
        <style>
        .main-header {
            text-align: center;
            padding: 1rem;
            background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .stButton > button {
            width: 100%;
            margin-top: 1rem;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # Lógica de autenticación
        if not hasattr(st.session_state, "logged_in") or not st.session_state.logged_in:
            self.login_interface()
        else:
            self.main_app()


def main():
    """Función principal"""
    app = NetflixApp()
    app.run()


if __name__ == "__main__":
    main()

import streamlit as st


class AppManager:
    """
    Gestor de la aplicación principal después del login
    """

    # Número de columnas según ancho estimado
    GRID_COLS_DESKTOP = 4
    GRID_COLS_TABLET = 3
    GRID_COLS_MOBILE = 2

    def __init__(self, data_manager, auth_manager):
        """Inicializar el gestor de la aplicación"""
        self.data_manager = data_manager
        self.auth_manager = auth_manager

    def render_main_app(self):
        """Aplicación principal después del login"""
        # Sidebar con info del usuario
        self._render_sidebar()

        # Dialog de detalles (se muestra si hay película seleccionada)
        if "selected_movie_id" in st.session_state:
            self._render_movie_detail_dialog()

        # Contenido principal
        st.title("🎬 Recomendador de Películas")

        # Pestañas principales
        tab1, tab2, tab3 = st.tabs(["🔍 Explorar", "⭐ Mis Ratings", "📊 Estadísticas"])

        with tab1:
            self._render_movies_browser()

        with tab2:
            self._render_user_ratings()

        with tab3:
            self._render_user_statistics()

    def _render_sidebar(self):
        """Renderizar sidebar con información del usuario"""
        with st.sidebar:
            st.markdown("### 👤 Usuario Actual")
            user_data = self.auth_manager.get_current_user_data()

            if user_data.get("name"):
                st.write(f"**Nombre:** {user_data['name']}")
            st.write(f"**ID:** {self.auth_manager.get_current_user_id()}")
            st.write(f"**Tipo:** {user_data.get('user_type', 'N/A')}")
            st.write(f"**Ratings:** {user_data.get('total_ratings', 0)}")

            if user_data.get("preferences", {}).get("favorite_genres"):
                st.write("**Géneros favoritos:**")
                for genre in user_data["preferences"]["favorite_genres"]:
                    st.write(f"• {genre}")

            st.markdown("---")

            if st.button("🚪 Cerrar Sesión", width="stretch"):
                self.auth_manager.logout()

    def _render_movies_browser(self):
        """Navegador de películas con grid responsivo"""
        st.subheader("Explorar Películas")

        # Filtros
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

        with col1:
            search_term = st.text_input(
                "🔍 Buscar película:", placeholder="Nombre de la película..."
            )

        with col2:
            genre_filter = st.selectbox(
                "🎭 Filtrar por género:",
                options=["Todos"] + self.data_manager.get_available_genres(),
            )

        with col3:
            movies_per_page = st.selectbox(
                "Películas por página:", options=[12, 24, 48], index=0
            )

        with col4:
            cols_per_row = st.selectbox(
                "Columnas:",
                options=[2, 3, 4, 5, 6],
                index=2,
                help="Ajusta las columnas del grid según tu pantalla",
            )

        # Mostrar películas en grid
        filtered_movies = self.data_manager.filter_movies(search_term, genre_filter)

        if filtered_movies:
            st.write(f"**{len(filtered_movies)} películas encontradas**")

            # Convertir a lista para paginación
            movies_list = list(filtered_movies.items())

            # Paginación
            total_pages = (len(movies_list) + movies_per_page - 1) // movies_per_page

            if total_pages > 1:
                page = (
                    st.selectbox(
                        f"Página (Total: {total_pages}):",
                        options=list(range(1, total_pages + 1)),
                        index=0,
                    )
                    - 1
                )
            else:
                page = 0

            # Calcular índices para la página actual
            start_idx = page * movies_per_page
            end_idx = min(start_idx + movies_per_page, len(movies_list))
            page_movies = movies_list[start_idx:end_idx]

            # Renderizar grid con columnas seleccionadas
            self._render_movies_grid(page_movies, cols_per_row)

        else:
            st.info("No se encontraron películas con los filtros seleccionados.")

    def _render_movies_grid(self, movies_list, cols_per_row=4):
        """Renderizar grid de películas con columnas configurables"""
        # Crear filas del grid con gap uniforme
        for i in range(0, len(movies_list), cols_per_row):
            cols = st.columns(cols_per_row, gap="medium")

            for j, col in enumerate(cols):
                if i + j < len(movies_list):
                    movie_id, movie_data = movies_list[i + j]

                    with col:
                        self._render_grid_movie_item(movie_id, movie_data)

    def _render_grid_movie_item(self, movie_id, movie_data):
        """Renderizar item individual del grid de películas"""
        with st.container():
            # Poster centralizado usando el método unificado
            self.data_manager.render_poster(movie_id, movie_data, use_container=True)

            # Título de la película
            title = movie_data.get("titulo", "Sin título")
            display_title = title[:37] + "..." if len(title) > 40 else title
            st.markdown(
                f"<p class='movie-card-title'>{display_title}</p>",
                unsafe_allow_html=True,
            )

            # Géneros
            genre_names = self.data_manager.get_genre_names(
                movie_data.get("generos", [])
            )
            if genre_names:
                genres_text = ", ".join(genre_names[:2])
                if len(genre_names) > 2:
                    genres_text += "..."
                st.markdown(
                    f"<p class='movie-card-genres'>🎭 {genres_text}</p>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<p class='movie-card-genres'>🎭 —</p>",
                    unsafe_allow_html=True,
                )

            # Botón mejorado para "Ver detalles"
            if st.button(
                "🎬 Detalles",
                key=f"details_{movie_id}",
                width="stretch",
            ):
                st.session_state.selected_movie_id = movie_id

    def _render_movie_detail_dialog(self):
        """Mostrar detalles de película en un dialog modal.
        Se invoca desde render_main_app si hay película seleccionada.
        """
        movie_id = st.session_state.get("selected_movie_id")
        if not movie_id:
            return

        movie_data = self.data_manager.movies.get(str(movie_id))
        if not movie_data:
            return

        @st.dialog(f"🎬 {movie_data.get('titulo', 'Sin título')}", width="large")
        def _detail_dialog():
            self._render_movie_card(movie_id, movie_data)

        _detail_dialog()
        # Limpiar selección después de mostrar el dialog
        del st.session_state.selected_movie_id

    def _render_movie_card(self, movie_id, movie_data):
        """Renderizar tarjeta de película individual dentro de un dialog"""
        col_poster, col_info = st.columns([1, 2])

        with col_poster:
            # Poster centralizado
            self.data_manager.render_poster(movie_id, movie_data, use_container=True)

            # Estadísticas de la película
            rating_stats = movie_data.get("rating_stats", {})
            if rating_stats:
                sub1, sub2 = st.columns(2)
                with sub1:
                    st.metric("Ratings", rating_stats.get("num_ratings", 0))
                with sub2:
                    st.metric(
                        "Promedio",
                        f"{rating_stats.get('avg_rating', 0):.1f}⭐",
                    )

        with col_info:
            # Géneros
            genre_names = self.data_manager.get_genre_names(
                movie_data.get("generos", [])
            )
            if genre_names:
                genre_pills = " ".join([f"`{g}`" for g in genre_names])
                st.markdown(f"**Géneros:** {genre_pills}")

            # Keywords
            keywords = movie_data.get("keywords", [])
            if keywords:
                kw_text = ", ".join(keywords[:5])
                st.markdown(f"**Keywords:** {kw_text}")

            # Puntuación TMDB
            if "puntuacion_media" in movie_data:
                st.markdown(
                    f"**Puntuación TMDB:** ⭐ {movie_data['puntuacion_media']}/10"
                )

            st.markdown("---")

            # Rating del usuario
            st.markdown("**Tu calificación:**")
            user_rating = st.slider(
                "Calificación",
                min_value=0.5,
                max_value=5.0,
                step=0.5,
                value=2.5,
                key=f"rating_{movie_id}",
                label_visibility="collapsed",
            )

            # Mostrar estrellas visuales
            full_stars = int(user_rating)
            half_star = user_rating - full_stars >= 0.5
            stars_display = "⭐" * full_stars + ("✨" if half_star else "")
            st.markdown(f"### {stars_display} {user_rating}/5.0")

            if st.button(
                "💾 Guardar Rating",
                key=f"save_{movie_id}",
                width="stretch",
                type="primary",
            ):
                st.success(f"¡Rating guardado: {user_rating} ⭐!")

    def _render_user_ratings(self):
        """Sección de ratings del usuario"""
        st.subheader("Mis Calificaciones")

        user_data = self.auth_manager.get_current_user_data()
        watched_movies = user_data.get("preferences", {}).get("watched_movies", [])

        if watched_movies:
            st.write(f"Has calificado {len(watched_movies)} películas:")

            for movie_id in watched_movies[:10]:  # Mostrar solo 10
                if str(movie_id) in self.data_manager.movies:
                    movie = self.data_manager.movies[str(movie_id)]
                    st.write(f"• {movie.get('titulo', f'Película {movie_id}')}")
        else:
            st.info(
                "Aún no has calificado ninguna película. ¡Ve a la pestaña 'Explorar' para empezar!"
            )

    def _render_user_statistics(self):
        """Estadísticas del usuario"""
        st.subheader("Mis Estadísticas")

        user_data = self.auth_manager.get_current_user_data()

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
        st.write(
            f"**Total de películas disponibles:** {self.data_manager.get_movie_count():,}"
        )

    def render_custom_css(self):
        """Renderizar CSS personalizado"""
        st.markdown(
            """
        <style>
        /* ===== Layout general ===== */
        .main-header {
            text-align: center;
            padding: 1rem;
            background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
            border-radius: 10px;
            margin-bottom: 2rem;
        }

        /* ===== Tarjetas del grid ===== */
        /* Cada columna del grid de Streamlit */
        [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
            display: flex;
            flex-direction: column;
        }

        /* Títulos de tarjetas */
        .movie-card-title {
            font-weight: 700;
            font-size: 0.9rem;
            line-height: 1.25;
            margin: 6px 0 2px 0;
            min-height: 2.5em;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }

        /* Géneros en tarjetas */
        .movie-card-genres {
            color: #999;
            font-size: 0.78rem;
            margin: 0 0 4px 0;
            min-height: 1.3em;
        }

        /* Posters: aspecto consistente */
        [data-testid="stImage"] img {
            border-radius: 8px;
            object-fit: cover;
            aspect-ratio: 2/3;
            width: 100%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.12);
            transition: box-shadow 0.2s ease, transform 0.2s ease;
        }

        [data-testid="stImage"] img:hover {
            box-shadow: 0 6px 20px rgba(0,0,0,0.25);
            transform: translateY(-2px);
        }

        /* Botones del grid */
        .stButton > button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }

        /* ===== Dialog de detalles ===== */
        [data-testid="stDialog"] {
            border-radius: 16px;
        }

        [data-testid="stDialog"] [data-testid="stImage"] img {
            border-radius: 12px;
        }

        /* ===== Responsive ===== */
        @media (max-width: 768px) {
            .movie-card-title {
                font-size: 0.8rem;
            }
            .movie-card-genres {
                font-size: 0.7rem;
            }
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

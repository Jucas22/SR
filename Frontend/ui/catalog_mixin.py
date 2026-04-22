import streamlit as st


class CatalogMixin:
    def _render_movies_browser(self):
        """Navegador de peliculas con grid responsivo."""
        st.subheader("Explorar Peliculas")

        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

        with col1:
            search_term = st.text_input(
                "Buscar pelicula:",
                placeholder="Nombre de la pelicula...",
            )

        with col2:
            genre_filter = st.selectbox(
                "Filtrar por genero:",
                options=["Todos"] + self.data_manager.get_available_genres(),
            )

        with col3:
            movies_per_page = st.selectbox(
                "Peliculas por pagina:",
                options=[12, 24, 48],
                index=0,
            )

        with col4:
            cols_per_row = st.selectbox(
                "Columnas:",
                options=[2, 3, 4, 5, 6],
                index=2,
                help="Ajusta las columnas del grid segun tu pantalla",
            )

        filtered_movies = self.data_manager.filter_movies(search_term, genre_filter)

        if not filtered_movies:
            st.info("No se encontraron peliculas con los filtros seleccionados.")
            return

        st.write(f"**{len(filtered_movies)} peliculas encontradas**")

        movies_list = list(filtered_movies.items())
        total_pages = (len(movies_list) + movies_per_page - 1) // movies_per_page

        if total_pages > 1:
            page = (
                st.selectbox(
                    f"Pagina (Total: {total_pages}):",
                    options=list(range(1, total_pages + 1)),
                    index=0,
                )
                - 1
            )
        else:
            page = 0

        start_idx = page * movies_per_page
        end_idx = min(start_idx + movies_per_page, len(movies_list))
        page_movies = movies_list[start_idx:end_idx]
        self._render_movies_grid(page_movies, cols_per_row)

    def _render_movies_grid(self, movies_list, cols_per_row=4):
        """Renderizar grid de peliculas con columnas configurables."""
        for i in range(0, len(movies_list), cols_per_row):
            cols = st.columns(cols_per_row, gap="medium")
            for j, col in enumerate(cols):
                if i + j >= len(movies_list):
                    continue
                movie_id, movie_data = movies_list[i + j]
                with col:
                    self._render_grid_movie_item(movie_id, movie_data)

    def _render_grid_movie_item(self, movie_id, movie_data):
        """Renderizar un item individual del grid de peliculas."""
        with st.container():
            self.data_manager.render_poster(movie_id, movie_data, use_container=True)

            title = movie_data.get("titulo", "Sin titulo")
            display_title = title[:37] + "..." if len(title) > 40 else title
            st.markdown(
                f"<p class='movie-card-title'>{display_title}</p>",
                unsafe_allow_html=True,
            )

            genre_names = self.data_manager.get_genre_names(
                movie_data.get("generos", [])
            )
            if genre_names:
                genres_text = ", ".join(genre_names[:2])
                if len(genre_names) > 2:
                    genres_text += "..."
                st.markdown(
                    f"<p class='movie-card-genres'>{genres_text}</p>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<p class='movie-card-genres'>-</p>",
                    unsafe_allow_html=True,
                )

            if st.button(
                "Detalles",
                key=f"details_{movie_id}",
                width="stretch",
            ):
                st.session_state.selected_movie_id = movie_id

    def _render_movie_detail_dialog(self):
        """Mostrar detalles de una pelicula en un dialog modal."""
        movie_id = st.session_state.get("selected_movie_id")
        if not movie_id:
            return

        movie_data = self.data_manager.movies.get(str(movie_id))
        if not movie_data:
            return

        @st.dialog(movie_data.get("titulo", "Sin titulo"), width="large")
        def _detail_dialog():
            self._render_movie_card(movie_id, movie_data)

        _detail_dialog()
        del st.session_state.selected_movie_id

    def _render_movie_card(self, movie_id, movie_data):
        """Renderizar la ficha detallada de una pelicula."""
        col_poster, col_info = st.columns([1, 2])

        with col_poster:
            self.data_manager.render_poster(movie_id, movie_data, use_container=True)

            rating_stats = movie_data.get("rating_stats", {})
            if rating_stats:
                sub1, sub2 = st.columns(2)
                with sub1:
                    st.metric("Ratings", rating_stats.get("num_ratings", 0))
                with sub2:
                    st.metric(
                        "Promedio",
                        f"{rating_stats.get('avg_rating', 0):.1f}*",
                    )

        with col_info:
            genre_names = self.data_manager.get_genre_names(
                movie_data.get("generos", [])
            )
            if genre_names:
                genre_pills = " ".join([f"`{g}`" for g in genre_names])
                st.markdown(f"**Generos:** {genre_pills}")

            keywords = movie_data.get("keywords", [])
            if keywords:
                st.markdown(f"**Keywords:** {', '.join(keywords[:5])}")

            if "puntuacion_media" in movie_data:
                st.markdown(
                    f"**Puntuacion TMDB:** {movie_data['puntuacion_media']}/10"
                )

            st.markdown("---")
            st.markdown("**Tu calificacion:**")
            user_rating = st.slider(
                "Calificacion",
                min_value=0.5,
                max_value=5.0,
                step=0.5,
                value=2.5,
                key=f"rating_{movie_id}",
                label_visibility="collapsed",
            )

            full_stars = int(user_rating)
            half_star = user_rating - full_stars >= 0.5
            stars_display = "*" * full_stars + ("+" if half_star else "")
            st.markdown(f"### {stars_display} {user_rating}/5.0")

            if st.button(
                "Guardar Rating",
                key=f"save_{movie_id}",
                width="stretch",
                type="primary",
            ):
                user_id = self.auth_manager.get_current_user_id()
                try:
                    result = self.data_manager.update_user_rating(
                        user_id,
                        int(movie_id),
                        user_rating,
                    )
                    if result:
                        st.success(f"Rating guardado: {user_rating}")
                        st.info(
                            "Tu perfil ha sido actualizado. Las recomendaciones mejoraran con mas ratings."
                        )
                        updated_user = self.data_manager.user_registry_manager.get_user(
                            int(user_id)
                        )
                        if updated_user:
                            st.session_state.user_data = updated_user
                    else:
                        st.error("No se pudo guardar el rating.")
                except Exception as exc:
                    st.error(f"Error inesperado: {exc}")

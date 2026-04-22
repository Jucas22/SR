import streamlit as st


class UserMixin:
    def _render_sidebar(self):
        """Renderizar sidebar con informacion del usuario."""
        with st.sidebar:
            st.markdown("### Usuario Actual")
            user_data = self.auth_manager.get_current_user_data()

            if user_data.get("name"):
                st.write(f"**Nombre:** {user_data['name']}")
            st.write(f"**ID:** {self.auth_manager.get_current_user_id()}")
            st.write(f"**Tipo:** {user_data.get('user_type', 'N/A')}")
            st.write(f"**Ratings:** {user_data.get('total_ratings', 0)}")

            favorite_genres = user_data.get("preferences", {}).get("favorite_genres", [])
            if favorite_genres:
                st.write("**Generos favoritos:**")
                for genre in favorite_genres:
                    st.write(f"- {genre}")

            st.markdown("---")
            if st.button("Cerrar Sesion", width="stretch"):
                self.auth_manager.logout()

    def _render_user_ratings(self):
        """Seccion de ratings del usuario."""
        st.subheader("Mis Calificaciones")

        user_data = self.auth_manager.get_current_user_data()
        watched_movies = user_data.get("preferences", {}).get("watched_movies", [])

        if not watched_movies:
            st.info(
                "Aun no has calificado ninguna pelicula. Ve a la pestana 'Explorar' para empezar."
            )
            return

        st.write(f"Has calificado {len(watched_movies)} peliculas:")
        for movie_id in watched_movies[:10]:
            if str(movie_id) not in self.data_manager.movies:
                continue
            movie = self.data_manager.movies[str(movie_id)]
            st.write(f"- {movie.get('titulo', f'Pelicula {movie_id}')}")

    def _render_user_statistics(self):
        """Mostrar estadisticas del usuario."""
        st.subheader("Mis Estadisticas")

        user_data = self.auth_manager.get_current_user_data()
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total de Ratings", user_data.get("total_ratings", 0))

        with col2:
            avg_rating = user_data.get("statistics", {}).get("avg_rating", 0)
            st.metric("Rating Promedio", f"{avg_rating:.1f}")

        with col3:
            st.metric("Tipo de Usuario", user_data.get("user_type", "N/A"))

        favorite_genres = user_data.get("preferences", {}).get("favorite_genres", [])
        if favorite_genres:
            st.subheader("Tus Generos Favoritos")
            for genre in favorite_genres:
                st.write(f"- {genre}")

        st.subheader("Informacion General")
        st.write(f"**Fecha de registro:** {user_data.get('created_date', 'N/A')}")
        st.write(f"**Ultima actividad:** {user_data.get('last_activity', 'N/A')}")
        st.write(
            f"**Total de peliculas disponibles:** {self.data_manager.get_movie_count():,}"
        )

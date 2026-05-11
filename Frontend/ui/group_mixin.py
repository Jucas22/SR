import streamlit as st


class GroupMixin:
    def _render_group_recommendations(self):
        """Renderiza la interfaz de recomendaciones para grupos de usuarios."""
        st.subheader("Recomendaciones para Grupos")
        st.info(
            "Selecciona varios usuarios y una estrategia de agregacion "
            "para obtener recomendaciones que funcionen para todo el grupo."
        )

        # Inicializar el recomendador de grupo si todavia no esta listo
        if self.data_manager.group_recommender is None:
            self.data_manager.group_recommender = (
                self.data_manager._initialize_group_recommender()
            )

        if self.data_manager.group_recommender is None:
            st.warning("El recomendador de grupo no esta disponible en este momento.")
            return

        # Cargar usuarios activos
        users = self.data_manager.user_registry_manager.users_data.get("users", {})
        user_options = {
            f"Usuario {uid}": uid
            for uid, data in users.items()
            if data.get("status") == "active"
        }

        col1, col2 = st.columns(2)

        with col1:
            selected_labels = st.multiselect(
                "Miembros del grupo:",
                options=list(user_options.keys()),
            )
            selected_users = [user_options[label] for label in selected_labels]

        with col2:
            strategy = st.selectbox(
                "Estrategia de agregacion:",
                options=["average", "least_misery", "most_pleasure"],
                format_func=lambda x: {
                    "average": "Media (equilibrio)",
                    "least_misery": "Least Misery (nadie sufre)",
                    "most_pleasure": "Most Pleasure (alguien disfruta)",
                }.get(x, x),
            )
            top_n = st.slider("Numero de recomendaciones:", min_value=5, max_value=20, value=10)

        if not selected_users:
            st.info("Selecciona al menos un usuario para generar recomendaciones.")
            return

        if st.button("Recomendar para el grupo", type="primary"):
            with st.spinner("Calculando recomendaciones..."):
                results = self.data_manager.group_recommender.recommend(
                    user_ids=selected_users,
                    top_n=top_n,
                    strategy=strategy,
                )

            if not results:
                st.warning("No se encontraron recomendaciones para este grupo.")
                return

            st.success(f"{len(results)} peliculas recomendadas para el grupo")

            movies_to_show = []
            for movie_id, score in results:
                if str(movie_id) not in self.data_manager.movies:
                    continue
                movie_data = dict(self.data_manager.movies[str(movie_id)])
                movie_data["_recommender_type"] = "group"
                movie_data["_score"] = min(score / 5.0, 1.0)
                movies_to_show.append((movie_id, movie_data))

            self._render_recommendations_grid(movies_to_show, cols_per_row=4)

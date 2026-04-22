import streamlit as st


class RecommendationsMixin:
    def _render_recommendations(self):
        """Mostrar recomendaciones personalizadas."""
        st.subheader("Peliculas Recomendadas para Ti")

        col1, col2 = st.columns([2, 1])
        with col1:
            recommender_type = st.selectbox(
                "Selecciona un sistema recomendador:",
                options=["content", "collaborative", "hybrid"],
                format_func=lambda x: {
                    "content": "Basado en Contenido",
                    "collaborative": "Colaborativo",
                    "hybrid": "Hibrido",
                }.get(x, x),
                key="recommender_type_selector",
            )

        with col2:
            st.write("")
            enable_recommender = st.checkbox(
                "Activar",
                value=True,
                key="enable_recommender",
            )

        if not enable_recommender:
            st.info(
                "Selecciona 'Activar' para generar recomendaciones con el sistema elegido."
            )
            return

        user_id = self.auth_manager.get_current_user_id()

        if recommender_type == "content":
            st.info(
                "**Basado en Contenido**: recomendaciones segun generos, palabras clave y caracteristicas de peliculas que te gustaron."
            )
        elif recommender_type == "collaborative":
            st.info(
                "**Colaborativo**: recomendaciones basadas en usuarios similares a ti y sus preferencias."
            )
        else:
            st.info(
                "**Hibrido**: combinacion de recomendaciones basadas en contenido y colaborativas."
            )

        self.data_manager.ensure_recommender_initialized(recommender_type)

        if self.data_manager.recommender is None:
            st.warning(
                "El recomendador no esta disponible en este momento. "
                "La app seguira funcionando, pero sin recomendaciones personalizadas."
            )
            st.info(
                "Puedes explorar manualmente en la seccion 'Explorar' y aun asi guardar tus ratings."
            )
            return

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            num_recommendations = st.slider(
                "Cuantas recomendaciones quieres?",
                min_value=5,
                max_value=20,
                value=10,
                step=5,
            )

        with col2:
            if recommender_type == "content":
                exclude_seen = st.checkbox("Excluir vistas", value=True)
            else:
                exclude_seen = False
                st.write("")

        with col3:
            st.write("")

        recommendations_df = self.data_manager.get_recommendations(
            user_id,
            top_k=num_recommendations,
            exclude_seen=exclude_seen,
            recommender_type=recommender_type,
        )

        if recommendations_df is None or recommendations_df.empty:
            st.info(
                "No hay suficientes datos para generar recomendaciones personalizadas. "
                "Comienza a calificar peliculas para obtener mejores recomendaciones."
            )
            st.info(
                "Ve a la seccion 'Explorar' y califica algunas peliculas que hayas visto."
            )
            return

        st.info(f"{len(recommendations_df)} peliculas recomendadas para ti")

        cols_per_row = st.selectbox(
            "Columnas:",
            options=[2, 3, 4, 5],
            index=2,
            key="rec_cols",
        )

        movies_to_show = []
        for _, rec in recommendations_df.iterrows():
            movie_id = int(rec["movie_id"])
            if str(movie_id) not in self.data_manager.movies:
                continue
            movie_data = dict(self.data_manager.movies[str(movie_id)])
            movie_data["_recommender_type"] = recommender_type
            movie_data["_score"] = rec.get("score")
            movie_data["_reasons"] = rec.get("reasons", [])
            if recommender_type == "collaborative":
                movie_data["_predicted_rating"] = rec.get("predicted_rating")
                movie_data["_confidence"] = rec.get("confidence")
                movie_data["_mean_similarity"] = rec.get("mean_similarity")
                movie_data["_num_contributors"] = rec.get("num_contributors")
                movie_data["_positive_contributors"] = rec.get("positive_contributors")
                movie_data["_neutral_contributors"] = rec.get("neutral_contributors")
                movie_data["_negative_contributors"] = rec.get("negative_contributors")
            elif recommender_type == "hybrid":
                movie_data["_alpha"] = rec.get("alpha")
                movie_data["_beta"] = rec.get("beta")
                movie_data["_content_score"] = rec.get("content_score")
                movie_data["_collaborative_score"] = rec.get("collaborative_score")
                movie_data["_appears_in_content"] = rec.get("appears_in_content")
                movie_data["_appears_in_collaborative"] = rec.get(
                    "appears_in_collaborative"
                )
            movies_to_show.append((movie_id, movie_data))

        if movies_to_show:
            self._render_recommendations_grid(movies_to_show, cols_per_row)

    def _render_recommendations_grid(self, movies_list, cols_per_row=4):
        """Renderizar grid de recomendaciones."""
        for i in range(0, len(movies_list), cols_per_row):
            cols = st.columns(cols_per_row, gap="medium")
            for j, col in enumerate(cols):
                if i + j >= len(movies_list):
                    continue
                movie_id, movie_data = movies_list[i + j]
                with col:
                    self._render_recommendation_card(movie_id, movie_data)

    def _render_recommendation_card(self, movie_id, movie_data):
        """Renderizar tarjeta de recomendacion individual."""
        with st.container(border=True):
            self.data_manager.render_poster(
                str(movie_id),
                movie_data,
                use_container=True,
            )

            title = movie_data.get("titulo", "Sin titulo")
            display_title = title[:35] + "..." if len(title) > 38 else title
            st.markdown(
                f"<p class='movie-card-title'>{display_title}</p>",
                unsafe_allow_html=True,
            )

            recommender_type = movie_data.get("_recommender_type")
            if recommender_type == "collaborative":
                metric_col_1, metric_col_2 = st.columns(2)
                with metric_col_1:
                    predicted_rating = movie_data.get("_predicted_rating")
                    if predicted_rating is not None:
                        st.metric("Rating Predicho", f"{float(predicted_rating):.2f}/5")
                with metric_col_2:
                    confidence = movie_data.get("_confidence")
                    if confidence is not None:
                        st.metric("Confianza", f"{float(confidence):.0%}")

                num_contributors = movie_data.get("_num_contributors")
                mean_similarity = movie_data.get("_mean_similarity")
                if num_contributors is not None and mean_similarity is not None:
                    st.caption(
                        "Soporte: "
                        f"{int(num_contributors)} vecinos, "
                        f"similitud media {float(mean_similarity):.0%}"
                    )
            elif recommender_type == "hybrid":
                score = movie_data.get("_score")
                if score is not None:
                    st.metric("Compatibilidad", f"{float(score):.1%}")

                alpha = movie_data.get("_alpha")
                beta = movie_data.get("_beta")
                if alpha is not None and beta is not None:
                    st.caption(
                        "Mezcla actual: "
                        f"contenido {float(alpha):.0%}, "
                        f"colaborativo {float(beta):.0%}"
                    )
            elif movie_data.get("_score") is not None:
                st.metric("Compatibilidad", f"{float(movie_data['_score']):.1%}")

            if "_reasons" in movie_data and movie_data["_reasons"]:
                with st.expander("Por que se recomienda"):
                    reason_limit = 3 if recommender_type == "hybrid" else 2
                    for reason in movie_data["_reasons"][:reason_limit]:
                        st.write(f"- {reason}")

            if st.button("Ver Detalles", key=f"rec_{movie_id}", width="stretch"):
                st.session_state.selected_movie_id = movie_id
                st.rerun()

    def _render_colaborative_recommendations_grid(
        self,
        user_id,
        recommendations,
        cols_per_row=4,
    ):
        """Renderizar grid de recomendaciones colaborativas."""
        movies_to_show = []
        for movie_id, predicted_rating in recommendations:
            if str(movie_id) not in self.data_manager.movies:
                continue
            movie_data = self.data_manager.movies[str(movie_id)]
            movie_data["_predicted_rating"] = predicted_rating
            movies_to_show.append((movie_id, movie_data))

        for i in range(0, len(movies_to_show), cols_per_row):
            cols = st.columns(cols_per_row, gap="medium")
            for j, col in enumerate(cols):
                if i + j >= len(movies_to_show):
                    continue
                movie_id, movie_data = movies_to_show[i + j]
                with col:
                    self._render_colaborative_recommendation_card(
                        user_id=user_id,
                        movie_id=movie_id,
                        movie_data=movie_data,
                    )

    def _render_colaborative_recommendation_card(self, user_id, movie_id, movie_data):
        """Renderizar tarjeta colaborativa con explicacion basada en vecinos."""
        with st.container(border=True):
            self.data_manager.render_poster(
                str(movie_id),
                movie_data,
                use_container=True,
            )

            title = movie_data.get("titulo", "Sin titulo")
            display_title = title[:35] + "..." if len(title) > 38 else title
            st.markdown(
                f"<p class='movie-card-title'>{display_title}</p>",
                unsafe_allow_html=True,
            )

            predicted_rating = movie_data.get("_predicted_rating")
            confidence = movie_data.get("_confidence")
            if predicted_rating is not None or confidence is not None:
                metric_col_1, metric_col_2 = st.columns(2)
                with metric_col_1:
                    if predicted_rating is not None:
                        st.metric("Rating Predicho", f"{float(predicted_rating):.2f}/5")
                with metric_col_2:
                    if confidence is not None:
                        st.metric("Confianza", f"{float(confidence):.0%}")

            explanation = self.data_manager.get_colaborative_recommendation_explanation(
                user_id=user_id,
                movie_id=movie_id,
            )
            contributors = explanation.get("contributors", []) if explanation else []
            with st.expander("Por que se recomienda"):
                if explanation:
                    confidence = explanation.get("confidence")
                    num_contributors = explanation.get("num_contributors")
                    mean_similarity = explanation.get("mean_similarity")
                    (
                        positive_contributors,
                        neutral_contributors,
                        negative_contributors,
                        breakdown_consistent,
                    ) = self._resolve_contributor_breakdown(explanation)
                    if confidence is not None:
                        confidence_value = float(confidence)
                        st.write(
                            "- Confianza de la recomendacion: "
                            f"{self._format_confidence_label(confidence_value)} "
                            f"({confidence_value:.0%})"
                        )
                    if num_contributors is not None and mean_similarity is not None:
                        st.write(
                            "- Evidencia colaborativa: "
                            f"{int(num_contributors)} usuarios parecidos han visto esta pelicula, "
                            f"con una similitud media del {float(mean_similarity):.0%}."
                        )
                        if breakdown_consistent:
                            st.write(
                                "- Como la valoraron esos usuarios: "
                                f"{int(positive_contributors)} positivamente, "
                                f"{int(neutral_contributors)} de forma tibia y "
                                f"{int(negative_contributors)} negativamente."
                            )
                        else:
                            st.write(
                                "- El desglose exacto de valoraciones no estaba disponible, "
                                "pero si se ha confirmado que esos usuarios contribuyeron a la recomendacion."
                            )
                if contributors:
                    st.write("Vecinos que tambien han valorado esta pelicula:")
                    for contributor in contributors[:5]:
                        sentiment_label = self._format_rating_sentiment(
                            float(contributor["rating"])
                        )
                        st.write(
                            f"- Usuario {contributor['neighbor_id']} | "
                            f"Similitud: {contributor['similarity']:.0%} | "
                            f"Rating: {contributor['rating']:.1f} | "
                            f"{sentiment_label}"
                        )
                else:
                    st.write(
                        "No hay vecinos contribuyentes disponibles para esta pelicula."
                    )

            if st.button(
                "Ver Detalles",
                key=f"colab_rec_{movie_id}",
                width="stretch",
            ):
                st.session_state.selected_movie_id = movie_id
                st.rerun()

    def _format_confidence_label(self, confidence: float) -> str:
        """Convierte la confianza numerica en un texto mas natural."""
        if confidence >= 0.70:
            return "Alta"
        if confidence >= 0.45:
            return "Media"
        return "Baja"

    def _format_rating_sentiment(self, rating: float) -> str:
        """Resume si a un vecino le gusto, no le gusto o le dejo tibio."""
        if rating >= 3.5:
            return "Le gusto"
        if rating <= 2.5:
            return "No le gusto"
        return "Opinion tibia"

    def _resolve_contributor_breakdown(self, explanation):
        """
        Asegura que el desglose positivo/tibio/negativo sea coherente.
        Si los contadores no cuadran, intenta reconstruirlos desde contributors.
        """
        num_contributors = int(explanation.get("num_contributors", 0) or 0)
        positive_contributors = int(explanation.get("positive_contributors", 0) or 0)
        neutral_contributors = int(explanation.get("neutral_contributors", 0) or 0)
        negative_contributors = int(explanation.get("negative_contributors", 0) or 0)

        total = (
            positive_contributors + neutral_contributors + negative_contributors
        )
        if total == num_contributors:
            return (
                positive_contributors,
                neutral_contributors,
                negative_contributors,
                True,
            )

        contributors = explanation.get("contributors", []) or []
        if contributors:
            positive_contributors = 0
            neutral_contributors = 0
            negative_contributors = 0
            for contributor in contributors:
                rating = float(contributor["rating"])
                if rating >= 3.5:
                    positive_contributors += 1
                elif rating <= 2.5:
                    negative_contributors += 1
                else:
                    neutral_contributors += 1

            total = (
                positive_contributors + neutral_contributors + negative_contributors
            )
            if total == num_contributors:
                return (
                    positive_contributors,
                    neutral_contributors,
                    negative_contributors,
                    True,
                )

        return 0, num_contributors, 0, False

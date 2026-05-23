from typing import Any, Dict, List, Set, Tuple


class GroupRecommender:
    """
    Recomendador para grupos de usuarios.

    Genera recomendaciones agregando los scores individuales de cada miembro
    del grupo segun una de las siguientes estrategias:

    - average:       media de los scores de todos los miembros (los ausentes cuentan como 0)
    - least_misery:  minimo score del grupo (los ausentes cuentan como 0)
    - most_pleasure: maximo score del grupo (alguien disfruta)
    """

    STRATEGIES = ["average", "least_misery", "most_pleasure"]

    def __init__(self, recommender):
        self.recommender = recommender

    def recommend(
        self,
        user_ids: List[str],
        top_n: int = 10,
        strategy: str = "average",
    ) -> List[Tuple[int, float]]:
        """
        Genera recomendaciones para un grupo de usuarios.

        Args:
            user_ids: lista de IDs de los miembros del grupo
            top_n:    numero de recomendaciones a devolver
            strategy: estrategia de agregacion ('average', 'least_misery', 'most_pleasure')

        Returns:
            Lista de (movie_id, score) ordenada de mayor a menor score
        """
        if not user_ids:
            return []

        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Estrategia desconocida: '{strategy}'. "
                f"Opciones validas: {self.STRATEGIES}"
            )

        group_seen_movies = self._get_group_seen_movies(user_ids)

        # Paso 1: recoger scores individuales por pelicula para cada miembro del grupo.
        # Se guarda un dict por usuario para poder asignar 0.0 a peliculas ausentes.
        user_scores: Dict[str, Dict[int, float]] = {}
        all_movie_ids: Set[int] = set()

        for user_id in user_ids:
            user_recs = self.recommender.recommend(user_id, top_n=200, return_details=True)
            user_scores[user_id] = {}
            for rec in user_recs:
                movie_id = int(rec["movie_id"])
                if movie_id in group_seen_movies:
                    continue
                # ranking_score combina rating predicho y confianza: mejor discriminacion que predicted_rating
                score = float(rec.get("ranking_score", rec.get("predicted_rating", 0.0)))
                user_scores[user_id][movie_id] = score
                all_movie_ids.add(movie_id)

        if not all_movie_ids:
            return []

        # Paso 2: agregar los scores segun la estrategia elegida.
        # Las peliculas ausentes en la lista de un usuario reciben score 0.0,
        # lo que penaliza correctamente a least_misery y average.
        n_users = len(user_ids)
        aggregated: Dict[int, float] = {}

        for movie_id in all_movie_ids:
            scores = [user_scores[uid].get(movie_id, 0.0) for uid in user_ids]

            if strategy == "average":
                aggregated[movie_id] = sum(scores) / n_users
            elif strategy == "least_misery":
                aggregated[movie_id] = min(scores)
            elif strategy == "most_pleasure":
                aggregated[movie_id] = max(scores)

        # Paso 3: ordenar por score agregado y devolver los top_n
        return sorted(aggregated.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def _get_group_seen_movies(self, user_ids: List[str]) -> Set[int]:
        """Devuelve las peliculas vistas por cualquier miembro del grupo."""
        registry = self._resolve_user_registry()
        users = registry.get("users", registry) if isinstance(registry, dict) else {}

        seen_movies: Set[int] = set()
        for user_id in user_ids:
            user_data = users.get(str(user_id), {}) if isinstance(users, dict) else {}
            preferences = user_data.get("preferences", {}) if isinstance(user_data, dict) else {}
            for movie_id in preferences.get("watched_movies", []):
                try:
                    seen_movies.add(int(movie_id))
                except (TypeError, ValueError):
                    continue

        return seen_movies

    def _resolve_user_registry(self) -> Dict[str, Any]:
        """Localiza el registro de usuarios aunque el recomendador base este envuelto."""
        if hasattr(self.recommender, "user_registry"):
            return getattr(self.recommender, "user_registry")
        if hasattr(self.recommender, "collaborative_model"):
            collaborative_model = getattr(self.recommender, "collaborative_model")
            if hasattr(collaborative_model, "user_registry"):
                return getattr(collaborative_model, "user_registry")
        if hasattr(self.recommender, "content_model"):
            content_model = getattr(self.recommender, "content_model")
            if hasattr(content_model, "user_registry"):
                return getattr(content_model, "user_registry")
        return {}

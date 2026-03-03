"""
Ejemplo de integración del sistema de gestión de usuarios
con el sistema de recomendaciones.
"""

import json
import pandas as pd
from Backend.user_registry_manager import UserRegistryManager
from typing import List, Dict, Tuple, Optional


class RecommendationSystemWithUserManagement:
    """
    Ejemplo de sistema de recomendaciones integrado con gestión de usuarios.
    """

    def __init__(self):
        """Inicializar sistema con gestión de usuarios."""
        self.user_manager = UserRegistryManager()
        self.movies_data = self._load_movies_data()
        self.train_ratings = self._load_train_ratings()

        print("🎬 Sistema de recomendaciones iniciado")
        print(f"📊 {len(self.movies_data)} películas disponibles")
        print(f"👥 {len(self.user_manager.get_active_users())} usuarios registrados")

    def _load_movies_data(self) -> Dict:
        """Cargar datos de películas."""
        try:
            with open(
                "Data_set/Clean_data/peliculas_clean.json", "r", encoding="utf-8"
            ) as f:
                movies = json.load(f)
            return {movie["id"]: movie for movie in movies}
        except FileNotFoundError:
            print("⚠️  Archivo de películas no encontrado")
            return {}

    def _load_train_ratings(self) -> pd.DataFrame:
        """Cargar ratings de entrenamiento."""
        try:
            return pd.read_csv("Data_set/Train/train_ratings.csv")
        except FileNotFoundError:
            print("⚠️  Archivo de ratings de entrenamiento no encontrado")
            return pd.DataFrame()

    def register_new_user(
        self, user_preferences: Dict, user_name: Optional[str] = None
    ) -> int:
        """
        Registrar un nuevo usuario en el sistema.

        Args:
            user_preferences: Preferencias del nuevo usuario
            user_name: Nombre opcional del usuario

        Returns:
            int: ID del nuevo usuario
        """
        # Validar preferencias básicas
        if "favorite_genres" not in user_preferences:
            user_preferences["favorite_genres"] = []
        if "watched_movies" not in user_preferences:
            user_preferences["watched_movies"] = []

        # Crear usuario
        new_user_id = self.user_manager.create_new_user(user_preferences, user_name)

        name_display = f" ({user_name})" if user_name else ""
        print(f"✅ Nuevo usuario registrado: ID {new_user_id}{name_display}")
        return new_user_id

    def add_user_rating(self, user_id: int, movie_id: int, rating: float) -> bool:
        """
        Agregar una nueva valoración de usuario.

        Args:
            user_id: ID del usuario
            movie_id: ID de la película
            rating: Valoración (1.0 - 5.0)

        Returns:
            bool: True si se agregó exitosamente
        """
        # Verificar que el usuario existe
        if not self.user_manager.user_exists(user_id):
            print(f"❌ Usuario {user_id} no encontrado")
            return False

        # Verificar que la película existe
        if movie_id not in self.movies_data:
            print(f"❌ Película {movie_id} no encontrada")
            return False

        # Agregar rating a datos de entrenamiento (simulación)
        # En un sistema real, esto se guardaría en una base de datos
        new_rating = pd.DataFrame(
            {"userId": [user_id], "movieId": [movie_id], "rating": [rating]}
        )

        # Actualizar estadísticas del usuario
        user = self.user_manager.get_user(user_id)
        current_watched = user["preferences"]["watched_movies"]

        if movie_id not in current_watched:
            current_watched.append(movie_id)

        # Recalcular estadísticas
        new_rating_count = user["statistics"]["rating_count"] + 1
        current_total = (
            user["statistics"]["avg_rating"] * user["statistics"]["rating_count"]
        )
        new_avg_rating = (current_total + rating) / new_rating_count

        # Actualizar usuario
        self.user_manager.update_user_activity(
            user_id=user_id,
            new_rating_count=new_rating_count,
            new_avg_rating=new_avg_rating,
            watched_movies=current_watched,
        )

        print(
            f"✅ Rating agregado: Usuario {user_id} valoró película {movie_id} con {rating}"
        )
        return True

    def get_user_profile(self, user_id: int) -> Optional[Dict]:
        """
        Obtener perfil completo de usuario para recomendaciones.

        Args:
            user_id: ID del usuario

        Returns:
            Dict con información completa del usuario
        """
        user = self.user_manager.get_user(user_id)
        if not user:
            return None

        # Enriquecer perfil con información adicional
        profile = user.copy()

        # Agregar información de películas vistas
        watched_movies_info = []
        for movie_id in user["preferences"]["watched_movies"]:
            if movie_id in self.movies_data:
                movie_info = self.movies_data[movie_id]
                watched_movies_info.append(
                    {
                        "id": movie_id,
                        "title": movie_info.get("title", "Unknown"),
                        "genres": [g["name"] for g in movie_info.get("genres", [])],
                    }
                )

        profile["watched_movies_details"] = watched_movies_info

        return profile

    def recommend_for_new_user(
        self, user_id: int, num_recommendations: int = 10
    ) -> List[Dict]:
        """
        Generar recomendaciones para un usuario nuevo (cold-start problem).

        Args:
            user_id: ID del usuario
            num_recommendations: Número de recomendaciones

        Returns:
            List de películas recomendadas
        """
        user = self.user_manager.get_user(user_id)
        if not user:
            return []

        # Para usuarios nuevos, usar géneros favoritos
        favorite_genres = user["preferences"]["favorite_genres"]
        watched_movies = set(user["preferences"]["watched_movies"])

        if not favorite_genres:
            # Si no hay géneros favoritos, recomendar películas populares
            return self._get_popular_movies(num_recommendations, watched_movies)

        # Buscar películas de géneros favoritos
        recommendations = []

        for movie_id, movie in self.movies_data.items():
            # Saltar películas ya vistas
            if movie_id in watched_movies:
                continue

            # Verificar si la película tiene géneros favoritos
            movie_genres = [g["name"] for g in movie.get("genres", [])]
            genre_match = any(genre in favorite_genres for genre in movie_genres)

            if genre_match:
                recommendations.append(
                    {
                        "movie_id": movie_id,
                        "title": movie.get("title", "Unknown"),
                        "genres": movie_genres,
                        "reason": f'Coincide con tus géneros favoritos: {", ".join(set(movie_genres) & set(favorite_genres))}',
                    }
                )

        # Ordenar por número de géneros coincidentes y tomar los mejores
        recommendations = sorted(
            recommendations,
            key=lambda x: len(set(x["genres"]) & set(favorite_genres)),
            reverse=True,
        )[:num_recommendations]

        return recommendations

    def recommend_for_existing_user(
        self, user_id: int, num_recommendations: int = 10
    ) -> List[Dict]:
        """
        Generar recomendaciones para usuario existente con historial.

        Args:
            user_id: ID del usuario
            num_recommendations: Número de recomendaciones

        Returns:
            List de películas recomendadas
        """
        user = self.user_manager.get_user(user_id)
        if not user:
            return []

        # Para usuarios existentes, usar datos de entrenamiento
        user_ratings = self.train_ratings[self.train_ratings["userId"] == user_id]
        watched_movies = set(user["preferences"]["watched_movies"])

        if user_ratings.empty:
            # Si no hay ratings en entrenamiento, tratar como usuario nuevo
            return self.recommend_for_new_user(user_id, num_recommendations)

        # Calcular preferencias basadas en ratings históricos
        genre_scores = self._calculate_genre_preferences_from_ratings(user_ratings)

        # Encontrar películas similares
        recommendations = self._find_similar_movies(
            genre_scores, watched_movies, num_recommendations
        )

        return recommendations

    def _get_popular_movies(
        self, num_recommendations: int, exclude_movies: set
    ) -> List[Dict]:
        """Obtener películas populares como fallback."""
        # Simplificación: tomar primeras películas disponibles
        recommendations = []

        for movie_id, movie in list(self.movies_data.items())[
            : num_recommendations * 2
        ]:
            if movie_id not in exclude_movies:
                recommendations.append(
                    {
                        "movie_id": movie_id,
                        "title": movie.get("title", "Unknown"),
                        "genres": [g["name"] for g in movie.get("genres", [])],
                        "reason": "Película popular recomendada",
                    }
                )

                if len(recommendations) >= num_recommendations:
                    break

        return recommendations

    def _calculate_genre_preferences_from_ratings(
        self, user_ratings: pd.DataFrame
    ) -> Dict[str, float]:
        """Calcular preferencias de género basadas en ratings."""
        genre_scores = {}

        for _, rating_row in user_ratings.iterrows():
            movie_id = rating_row["movieId"]
            rating = rating_row["rating"]

            if movie_id in self.movies_data:
                movie_genres = [
                    g["name"] for g in self.movies_data[movie_id].get("genres", [])
                ]

                for genre in movie_genres:
                    if genre not in genre_scores:
                        genre_scores[genre] = []
                    genre_scores[genre].append(rating)

        # Calcular promedios
        genre_averages = {}
        for genre, ratings in genre_scores.items():
            genre_averages[genre] = sum(ratings) / len(ratings)

        return genre_averages

    def _find_similar_movies(
        self,
        user_genre_preferences: Dict[str, float],
        exclude_movies: set,
        num_recommendations: int,
    ) -> List[Dict]:
        """Encontrar películas similares basadas en preferencias de género."""
        recommendations = []

        # Obtener géneros mejor valorados (rating >= 3.5)
        preferred_genres = [
            genre for genre, score in user_genre_preferences.items() if score >= 3.5
        ]

        for movie_id, movie in self.movies_data.items():
            if movie_id in exclude_movies:
                continue

            movie_genres = [g["name"] for g in movie.get("genres", [])]

            # Calcular score de coincidencia
            match_score = 0
            matching_genres = []

            for genre in movie_genres:
                if genre in preferred_genres:
                    match_score += user_genre_preferences.get(genre, 0)
                    matching_genres.append(genre)

            if match_score > 0:
                recommendations.append(
                    {
                        "movie_id": movie_id,
                        "title": movie.get("title", "Unknown"),
                        "genres": movie_genres,
                        "match_score": match_score,
                        "reason": f'Basado en tus gustos en: {", ".join(matching_genres)}',
                    }
                )

        # Ordenar por score y tomar los mejores
        recommendations = sorted(
            recommendations, key=lambda x: x["match_score"], reverse=True
        )
        return recommendations[:num_recommendations]

    def get_recommendations(
        self, user_id: int, num_recommendations: int = 10
    ) -> List[Dict]:
        """
        Método principal para obtener recomendaciones.

        Args:
            user_id: ID del usuario
            num_recommendations: Número de recomendaciones

        Returns:
            List de recomendaciones
        """
        user = self.user_manager.get_user(user_id)
        if not user:
            print(f"❌ Usuario {user_id} no encontrado")
            return []

        # Determinar tipo de recomendación según el tipo de usuario
        if user["user_type"] == "new" or user["statistics"]["rating_count"] < 5:
            recommendations = self.recommend_for_new_user(user_id, num_recommendations)
            print(f"🆕 Recomendaciones para usuario nuevo/con pocos datos")
        else:
            recommendations = self.recommend_for_existing_user(
                user_id, num_recommendations
            )
            print(f"👤 Recomendaciones personalizadas basadas en historial")

        print(
            f"✅ {len(recommendations)} recomendaciones generadas para usuario {user_id}"
        )
        return recommendations


# Ejemplo de uso
def demo_system():
    """Demostración del sistema integrado."""
    print("🎬 DEMO: Sistema de Recomendaciones con Gestión de Usuarios")
    print("=" * 60)

    # Inicializar sistema
    rs = RecommendationSystemWithUserManagement()

    # 1. Registrar nuevo usuario con nombre
    print("\n👤 1. Registrando nuevo usuario con nombre...")
    new_user_id = rs.register_new_user(
        {"favorite_genres": ["Action", "Sci-Fi", "Thriller"], "watched_movies": []},
        user_name="Ana García",
    )

    # 1.5. Registrar usuario anónimo (sin nombre)
    print("\n👤 1.5. Registrando usuario anónimo...")
    anonymous_user_id = rs.register_new_user(
        {"favorite_genres": ["Comedy", "Romance"], "watched_movies": []}
    )

    # 2. Obtener recomendaciones para usuario nuevo
    print(f"\n🎯 2. Generando recomendaciones para usuario nuevo {new_user_id}...")
    recommendations = rs.get_recommendations(new_user_id, 5)

    print("\nRecomendaciones:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']} ({', '.join(rec['genres'])}) - {rec['reason']}")

    # 3. Simular que el usuario ve una película y la valora
    if recommendations:
        print(f"\n⭐ 3. Usuario {new_user_id} valora películas...")
        first_movie = recommendations[0]
        rs.add_user_rating(new_user_id, first_movie["movie_id"], 4.5)

        second_movie = (
            recommendations[1] if len(recommendations) > 1 else recommendations[0]
        )
        rs.add_user_rating(new_user_id, second_movie["movie_id"], 3.0)

    # 4. Obtener nuevas recomendaciones actualizadas
    print(f"\n🔄 4. Nuevas recomendaciones después de valoraciones...")
    updated_recommendations = rs.get_recommendations(new_user_id, 5)

    print("\nRecomendaciones actualizadas:")
    for i, rec in enumerate(updated_recommendations, 1):
        print(f"{i}. {rec['title']} ({', '.join(rec['genres'])}) - {rec['reason']}")

    # 5. Mostrar perfil del usuario
    print(f"\n👤 5. Perfil actualizado del usuario {new_user_id}:")
    profile = rs.get_user_profile(new_user_id)
    if profile:
        print(f"   📊 Total de valoraciones: {profile['statistics']['rating_count']}")
        print(f"   ⭐ Rating promedio: {profile['statistics']['avg_rating']}")
        print(
            f"   🎭 Géneros favoritos: {', '.join(profile['preferences']['favorite_genres'])}"
        )
        print(f"   🎬 Películas vistas: {len(profile['watched_movies_details'])}")

    # 6. Demostrar funcionalidades de nombres
    print(f"\n👤 6. Demostrando gestión de nombres de usuarios...")

    # Mostrar nombres para display
    print(
        f"   📺 Usuario con nombre: {rs.user_manager.get_user_display_name(new_user_id)}"
    )
    print(
        f"   📺 Usuario anónimo: {rs.user_manager.get_user_display_name(anonymous_user_id)}"
    )

    # Añadir nombre al usuario anónimo
    print(f"\n   🏷️  Añadiendo nombre al usuario anónimo...")
    rs.user_manager.update_user_name(anonymous_user_id, "Carlos López")
    print(
        f"   📺 Usuario actualizado: {rs.user_manager.get_user_display_name(anonymous_user_id)}"
    )

    # Mostrar estadísticas de nombres
    users_with_names = rs.user_manager.get_users_with_names()
    users_without_names = rs.user_manager.get_users_without_names()

    print(f"\n   📊 Usuarios con nombres: {len(users_with_names)}")
    if users_with_names:
        for user in users_with_names[-3:]:  # Mostrar últimos 3
            print(
                f"      • {user['name']} (ID: {user['user_id']}, Tipo: {user['user_type']})"
            )

    print(f"   📊 Usuarios sin nombres: {len(users_without_names)}")
    if users_without_names:
        print(
            f"      • {len(users_without_names)} usuarios del dataset original sin nombre"
        )

    print("\n✅ Demo completada exitosamente!")


if __name__ == "__main__":
    demo_system()

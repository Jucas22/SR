import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from project_paths import ROOT_DIR, USER_REGISTRY_PATH


class UserRegistryManager:
    """
    Clase para gestionar el registro de usuarios del sistema de recomendaciones.
    Permite crear, actualizar, eliminar y consultar usuarios.
    """

    def __init__(self, registry_path: Optional[str] = None):
        """
        Inicializa el gestor de registro de usuarios.

        Args:
            registry_path: Ruta al archivo JSON de registro de usuarios. Si no se
                proporciona, se utilizará el archivo dentro del directorio
                `Data` relativo al proyecto.
        """
        if registry_path:
            candidate = Path(registry_path)
            if not candidate.is_absolute():
                candidate = ROOT_DIR / candidate
        else:
            candidate = USER_REGISTRY_PATH

        self.registry_path = candidate.resolve()
        self.users_data = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Carga el registro de usuarios desde el archivo JSON."""
        # Asegurar que el directorio del registro existe
        dirpath = self.registry_path.parent
        dirpath.mkdir(parents=True, exist_ok=True)

        if self.registry_path.exists():
            with self.registry_path.open("r", encoding="utf-8") as file:
                return json.load(file)
        else:
            # Crear estructura inicial si no existe
            return {
                "metadata": {
                    "last_user_id": 0,
                    "total_users": 0,
                    "active_users": 0,
                    "created_date": datetime.now().strftime("%Y-%m-%d"),
                    "last_updated": datetime.now().strftime("%Y-%m-%d"),
                },
                "users": {},
            }

    def _save_registry(self):
        """Guarda el registro de usuarios en el archivo JSON."""        # Asegurar que el directorio exista antes de escribir
        dirpath = self.registry_path.parent
        dirpath.mkdir(parents=True, exist_ok=True)
        self.users_data["metadata"]["last_updated"] = datetime.now().strftime(
            "%Y-%m-%d"
        )

        with self.registry_path.open("w", encoding="utf-8") as file:
            json.dump(self.users_data, file, indent=2, ensure_ascii=False)

    def reload_registry(self) -> Dict[str, Any]:
        """Recarga el registro desde disco para evitar trabajar con datos obsoletos."""
        self.users_data = self._load_registry()
        return self.users_data

    def create_new_user(
        self, user_preferences: Optional[Dict] = None, user_name: Optional[str] = None
    ) -> int:
        """
        Crea un nuevo usuario en el sistema.

        Args:
            user_preferences: Preferencias opcionales del usuario
            user_name: Nombre opcional del usuario

        Returns:
            int: ID del nuevo usuario creado
        """
        # Generar nuevo ID de usuario
        new_user_id = self.users_data["metadata"]["last_user_id"] + 1
        preferences = self._normalize_preferences(user_preferences)

        # Crear estructura del nuevo usuario
        new_user = {
            "user_id": new_user_id,
            "name": user_name,  # Campo opcional para nombre de usuario
            "status": "active",
            "created_date": datetime.now().strftime("%Y-%m-%d"),
            "last_activity": datetime.now().strftime("%Y-%m-%d"),
            "total_ratings": 0,
            "user_type": "new",
            "preferences": preferences,
            "statistics": {
                "avg_rating": 0.0,
                "rating_count": 0,
                "active_since": datetime.now().strftime("%Y-%m-%d"),
            },
        }

        # Agregar usuario al registro
        self.users_data["users"][str(new_user_id)] = new_user

        # Actualizar metadata
        self.users_data["metadata"]["last_user_id"] = new_user_id
        self.users_data["metadata"]["total_users"] += 1
        self.users_data["metadata"]["active_users"] += 1

        # Guardar cambios
        self._save_registry()

        name_display = f" ({user_name})" if user_name else ""
        print(f"✅ Nuevo usuario creado con ID: {new_user_id}{name_display}")
        return new_user_id

    def get_user(self, user_id: int) -> Optional[Dict]:
        """
        Obtiene la información de un usuario específico.

        Args:
            user_id: ID del usuario

        Returns:
            Dict con información del usuario o None si no existe
        """
        return self.users_data["users"].get(str(user_id))

    def user_exists(self, user_id: int) -> bool:
        """
        Verifica si un usuario existe en el registro.

        Args:
            user_id: ID del usuario

        Returns:
            bool: True si el usuario existe, False en caso contrario
        """
        return str(user_id) in self.users_data["users"]

    def update_user_activity(
        self,
        user_id: int,
        new_rating_count: Optional[int] = None,
        new_avg_rating: Optional[float] = None,
        watched_movies: Optional[List[int]] = None,
        genre_vector: Optional[Dict[str, float]] = None,
        clear_neighbors: bool = False,
    ):
        """
        Actualiza la actividad de un usuario.

        Args:
            user_id: ID del usuario
            new_rating_count: Nuevo número total de ratings
            new_avg_rating: Nueva calificación promedio
            watched_movies: Lista actualizada de películas vistas
        """
        user_str_id = str(user_id)

        if not self.user_exists(user_id):
            print(f"❌ Usuario {user_id} no encontrado")
            return False

        user = self.users_data["users"][user_str_id]
        user["last_activity"] = datetime.now().strftime("%Y-%m-%d")

        if new_rating_count is not None:
            user["total_ratings"] = new_rating_count
            user["statistics"]["rating_count"] = new_rating_count

        if new_avg_rating is not None:
            user["statistics"]["avg_rating"] = round(new_avg_rating, 2)

        if watched_movies is not None:
            user["preferences"]["watched_movies"] = watched_movies

        if genre_vector is not None:
            user["preferences"]["genre_vector"] = {
                str(genre): round(float(value), 2)
                for genre, value in genre_vector.items()
                if float(value) > 0
            }

        if clear_neighbors:
            user.pop("neighbors", None)

        self._save_registry()
        print(f"✅ Usuario {user_id} actualizado")
        return True

    def _normalize_preferences(self, preferences: Optional[Dict]) -> Dict[str, Any]:
        """Normaliza la estructura de preferencias al crear un usuario."""
        base_preferences = dict(preferences or {})
        favorite_genres = base_preferences.get("favorite_genres", [])
        watched_movies = base_preferences.get("watched_movies", [])
        genre_vector = base_preferences.get("genre_vector")

        if not isinstance(favorite_genres, list):
            favorite_genres = []
        if not isinstance(watched_movies, list):
            watched_movies = []
        if not isinstance(genre_vector, dict):
            genre_vector = self._build_initial_genre_vector(favorite_genres)

        base_preferences["favorite_genres"] = favorite_genres
        base_preferences["watched_movies"] = watched_movies
        base_preferences["genre_vector"] = genre_vector
        return base_preferences

    def _build_initial_genre_vector(self, favorite_genres: List[str]) -> Dict[str, float]:
        """Genera un perfil inicial simple a partir de los generos favoritos elegidos."""
        vector: Dict[str, float] = {}
        for genre in favorite_genres:
            genre_name = str(genre).strip()
            if genre_name:
                vector[genre_name] = 100.0
        return vector

    def update_user_name(self, user_id: int, new_name: Optional[str]) -> bool:
        """
        Actualiza el nombre de un usuario.

        Args:
            user_id: ID del usuario
            new_name: Nuevo nombre del usuario (None para eliminar el nombre)

        Returns:
            bool: True si se actualizó exitosamente
        """
        if not self.user_exists(user_id):
            print(f"❌ Usuario {user_id} no encontrado")
            return False

        user_str_id = str(user_id)
        old_name = self.users_data["users"][user_str_id].get("name")
        self.users_data["users"][user_str_id]["name"] = new_name
        self.users_data["users"][user_str_id][
            "last_activity"
        ] = datetime.now().strftime("%Y-%m-%d")

        self._save_registry()

        if old_name and new_name:
            print(
                f"✅ Nombre del usuario {user_id} cambiado de '{old_name}' a '{new_name}'"
            )
        elif new_name:
            print(f"✅ Nombre '{new_name}' añadido al usuario {user_id}")
        elif old_name:
            print(f"✅ Nombre '{old_name}' eliminado del usuario {user_id}")
        else:
            print(f"✅ Usuario {user_id} actualizado (sin cambios de nombre)")

        return True

    def deactivate_user(self, user_id: int):
        """
        Desactiva un usuario sin eliminarlo del registro.

        Args:
            user_id: ID del usuario a desactivar
        """
        if not self.user_exists(user_id):
            print(f"❌ Usuario {user_id} no encontrado")
            return False

        self.users_data["users"][str(user_id)]["status"] = "inactive"
        self.users_data["metadata"]["active_users"] -= 1

        self._save_registry()
        print(f"✅ Usuario {user_id} desactivado")
        return True

    def get_active_users(self) -> List[int]:
        """
        Obtiene una lista de todos los usuarios activos.

        Returns:
            List[int]: Lista de IDs de usuarios activos
        """
        active_users = []
        for user_id, user_data in self.users_data["users"].items():
            if user_data["status"] == "active":
                active_users.append(int(user_id))

        return sorted(active_users)

    def get_new_users(self) -> List[int]:
        """
        Obtiene una lista de usuarios nuevos (creados por el sistema).

        Returns:
            List[int]: Lista de IDs de usuarios nuevos
        """
        new_users = []
        for user_id, user_data in self.users_data["users"].items():
            if user_data["user_type"] == "new" and user_data["status"] == "active":
                new_users.append(int(user_id))

        return sorted(new_users)

    def get_existing_users(self) -> List[int]:
        """
        Obtiene una lista de usuarios existentes (del dataset original).

        Returns:
            List[int]: Lista de IDs de usuarios existentes
        """
        existing_users = []
        for user_id, user_data in self.users_data["users"].items():
            if user_data["user_type"] == "existing" and user_data["status"] == "active":
                existing_users.append(int(user_id))

        return sorted(existing_users)

    def get_users_with_names(self) -> List[Dict]:
        """
        Obtiene una lista de usuarios que tienen nombre asignado.

        Returns:
            List[Dict]: Lista de usuarios con nombre
        """
        users_with_names = []
        for user_id, user_data in self.users_data["users"].items():
            if user_data.get("name") and user_data["status"] == "active":
                users_with_names.append(
                    {
                        "user_id": int(user_id),
                        "name": user_data["name"],
                        "user_type": user_data["user_type"],
                    }
                )

        return sorted(users_with_names, key=lambda x: x["user_id"])

    def get_users_without_names(self) -> List[int]:
        """
        Obtiene una lista de IDs de usuarios que no tienen nombre asignado.

        Returns:
            List[int]: Lista de IDs de usuarios sin nombre
        """
        users_without_names = []
        for user_id, user_data in self.users_data["users"].items():
            if not user_data.get("name") and user_data["status"] == "active":
                users_without_names.append(int(user_id))

        return sorted(users_without_names)

    def get_user_display_name(self, user_id: int) -> str:
        """
        Obtiene el nombre para mostrar de un usuario (nombre si existe, sino ID).

        Args:
            user_id: ID del usuario

        Returns:
            str: Nombre para mostrar del usuario
        """
        user = self.get_user(user_id)
        if not user:
            return f"Usuario #{user_id} (no encontrado)"

        user_name = user.get("name")
        if user_name:
            return f"{user_name} (ID: {user_id})"
        else:
            return f"Usuario #{user_id}"

    def get_statistics(self) -> Dict:
        """
        Obtiene estadísticas generales del registro de usuarios.

        Returns:
            Dict: Estadísticas del registro
        """
        stats = self.users_data["metadata"].copy()

        # Recalcular estadísticas actuales
        active_count = len(
            [u for u in self.users_data["users"].values() if u["status"] == "active"]
        )
        new_user_count = len(
            [
                u
                for u in self.users_data["users"].values()
                if u["user_type"] == "new" and u["status"] == "active"
            ]
        )
        existing_user_count = len(
            [
                u
                for u in self.users_data["users"].values()
                if u["user_type"] == "existing" and u["status"] == "active"
            ]
        )

        # Calcular estadísticas de nombres
        users_with_names = len(self.get_users_with_names())
        users_without_names = len(self.get_users_without_names())

        stats.update(
            {
                "current_active_users": active_count,
                "new_users_created": new_user_count,
                "existing_users": existing_user_count,
                "users_with_names": users_with_names,
                "users_without_names": users_without_names,
            }
        )

        return stats

    def print_summary(self):
        """Imprime un resumen del registro de usuarios."""
        stats = self.get_statistics()

        print("\n" + "=" * 50)
        print("📊 RESUMEN DEL REGISTRO DE USUARIOS")
        print("=" * 50)
        print(f"📅 Fecha de creación: {stats['created_date']}")
        print(f"🔄 Última actualización: {stats['last_updated']}")
        print(f"👥 Total de usuarios registrados: {stats['total_users']}")
        print(f"✅ Usuarios activos: {stats['current_active_users']}")
        print(f"🆕 Usuarios nuevos creados: {stats['new_users_created']}")
        print(f"📚 Usuarios del dataset original: {stats['existing_users']}")
        print(f"👤 Usuarios con nombre: {stats['users_with_names']}")
        print(f"🏷️  Usuarios sin nombre: {stats['users_without_names']}")
        print(f"🆔 Último ID asignado: {stats['last_user_id']}")
        print("=" * 50)


# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar el gestor de usuarios
    user_manager = UserRegistryManager()

    # Mostrar resumen inicial
    user_manager.print_summary()

    # Ejemplo: Crear un nuevo usuario con nombre
    new_user_id = user_manager.create_new_user(
        {"favorite_genres": ["Action", "Sci-Fi"], "watched_movies": []},
        user_name="Juan Pérez",
    )

    # Ejemplo: Crear un usuario sin nombre
    anonymous_user_id = user_manager.create_new_user(
        {"favorite_genres": ["Comedy", "Drama"], "watched_movies": []}
    )

    # Verificar si un usuario existe
    print(f"\n¿Usuario {new_user_id} existe? {user_manager.user_exists(new_user_id)}")

    # Obtener información del usuario
    user_info = user_manager.get_user(new_user_id)
    print(f"Información del usuario {new_user_id}: {user_info}")

    # Ejemplo: Actualizar nombre de usuario
    user_manager.update_user_name(anonymous_user_id, "María García")

    # Mostrar usuarios con y sin nombres
    print(f"\n👤 Usuarios con nombres: {user_manager.get_users_with_names()}")
    print(f"🏷️  Usuarios sin nombres: {user_manager.get_users_without_names()}")

    # Mostrar nombres para display
    print(
        f"\n📺 Nombre para mostrar del usuario {new_user_id}: {user_manager.get_user_display_name(new_user_id)}"
    )
    print(
        f"📺 Nombre para mostrar del usuario {anonymous_user_id}: {user_manager.get_user_display_name(anonymous_user_id)}"
    )

    # Mostrar resumen final
    user_manager.print_summary()

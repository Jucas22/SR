"""
Script para poblar inicialmente el registro de usuarios con los usuarios existentes
del dataset original y preparar el sistema para nuevos usuarios.

IMPORTANTE: Solo usa datos de TRAIN para evitar fuga de información de test.
De esta forma se puede evaluar con test_ratings sin sesgo.
"""

import json
import sys
import os
from collections import defaultdict

# Añadir el directorio padre al path para importar Backend
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from Backend.user_registry_manager import UserRegistryManager
from datetime import datetime


def populate_existing_users():
    """
    Pobla el registro de usuarios SOLO con datos de train_ratings.json.
    No se usa ningún dato de test para evitar data leakage.
    """
    print("🚀 Iniciando población del registro de usuarios (SOLO TRAIN)...")

    # Cargar solo ratings de entrenamiento
    print("📖 Cargando train_ratings.json...")
    with open("Data/Clean_data/train_ratings.json", "r", encoding="utf-8") as f:
        train_ratings_list = json.load(f)

    print(f"   {len(train_ratings_list)} ratings de entrenamiento cargados")

    # Agrupar ratings por usuario
    ratings_by_user = defaultdict(list)
    for r in train_ratings_list:
        ratings_by_user[int(r["userId"])].append(r)

    print(f"📊 Procesando {len(ratings_by_user)} usuarios desde datos de train...")

    # Inicializar gestor de usuarios
    user_manager = UserRegistryManager()

    users_processed = 0

    for user_id, user_ratings in ratings_by_user.items():
        user_id_str = str(user_id)

        # Calcular estadísticas SOLO con datos de train
        avg_rating = sum(r["rating"] for r in user_ratings) / len(user_ratings)
        watched_movies = [int(r["movieId"]) for r in user_ratings]

        existing_user = {
            "user_id": user_id,
            "name": None,
            "status": "active",
            "created_date": "2024-01-01",
            "last_activity": "2024-12-31",
            "total_ratings": len(user_ratings),
            "user_type": "existing",
            "preferences": {
                "favorite_genres": [],
                "watched_movies": watched_movies,
            },
            "statistics": {
                "avg_rating": round(avg_rating, 2),
                "rating_count": len(user_ratings),
                "active_since": "2024-01-01",
            },
        }

        user_manager.users_data["users"][user_id_str] = existing_user
        users_processed += 1

        if users_processed % 100 == 0:
            print(f"✅ Procesados {users_processed} usuarios...")

    # Actualizar metadata
    max_user_id = max([int(uid) for uid in user_manager.users_data["users"].keys()])
    user_manager.users_data["metadata"] = {
        "last_user_id": max_user_id,
        "total_users": len(user_manager.users_data["users"]),
        "active_users": len(user_manager.users_data["users"]),
        "created_date": datetime.now().strftime("%Y-%m-%d"),
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
    }

    # Guardar registro poblado
    user_manager._save_registry()

    print(f"\n🎉 ¡Registro poblado exitosamente!")
    print(f"👥 Total de usuarios registrados: {len(user_manager.users_data['users'])}")
    print(f"🆔 Rango de IDs: 1 - {max_user_id}")
    print(f"📊 Próximo ID disponible para nuevos usuarios: {max_user_id + 1}")

    return user_manager


def main():
    """
    Función principal para ejecutar todo el proceso de inicialización.
    """
    print("🎬 INICIALIZANDO SISTEMA DE GESTIÓN DE USUARIOS")
    print("=" * 60)

    # Poblar usuarios existentes
    user_manager = populate_existing_users()

    # Mostrar resumen final
    print("\n" + "=" * 60)
    user_manager.print_summary()

    print("\n✅ ¡Sistema de usuarios inicializado correctamente!")
    print(f"📝 Archivo de registro: {user_manager.registry_path}")
    print(f"🔧 Usa UserRegistryManager para gestionar usuarios en tu aplicación")

    return user_manager


if __name__ == "__main__":
    # Ejecutar inicialización completa
    user_manager = main()

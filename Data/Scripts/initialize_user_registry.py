"""
Script para poblar inicialmente el registro de usuarios con los usuarios existentes
del dataset original y preparar el sistema para nuevos usuarios.
"""

import json
import pandas as pd
import sys
import os

# Añadir el directorio padre al path para importar Backend
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from Backend.user_registry_manager import UserRegistryManager
from datetime import datetime


def populate_existing_users():
    """
    Pobla el registro de usuarios con los usuarios existentes del dataset original.
    """
    print("🚀 Iniciando población del registro de usuarios...")

    # Cargar datos existentes
    print("📖 Cargando datos existentes...")

    # Cargar ratings por usuario (datos completos)
    with open("Data/Clean_data/ratings_por_usuario.json", "r", encoding="utf-8") as f:
        ratings_por_usuario = json.load(f)

    # Cargar análisis de usuarios (estadísticas)
    with open(
        "Data/train-test-info/user_train_test_analysis.json", "r", encoding="utf-8"
    ) as f:
        user_analysis = json.load(f)

    # Crear diccionario de análisis por ID para acceso rápido
    analysis_by_id = {str(user["user_id"]): user for user in user_analysis}

    # Inicializar gestor de usuarios
    user_manager = UserRegistryManager()

    print(f"📊 Procesando {len(ratings_por_usuario)} usuarios existentes...")

    users_processed = 0

    for user_id_str, user_ratings in ratings_por_usuario.items():
        user_id = int(user_id_str)

        # Obtener estadísticas del análisis
        user_stats = analysis_by_id.get(user_id_str, {})

        # Calcular promedio de ratings
        if user_ratings:
            avg_rating = sum([rating["rating"] for rating in user_ratings]) / len(
                user_ratings
            )
            watched_movies = [rating["movieId"] for rating in user_ratings]
        else:
            avg_rating = 0.0
            watched_movies = []

        # Crear estructura de usuario existente
        existing_user = {
            "user_id": user_id,
            "name": None,  # Usuarios del dataset original no tienen nombre
            "status": "active",
            "created_date": "2024-01-01",  # Fecha aproximada para usuarios del dataset
            "last_activity": "2024-12-31",  # Última actividad aproximada
            "total_ratings": user_stats.get("total_ratings", len(user_ratings)),
            "user_type": "existing",
            "preferences": {
                "favorite_genres": [],  # Se puede calcular posteriormente
                "watched_movies": watched_movies,
            },
            "statistics": {
                "avg_rating": round(avg_rating, 2),
                "rating_count": len(user_ratings),
                "active_since": "2024-01-01",
                "train_ratings": user_stats.get("train_ratings", 0),
                "test_ratings": user_stats.get("test_ratings", 0),
                "train_percentage": user_stats.get("train_percentage", 0),
                "test_percentage": user_stats.get("test_percentage", 0),
            },
        }

        # Agregar usuario directamente al registro
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

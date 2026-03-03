"""
Script de prueba para demostrar la nueva funcionalidad de nombres de usuario.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Backend.user_registry_manager import UserRegistryManager


def demo_user_names():
    """Demuestra las nuevas funcionalidades de nombres de usuario."""
    print("🎬 DEMO: Nuevas Funcionalidades de Nombres de Usuario")
    print("=" * 60)

    # Inicializar gestor
    user_manager = UserRegistryManager()

    print("📊 Estado inicial del sistema:")
    user_manager.print_summary()

    print("\n" + "=" * 60)
    print("1️⃣ CREANDO USUARIOS CON Y SIN NOMBRES")
    print("=" * 60)

    # Crear usuario con nombre
    user_with_name = user_manager.create_new_user(
        {"favorite_genres": ["Action", "Adventure", "Sci-Fi"], "watched_movies": []},
        user_name="María González",
    )

    # Crear usuario sin nombre
    user_without_name = user_manager.create_new_user(
        {"favorite_genres": ["Comedy", "Romance", "Drama"], "watched_movies": []}
    )

    # Crear otro usuario con nombre
    user_with_name2 = user_manager.create_new_user(
        {"favorite_genres": ["Horror", "Thriller"], "watched_movies": []},
        user_name="Juan Pérez",
    )

    print("\n" + "=" * 60)
    print("2️⃣ CONSULTANDO INFORMACIÓN DE USUARIOS")
    print("=" * 60)

    # Mostrar información de usuarios
    print(
        f"👤 Usuario {user_with_name}: {user_manager.get_user_display_name(user_with_name)}"
    )
    print(
        f"👤 Usuario {user_without_name}: {user_manager.get_user_display_name(user_without_name)}"
    )
    print(
        f"👤 Usuario {user_with_name2}: {user_manager.get_user_display_name(user_with_name2)}"
    )

    print("\n" + "=" * 60)
    print("3️⃣ ACTUALIZANDO NOMBRES DE USUARIOS")
    print("=" * 60)

    # Añadir nombre al usuario sin nombre
    print("🏷️ Añadiendo nombre al usuario anónimo...")
    user_manager.update_user_name(user_without_name, "Ana Rodríguez")

    # Cambiar nombre de usuario existente
    print("✏️ Cambiando nombre de usuario existente...")
    user_manager.update_user_name(user_with_name2, "Juan Carlos Pérez López")

    # Intentar eliminar nombre (establecer a None)
    print("🗑️ Eliminando nombre de usuario...")
    user_manager.update_user_name(user_with_name, None)

    print("\n📝 Nombres actualizados:")
    print(
        f"👤 Usuario {user_with_name}: {user_manager.get_user_display_name(user_with_name)}"
    )
    print(
        f"👤 Usuario {user_without_name}: {user_manager.get_user_display_name(user_without_name)}"
    )
    print(
        f"👤 Usuario {user_with_name2}: {user_manager.get_user_display_name(user_with_name2)}"
    )

    print("\n" + "=" * 60)
    print("4️⃣ ESTADÍSTICAS DE NOMBRES")
    print("=" * 60)

    # Obtener usuarios con nombres
    users_with_names = user_manager.get_users_with_names()
    users_without_names = user_manager.get_users_without_names()

    print(f"👥 Usuarios con nombres ({len(users_with_names)}):")
    for user in users_with_names:
        print(f"   • {user['name']} (ID: {user['user_id']}, Tipo: {user['user_type']})")

    print(f"\n🏷️ Usuarios sin nombres ({len(users_without_names)}):")
    if len(users_without_names) <= 10:
        for user_id in users_without_names:
            print(f"   • Usuario #{user_id}")
    else:
        print(
            f"   • {len(users_without_names)} usuarios (principalmente del dataset original)"
        )
        print(f"   • Últimos 5: {users_without_names[-5:]}")

    print("\n" + "=" * 60)
    print("5️⃣ ESTADO FINAL DEL SISTEMA")
    print("=" * 60)

    user_manager.print_summary()

    print("\n✅ Demo de nombres de usuario completado exitosamente!")
    print("💡 Los usuarios del dataset original no tienen nombre por defecto")
    print("💡 Los nuevos usuarios pueden crearse con o sin nombre")
    print("💡 Los nombres se pueden añadir, cambiar o eliminar en cualquier momento")


if __name__ == "__main__":
    demo_user_names()

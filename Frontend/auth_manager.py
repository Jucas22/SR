import streamlit as st
import sys
from pathlib import Path

# Agregar el directorio backend al path
sys.path.append(str(Path(__file__).parent.parent / "Backend"))

from user_registry_manager import UserRegistryManager


class AuthManager:
    """
    Gestor de autenticación y registro de usuarios
    """

    def __init__(self, data_manager):
        """Inicializar el gestor de autenticación"""
        self.user_manager = UserRegistryManager()
        self.data_manager = data_manager

    def render_login_interface(self):
        """Interfaz de inicio de sesión"""
        st.title("🎬 Sistema de Recomendación de Películas")
        st.markdown("---")

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.subheader("🔑 Iniciar Sesión")

            # Opciones de login
            login_option = st.radio(
                "Selecciona una opción:",
                ["👤 Usuario Existente", "➕ Nuevo Usuario"],
                horizontal=True,
            )

            if login_option == "👤 Usuario Existente":
                self._render_existing_user_login()
            else:
                self._render_new_user_registration()

    def _render_existing_user_login(self):
        """Login para usuarios existentes"""
        st.markdown("### Acceso para usuarios registrados")

        # Obtener lista de usuarios activos
        active_users = self._get_active_users()

        if not active_users:
            st.warning("No hay usuarios registrados. Por favor, crea un nuevo usuario.")
            return

        # Crear opciones de selección
        user_options = {}
        for user_id, user_data in active_users.items():
            if user_data.get("name"):
                display_name = f"ID: {user_id} - {user_data['name']}"
            else:
                display_name = f"Usuario ID: {user_id}"
            user_options[display_name] = user_id

        selected_user_display = st.selectbox(
            "Selecciona tu usuario:", options=list(user_options.keys())
        )

        if st.button("🚀 Iniciar Sesión", type="primary", use_container_width=True):
            selected_user_id = user_options[selected_user_display]
            st.session_state.logged_in = True
            st.session_state.user_id = selected_user_id
            st.session_state.user_data = active_users[selected_user_id]
            st.success(f"¡Bienvenido de vuelta!")
            st.rerun()

    def _render_new_user_registration(self):
        """Registro de nuevo usuario"""
        st.markdown("### Crear nueva cuenta")

        user_name = st.text_input(
            "Nombre de usuario (opcional):", placeholder="Ej: Juan Pérez"
        )

        # Géneros favoritos opcionales
        available_genres = self.data_manager.get_available_genres()

        if available_genres:
            favorite_genres = st.multiselect(
                "Géneros favoritos (opcional):",
                options=available_genres,
                help="Selecciona tus géneros favoritos para mejores recomendaciones",
            )
        else:
            favorite_genres = []

        if st.button("✨ Crear Usuario", type="primary", use_container_width=True):
            try:
                # Crear preferencias del usuario
                user_preferences = {
                    "favorite_genres": favorite_genres,
                    "watched_movies": [],
                }

                # Crear usuario
                new_user_id = self.user_manager.create_new_user(
                    user_preferences=user_preferences,
                    user_name=user_name if user_name.strip() else None,
                )

                # Iniciar sesión automáticamente
                st.session_state.logged_in = True
                st.session_state.user_id = new_user_id
                st.session_state.user_data = self.user_manager.get_user(new_user_id)

                st.success(f"¡Usuario creado con ID: {new_user_id}!")
                st.balloons()
                st.rerun()

            except Exception as e:
                st.error(f"Error al crear usuario: {e}")

    def _get_active_users(self):
        """Obtener usuarios activos"""
        users = self.user_manager.users_data.get("users", {})
        return {k: v for k, v in users.items() if v.get("status") == "active"}

    def logout(self):
        """Cerrar sesión del usuario actual"""
        # Limpiar sesión
        for key in ["logged_in", "user_id", "user_data"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    def is_logged_in(self):
        """Verificar si hay un usuario logueado"""
        return hasattr(st.session_state, "logged_in") and st.session_state.logged_in

    def get_current_user_data(self):
        """Obtener datos del usuario actual"""
        if self.is_logged_in():
            return st.session_state.user_data
        return None

    def get_current_user_id(self):
        """Obtener ID del usuario actual"""
        if self.is_logged_in():
            return st.session_state.user_id
        return None

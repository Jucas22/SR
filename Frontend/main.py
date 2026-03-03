import streamlit as st
from data_manager import DataManager
from auth_manager import AuthManager
from app_manager import AppManager


class NetflixApp:
    """
    Aplicación principal coordinadora de recomendación de películas
    """

    def __init__(self):
        """Inicializar la aplicación y sus gestores"""
        # Inicializar gestores
        self.data_manager = DataManager()
        self.auth_manager = AuthManager(self.data_manager)
        self.app_manager = AppManager(self.data_manager, self.auth_manager)

    def run(self):
        """Ejecutar la aplicación"""
        # Configuración de la página
        st.set_page_config(
            page_title="Recomendador de Películas",
            page_icon="🎬",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # CSS personalizado
        self.app_manager.render_custom_css()

        # Lógica de autenticación
        if not self.auth_manager.is_logged_in():
            self.auth_manager.render_login_interface()
        else:
            self.app_manager.render_main_app()


def main():
    """Función principal"""
    app = NetflixApp()
    app.run()


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Frontend.app_manager import AppManager
from Frontend.auth_manager import AuthManager
from Frontend.data_manager import DataManager


def get_session_data_manager():
    """Mantener una unica instancia del gestor de datos durante la sesion."""
    if "data_manager" not in st.session_state:
        st.session_state.data_manager = DataManager()
    return st.session_state.data_manager


class NetflixApp:
    """Aplicacion principal coordinadora de recomendacion de peliculas."""

    def __init__(self):
        self.data_manager = get_session_data_manager()
        self.auth_manager = AuthManager(self.data_manager)
        self.app_manager = AppManager(self.data_manager, self.auth_manager)

    def run(self):
        """Ejecutar la aplicacion."""
        self.app_manager.render_custom_css()

        if not self.auth_manager.is_logged_in():
            self.auth_manager.render_login_interface()
        else:
            self.app_manager.render_main_app()


def main():
    """Funcion principal."""
    st.set_page_config(
        page_title="Recomendador de Peliculas",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    app = NetflixApp()
    app.run()


if __name__ == "__main__":
    main()

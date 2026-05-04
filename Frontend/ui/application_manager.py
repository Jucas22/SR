import streamlit as st

from Frontend.ui.catalog_mixin import CatalogMixin
from Frontend.ui.group_mixin import GroupMixin
from Frontend.ui.recommendations_mixin import RecommendationsMixin
from Frontend.ui.styles import APP_STYLES
from Frontend.ui.user_mixin import UserMixin


class AppManager(UserMixin, RecommendationsMixin, CatalogMixin, GroupMixin):
    """Coordinador principal de la UI tras el login."""

    GRID_COLS_DESKTOP = 4
    GRID_COLS_TABLET = 3
    GRID_COLS_MOBILE = 2

    def __init__(self, data_manager, auth_manager):
        self.data_manager = data_manager
        self.auth_manager = auth_manager

    def render_main_app(self):
        self._render_sidebar()

        if "selected_movie_id" in st.session_state:
            self._render_movie_detail_dialog()

        st.title("Recomendador de Peliculas")
        section = self._render_section_navigation()

        if section == "Recomendadas":
            self._render_recommendations()
        elif section == "Explorar":
            self._render_movies_browser()
        elif section == "Mis Ratings":
            self._render_user_ratings()
        elif section == "Grupo":
            self._render_group_recommendations()
        else:
            self._render_user_statistics()

    def render_custom_css(self):
        st.markdown(APP_STYLES, unsafe_allow_html=True)

    def _render_section_navigation(self) -> str:
        """Renderiza una navegacion ligera sin evaluar secciones ocultas."""
        if "main_section" not in st.session_state:
            st.session_state.main_section = "Explorar"

        return st.radio(
            "Seccion principal",
            options=["Recomendadas", "Explorar", "Mis Ratings", "Grupo", "Estadisticas"],
            horizontal=True,
            key="main_section",
            label_visibility="collapsed",
        )

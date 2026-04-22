# Arquitectura del Proyecto

## Objetivo

Este documento resume como queda organizado el repositorio tras la reorganizacion y que responsabilidad tiene cada bloque.

## Capas principales

### `Backend/`

Contiene la logica del dominio:

- `recommenders/`: punto de acceso unificado a los algoritmos
- `services/user_registry.py`: gestion del registro de usuarios
- `Colaborativo/`: implementacion historica del colaborativo y cache de la matriz de preferencias

### `Frontend/`

Contiene la app Streamlit:

- `main.py`: punto de entrada
- `auth_manager.py`: login y alta de usuarios
- `services/app_data_manager.py`: fachada actual para carga de datos y acceso a recomendadores
- `ui/application_manager.py`: coordinador de la UI tras el login
- `ui/*_mixin.py`: separacion por responsabilidades visuales

### `scripts/`

Scripts operativos fuera de `Data/`:

- `scripts/data/`: pipeline de preparacion y enriquecimiento de datos
- `scripts/diagnostics/`: pruebas manuales y validaciones rapidas

### `Data/`

Solo datos y artefactos:

- `Raw_data/`: origen
- `Clean_data/`: datasets procesados
- `user_registry.json`: estado de usuarios

## Decisiones de reorganizacion

### 1. Introducir wrappers de compatibilidad

Se han dejado wrappers en rutas antiguas como:

- `Backend/content_recommender.py`
- `Backend/hybrid_recommender.py`
- `Backend/user_registry_manager.py`
- `Frontend/data_manager.py`
- `Frontend/app_manager.py`
- `Data/Scripts/*`

Esto permite seguir usando rutas antiguas mientras el repo converge hacia la nueva estructura.

### 2. Separar UI en modulos pequenos

El antiguo `Frontend/app_manager.py` acumulaba demasiadas responsabilidades. Ahora la UI esta dividida en:

- recomendaciones
- catalogo y detalle de peliculas
- informacion del usuario
- estilos

### 3. Centralizar rutas del proyecto

El archivo `project_paths.py` evita recomponer rutas relativas en cada modulo y reduce errores al mover archivos.

## Flujo principal en tiempo de ejecucion

1. `Frontend/main.py` arranca Streamlit.
2. `AuthManager` resuelve el usuario actual.
3. `AppManager` dibuja la UI principal.
4. `DataManager` carga catalogo, generos y recomendadores bajo demanda.
5. Los recomendadores leen:
   - `Data/Clean_data/enhanced_movies.json`
   - `Data/Clean_data/train_ratings.json`
   - `Data/user_registry.json`

## Puntos a mejorar despues de esta limpieza

- Reducir todavia mas el tamano del gestor legado de datos.
- Unificar nombres de campos entre ingles y espanol.
- Mover definitivamente la implementacion colaborativa a `Backend/recommenders/`.
- Incorporar tests automaticos reales en una carpeta `tests/`.

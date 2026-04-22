# Sistemas Recomendadores de Peliculas

Proyecto de la asignatura de Sistemas Recomendadores orientado a recomendar peliculas mediante tres enfoques:

- Recomendador basado en contenido
- Recomendador colaborativo
- Recomendador hibrido

El repositorio incluye un backend con la logica de recomendacion, un frontend en Streamlit para explorar peliculas y recibir recomendaciones, y varios scripts para preparar y enriquecer los datos.

## Estado actual

Se ha reorganizado el proyecto para separar mejor:

- `Backend/`: algoritmos y servicios del dominio
- `Frontend/`: interfaz Streamlit y gestores de UI
- `scripts/`: scripts operativos y pruebas manuales
- `Data/`: datasets y artefactos generados
- `docs/`: documentacion de arquitectura y algoritmos

Los antiguos puntos de entrada se mantienen como wrappers para no romper rutas previas.

## Estructura recomendada

```text
.
|- Backend/
|  |- recommenders/
|  |  |- content_based.py
|  |  |- collaborative.py
|  |  `- hybrid.py
|  |- services/
|  |  `- user_registry.py
|  `- Colaborativo/
|     `- preference_matrix.npz
|- Frontend/
|  |- services/
|  |  |- app_data_manager.py
|  |  `- data_manager.py
|  |- ui/
|  |  |- application_manager.py
|  |  |- recommendations_mixin.py
|  |  |- catalog_mixin.py
|  |  |- user_mixin.py
|  |  `- styles.py
|  |- auth_manager.py
|  `- main.py
|- scripts/
|  |- data/
|  `- diagnostics/
|- Data/
|  |- Raw_data/
|  |- Clean_data/
|  `- user_registry.json
`- docs/
   |- ARCHITECTURE.md
   `- ALGORITHMS.md
```

## Requisitos

Usa Python 3.11 o superior.

Instalacion recomendada desde la raiz del proyecto:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Nota:

- La carpeta `SR/` que aparece en el repo es un entorno virtual local ya existente.
- No forma parte del codigo fuente ni de la arquitectura del proyecto.

## Como ejecutar

### Frontend

```bash
streamlit run Frontend/main.py
```

### Diagnosticos manuales

Recomendador basado en contenido:

```bash
python scripts/diagnostics/content_recommender_smoke.py
```

Recomendador colaborativo:

```bash
python scripts/diagnostics/collaborative_recommender_smoke.py
```

Tambien se conservan los wrappers heredados:

```bash
python test_recommender.py
python Backend/Colaborativo/test_colaborativo.py
```

## Flujo de datos

### 1. Datos crudos

Los archivos base viven en `Data/Raw_data/`:

- `peliculas.csv`
- `ratings_small.csv`
- `links.csv`
- `keywords.csv`
- `generos.csv`

### 2. Procesado inicial

```bash
python scripts/data/process_data.py
```

Genera, entre otros:

- `Data/Clean_data/cleaned_movies.json`
- `Data/Clean_data/cleaned_ratings.json`
- `Data/Clean_data/train_ratings.json`
- `Data/Clean_data/test_ratings.json`

### 3. Registro de usuarios

```bash
python scripts/data/initialize_user_registry.py
```

Genera o actualiza:

- `Data/user_registry.json`

### 4. Vector de preferencias por genero

```bash
python scripts/data/initialize_user_vector.py
```

Actualiza en el registro:

- `favorite_genres`
- `genre_vector`
- estadisticas agregadas del usuario

### 5. Enriquecimiento de peliculas con TMDb

```bash
python scripts/data/enrich_movies_tmdb.py
```

Genera o actualiza:

- `Data/Clean_data/enhanced_movies.json`

## Resumen de algoritmos

### 1. Basado en contenido

Se construye un espacio de items usando:

- generos
- overview
- tagline
- keywords
- tags
- reparto y direccion cuando estan disponibles

El perfil del usuario combina:

- `genre_vector` del registro de usuario
- historial de ratings y peliculas vistas

La puntuacion final mezcla similitud por genero, similitud textual, calidad y popularidad.

### 2. Colaborativo

Se calcula una matriz usuario-genero a partir del `genre_vector` de cada usuario.

Despues:

- se buscan vecinos similares con correlacion de Pearson
- se recuperan items bien valorados por esos vecinos
- se predice el rating del usuario con una media ponderada por similitud

### 3. Hibrido

Combina la puntuacion del recomendador basado en contenido y la del colaborativo con pesos configurables.

## Documentacion adicional

- Arquitectura: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Algoritmos: [docs/ALGORITHMS.md](docs/ALGORITHMS.md)
- Basado en contenido en detalle: [docs/CONTENT_BASED_RECOMMENDER.md](docs/CONTENT_BASED_RECOMMENDER.md)
- Colaborativo en detalle: [docs/COLLABORATIVE_RECOMMENDER.md](docs/COLLABORATIVE_RECOMMENDER.md)
- Integracion previa: [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
- Historial de cambios: [CHANGELOG.md](CHANGELOG.md)

## Proximo paso recomendado

Con esta base ya mas ordenada, el siguiente paso natural es revisar:

- calidad de las implementaciones de los algoritmos
- consistencia de nombres y esquemas de datos
- como se explican y presentan las recomendaciones en el frontend

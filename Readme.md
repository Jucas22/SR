# Trabajo de la Asignatura de Sistemas Recomendadores.

## Primera parte: division del dataset

En esta primera parte debemos preparar el sistema para posteriormente empezar a implementar los sitemas recomendadores. Primero empezaremos generando el grupo de usuarios test y el grupo de train.

Cogemos todos los usuarios del sistema (fichero de Data_set/ratings_small.csv) y los dividimos en entranmiento y test, 70% y 30% respectivamente. De esta manera, tras realizar el sistema recomendador podremos evaluar su calidad utilizando el test, comprobando la recomendacion que nos ofrece el SR y el rating real que le ha dado el usuario.

Perfil del usuario:

- Demografia: NO SE TNDRÁ EN CUENTA EN ESTE TRABAJO
- Preferencias del usuario: genero de peliculas y la puntuacion de cada una de las pelis.
- Historico de interacciones: que le ha gustado, que ha visto, que no...
- Informacion interna del SR: que algoritmos se han usado, que resultados han dado, etc.

**Para el usuario de la DB:**
Se tendráe en cuenta la lista de peliculas puntuadas de cada usuario y la clasificacion de peliculas por genero de cada una de ellas. Con esto se va infiriendo a que grado le gusta cada genero, y se le pueden recomendar peliculas de ese genero. Con eso se puede obtener un vector de preferencias del usuario, con un valor para cada genero. Se puede hacer lo mismo con los actores, directores, etc.

TODO: decision de como rellenar el vector de preferencias del usuario, de la DB. Intentar tener un valor para cada uno de los generos, sin embargo, si tenemos todo rellenado puede haber un problema a la hora de recomendar pelculas. Por tanto, a la hora de recomendar en vez de utilizar todo el vector, se utilizará por ejemplo las mejores 5 generos posibles. De esta forma, habrá una mayor distintición entre generos.
Usar datos de training para rellenar el vector de preferencias del usuario, y luego usar ese vector para recomendar peliculas en el test. De esta forma, se puede evaluar la calidad de las recomendaciones.

**Para los usuarios que se registren**, habra que preguntarle direcatmente sus preferencias en el registro.

**Lista de recomendaciones: (futuras partes)** sera una lista de peliculas que el usuario no ha visto y se le recomienda ver, ordenada por probabilidad o por algun cirterio de relevancia. Esta lista se le pasará tal cual a la UI y esta ya mostrará las peliculas al usuario, con la informcaion que necesite.

TODO: Ya se pede empezar a hacer la interfaz para generar nuevo usuario

TODO: Generar las fichas de usuario de la DB (creo que esto con el archivo de usuarios ya esta perfe, pero hay que revisarlo)

### Lo que se ha realizado:

#### 1. **Limpieza y Transformación de Datos**

**Script:** `clean_and_transform_data.py`

**Proceso de limpieza:**

- ✅ **Eliminación de ratings huérfanos:** Se identificaron y eliminaron ratings que referencian películas inexistentes en el dataset de películas
- ✅ **Integración de keywords:** Se procesaron y añadieron las palabras clave de cada película desde `keywords.csv`
- ✅ **Normalización de datos:** Conversión a formatos JSON estructurados para facilitar el acceso y manipulación
- ✅ **Validación de integridad:** Verificación de consistencia entre los diferentes archivos CSV

#### 2. **División Train/Test (70%-30%)**

**Metodología:** División por usuario de sus ratings individuales (no separación completa de usuarios)

- **670 usuarios totales** aparecen en ambos conjuntos
- **Para cada usuario:** ~70% de sus ratings → entrenamiento, ~30% → evaluación
- **Usuarios con 1 solo rating:** Asignados completamente a entrenamiento
- **Resultado:** 22,114 ratings de entrenamiento + 9,902 ratings de test

**¿Por qué esta división?**

- Permite entrenar con conocimiento parcial de cada usuario
- Evalúa la capacidad de predecir ratings faltantes de usuarios conocidos
- Simula el escenario real de recomendación a usuarios existentes

#### 3. **Archivos Generados**

##### 📊 **Datasets Completos (para análisis general):**

- `peliculas_con_ratings.json` - Dataset integrado con películas, ratings y keywords
- `peliculas_clean.json` - Solo información de películas con keywords incluidas
- `ratings_por_pelicula.json` - Todos los ratings agrupados por película
- `ratings_por_usuario.json` - Todos los ratings agrupados por usuario
- `ratings_clean.csv` - Todos los ratings limpios en formato CSV
- `keywords_por_pelicula.json` - Keywords organizadas por ID de película
- `estadisticas_peliculas.json` - Estadísticas completas de ratings

##### 🎯 **Datasets de Entrenamiento (USAR SOLO ESTOS PARA ENTRENAR):**

- `peliculas_con_ratings_train.json` - Dataset integrado solo con datos de entrenamiento
- `train_ratings_por_pelicula.json` - Ratings de entrenamiento por película
- `train_ratings_por_usuario.json` - Ratings de entrenamiento por usuario
- `train_ratings.csv` - Ratings de entrenamiento en CSV
- `train_estadisticas_peliculas.json` - Estadísticas basadas solo en entrenamiento

##### 🧪 **Dataset de Evaluación:**

- `test_ratings.csv` - **Ratings para evaluar la precisión del sistema de recomendación**

##### 📋 **Análisis de División:**

- `user_train_test_analysis.json` - Análisis detallado por usuario de la división
- `user_ids_by_set.json` - IDs de usuarios organizados por conjunto

#### 4. **Cómo Usar los Datos para el Sistema de Recomendación**

##### **ENTRENAMIENTO (construir vectores de preferencias):**

```python
# Usar SOLO los datos de entrenamiento
train_ratings = pd.read_csv('Data_set/train_ratings.csv')
train_movies = json.load(open('Data_set/peliculas_con_ratings_train.json'))

# Construir vectores de preferencias por género usando solo train data
# Ejemplo: Para cada usuario, calcular preferencia por género basada en ratings de entrenamiento
```

##### **EVALUACIÓN (medir precisión):**

```python
# Usar datos de test para evaluar recomendaciones
test_ratings = pd.read_csv('Data_set/test_ratings.csv')

# Para cada rating en test:
# 1. Generar recomendación usando modelo entrenado
# 2. Comparar con rating real del usuario
# 3. Calcular métricas (MAE, RMSE, etc.)
```

#### 5. **Gestión de Usuarios del Sistema**

**Script:** `user_registry_manager.py`
**Archivo de registro:** `Data_set/user_registry.json`

**⚠️ IMPORTANTE:** Los archivos existentes (`ratings_por_usuario.json`, etc.) están optimizados para **análisis de datos y entrenamiento**, pero NO para **gestión dinámica de usuarios en producción**.

##### **¿Por qué necesitas un registro adicional de usuarios?**

Los archivos actuales tienen limitaciones para un sistema real:

- ❌ **Solo análisis:** Diseñados para analizar usuarios del dataset histórico
- ❌ **Sin metadata:** No tienen información de estado, fechas de creación, preferencias explícitas
- ❌ **No escalable:** No permite agregar nuevos usuarios fácilmente
- ❌ **Sin gestión:** No distinguish entre usuarios existentes vs. nuevos

##### **✅ Solución: Sistema de Registro de Usuarios**

**Funcionalidades del `UserRegistryManager`:**

- 🆔 **Gestión de IDs únicos:** Genera nuevos IDs de usuario automáticamente
- 👥 **Tipos de usuario:** Distingue entre usuarios originales del dataset y nuevos usuarios
- 📊 **Estadísticas:** Tracking de actividad, ratings promedio, películas vistas
- 🔄 **Estados:** Usuarios activos/inactivos
- 📅 **Metadata temporal:** Fechas de creación, última actividad
- ⚙️ **Preferencias:** Géneros favoritos, configuraciones personalizadas

##### **Estructura del registro:**

```json
{
  "metadata": {
    "last_user_id": 672,
    "total_users": 672,
    "active_users": 672,
    "created_date": "2026-03-03"
  },
  "users": {
    "1": {
      "user_id": 1,
      "status": "active",
      "user_type": "existing", // "existing" o "new"
      "total_ratings": 5,
      "preferences": {
        "favorite_genres": ["Action", "Sci-Fi"],
        "watched_movies": [1371, 1405, 2105]
      },
      "statistics": {
        "avg_rating": 2.3,
        "rating_count": 5
      }
    }
  }
}
```

##### **Uso del UserRegistryManager:**

```python
from user_registry_manager import UserRegistryManager

# Inicializar gestor
user_manager = UserRegistryManager()

# Crear nuevo usuario
new_user_id = user_manager.create_new_user({
    "favorite_genres": ["Action", "Comedy"],
    "watched_movies": []
})

# Verificar existencia
if user_manager.user_exists(user_id):
    user_info = user_manager.get_user(user_id)

# Actualizar actividad después de nuevo rating
user_manager.update_user_activity(
    user_id=673,
    new_rating_count=5,
    new_avg_rating=4.2,
    watched_movies=[1, 2, 3, 4, 5]
)

# Obtener estadísticas
stats = user_manager.get_statistics()
```

##### **Flujo de trabajo recomendado:**

1. **Para usuarios existentes (dataset original):**
   - Usar `train_ratings_por_usuario.json` para entrenar modelo
   - Registrar en `user_registry.json` con `user_type: "existing"`

2. **Para nuevos usuarios (registro en la app):**
   - Crear con `user_manager.create_new_user()`
   - Capturar preferencias explícitas en el registro
   - Usar sistema de recomendaciones para cold-start

3. **Para recomendaciones activas:**
   - Consultar `user_registry.json` para obtener perfil completo
   - Combinar con datos históricos de ratings
   - Actualizar actividad después de cada interacción

#### 6. **Estadísticas del Dataset Procesado**

- **Películas procesadas:** 27,841
- **Ratings válidos totales:** 32,016
- **Ratings de entrenamiento:** 22,114 (69.1%)
- **Ratings de test:** 9,902 (30.9%)
- **Usuarios únicos del dataset:** 670
- **Películas con keywords:** Variable según disponibilidad
- **Géneros únicos:** Multiple IDs de géneros disponibles

# 🎬 Sistema de Recomendación de Películas - Guía de Usuario

## 📋 Descripción

Aplicación web de recomendación de películas con interfaz de login simplificada (sin contraseña) desarrollada en Streamlit.

## 🚀 Cómo ejecutar la aplicación

### 1. Activar el entorno virtual

```bash
.\SR\Scripts\Activate.ps1
```

### 2. Ejecutar la aplicación

```bash
streamlit run .\Frontend\main.py
```

### 3. Acceder a la aplicación

La aplicación se abrirá automáticamente en tu navegador en:

- **Local:** http://localhost:8501
- **Red:** http://192.168.x.x:8501

## 🔑 Funcionalidades de Login

### 👤 Usuario Existente

- Selecciona un usuario de la lista de usuarios registrados
- Los usuarios aparecen con su ID y nombre (si lo tienen)
- Click en "Iniciar Sesión" para acceder

### ➕ Nuevo Usuario

- Introduce un nombre de usuario (opcional)
- Selecciona géneros favoritos para mejores recomendaciones
- Click en "Crear Usuario" para registrarte e iniciar sesión automáticamente

## 🎯 Funcionalidades Principales

### 🔍 Explorar Películas

- **Búsqueda:** Busca películas por título
- **Filtros:** Filtra por género usando el selector
- **Información de películas:**
  - Título, géneros, palabras clave
  - Puntuación TMDB
  - Estadísticas de ratings de usuarios
  - Sistema de calificación personal (0.5 - 5.0 estrellas)

### ⭐ Mis Ratings

- Visualiza las películas que has calificado
- Historial de tus valoraciones personales

### 📊 Estadísticas

- **Métricas personales:**
  - Total de ratings dados
  - Rating promedio personal
  - Tipo de usuario (nuevo/existente)
- **Información del perfil:**
  - Géneros favoritos seleccionados
  - Fecha de registro
  - Última actividad
  - Total de películas disponibles

## 👤 Panel de Usuario (Sidebar)

- **Información actual:** Nombre, ID, tipo, total de ratings
- **Géneros favoritos:** Lista de géneros preferidos
- **Cerrar sesión:** Botón para salir de la cuenta

## 🎨 Características Técnicas

### 🗃️ Integración de Datos

- **Películas:** Carga desde `peliculas_clean.json`
- **Géneros:** Mapeo automático desde `generos.csv`
- **Usuarios:** Gestión completa con `UserRegistryManager`
- **Estadísticas:** Métricas de películas y usuarios

### 🛠️ Funcionalidades Avanzadas

- **Filtrado inteligente:** Por título y género
- **Interfaz responsiva:** Diseño adaptativo para diferentes pantallas
- **Gestión de sesiones:** Estado persistente durante la navegación
- **Validación de datos:** Manejo robusto de errores

## 🔧 Estructura del Proyecto

```
Frontend/
├── main.py                 # Aplicación principal
Backend/
├── user_registry_manager.py  # Gestión de usuarios
Data/
├── Clean_data/            # Datos procesados
│   ├── peliculas_clean.json
│   ├── ratings_por_pelicula.json
│   └── estadisticas_peliculas.json
├── Raw_data/              # Datos originales
│   └── generos.csv
└── user_registry.json     # Base de datos de usuarios
```

## 🎪 Experiencia de Usuario

### 🌟 Al Iniciar Sesión

1. **Pantalla de bienvenida** con opciones de login
2. **Selección de usuario** o **registro nuevo**
3. **Acceso inmediato** a la aplicación principal

### 🎭 Explorando Películas

1. **Navegación intuitiva** por las 3 pestañas principales
2. **Filtros dinámicos** para encontrar contenido específico
3. **Sistema de rating** fácil de usar
4. **Información completa** de cada película

### 📱 Interfaz Moderna

- **Emojis descriptivos** para mejor experiencia visual
- **Métricas en tiempo real** con st.metric()
- **Layouts responsivos** con columnas
- **CSS personalizado** para mejorar la apariencia

## 🚨 Notas Importantes

### 🔒 Seguridad Simplificada

- **Sin contraseñas:** Login basado solo en selección de usuario
- **Sesiones temporales:** Los datos se mantienen solo durante la sesión del navegador
- **Ideal para demos** y entornos de desarrollo

### ⚡ Rendimiento

- **Carga limitada:** Solo se muestran las primeras 10 películas por consulta
- **Datos en memoria:** Carga rápida de información
- **Filtrado eficiente:** Búsquedas optimizadas

## 🛠️ Posibles Mejoras Futuras

1. **Sistema de recomendaciones:** Algoritmos basados en preferencias
2. **Ratings persistentes:** Guardar calificaciones en la base de datos
3. **Imágenes de posters:** Mostrar carátulas de películas
4. **Estadísticas avanzadas:** Gráficos y analíticas detalladas
5. **Sistema de favoritos:** Marcar películas como favoritas
6. **Búsqueda avanzada:** Filtros por año, rating, etc.

## 💡 Tips de Uso

- **Crea un usuario con nombre** para una experiencia más personalizada
- **Selecciona géneros favoritos** al registrarte para mejores recomendaciones futuras
- **Explora diferentes géneros** para descubrir nuevo contenido
- **Usa la búsqueda** para encontrar películas específicas rápidamente

¡Disfruta explorando películas! 🍿✨

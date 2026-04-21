# Guía de Integración: 3 Sistemas Recomendadores

## 📋 Resumen de Cambios

Se ha integrado exitosamente **3 sistemas recomendadores** en el frontend (Streamlit):
- **📚 Basado en Contenido** (ya existía, mejorado)
- **👥 Colaborativo** (nuevo)
- **🔄 Híbrido** (nuevo)

---

## 🎯 Características por Sistema

### 📚 Basado en Contenido
- **Qué hace**: Recomienda películas similares a las que te gustaron (géneros, actores, director)
- **Ventajas**: Explicaciones detalladas, rápido, sin data sparsity
- **Desventajas**: Puede ser repetitivo

### 👥 Colaborativo
- **Qué hace**: Recomienda películas que usuarios similares a ti han visto
- **Ventajas**: Descubre películas nuevas, sorpresas
- **Desventajas**: Requiere muchos usuarios/ratings, cold-start problem
- **Scores**: Normalizados a porcentaje (0-100%) desde ratings absolutos (0-5)

### 🔄 Híbrido
- **Qué hace**: Combina ambos enfoques
- **Ventajas**: Lo mejor de ambos mundos
- **Desventajas**: Más lento (inicializa ambos modelos)
- **Scores**: Normalizados a porcentaje (0-100%) desde ratings absolutos (0-5)

---

## 🚀 Cómo Usar

### 1. Inicia la Aplicación
```bash
cd Frontend
streamlit run main.py
```

### 2. Login
- Ingresa con un usuario existente o crea uno nuevo

### 3. Accede a "🎯 Recomendadas"
La nueva interfaz mostrará:
```
┌─────────────────────────────────────────┐
│ Selecciona un sistema recomendador:     │
│                                         │
│ [📚 Basado en Contenido ▼] [Activar ☑️] │
│                                         │
│ 📚 Basado en Contenido: Recomienda...  │
│                                         │
│ Controles:                              │
│ [Slider: 5-20 películas] [Excluir vistas]│
│                                         │
│ Películas recomendadas...               │
└─────────────────────────────────────────┘
```

### 4. Selecciona el Sistema
Haz clic en el dropdown y elige:
- **📚 Basado en Contenido**
- **👥 Colaborativo**
- **🔄 Híbrido**

### 5. Ajusta Preferencias
- **Slider**: Número de recomendaciones (5-20)
- **Excluir vistas**: Solo aparece en modo Contenido

---

## 📊 Archivos Modificados

### `Frontend/data_manager.py`
✅ Agregados:
- Soporte para 3 tipos de recomendadores
- Métodos de inicialización para cada tipo
- Conversión automática de formatos

### `Frontend/app_manager.py`
✅ Modificado:
- Interface mejorada con selectbox
- Información contextual para cada tipo
- Lógica de paso de parámetros

---

## 🔧 Requisitos Técnicos

### Archivos de Datos Necesarios
```
Data/
├── Clean_data/
│   ├── enhanced_movies.json        ✅ Requerido
│   ├── train_ratings.json          ✅ Recomendado
│   └── (otras películas)
├── user_registry.json              ✅ Auto-creado
└── Raw_data/
    ├── generos.csv                 ✅ Requerido
    └── (otros csvs)

Backend/
├── Colaborativo/
│   └── preference_matrix.npz       ⚡ Se genera automáticamente
```

### Instalación de Dependencias
```bash
pip install -r requirements.txt
```

Asegúrate de que tienes:
- `pandas`
- `scikit-learn`
- `scipy`
- `streamlit`

---

## 🧪 Testing

### Prueba Rápida por Tipo

#### 1. Sistema Basado en Contenido
1. Selecciona "📚 Basado en Contenido"
2. Verifica que muestra películas con géneros similares
3. Haz clic en "Por qué se recomienda" para ver razones

#### 2. Sistema Colaborativo
1. Selecciona "👥 Colaborativo"
2. Verifica que muestra diferentes películas
3. Nota: No mostrará explicaciones (usa matriz de preferencias)

#### 3. Sistema Híbrido
1. Selecciona "🔄 Híbrido"
2. Verifica que combina características de ambos
3. Inicialización más lenta (normal, carga ambos modelos)

### Cambio de Sistema sin Reiniciar
✅ Puedes cambiar entre sistemas múltiples veces sin reiniciar la app
✅ Los recomendadores se cachean automáticamente

---

## 🐛 Troubleshooting

### "El recomendador no está disponible"
- ✅ Verifica que `enhanced_movies.json` existe
- ✅ Revisa los logs de la consola para más detalles
- ✅ Asegúrate de que el usuario tiene ratings registrados

### "Error en recomendador colaborativo"
- ✅ Verifica que `user_registry.json` existe
- ✅ Prueba con otro usuario que tenga más ratings
- ✅ Revisa que `preference_matrix.npz` se generó

### "El híbrido es muy lento"
- ✅ Esto es normal en la primera inicialización
- ✅ Después se cachea y debería ser rápido
- ✅ Si persiste, revisa logs en consola

### "Aparecen porcentajes muy altos (>100%)"
- ✅ **Problema solucionado**: Los scores del colaborativo e híbrido ahora se normalizan automáticamente
- ✅ Los ratings absolutos (0-5) se convierten a porcentajes (0-100%)
- ✅ Si ves valores >100%, indica que la normalización falló

---

## 📈 Próximas Mejoras (Opcionales)

1. **Ajuste de Pesos Híbrido**
   - Permitir usuario elegir proportions (70% contenido, 30% colaborativo)

2. **Explicaciones en Colaborativo**
   - Mostrar "Usuarios similares vieron..."

3. **Visualización Comparativa**
   - Mostrar lado a lado recomendaciones de los 3 sistemas

4. **Feedback Loop**
   - Guardar cuáles recomendaciones el usuario vio/calificó
   - Mejorar siguientes recomendaciones de forma adaptativa

5. **Caché Inteligente**
   - Sesiones de usuario con preferencias recomendador
   - Historial de recomendaciones

---

## 📝 Notas Técnicas

### Flujo de Ejecución
```
Usuario selecciona tipo en selectbox
    ↓
Llamada: get_recommendations(..., recommender_type="content")
    ↓
ensure_recommender_initialized(type)
    ├─ Si ya está cargado → usa el cacheado
    └─ Si no → lo inicializa
    ↓
recommender.recommend(...) → devuelve resultados
    ├─ Content: DataFrame directo
    ├─ Collaborative: Lista de tuplas → convierte a DataFrame
    └─ Hybrid: Lista de tuplas → convierte a DataFrame
    ↓
DataFrame uniforme: {movie_id, score, reasons}
    ↓
app_manager renderiza grid de películas
```

### Conversión de Formatos
- **Content**: `recommend()` → DataFrame ✅
- **Collaborative**: `recommend()` → List[Tuple[movie_id, score]]
  - Convertido a DataFrame con estructura uniforme
- **Hybrid**: `recommend()` → List[Tuple[movie_id, score]]
  - Convertido a DataFrame con estructura uniforme

---

## ✨ Conclusión

La integración permite al usuario:
✅ Elegir entre 3 sistemas recomendadores
✅ Ver recomendaciones sin reiniciar
✅ Cambiar de sistema en tiempo real
✅ Entender por qué (para basado en contenido)

**¡Disfruta usando los 3 sistemas!** 🎬

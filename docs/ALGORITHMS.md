# Algoritmos de Recomendacion

## 1. Recomendador Basado en Contenido

### Idea

Recomienda peliculas parecidas a los gustos del usuario analizando las caracteristicas de las peliculas y el perfil del usuario.

### Senales utilizadas

- Generos
- `overview`
- `tagline`
- `keywords`
- `tags`
- `cast`
- `crew`
- `director`

### Perfil del usuario

Se construye combinando:

- el `genre_vector` almacenado en `user_registry.json`
- el historial de ratings del usuario
- las peliculas vistas como fallback para cold start

### Espacio de items

- Los generos se codifican con `MultiLabelBinarizer`
- El texto se vectoriza con `TF-IDF`
- La similitud se calcula con coseno

### Formula general

El score final combina:

```text
score =
    w_genre * sim_genre
  + w_text * sim_text
  + w_quality * quality
  + w_popularity * popularity
```

### Ventajas

- Explicable
- Funciona incluso con pocos usuarios
- Aprovecha bien metadatos enriquecidos

### Limitaciones

- Depende de la calidad de los metadatos
- Puede sobreespecializar recomendaciones

## 2. Recomendador Colaborativo

### Idea

Busca usuarios parecidos y recomienda peliculas bien valoradas por esos vecinos.

### Representacion del usuario

Cada usuario se representa con su `genre_vector`, construido a partir de ratings historicos.

### Similitud

Se usa correlacion de Pearson sobre el vector de preferencias por genero.

Hay una estrategia hibrida:

- si dos usuarios comparten suficientes preferencias no nulas, se compara sobre la interseccion
- si no, se compara sobre la union completa

### Pipeline

1. Construir o cargar la matriz usuario-genero
2. Buscar vecinos mas similares
3. Recuperar candidatos bien valorados por esos vecinos
4. Predecir rating con media ponderada por similitud
5. Ordenar y devolver top-N

### Decisiones metodologicas clave

- Comparacion hibrida de preferencias:
  interseccion si hay suficiente solapamiento, union si no lo hay
- Seleccion de vecinos:
  similitud de Pearson positiva y `top-N` acotado, normalmente `40`
- Candidatos:
  items valorados favorablemente con umbral `>= 3.5`
- Combinacion de items repetidos:
  prediccion final ponderada por similitud y reforzada por soporte
- Usuarios nuevos:
  perfil inicial desde generos favoritos y actualizacion progresiva con ratings

### Documentacion detallada

Para la explicacion completa pensada para memoria o presentacion:

- [docs/COLLABORATIVE_RECOMMENDER.md](</c:/Users/juanc/Documents/UPV/Master/SR/docs/COLLABORATIVE_RECOMMENDER.md>)

### Prediccion

```text
r(u, i) = sum(sim(u, v) * r(v, i)) / sum(abs(sim(u, v)))
```

### Ventajas

- Usa comportamiento colectivo
- Puede descubrir peliculas alejadas del historial directo del usuario

### Limitaciones

- Sensible a sparsity
- La calidad depende mucho del vector de usuario

## 3. Recomendador Hibrido

### Idea

Mezcla ambos mundos:

- score del recomendador basado en contenido
- score del recomendador colaborativo

### Formula simplificada

```text
score_hibrido = alpha * score_content + beta * score_collaborative
```

### Ventajas

- Reduce algunas debilidades del enfoque individual
- Mejora cobertura y robustez

### Limitaciones

- Ahora mismo la mezcla es sencilla y todavia admite mejora
- Falta una calibracion mas fina de scores y pesos

## 4. Cold Start

### Usuarios nuevos

Se apoya principalmente en:

- `favorite_genres`
- `genre_vector`
- peliculas marcadas o vistas durante el uso

### Catalogo

El enfoque basado en contenido permite recomendar peliculas nuevas siempre que tengan metadatos suficientes.

## 5. Artefactos clave

- `Data/Clean_data/train_ratings.json`
- `Data/Clean_data/test_ratings.json`
- `Data/Clean_data/enhanced_movies.json`
- `Data/user_registry.json`
- `Backend/Colaborativo/preference_matrix.npz`

## 6. Siguiente iteracion recomendada

Tras la reorganizacion del repo, lo mas valioso es mejorar:

- la calidad del vector de usuario
- la estrategia de vecinos del colaborativo
- la normalizacion de scores para el hibrido
- la explicabilidad visible en el frontend

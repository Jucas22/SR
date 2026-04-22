# Recomendador Basado en Contenido

## Objetivo

Este documento describe en detalle el funcionamiento del sistema recomendador
basado en contenido del proyecto, las señales que utiliza y la lógica que sigue
para explicar por qué una película aparece recomendada.

## Idea general

El recomendador basado en contenido no busca usuarios parecidos. Lo que hace es:

1. Representar cada película mediante sus metadatos.
2. Construir un perfil de gustos del usuario en ese mismo espacio de features.
3. Calcular qué películas no vistas están más cerca de ese perfil.

La lógica base es:

```text
perfil_usuario  <->  pelicula
```

Cuanto más cerca esté una película del perfil del usuario, mayor score recibe.

## Datos de entrada

El algoritmo usa principalmente estas fuentes:

- `Data/Clean_data/enhanced_movies.json`
- `Data/Clean_data/train_ratings.json`
- `Data/user_registry.json`

### `enhanced_movies.json`

De aquí obtiene la descripción de cada película:

- géneros
- `overview`
- `tagline`
- `keywords`
- `tags`
- `cast`
- `crew`
- `director`
- señales adicionales como popularidad o valoración media si existen

### `train_ratings.json`

De aquí obtiene el historial explícito del usuario:

- películas valoradas
- rating asociado a cada película
- `timestamp` si está disponible

### `user_registry.json`

De aquí obtiene la parte más declarativa del perfil:

- `favorite_genres`
- `genre_vector`
- `watched_movies`

## Representación de las películas

Cada película se convierte en un vector numérico que combina señales
categóricas y textuales.

## 1. Géneros

La columna `generos` puede llegar como nombres o como IDs numéricos.
Antes de construir el modelo se normaliza para que todos los géneros queden en
el mismo formato canónico que usa el perfil del usuario.

Ejemplo conceptual:

```text
[35, 10749]  ->  ["comedia", "romance"]
```

Esto es importante porque el `genre_vector` del usuario también está definido
por nombres de género. Si no se hace esta normalización, la parte de géneros del
perfil y la de las películas no hablan el mismo idioma.

Después, los géneros se codifican con `MultiLabelBinarizer`.

Ejemplo:

```text
pelicula_i = [accion=0, comedia=1, drama=0, romance=1, ...]
```

### Por qué `MultiLabelBinarizer`

Porque una película puede pertenecer a varios géneros a la vez.
No es un problema de clasificación simple con una única etiqueta.

## 2. Texto y metadatos semánticos

Se construye un documento textual por película concatenando:

- `overview`
- `tagline`
- `keywords`
- `tags`
- `cast`
- `crew`
- `director`
- géneros ya normalizados

Ese documento se limpia y vectoriza con `TF-IDF`.

### Por qué `TF-IDF`

Porque permite:

- representar texto de forma numérica
- dar más peso a términos informativos
- reducir el impacto de palabras muy frecuentes y poco discriminativas
- trabajar bien en espacios dispersos y de alta dimensión

En la aplicación se limita además el número de features textuales para mantener
el sistema más ligero y evitar un coste excesivo al iniciar el recomendador.

## Espacio final de ítems

El vector final de una película es la concatenación de:

- bloque de géneros
- bloque TF-IDF de texto

Conceptualmente:

```text
item_vector = [genre_features | text_features]
```

Esto permite medir por separado:

- cercanía por géneros
- cercanía por contenido textual

## Construcción del perfil del usuario

El perfil del usuario no sale de una única fuente. Se construye combinando dos
componentes.

## 1. Perfil por géneros

Se parte del `genre_vector` guardado en `user_registry.json`.

Ese vector original contiene afinidad por unos 20 géneros. Antes de usarlo, se
reduce a un subconjunto más representativo:

1. se eliminan géneros con peso 0 o muy bajo
2. se aplica un umbral absoluto y relativo
3. si quedan pocos, se completa hasta un mínimo
4. si quedan demasiados, se recorta al top más relevante
5. se normaliza para que la suma sea 1

El objetivo de esta reducción es evitar ruido.

En vez de decir que al usuario le gusta "un poco de todo", nos quedamos con
los 5-8 géneros más útiles para definir su perfil.

### Por qué reducir el vector

Porque un vector demasiado ancho y poco discriminativo:

- mete ruido en la similitud
- hace las explicaciones menos claras
- diluye los géneros realmente dominantes

## 2. Perfil histórico

Se construye a partir de las películas que el usuario ha valorado.

Para cada rating:

1. se localiza el vector de la película
2. se calcula un peso según la valoración del usuario
3. se hace una media ponderada de esos vectores

En la estrategia actual, el peso principal se obtiene centrando los ratings del
usuario respecto a su media:

```text
peso = rating_pelicula - media_usuario
```

Así se distingue mejor entre:

- películas que el usuario ha valorado por encima de su media
- películas que ha valorado por debajo de su media

Si no hay ratings disponibles, se usa `watched_movies` como fallback de cold
start y se hace una media simple de las películas vistas.

### Por qué usar ratings centrados

Porque un `4` no significa lo mismo para todos los usuarios.

Hay usuarios más generosos y otros más estrictos. Al centrar respecto a la
media personal:

- se captura mejor la preferencia relativa
- el perfil se vuelve más estable entre usuarios distintos

## 3. Perfil final

El perfil final del usuario combina:

```text
perfil_final =
    genre_profile_weight   * perfil_generos
  + history_profile_weight * perfil_historico
```

Después se normaliza.

La idea es mezclar:

- gustos declarados y agregados por género
- comportamiento observado en ratings y películas vistas

## Cálculo del score de recomendación

Una vez tenemos el perfil del usuario y el espacio de películas, se calcula una
puntuación para cada candidata no vista.

## 1. Similaridad por géneros

Se compara el perfil del usuario y la película solo en el subespacio de
géneros.

Se usa similitud coseno:

```text
sim_genre = cosine(perfil_generos, pelicula_generos)
```

## 2. Similaridad textual

Se repite el mismo proceso pero en el subespacio textual:

```text
sim_text = cosine(perfil_texto, pelicula_texto)
```

## 3. Calidad y popularidad

Además, el recomendador incorpora dos señales auxiliares:

- `quality`
- `popularity`

La calidad se extrae de columnas disponibles como:

- `vote_average_tmdb`
- `vote_average`
- `imdb_rating`
- `tmdb_score`
- `puntuacion_media`

La popularidad se toma de `popularity` si existe, o se aproxima con el número de
ratings cuando esa columna no está disponible.

Ambas señales se normalizan a `[0, 1]`.

## 4. Fórmula final

El score final se calcula como:

```text
score =
    w_genre * sim_genre
  + w_text * sim_text
  + w_quality * quality
  + w_popularity * popularity
```

La idea de diseño es que el grueso del score venga de la afinidad de contenido,
y que calidad y popularidad solo actúen como señales de apoyo.

## Selección final y diversidad

Después de ordenar las películas por score:

1. se recorre el ranking de mayor a menor
2. se van seleccionando candidatas
3. se evita añadir películas demasiado parecidas entre sí

Esto se hace comparando la similitud coseno entre candidatas ya seleccionadas y
la nueva candidata.

Si una película es casi un duplicado de otra ya elegida, se penaliza y se busca
otra opción.

### Por qué introducir diversidad

Porque si solo se ordena por score puro, el top puede llenarse de películas muy
parecidas entre sí y la experiencia de usuario empeora.

## Explicación de la recomendación

Una parte importante del proyecto es la explicabilidad visible en el frontend.

La explicación actual del SR basado en contenido se construye en tres capas.

## 1. Ajuste al perfil

Primero se busca qué comparte la película candidata con el perfil del usuario:

- géneros dominantes del `genre_vector`
- temas frecuentes en su historial a partir de `keywords` y `tags`

El mensaje resultante es del estilo:

```text
Encaja con tu perfil porque combina géneros que suelen gustarte, como comedia y romance,
y temas recurrentes en tu historial, como adolescencia y amistad.
```

## 2. Película puente del historial

Después se busca una película del historial del usuario que sea especialmente
parecida a la candidata.

La similitud se calcula con coseno sobre el vector completo de contenido.

Una vez encontrada esa película puente, la explicación se expresa de forma
mucho más comprensible:

```text
Se parece a 'American Pie', que ya viste, porque comparte géneros como comedia y romance
y temas como adolescencia e instituto.
```

## 3. Refuerzo por calidad o popularidad

Como razón secundaria, si la película destaca por valoración media o
popularidad, se añade una frase de apoyo.

Ejemplo:

```text
Además, cuenta con una valoración media de 6.8/10 en TMDb.
```

## Por qué ya no se usa la sinopsis cruda como explicación principal

La sinopsis puede ser útil para describir la película, pero no explica bien la
relación entre la película y el usuario.

Una frase como:

```text
Su contenido encaja con tu perfil temático: <sinopsis recortada>
```

describe el ítem, pero no justifica la recomendación.

Por eso la nueva explicación prioriza:

- géneros compartidos
- temas compartidos
- una película puente del historial

Es decir, señales que sí conectan la recomendación con el perfil del usuario.

## Resumen del pipeline completo

1. Cargar películas, ratings y registro de usuarios.
2. Normalizar géneros y títulos.
3. Construir features de películas con `MultiLabelBinarizer` y `TF-IDF`.
4. Construir perfil del usuario combinando `genre_vector` e historial.
5. Calcular similitud por géneros y por texto.
6. Mezclar similitud, calidad y popularidad en un score final.
7. Eliminar películas ya vistas.
8. Aplicar una selección con diversidad.
9. Generar explicaciones basadas en perfil y películas puente.

## Justificación de los algoritmos elegidos

## `MultiLabelBinarizer`

Se elige porque los géneros son etiquetas múltiples y discretas.

## `TF-IDF`

Se elige porque el contenido textual del catálogo es disperso, ruidoso y de alta
dimensión, y TF-IDF es una forma robusta y muy interpretable de representarlo.

## Similitud coseno

Se elige porque funciona especialmente bien en vectores dispersos y compara la
orientación del perfil y de la película, no solo su magnitud.

## Combinación ponderada de señales

Se elige porque el problema no se resuelve bien con una sola señal:

- los géneros aportan estructura
- el texto aporta matiz semántico
- la calidad y la popularidad ayudan a desempatar y estabilizar

## Diversidad en el top final

Se introduce para evitar recomendaciones redundantes y mejorar la experiencia de
exploración.

## Limitaciones actuales

- Depende de la calidad real de `keywords`, `tags` y `overview`.
- Si faltan metadatos, la explicación puede ser menos rica.
- El texto todavía se modela con TF-IDF y no con embeddings semánticos más
  avanzados.
- La señal de calidad depende de qué columnas estén realmente disponibles en el
  dataset enriquecido.

## Posibles mejoras futuras

- usar embeddings semánticos para el texto
- ponderar mejor `cast`, `director` y otros créditos
- aprender pesos automáticamente en vez de fijarlos a mano
- añadir explicaciones más visuales en el frontend
- incorporar diversidad explícita por género o década, no solo por similitud


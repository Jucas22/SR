# Recomendador Hibrido

## Objetivo

Este documento describe como se ha implementado el sistema recomendador
hibrido del proyecto, por que mezcla el enfoque basado en contenido con el
colaborativo y como se justifican las decisiones principales de diseño.

## Idea general

El recomendador hibrido no sustituye a los dos sistemas base. Lo que hace es:

1. obtener una lista del recomendador basado en contenido
2. obtener una lista del recomendador colaborativo
3. mezclar ambas listas en una nueva lista final
4. ordenar los items por un nuevo ratio hibrido

La motivacion es aprovechar lo mejor de cada enfoque:

- el basado en contenido funciona bien cuando conocemos los gustos del usuario
- el colaborativo aporta descubrimiento a partir de usuarios similares

## Relacion con las indicaciones de la profesora

La implementacion sigue el esquema pedido para la recomendacion hibrida:

### Paso 0

Este proceso solo se aplica cuando el usuario selecciona el modo hibrido.

### Paso 1

Se obtienen dos listas base:

- lista `BC` del recomendador basado en contenido
- lista `Col` del recomendador colaborativo

### Paso 2

Se mezclan los items de ambas listas:

- se unifican los ids de pelicula
- se eliminan duplicados
- para cada item se calcula un nuevo ratio
- si el item aparece en ambas listas, se combinan los dos ratios

### Paso 3

Se ordenan los items por el nuevo ratio hibrido y se devuelven los `N`
elementos de mayor puntuacion.

## Modelos base que utiliza

El hibrido se apoya en dos recomendadores ya existentes:

- [docs/CONTENT_BASED_RECOMMENDER.md](</c:/Users/juanc/Documents/UPV/Master/SR/docs/CONTENT_BASED_RECOMMENDER.md>)
- [docs/COLLABORATIVE_RECOMMENDER.md](</c:/Users/juanc/Documents/UPV/Master/SR/docs/COLLABORATIVE_RECOMMENDER.md>)

## Datos que recibe de cada modelo

## 1. Recomendador basado en contenido

Del modelo de contenido se recupera una lista amplia de candidatos, con campos
como:

- `movie_id`
- `title`
- `score`
- `genre_score`
- `text_score`
- `quality_score`
- `popularity_score`
- `reasons`

Su `score` ya esta normalizado aproximadamente en `[0, 1]`, por lo que puede
interpretarse como una compatibilidad por contenido.

## 2. Recomendador colaborativo

Del modelo colaborativo se recupera una lista amplia de candidatos con detalle:

- `movie_id`
- `predicted_rating`
- `confidence`
- `mean_similarity`
- `num_contributors`
- `ranking_score`
- contadores de contribuyentes positivos, neutros y negativos

En el colaborativo, el valor importante para mezclar no es solo el
`predicted_rating`, sino el `ranking_score`, porque ya incorpora:

- el rating esperado
- la confianza
- el soporte de vecinos

## Problema metodologico que habia en la version anterior

La implementacion anterior del hibrido era demasiado simple:

- usaba `alpha=0.5` y `beta=0.5` fijos
- combinaba directamente el `score` de contenido con el rating colaborativo
- no calibraba bien las escalas
- no justificaba los pesos para cada usuario

Esto generaba un problema claro:

- el colaborativo devolvia valores en una escala cercana a `0-5`
- el contenido devolvia valores en una escala cercana a `0-1`

Por tanto, el colaborativo dominaba la mezcla aunque ambos tuvieran el mismo
peso nominal.

## Como se ha corregido

La nueva implementacion hace tres cosas:

1. normaliza los ratios de ambos modelos a `[0, 1]`
2. calcula `alpha` y `beta` dinamicamente para cada usuario
3. mezcla los items con un score interpretable y trazable

## Paso 1. Obtener las dos listas base

Se recupera un conjunto amplio de candidatos de cada modelo.

En la implementacion actual:

- se pide un pool de candidatos mayor que el `top_k` final
- como referencia minima se usan `50` candidatos por lista

La idea es no mezclar solo el top-10 de cada sistema, sino tener suficiente
espacio para que el nuevo ranking hibrido pueda reordenar con libertad.

## Paso 2. Normalizar los ratios

Antes de mezclar, ambos modelos deben hablar en la misma escala.

## Ratio del basado en contenido

Se usa directamente:

```text
r_content(i) = score_content(i)
```

porque ese valor ya esta acotado en `[0, 1]`.

## Ratio del colaborativo

Se usa como señal principal:

```text
r_collab(i) = ranking_score(i) / 5
```

donde:

```text
ranking_score = predicted_rating * confidence
```

### Por que se usa `ranking_score` y no solo `predicted_rating`

Porque en el colaborativo no basta con saber que el rating esperado es alto.
Tambien importa:

- cuanta confianza tenemos en esa prediccion
- cuantos vecinos la sostienen

Por eso el hibrido reutiliza la señal ya refinada del colaborativo en vez de
volver a una combinacion ingenua.

## Paso 3. Calcular los coeficientes de ponderacion

Una de las condiciones de la profesora era que:

- `alpha + beta = 1`
- `alpha` no sea fijo
- `beta` no sea fijo
- ambos dependan de la recomendacion actual

En esta implementacion, los coeficientes se calculan para cada peticion de un
usuario concreto.

## 3.1. Coeficiente del contenido `alpha`

`alpha` depende de la calidad y cantidad de informacion de contenido del
usuario.

Se calculan varias señales:

- `genre_signal`
  depende del numero de generos relevantes disponibles en su perfil
- `ratings_signal`
  depende del numero de ratings historicos
- `watched_signal`
  depende de cuantas peliculas vistas tenemos como respaldo
- `candidate_signal`
  depende de que el modelo de contenido pueda generar una lista amplia de candidatos

La señal agregada del contenido es:

```text
content_signal =
    0.45 * genre_signal
  + 0.35 * history_signal
  + 0.20 * candidate_signal
```

donde `history_signal` se apoya en ratings explicitos y, si faltan, usa
`watched_movies` como respaldo mas debil.

### Caso especial

Si no existe informacion real de contenido del usuario:

```text
alpha = 0
```

Esto cumple con la idea metodologica de que el peso del contenido no debe
inventarse si no tenemos base suficiente.

## 3.2. Coeficiente del colaborativo `beta`

`beta` depende de la calidad y cantidad de los vecinos.

Se calculan estas señales:

- `neighbor_count_signal`
  refleja cuantos vecinos utiles tenemos, acotando la referencia en `40`
- `mean_similarity`
  refleja la calidad media de esos vecinos
- `candidate_signal`
  refleja que el colaborativo puede producir una lista suficiente de candidatos

La señal agregada del colaborativo es:

```text
collaborative_signal =
    0.55 * mean_similarity
  + 0.30 * neighbor_count_signal
  + 0.15 * candidate_signal
```

### Caso especial

Si no tenemos vecinos:

```text
beta = 0
```

Esto vuelve a seguir directamente la regla pedida por la profesora.

## 3.3. Normalizacion final de pesos

Una vez calculadas ambas señales:

```text
alpha_raw = content_signal
beta_raw  = collaborative_signal
```

se normalizan para que sumen 1.

En la implementacion se aplica una suavizacion previa con raiz cuadrada:

```text
alpha = sqrt(alpha_raw) / (sqrt(alpha_raw) + sqrt(beta_raw))
beta  = sqrt(beta_raw)  / (sqrt(alpha_raw) + sqrt(beta_raw))
```

### Por que se usa esta suavizacion

Porque si un modelo tiene una señal solo moderadamente mejor que el otro, no
queremos que absorba casi toda la mezcla y el sistema deje de ser realmente
hibrido.

La raiz cuadrada:

- mantiene la diferencia entre modelos
- pero la comprime
- evita mezclas demasiado extremas cuando ambos aportan informacion util

## Paso 4. Mezclar los items

Una vez tenemos:

- la lista `BC`
- la lista `Col`
- los pesos `alpha` y `beta`

se unifican todos los ids de pelicula en una sola lista.

Para cada item `i`:

```text
score_hibrido(i) =
    alpha * r_content(i)
  + beta  * r_collab(i)
```

Si la pelicula aparece solo en una lista, el ratio de la otra es `0`.

## Como se combinan los items repetidos

Si una pelicula aparece en ambos recomendadores:

- no se duplica en la lista final
- se crea una sola entrada
- se combinan sus dos ratios con la formula anterior

Esto implementa exactamente la idea de:

- eliminar repetidos
- combinar ratios
- favorecer los items con acuerdo entre modelos

## Paso 5. Ordenacion final

Los items se ordenan de mayor a menor segun:

```text
score_hibrido
```

Como criterio secundario se favorecen:

- items que aparecen en ambos modelos
- despues, mayor señal colaborativa
- despues, mayor señal de contenido

La idea es que, en caso de empate o casi empate, el sistema prefiera el acuerdo
entre ambos enfoques.

## Explicacion visible en la interfaz

Cada recomendacion hibrida devuelve informacion interpretable:

- si la pelicula aparece en contenido, en colaborativo o en ambos
- cuanto peso se ha dado al contenido
- cuanto peso se ha dado al colaborativo
- que ratio aporta cada parte

Ademas:

- si la pelicula viene del contenido, se puede reutilizar una razon del modelo de contenido
- si viene del colaborativo, se indica confianza y numero de vecinos contribuyentes

## Ejemplo conceptual

Supongamos:

```text
alpha = 0.45
beta  = 0.55

r_content(i) = 0.72
r_collab(i)  = 0.60
```

Entonces:

```text
score_hibrido(i) =
    0.45 * 0.72
  + 0.55 * 0.60
  = 0.324 + 0.330
  = 0.654
```

Si la pelicula solo apareciera en contenido:

```text
score_hibrido(i) = 0.45 * 0.72 = 0.324
```

Esto hace que los items compartidos por ambos modelos tiendan a subir en el
ranking, que es precisamente lo que se espera de una mezcla hibrida.

## Decisiones de diseño y justificacion

## 1. Usar pesos dinamicos y no fijos

Se hace porque el usuario puede tener:

- mucha informacion de contenido y pocos vecinos
- pocos datos de contenido y muchos vecinos
- informacion fuerte en ambos modelos

Un `alpha=0.5` fijo no refleja esa situacion real.

## 2. Reutilizar `ranking_score` del colaborativo

Se hace porque ya es una señal mas robusta que el rating predicho puro.

## 3. Normalizar ambas ramas a `[0, 1]`

Se hace para evitar dominancias artificiales por una diferencia de escala y no
por una diferencia real de calidad.

## 4. Mantener una mezcla conservadora con suavizacion

Se hace para que el hibrido no se degrade facilmente a:

- "casi todo colaborativo"
- o "casi todo contenido"

cuando ambos modelos siguen teniendo valor.

## 5. Usar un pool amplio antes del ranking final

Se hace para que la mezcla no dependa solo del top mas corto de cada sistema y
pueda reorganizar mejor las candidatas.

## Limitaciones actuales

- El calculo de `alpha` y `beta` sigue siendo heuristico, no aprendido.
- Si no hay solapamiento entre listas, el hibrido funciona como una mezcla de
  listas separadas, no como un acuerdo fuerte entre ambas.
- Todavia no hay una calibracion automatica basada en evaluacion offline.
- No se esta aprendiendo la mejor funcion de mezcla a partir de datos reales de uso.

## Posibles mejoras futuras

- aprender `alpha` y `beta` automaticamente
- calibrar las puntuaciones con evaluacion offline
- usar blending por item, no solo por peticion
- introducir bonificaciones explicitas por consenso entre modelos
- mostrar en la interfaz un desglose visual mas rico de la mezcla


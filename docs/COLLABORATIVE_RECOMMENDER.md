# Recomendador Colaborativo

## Objetivo

Este documento describe en detalle el funcionamiento del sistema recomendador colaborativo del proyecto, las metricas que calcula y como se interpretan en la interfaz.

## Idea general

El recomendador colaborativo no compara peliculas por contenido. En su lugar:

1. Representa cada usuario con un vector de preferencias por genero.
2. Busca usuarios parecidos al usuario objetivo.
3. Usa las valoraciones de esos vecinos para predecir como puntuaria el usuario objetivo una pelicula que todavia no ha visto.

## Datos de entrada

El algoritmo usa principalmente tres fuentes:

- `Data/user_registry.json`
- `Data/Clean_data/train_ratings.json`
- `Data/Clean_data/enhanced_movies.json`

Ademas, cuando la matriz usuario-genero ya se ha calculado una vez, se cachea en:

- `Data/Clean_data/collaborative_preference_matrix.npz`

### `user_registry.json`

De aqui obtiene:

- usuarios activos
- `watched_movies`
- `genre_vector`

El `genre_vector` representa la afinidad del usuario con cada genero en una escala `[0, 100]`.

### `train_ratings.json`

De aqui obtiene los ratings historicos para:

- recuperar las peliculas valoradas por los vecinos
- construir la prediccion de rating para cada candidato

## Representacion del usuario

Cada usuario se representa mediante un vector:

```text
u = [pref_accion, pref_comedia, pref_drama, ...]
```

Cada componente es la afinidad de ese usuario con un genero concreto.

La matriz final es:

```text
M in R^(num_usuarios x num_generos)
```

donde:

- cada fila es un usuario
- cada columna es un genero

En esta implementacion no se usa todavia una matriz usuario-item clasica para medir similitud.
La similitud entre usuarios se calcula sobre su perfil de gustos por genero, que viene del
`genre_vector` guardado en el registro de usuarios.

## Como se construye la matriz usuario-genero

La matriz se construye en `_build_preference_matrix()` y sigue estos pasos:

### 1. Se fija un orden estable de generos

Primero se crea `genre_map` en `_build_genre_map()`.

Ese mapa:

- toma los nombres de genero del `genre_vector` de un usuario
- los ordena alfabeticamente
- asigna un indice fijo a cada genero

Por ejemplo, conceptualmente podria quedar algo asi:

```text
{
  "Action": 0,
  "Adventure": 1,
  "Animation": 2,
  "Comedy": 3,
  ...
}
```

Este detalle es importante porque garantiza que la columna 0 siempre represente el mismo
genero para todos los usuarios, la columna 1 otro genero concreto, y asi sucesivamente.

### 2. Solo se incluyen usuarios activos

El bucle recorre `self.user_registry["users"]`, pero filtra:

- usuarios con `status != "active"`

Por tanto, cada fila de la matriz representa un usuario activo del sistema.

### 3. Para cada usuario se crea una fila numerica

Para cada usuario activo:

- se lee `user["preferences"]["genre_vector"]`
- se crea un array siguiendo el orden exacto de `genre_names`
- si un genero no aparece, se rellena con `0.0`

Conceptualmente, una fila puede verse asi:

```text
usuario_42 = [82.0, 15.0, 0.0, 67.0, 40.0, ...]
```

Eso significa:

- afinidad alta con algunos generos
- afinidad baja con otros
- o ausencia total de interes si el valor es `0`

### 4. Se apilan todas las filas

Cuando ya se han generado todas las filas:

- se hace `np.vstack(rows)`
- se obtiene una matriz densa `num_usuarios x num_generos`
- se convierte a formato disperso `csr_matrix`

El formato disperso no cambia la semantica de los datos, pero ayuda a almacenar y reutilizar
la matriz de manera mas eficiente.

### 5. Se guardan tambien los mapeos de usuarios

Mientras se construye la matriz se mantienen dos diccionarios:

- `user_id_to_idx`: traduce `user_id -> fila`
- `idx_to_user_id`: traduce `fila -> user_id`

Esto permite:

- localizar rapidamente la fila del usuario objetivo
- saber a que usuario corresponde cada fila comparada

### 6. La matriz se cachea para no reconstruirla siempre

Una vez creada, se guarda en `collaborative_preference_matrix.npz`.

En ejecuciones posteriores:

- si el fichero existe y es compatible con el numero de usuarios activos y generos actuales,
  se reutiliza
- si no, se reconstruye desde `user_registry.json`

## Lectura practica de la matriz

La matriz usuario-genero puede leerse asi:

- fila `i`: el perfil de gustos del usuario `i`
- columna `j`: la intensidad de preferencia por el genero `j`
- celda `M[i, j]`: cuanto le gusta al usuario `i` el genero `j`

Por tanto, el algoritmo no esta preguntando todavia:

- "que items han puntuado igual estos dos usuarios"

Sino:

- "que parecido tienen sus perfiles globales de gustos por genero"

## Calculo de similitud entre usuarios

La similitud se calcula con correlacion de Pearson.

### Caso 1: suficiente solapamiento

Si dos usuarios comparten suficientes generos con valor no nulo, se calcula Pearson solo sobre esa interseccion.

### Caso 2: poco solapamiento

Si no comparten suficientes generos no nulos, se calcula Pearson sobre la union completa del vector.

Esto intenta equilibrar:

- precision cuando hay informacion comun suficiente
- robustez cuando el perfil es mas disperso

## Decision de diseno: interseccion o union de preferencias

Uno de los puntos clave del trabajo es decidir sobre que parte del vector se compara a dos usuarios.
Habia varias alternativas razonables:

- usar solo la interseccion de preferencias no nulas
- usar siempre la union completa del vector
- usar una estrategia hibrida

### Alternativa 1. Solo interseccion

Ventaja:

- compara unicamente sobre los generos donde ambos usuarios realmente han expresado preferencia

Inconveniente:

- si la interseccion es muy pequena, Pearson puede salir poco estable
- dos usuarios pueden parecer muy correlacionados por coincidir solo en pocos generos

### Alternativa 2. Solo union completa

Ventaja:

- la comparacion siempre usa el mismo espacio de generos
- evita depender demasiado de coincidencias pequenas

Inconveniente:

- cuando un usuario tiene muchas componentes a `0`, la comparacion puede diluir la informacion
- mezcla zonas con preferencia real y zonas sin senal

### Alternativa 3. Estrategia hibrida

Esta es la tecnica elegida en el proyecto.

La regla actual es:

- si dos usuarios comparten suficientes preferencias no nulas, se compara sobre la interseccion
- si no, se compara sobre la union completa

Concretamente:

```text
MIN_COMMON_PREFS = 10
```

### Justificacion de la tecnica elegida

La eleccion esta justificada por equilibrio entre precision y robustez:

- cuando hay bastante solapamiento, la interseccion es mas informativa porque compara gustos realmente compartidos
- cuando hay poco solapamiento, usar solo la interseccion puede producir similitudes artificiales o muy inestables
- usar la union en esos casos introduce mas contexto y hace el sistema mas conservador

Dicho de forma mas simple para memoria o presentacion:

- no queriamos ni un sistema demasiado estricto ni uno demasiado ruidoso
- por eso usamos una estrategia adaptativa segun la cantidad de informacion comun

## Como se buscan los vecinos similares

La busqueda de vecinos se hace en `_find_neighbors(user_id, top_n=40)`.

El procedimiento exacto es:

### 1. Se recupera el vector del usuario objetivo

Con `user_id_to_idx` se localiza la fila del usuario y se extrae su vector de preferencias:

```text
user_vec = preference_matrix[user_idx]
```

### 2. Se compara contra todos los demas usuarios activos

Para cada `other_uid`:

- se ignora el propio usuario
- se extrae `other_vec`
- se calcula `sim = _pearson_similarity(user_vec, other_vec)`

### 3. Se decide sobre que dimensiones comparar

Dentro de `_pearson_similarity()` se comprueba:

- que componentes de `vec_a` son no nulas
- que componentes de `vec_b` son no nulas
- cuantas coinciden en no ser cero

Eso crea una mascara booleana `common_nonzero`.

Luego ocurren dos casos:

#### Caso A. Hay suficiente informacion comun

Si `n_common >= MIN_COMMON_PREFS` y actualmente:

```text
MIN_COMMON_PREFS = 10
```

entonces Pearson se calcula solo sobre esos generos compartidos con valor no nulo.

La idea es:

- si ambos usuarios expresan preferencias en bastantes generos comunes,
  conviene comparar solo esa parte realmente compartida

#### Caso B. Hay poco solapamiento

Si no llegan a ese minimo, Pearson se calcula sobre el vector completo.

La idea es:

- evitar que una comparacion sobre muy pocos generos produzca una correlacion demasiado inestable

### 4. Se calcula Pearson

Una vez elegidos los componentes a comparar:

- se calcula la media de cada vector
- se centran ambos vectores restando su media
- se obtiene la covarianza normalizada

### Formula

Si tenemos dos vectores `a` y `b`:

```text
sim(a, b) = sum((a_i - mean(a)) * (b_i - mean(b))) /
            (sqrt(sum((a_i - mean(a))^2)) * sqrt(sum((b_i - mean(b))^2)))
```

El resultado esta en `[-1, 1]`, pero el sistema solo conserva vecinos con similitud positiva.

En esta implementacion:

- si el denominador sale `0`, la similitud se fuerza a `0.0`
- si `sim <= 0`, ese usuario no entra como vecino util

Por tanto, la lista final de vecinos contiene usuarios que:

- no son el propio usuario
- tienen correlacion positiva
- y estan entre los mas parecidos tras ordenar de mayor a menor

## Seleccion de vecinos

Una vez calculadas las similitudes:

- se descarta el propio usuario
- se descartan similitudes `<= 0`
- se ordena de mayor a menor
- se conservan los `top_n` vecinos

En la practica:

- `recommend()` usa vecinos ya precalculados si existen en `user_registry.json`
- si no existen, los calcula al vuelo

Tambien existe `precompute_all_neighbors()` para dejar ese paso persistido y no repetirlo
en cada recomendacion.

## Decision de diseno: como se obtienen los vecinos

Otro punto importante del trabajo es decidir quien entra realmente en el conjunto de vecinos.
Las opciones tipicas eran:

- todos los usuarios con correlacion positiva
- los usuarios que superen un umbral fijo
- los `N` usuarios con mayor similitud
- una combinacion de umbral y `top-N`

### Regla adoptada en este proyecto

La implementacion actual hace esto:

1. Calcula Pearson entre el usuario objetivo y el resto de usuarios activos.
2. Descarta usuarios con similitud `<= 0`.
3. Ordena los restantes de mayor a menor.
4. Conserva como maximo los `top_n` vecinos.

En la practica, el valor por defecto es:

```text
top_n = 40
```

### Por que esta decision es razonable

La justificacion es la siguiente:

- exigir `sim > 0` garantiza que solo entren usuarios con correlacion positiva
- ordenar por similitud da prioridad a los perfiles realmente mas afines
- limitar a unos `40` vecinos evita que entren usuarios demasiado lejanos y mantiene el coste computacional controlado
- este valor esta alineado con la recomendacion habitual de no usar conjuntos demasiado grandes, por ejemplo no mas de `40-50` vecinos

### Alternativas que podrian haberse usado

Tambien hubiera sido valido usar:

- un umbral minimo, por ejemplo `sim >= 0.30`
- o una regla mixta del tipo: vecinos con `sim > 0.20`, pero nunca mas de `40`

En nuestro caso preferimos el esquema `sim > 0` + `top-40` porque:

- se adapta mejor a usuarios con distinta densidad de informacion
- evita dejar a algunos usuarios sin vecinos por culpa de un umbral demasiado agresivo
- y mantiene una lista acotada y facil de interpretar

## Generacion de candidatos

Una pelicula candidata es una pelicula que:

- ha sido puntuada favorablemente por algun vecino
- no esta en `watched_movies` del usuario objetivo

Actualmente se considera favorable un rating `>= 3.5`.

## Como se construyen los candidatos bien valorados

La generacion de candidatos se hace en `_get_candidate_items(user_id, neighbors)`.

El flujo es este:

### 1. Se recuperan las peliculas ya vistas por el usuario objetivo

Desde `user_registry.json` se lee:

- `user["preferences"]["watched_movies"]`

Eso se convierte en un `set` llamado `seen`.

### 2. Se recorren los vecinos seleccionados

Para cada vecino:

- se consulta `ratings_by_user[int(neighbor_id)]`
- ahi se encuentran sus ratings historicos cargados desde `train_ratings.json`

La estructura en memoria queda organizada como:

```text
ratings_by_user = {
  userId: {
    movieId: rating,
    ...
  }
}
```

### 3. Se filtran solo los items con evidencia positiva

Una pelicula entra como candidata si cumple simultaneamente:

- el vecino le ha dado `rating >= FAVORABLE_RATING_THRESHOLD`
- el usuario objetivo todavia no la ha visto

En la implementacion actual:

```text
FAVORABLE_RATING_THRESHOLD = 3.5
```

### 4. Se usa un conjunto para eliminar duplicados

Si varios vecinos bien valoran la misma pelicula:

- la pelicula aparece una sola vez como candidata

Esto es importante porque la duplicacion no se gestiona en esta fase, sino despues,
cuando se calcula la prediccion usando todos los vecinos contribuyentes reales.

## Que significa exactamente "candidato bien valorado"

Hay un matiz importante:

- para que una pelicula sea candidata basta con que al menos un vecino la valore bien

Pero eso no significa todavia que la prediccion final vaya a ser alta.

En la fase de prediccion:

- entran todos los vecinos similares que hayan valorado esa pelicula
- no solo los que la valoraron positivamente

Asi, una pelicula puede:

- entrar como candidata porque un vecino la puntuo con `4.5`
- pero bajar despues si otros vecinos parecidos le dieron `3.0` o `2.0`

## Decision de diseno: como se obtienen y combinan los items recomendados

Esta fase corresponde a la parte del trabajo donde hay que decidir:

- que significa que un item este "bien valorado"
- como eliminar duplicados
- como combinar la informacion cuando varios vecinos apuntan al mismo item
- como calcular el interes final del item para el usuario

### 1. Que significa "favorablemente"

En este proyecto se adopta:

```text
FAVORABLE_RATING_THRESHOLD = 3.5
```

Es decir:

- un item entra como candidato si algun vecino similar le ha dado `3.5` o mas

La justificacion es que, en una escala tipica `0-5`, un `3.5` ya indica una opinion claramente positiva,
sin exigir que solo entren peliculas puntuadas con `4.5` o `5.0`.

### 2. Como se recorren los historicos de los vecinos

Para cada vecino seleccionado:

- se consulta su historico en `train_ratings.json`
- se recuperan las peliculas que puntuo favorablemente
- se descartan directamente las peliculas ya vistas por el usuario objetivo

### 3. Como se eliminan los duplicados

Si varios vecinos recomiendan la misma pelicula:

- la pelicula aparece una sola vez en el conjunto de candidatos

Esto se hace usando un `set`.

### 4. Como se combinan los items repetidos

Aqui habia varias opciones:

- media simple de ratings
- media ponderada por similitud
- contar repeticiones y despues combinar con similitud
- o predecir el rating final usando todos los vecinos contribuyentes

La tecnica elegida en el proyecto es la ultima.

No se combina el item repetido con una media previa "rapida", sino que:

- primero se genera una lista unica de candidatos
- despues, para cada pelicula, se reunen todos los vecinos que realmente la valoraron
- y la prediccion final se calcula con una media ponderada por similitud

Esto hace que la combinacion tenga en cuenta a la vez:

- la afinidad con cada vecino
- el numero de vecinos que apoyan el item
- y el valor exacto del rating de cada uno

### 5. Como se calcula el ratio de interes del item

El "ratio de interes" que finalmente usamos es el `predicted_rating`.

Ese valor se calcula asi:

```text
predicted_rating = sum(sim(u, v) * rating(v, i)) / sum(abs(sim(u, v)))
```

Por tanto, un item tendra un ratio de interes alto si:

- lo han visto varios vecinos parecidos
- esos vecinos le han dado ratings altos
- y la similitud media con el usuario objetivo es buena

### 6. Como se tiene en cuenta el numero de repeticiones

El numero de repeticiones no se ignora.

Aunque no aparezca como una simple cuenta sumada al score, si influye en dos puntos:

- aumenta `num_contributors`
- aumenta `support_ratio`

Y eso repercute en la `confidence` y en el `ranking_score`.

Por tanto, un item apoyado por varios vecinos no solo tiene mas evidencia, sino que queda mejor posicionado.

### 7. Eliminacion final de items ya vistos

Los items ya vistos por el usuario se eliminan desde el principio usando `watched_movies`.

Esto evita recomendar:

- peliculas ya valoradas
- peliculas ya vistas aunque todavia no tuvieran explicacion colaborativa fuerte

## Prediccion del rating

Para cada pelicula candidata, se buscan los vecinos que realmente la han valorado.

Si los vecinos contribuyentes son `v1, v2, ..., vk`, la prediccion se calcula como:

```text
r_hat(u, i) = sum(sim(u, v) * r(v, i)) / sum(abs(sim(u, v)))
```

donde:

- `r_hat(u, i)` es el rating predicho para el usuario `u` sobre la pelicula `i`
- `sim(u, v)` es la similitud con el vecino `v`
- `r(v, i)` es el rating que ese vecino dio a la pelicula

El `predicted_rating` resultante esta en la escala de ratings del sistema, normalmente `0-5`.

## Como se calcula exactamente la prediccion

Esta parte se reparte entre `_get_contributors()`, `_predict_ratings()` y `_build_prediction()`.

### 1. Se obtienen los contribuyentes reales de cada pelicula

No todos los vecinos seleccionados han visto cada candidata.

Por eso, para cada `movie_id`, `_get_contributors()` filtra solo los vecinos que:

- tienen rating registrado para esa pelicula

Cada contribuyente se guarda como:

```text
{
  "neighbor_id": ...,
  "similarity": ...,
  "rating": ...,
  "weight": ...
}
```

En esta implementacion:

- `weight = similarity`

Como previamente solo se han retenido similitudes positivas, los pesos tambien son positivos.

### 2. Se hace una media ponderada por similitud

La intuicion es:

- cuanto mas parecido es un vecino al usuario objetivo, mas peso tiene su rating

Por eso el numerador acumula:

```text
sum(weight * rating)
```

y el denominador:

```text
sum(abs(weight))
```

Aunque aqui `abs(weight)` y `weight` coinciden en la practica porque no hay similitudes negativas,
se deja asi para que la formula sea robusta.

### 3. Se calcula el rating predicho

Finalmente:

```text
predicted_rating = numerator / denominator
```

Si por algun motivo no hubiera peso acumulado, el sistema devuelve `0.0`.

### 4. Se calculan metricas auxiliares para interpretar la recomendacion

Ademas del rating predicho, `_build_prediction()` genera:

- `num_contributors`: cuantos vecinos han aportado rating para esa pelicula
- `mean_similarity`: similitud media de esos contribuyentes
- `support_ratio`: cuanto soporte tiene la prediccion
- `confidence`: confianza global de la recomendacion
- `ranking_score`: score interno usado para ordenar
- `average_contributor_rating`: media simple de ratings de los contribuyentes
- `positive_contributors`: cuantos la valoraron con `>= 3.5`
- `neutral_contributors`: cuantos quedaron en zona intermedia
- `negative_contributors`: cuantos la valoraron con `<= 2.5`

### 5. Se calcula la confianza

La confianza no es el rating esperado, sino una medida de respaldo colaborativo:

```text
mean_similarity = average(abs(similarity))
support_ratio = min(1.0, num_contributors / CONFIDENCE_NEIGHBOR_TARGET)
confidence = mean_similarity * support_ratio
```

Y actualmente:

```text
CONFIDENCE_NEIGHBOR_TARGET = 3
```

Eso implica:

- con 1 contribuyente, la confianza se reduce bastante
- con 2, mejora
- con 3 o mas, el soporte ya se considera suficiente

### 6. Se calcula el score final de ranking

Para ordenar recomendaciones no se usa solo `predicted_rating`, sino:

```text
ranking_score = predicted_rating * confidence
```

Con esto se evita que una pelicula con una unica valoracion muy alta quede automaticamente
por encima de otra con mejor respaldo colaborativo.

## Pipeline completo de una recomendacion colaborativa

Visto de extremo a extremo, el algoritmo sigue esta cadena:

1. Leer usuarios activos, `genre_vector` y peliculas vistas.
2. Construir o cargar la matriz usuario-genero.
3. Extraer la fila del usuario objetivo.
4. Compararla con el resto de usuarios usando Pearson.
5. Quedarse con los vecinos de similitud positiva mejor rankeados.
6. Buscar peliculas no vistas que al menos un vecino haya valorado con `>= 3.5`.
7. Para cada candidata, reunir a todos los vecinos similares que realmente la puntuaron.
8. Calcular el `predicted_rating` con media ponderada por similitud.
9. Calcular soporte, confianza y `ranking_score`.
10. Ordenar priorizando primero recomendaciones con soporte suficiente y despues
    el `ranking_score`.

## Proceso de recomendacion segun tipo de usuario

De cara a memoria o presentacion, conviene distinguir dos escenarios.

### Caso 1. Usuario del dataset o usuario ya registrado

El flujo es:

1. El usuario ya existe en `user_registry.json`.
2. Ya dispone de preferencias almacenadas:
   `favorite_genres`, `watched_movies` y normalmente `genre_vector`.
3. El sistema construye o recupera su fila en la matriz usuario-genero.
4. Se obtienen sus vecinos:
   precalculados si existen, o calculados al vuelo si hace falta.
5. Se generan los candidatos a partir de los historicos favorables de esos vecinos.
6. Se predice el rating de cada candidato.
7. Se ordenan los resultados y se muestra la recomendacion final.

En otras palabras:

- el usuario solicita recomendacion
- el sistema usa sus vecinos ya disponibles
- y devuelve la lista recomendada

### Caso 2. Usuario nuevo

Para un usuario nuevo hay un pequeno proceso previo antes de poder recomendar.

El flujo es:

1. Se registra el usuario.
2. Se recogen sus preferencias iniciales, normalmente:
   - generos favoritos
   - peliculas que va puntuando
3. Se genera o completa su `genre_vector`.
4. Se actualiza `watched_movies`.
5. Se reconstruye o actualiza la matriz usuario-genero.
6. Se calculan sus vecinos.
7. Se aplica el mismo pipeline de recomendacion que en un usuario ya existente.
8. Se muestra la recomendacion obtenida.

### Como se implementa actualmente el caso de usuario nuevo

En la version actual del proyecto:

- al crear un usuario nuevo se genera un `genre_vector` inicial a partir de sus generos favoritos
- al guardar ratings se actualizan:
  - `watched_movies`
  - la media de ratings del usuario
  - el `genre_vector`
  - los ratings persistidos en `train_ratings.json`
- si el usuario aun no tenia `genre_vector`, el colaborativo puede reconstruirlo usando:
  - generos favoritos
  - peliculas vistas

Esto es importante porque evita el problema clasico de cold start absoluto:

- un usuario nuevo no necesita parecerse a nadie desde el primer segundo
- pero si necesita una senal minima de preferencias para poder buscar vecinos

### Resumen muy corto para presentacion

Una forma clara de contarlo en diapositivas es:

- usuario existente: cargar perfil -> buscar vecinos -> generar candidatos -> predecir rating -> recomendar
- usuario nuevo: registrar preferencias -> construir perfil inicial -> calcular vecinos -> recomendar

## Ejemplo conceptual

Supongamos este escenario simplificado:

- usuario objetivo `U`
- vecinos similares:
  - `V1` con similitud `0.82`
  - `V2` con similitud `0.70`
  - `V3` con similitud `0.45`
- pelicula `P` no vista por `U`
- ratings de `P`:
  - `V1 -> 5.0`
  - `V2 -> 4.0`
  - `V3 -> 3.5`

Entonces:

```text
predicted_rating(P) =
  (0.82 * 5.0 + 0.70 * 4.0 + 0.45 * 3.5) /
  (0.82 + 0.70 + 0.45)
  = 4.31 aprox.
```

Y ademas:

```text
num_contributors = 3
mean_similarity = (0.82 + 0.70 + 0.45) / 3 = 0.656 aprox.
support_ratio = min(1, 3 / 3) = 1.0
confidence = 0.656 * 1.0 = 0.656
ranking_score = 4.31 * 0.656 = 2.83 aprox.
```

Interpretacion:

- la pelicula parece gustarle bastante al usuario
- la recomendacion tiene una confianza razonable
- y esa confianza viene de que hay varios vecinos parecidos que la han visto

## Problema detectado originalmente

Antes de esta mejora, el frontend trataba `predicted_rating / 5` como si fuera una "compatibilidad".

Eso provocaba un problema importante:

- si una pelicula era recomendada por un unico vecino con rating `5.0`
- entonces el `predicted_rating` podia ser `5.0`
- y la interfaz mostraba `100%`

Pero ese `100%` no significaba:

- ni similitud perfecta con otros usuarios
- ni confianza alta en la recomendacion

Solo significaba:

- "el rating predicho salio 5.0"

## Nueva interpretacion: rating + confianza

Ahora el colaborativo separa dos metricas:

### 1. `predicted_rating`

Representa cuanto se espera que le guste la pelicula al usuario.

Ejemplo:

- `4.72/5`

### 2. `confidence`

Representa cuanta evidencia hay detras de esa prediccion.

No depende solo del rating esperado, sino de:

- cuantos vecinos contribuyen realmente
- que similitud media tienen esos vecinos con el usuario objetivo

## Calculo de confianza

La confianza combina dos factores.

### Similitud media

```text
mean_similarity = average(abs(sim(u, v_j)))
```

### Soporte

```text
support_ratio = min(1, num_contributors / confidence_neighbor_target)
```

En la implementacion actual:

- `confidence_neighbor_target = 3`

Eso significa:

- con 1 vecino, el soporte es bajo
- con 2 vecinos, mejora
- con 3 o mas vecinos, el soporte se considera suficiente

### Formula final

```text
confidence = mean_similarity * support_ratio
```

Como ambos factores estan en `[0, 1]`, la confianza tambien queda en `[0, 1]`.

## Penalizacion por poco soporte

Ademas del `predicted_rating`, el ranking usa un score interno:

```text
ranking_score = predicted_rating * confidence
```

Con esto, una pelicula con:

- `predicted_rating = 5.0`
- pero solo 1 vecino poco parecido

deja de parecer mas fuerte que otra con:

- `predicted_rating = 4.4`
- varios vecinos
- similitud media mayor

## Minimo de contribuyentes

El sistema tambien prioriza primero las recomendaciones con al menos:

```text
MIN_RECOMMENDATION_CONTRIBUTORS = 2
```

Si hay suficientes candidatas con soporte fuerte, esas aparecen primero.

Si no hay suficientes, se pueden completar con candidatas mas debiles, pero siempre quedaran penalizadas por la confianza.

## Ordenacion final

La lista final se ordena priorizando:

1. `ranking_score`
2. `confidence`
3. `predicted_rating`
4. `mean_similarity`
5. `num_contributors`

## Metricas expuestas en frontend

Ahora la interfaz colaborativa muestra:

- `Rating Predicho`
- `Confianza`

Y, cuando hay detalle, tambien:

- numero de vecinos contribuyentes
- similitud media
- desglose de valoraciones positivas, tibias y negativas
- lista de vecinos que han aportado a la prediccion

## Explicacion mostrada al usuario

En la seccion "Por que se recomienda" ya no se enfatiza tanto el rating predicho.
Ahora se prioriza explicar la evidencia colaborativa:

- nivel de confianza de la recomendacion
- cuantos usuarios parecidos han visto la pelicula
- cuantos la valoraron positivamente
- cuantos la valoraron de forma tibia
- cuantos la valoraron negativamente

Esto hace que la recomendacion sea mas interpretable, porque separa:

- cuanto podria gustarle la pelicula al usuario
- cuanta evidencia real existe detras de esa prediccion

## Interpretacion recomendada

### Caso A

- `Rating Predicho: 4.8/5`
- `Confianza: 22%`

Interpretacion:

- parece gustarle mucho
- pero la evidencia es debil

### Caso B

- `Rating Predicho: 4.4/5`
- `Confianza: 68%`

Interpretacion:

- puede gustarle bastante
- y ademas hay una base colaborativa mucho mas solida

En la practica, el caso B suele ser una recomendacion mas fiable que el caso A.

## Limitaciones actuales

- La similitud se calcula sobre preferencias por genero, no sobre una matriz usuario-item completa.
- El sistema depende mucho de la calidad del `genre_vector`.
- Todavia no incorpora sesgos de usuario o item como en filtrado colaborativo mas avanzado.
- No hay un proceso automatico de calibracion de confianza.

## Mejoras futuras razonables

- Introducir baseline de usuario/item para corregir sesgos de rating.
- Pasar a un modelo usuario-item con factorizacion matricial.
- Ajustar dinamicamente el umbral de soporte minimo.
- Aprender una formula de confianza mas sofisticada.
- Mejorar la explicabilidad con ejemplos de vecinos y peliculas puente.

## Archivos clave

- Implementacion principal:
  [Backend/Colaborativo/colaborative_recommender.py](</c:/Users/juanc/Documents/UPV/Master/SR/Backend/Colaborativo/colaborative_recommender.py>)
- Capa de datos del frontend:
  [Frontend/services/app_data_manager.py](</c:/Users/juanc/Documents/UPV/Master/SR/Frontend/services/app_data_manager.py>)
- Presentacion en UI:
  [Frontend/ui/recommendations_mixin.py](</c:/Users/juanc/Documents/UPV/Master/SR/Frontend/ui/recommendations_mixin.py>)

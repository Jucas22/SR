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

### Lo que se ha realizado:

#### 1. **Limpieza y Transformación de Datos**

Dividiendo ratings en train (70%) y test (30%)...
Usuarios procesados: 670
Usuarios con un solo rating (asignados a train): 0
Ratings de entrenamiento: 22114
Ratings de test: 9902
Total: 32016 (original: 32016)
Guardado: Data/Clean_data/train_ratings.json
Guardado: Data/Clean_data/test_ratings.json

Que se a a;adido al trabajo original: - Se ha actualizado la info de las pelis y se ha a;adido mas campos como el overwiew, produccion, y algunas cosas mas

**Resultado de este paso**:

- **tes_ratings.json and train_ratings.json:** dos ficheros con la informacion de los ratings de cada usuario, divididos en train y test. Cada rating tiene el id del usuario, el id de la pelicula, y la puntuacion que le ha dado el usuario a esa pelicula.
- **user_registry.json:** un fichero con la informacion de cada usuario, con su id, su lista de peliculas puntuadas, y su vector de preferencias (con un valor para cada genero). Este fichero se usará posteriormente para recomendar peliculas a los usuarios. SOLO continene informacion de Train.
  - User vector: se ha decidido rellenar el vector de preferencias mediante **acumulación ponderada positiva normalizada a [0, 100]**:
    1. Para cada usuario se calcula su rating medio: $\bar{r}_u = \frac{1}{|R_u|}\sum_{i \in R_u} r_{u,i}$
    2. Solo se consideran las películas con rating por encima de la media ($r_{u,i} > \bar{r}_u$), es decir, las que le gustaron más de lo habitual.
    3. El peso de cada rating positivo es $w_{u,i} = r_{u,i} - \bar{r}_u$ y se distribuye equitativamente entre los géneros $G_i$ de la película: cada género recibe $\frac{w_{u,i}}{|G_i|}$.
    4. Para cada género $g$ se calcula la afinidad media: $a_{u,g} = \frac{\sum_{i: g \in G_i} w_{u,i}/|G_i|}{|\{i : g \in G_i\}|}$
    5. Se normaliza a [0, 100]: $v_{u,g} = \frac{a_{u,g}}{\max_g(a_{u,g})} \times 100$
    - Se usan solo valores positivos [0, 100] porque así la similitud coseno queda acotada a [0, 1] (interpretable como porcentaje de afinidad), se evita la cancelación entre géneros, y es consistente con los pesos TF-IDF no-negativos del espacio de ítems
- **enhanced_movies.json:** un fichero con la informacion de cada pelicula, con su id, su genero, su titulo, su sinopsis, etc. Este fichero se usará posteriormente para recomendar peliculas a los usuarios. En este fichero se ha añadido mas informacion que en el cleaned_movies.json, como el overview, la produccion, etc.


## SR colaborativo:

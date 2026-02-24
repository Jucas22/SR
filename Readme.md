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

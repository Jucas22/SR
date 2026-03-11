"""
Inicializa el vector de preferencias de género para cada usuario del sistema.

Método: Acumulación ponderada positiva normalizada a [0, 100].
  1. Para cada usuario, solo se consideran ratings por encima de un umbral
     (media del usuario), que representan películas que le gustaron.
  2. El peso de cada rating es (rating - umbral), siempre positivo.
  3. Se acumula y promedia por género.
  4. Se normaliza a [0, 100] donde 100 = género de máxima afinidad.

¿Por qué solo valores positivos en [0, 100]?
  - Al usar solo valores ≥ 0, la similitud coseno queda acotada a [0, 1],
    lo que la hace directamente interpretable como porcentaje de afinidad.
  - Vectores no-negativos evitan que géneros se "cancelen" entre sí durante
    el cálculo de similitud, lo que podría producir recomendaciones erráticas.
  - El rango 0-100 es intuitivo: 0 = sin interés, 100 = máxima afinidad.
  - Es consistente con enfoques estándar de perfiles TF-IDF en sistemas
    basados en contenido, donde los pesos son siempre no-negativos.

Solo usa datos de TRAIN para evitar fuga de información.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Para importar Backend
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# ── Rutas ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
TRAIN_RATINGS_PATH = BASE_DIR / "Clean_data" / "train_ratings.json"
MOVIES_PATH = BASE_DIR / "Clean_data" / "enhanced_movies.json"
CLEANED_MOVIES_PATH = BASE_DIR / "Clean_data" / "cleaned_movies.json"
GENRE_CSV_PATH = BASE_DIR / "Raw_data" / "generos.csv"
REGISTRY_PATH = BASE_DIR / "user_registry.json"
OUTPUT_PATH = BASE_DIR / "user_vectors.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_genre_map(csv_path):
    """Carga mapeo IdDataset → nombre en español desde generos.csv"""
    genre_map = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(";")
            if len(parts) >= 4:
                genre_id = int(parts[1])
                genre_name = parts[3]
                genre_map[genre_id] = genre_name
    return genre_map


def build_movie_genre_index(movies):
    """Crea dict movieId → lista de genre_ids"""
    index = {}
    for m in movies:
        mid = m.get("id") or m.get("movieId")
        genres = m.get("generos", [])
        if isinstance(genres, list):
            index[int(mid)] = [int(g) for g in genres]
    return index


def build_user_vectors(train_ratings, movie_genres, genre_map):
    """
    Construye vector de preferencias por género para cada usuario.

    Método: acumulación ponderada positiva.
    - Solo se usan ratings por encima de la media del usuario (películas que
      le gustaron más de lo habitual).
    - El peso es (rating - media), siempre > 0.
    - Se promedia por género y se normaliza a [0, 100].
    """
    # Agrupar ratings por usuario
    ratings_by_user = defaultdict(list)
    for r in train_ratings:
        ratings_by_user[int(r["userId"])].append(r)

    all_genre_ids = sorted(genre_map.keys())
    user_vectors = {}

    for user_id, ratings in ratings_by_user.items():
        # Media del usuario (umbral de "le gustó")
        mean_rating = sum(r["rating"] for r in ratings) / len(ratings)

        # Acumuladores por género (solo ratings positivos)
        genre_score_sum = defaultdict(float)
        genre_count = defaultdict(int)

        for r in ratings:
            movie_id = int(r["movieId"])
            weight = r["rating"] - mean_rating
            if weight <= 0:
                continue  # Ignorar películas que no le gustaron
            genres = movie_genres.get(movie_id, [])
            if not genres:
                continue
            # Distribuir el peso positivo entre los géneros de la película
            weight_per_genre = weight / len(genres)
            for g in genres:
                genre_score_sum[g] += weight_per_genre
                genre_count[g] += 1

        # Construir vector: media de los pesos positivos por género
        raw_vector = {}
        for g in all_genre_ids:
            if genre_count[g] > 0:
                raw_vector[g] = genre_score_sum[g] / genre_count[g]
            else:
                raw_vector[g] = 0.0

        # Normalizar a [0, 100]
        max_val = max(raw_vector.values(), default=1.0)
        if max_val > 0:
            normalized = {
                g: round((v / max_val) * 100, 2) for g, v in raw_vector.items()
            }
        else:
            normalized = {g: 0.0 for g in raw_vector}

        # Guardar con nombres legibles
        vector_named = {}
        for genre_id in all_genre_ids:
            name = genre_map.get(genre_id, f"genre_{genre_id}")
            vector_named[name] = normalized.get(genre_id, 0.0)

        # Top géneros favoritos (positivos, ordenados)
        top_genres = sorted(
            [(name, score) for name, score in vector_named.items() if score > 0],
            key=lambda x: x[1],
            reverse=True,
        )

        user_vectors[str(user_id)] = {
            "user_id": user_id,
            "genre_vector": vector_named,
            "top_genres": [g[0] for g in top_genres[:5]],
            "num_ratings": len(ratings),
            "mean_rating": round(mean_rating, 2),
        }

    return user_vectors


def update_registry_with_vectors(registry_path, user_vectors):
    """Actualiza user_registry.json con el vector de género y top géneros de cada usuario."""
    registry = load_json(registry_path)
    updated = 0
    for uid_str, vec in user_vectors.items():
        if uid_str in registry["users"]:
            user = registry["users"][uid_str]
            user["preferences"]["favorite_genres"] = vec["top_genres"]
            user["preferences"]["genre_vector"] = vec["genre_vector"]
            user["statistics"]["mean_rating"] = vec["mean_rating"]
            updated += 1
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    print(
        f"✅ user_registry.json actualizado: {updated} usuarios con genre_vector y favorite_genres"
    )


def main():
    print("=== INICIALIZACIÓN DE VECTORES DE USUARIO ===")
    print(f"  Solo datos de TRAIN (sin leakage de test)\n")

    # Cargar datos
    print("📖 Cargando datos...")
    train_ratings = load_json(TRAIN_RATINGS_PATH)
    genre_map = load_genre_map(GENRE_CSV_PATH)

    # Intentar enhanced_movies primero, fallback a cleaned_movies
    if MOVIES_PATH.exists():
        movies = load_json(MOVIES_PATH)
        print(f"   Películas (enhanced): {len(movies)}")
    else:
        movies = load_json(CLEANED_MOVIES_PATH)
        print(f"   Películas (cleaned): {len(movies)}")

    print(f"   Ratings de train: {len(train_ratings)}")
    print(f"   Géneros: {len(genre_map)}")

    # Construir índice película → géneros
    movie_genres = build_movie_genre_index(movies)
    print(f"   Películas con géneros: {len(movie_genres)}")

    # Construir vectores
    print("\n🔧 Construyendo vectores de usuario...")
    user_vectors = build_user_vectors(train_ratings, movie_genres, genre_map)
    print(f"   Vectores generados: {len(user_vectors)}")

    # Guardar vectores
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(user_vectors, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Guardado: {OUTPUT_PATH}")

    # Actualizar user_registry.json con vectores y favorite_genres
    if REGISTRY_PATH.exists():
        update_registry_with_vectors(REGISTRY_PATH, user_vectors)

    # Estadísticas
    print("\n📊 Estadísticas:")
    all_top = defaultdict(int)
    for vec in user_vectors.values():
        for g in vec["top_genres"]:
            all_top[g] += 1
    print("   Géneros más populares como favorito:")
    for genre, count in sorted(all_top.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"     {genre}: {count} usuarios")

    # Ejemplo
    sample_id = list(user_vectors.keys())[0]
    sample = user_vectors[sample_id]
    print(f"\n📋 Ejemplo (usuario {sample_id}):")
    print(f"   Top géneros: {sample['top_genres']}")
    print(f"   Ratings: {sample['num_ratings']}, media: {sample['mean_rating']}")
    print(f"   Vector:")
    for name, score in sorted(
        sample["genre_vector"].items(), key=lambda x: x[1], reverse=True
    ):
        bar = "█" * int(abs(score) * 20)
        sign = "+" if score >= 0 else "-"
        print(f"     {name:20s} {sign}{abs(score):.4f} {bar}")


if __name__ == "__main__":
    main()

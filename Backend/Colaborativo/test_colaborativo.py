import sys
import time

from colaborative_recommender import ColaborativeRecommender


def main():
    user_registry_path = "Data/user_registry.json"
    movie_data_path = "Data/Clean_data/enhanced_movies.json"
    rec = ColaborativeRecommender(user_registry_path, movie_data_path)

    # 1. Construir/cargar la matriz de preferencias
    print("=== Matriz de preferencias ===")
    matrix = rec.get_preference_matrix()
    print(f"Shape: {matrix.shape}\n")

    # 2. Buscar vecinos para un usuario de ejemplo
    test_user = "1"
    print(f"=== Vecinos del usuario {test_user} ===")
    t0 = time.time()
    neighbors, user_vec = rec._find_neighbors(test_user, top_n=40)
    print(f"Calculado en {time.time() - t0:.2f}s")
    for nid, sim in neighbors:
        print(f"  Vecino {nid}:     Pearson = {sim:.4f} Vector = {user_vec}")
    print()

    # 3. Recomendaciones para el usuario
    print(f"=== Recomendaciones para usuario {test_user} ===")
    recommendations = rec.recommend(test_user, top_n=10)
    for movie_id, pred_rating in recommendations:
        print(f"  Película {movie_id}: rating predicho = {pred_rating:.3f}")
    print()

    # 4. Explicación de la primera recomendación
    if recommendations:
        first_movie = recommendations[0][0]
        print(f"=== Explicación para película {first_movie} ===")
        explanation = rec.explain_recommendation(test_user, first_movie)
        print(f"  Rating predicho: {explanation.get('predicted_rating')}")
        print(f"  Vecinos contribuyentes: {explanation.get('num_contributors')}")
        for c in explanation.get("contributors", [])[:3]:
            print(
                f"    Vecino {c['neighbor_id']}: sim={c['similarity']}, rating={c['rating']}"
            )


if __name__ == "__main__":
    main()

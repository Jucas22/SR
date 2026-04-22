import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Backend.recommenders import ColaborativeRecommender
from project_paths import CLEAN_DATA_DIR, USER_REGISTRY_PATH


def build_recommender() -> ColaborativeRecommender:
    """Construye una instancia lista para pruebas manuales del SR colaborativo."""
    user_registry_path = str(USER_REGISTRY_PATH)
    movie_data_path = str(CLEAN_DATA_DIR / "enhanced_movies.json")
    return ColaborativeRecommender(user_registry_path, movie_data_path)


def refresh_user_matrix(
    recommender: ColaborativeRecommender | None = None,
) -> ColaborativeRecommender:
    """
    Reconstruye la matriz usuario-genero desde el registro actual de usuarios.

    Si no se pasa una instancia, crea una nueva y devuelve el recomendador ya
    actualizado para poder reutilizarlo en el resto del test.
    """
    recommender = recommender or build_recommender()
    print("=== Actualizando matriz de preferencias ===")
    t0 = time.time()
    matrix = recommender.refresh_preference_matrix()
    print(f"Actualizada en {time.time() - t0:.2f}s")
    print(f"Shape: {matrix.shape}\n")
    return recommender


def run_smoke_test(
    recommender: ColaborativeRecommender,
    test_user: str = "1",
    top_neighbors: int = 40,
    top_recommendations: int = 10,
):
    """Ejecuta una pasada rapida del recomendador colaborativo."""
    print("=== Matriz de preferencias ===")
    matrix = recommender.get_preference_matrix()
    print(f"Shape: {matrix.shape}\n")

    print(f"=== Vecinos del usuario {test_user} ===")
    t0 = time.time()
    neighbors = recommender._find_neighbors(test_user, top_n=top_neighbors)
    print(f"Calculado en {time.time() - t0:.2f}s")
    for neighbor_id, similarity, user_vec in neighbors:
        print(
            f"  Vecino {neighbor_id}: Pearson = {similarity:.4f}: "
            f"user_vec={user_vec}"
        )
    print()

    print(f"=== Recomendaciones para usuario {test_user} ===")
    recommendations = recommender.recommend(test_user, top_n=top_recommendations)
    for movie_id, predicted_rating in recommendations:
        print(f"  Pelicula {movie_id}: rating predicho = {predicted_rating:.3f}")
    print()

    if recommendations:
        first_movie = recommendations[0][0]
        print(f"=== Explicacion para pelicula {first_movie} ===")
        explanation = recommender.explain_recommendation(test_user, first_movie)
        print(f"  Rating predicho: {explanation.get('predicted_rating')}")
        print(f"  Vecinos contribuyentes: {explanation.get('num_contributors')}")
        for contributor in explanation.get("contributors", [])[:3]:
            print(
                f"    Vecino {contributor['neighbor_id']}: "
                f"sim={contributor['similarity']}, rating={contributor['rating']}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke test y utilidades del recomendador colaborativo.",
    )
    parser.add_argument(
        "--refresh-matrix",
        action="store_true",
        help="Reconstruye la matriz usuario-genero antes de ejecutar el test.",
    )
    parser.add_argument(
        "--user-id",
        default="1",
        help="Usuario de prueba para vecinos y recomendaciones.",
    )
    parser.add_argument(
        "--top-neighbors",
        type=int,
        default=40,
        help="Numero de vecinos a mostrar en el test.",
    )
    parser.add_argument(
        "--top-recommendations",
        type=int,
        default=10,
        help="Numero de recomendaciones a mostrar en el test.",
    )
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    recommender = build_recommender()

    if args.refresh_matrix:
        recommender = refresh_user_matrix(recommender)

    run_smoke_test(
        recommender,
        test_user=str(args.user_id),
        top_neighbors=args.top_neighbors,
        top_recommendations=args.top_recommendations,
    )


if __name__ == "__main__":
    main()

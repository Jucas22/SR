#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para analizar la división train/test de usuarios y ratings
"""

import pandas as pd
import json


def analyze_train_test_split():
    """Analiza la división train/test y genera estadísticas por usuario"""
    print("=== ANÁLISIS DE DIVISIÓN TRAIN/TEST ===\n")

    # Cargar datos
    print("Cargando datos...")
    train_df = pd.read_csv(
        "Data_set/train_ratings.csv", sep=";", decimal=",", encoding="utf-8"
    )
    test_df = pd.read_csv(
        "Data_set/test_ratings.csv", sep=";", decimal=",", encoding="utf-8"
    )
    ratings_clean = pd.read_csv(
        "Data_set/ratings_clean.csv", sep=";", decimal=",", encoding="utf-8"
    )

    print(f"Ratings de entrenamiento: {len(train_df)}")
    print(f"Ratings de test: {len(test_df)}")
    print(f"Ratings totales: {len(ratings_clean)}")

    # Analizar usuarios
    print("\n=== ANÁLISIS POR USUARIO ===")

    train_users = set(train_df["userId"].unique())
    test_users = set(test_df["userId"].unique())
    all_users = set(ratings_clean["userId"].unique())

    print(f"Usuarios únicos totales: {len(all_users)}")
    print(f"Usuarios con datos en TRAIN: {len(train_users)}")
    print(f"Usuarios con datos en TEST: {len(test_users)}")
    print(f"Usuarios en ambos conjuntos: {len(train_users & test_users)}")
    print(f"Usuarios SOLO en train: {len(train_users - test_users)}")
    print(f"Usuarios SOLO en test: {len(test_users - train_users)}")

    # Detalles por usuario
    user_analysis = []

    for user_id in all_users:
        train_count = len(train_df[train_df["userId"] == user_id])
        test_count = len(test_df[test_df["userId"] == user_id])
        total_count = len(ratings_clean[ratings_clean["userId"] == user_id])

        user_info = {
            "user_id": int(user_id),
            "total_ratings": total_count,
            "train_ratings": train_count,
            "test_ratings": test_count,
            "train_percentage": (
                round((train_count / total_count) * 100, 1) if total_count > 0 else 0
            ),
            "test_percentage": (
                round((test_count / total_count) * 100, 1) if total_count > 0 else 0
            ),
            "in_train": train_count > 0,
            "in_test": test_count > 0,
            "only_in_train": train_count > 0 and test_count == 0,
            "only_in_test": train_count == 0 and test_count > 0,
            "in_both": train_count > 0 and test_count > 0,
        }

        user_analysis.append(user_info)

    # Guardar análisis detallado
    with open("Data_set/user_train_test_analysis.json", "w", encoding="utf-8") as f:
        json.dump(user_analysis, f, ensure_ascii=False, indent=2)
    print("\nGuardado: Data_set/user_train_test_analysis.json")

    # Estadísticas resumidas
    users_only_train = [u for u in user_analysis if u["only_in_train"]]
    users_only_test = [u for u in user_analysis if u["only_in_test"]]
    users_in_both = [u for u in user_analysis if u["in_both"]]

    print(f"\n=== ESTADÍSTICAS RESUMIDAS ===")
    print(f"Usuarios con datos SOLO en train: {len(users_only_train)}")
    print(f"Usuarios con datos SOLO en test: {len(users_only_test)}")
    print(f"Usuarios con datos en AMBOS: {len(users_in_both)}")

    # IDs de usuarios por conjunto
    user_sets = {
        "usuarios_solo_train": [int(u["user_id"]) for u in users_only_train],
        "usuarios_solo_test": [int(u["user_id"]) for u in users_only_test],
        "usuarios_en_ambos": [int(u["user_id"]) for u in users_in_both],
        "todos_usuarios_train": [int(x) for x in train_users],
        "todos_usuarios_test": [int(x) for x in test_users],
    }

    with open("Data_set/user_ids_by_set.json", "w", encoding="utf-8") as f:
        json.dump(user_sets, f, ensure_ascii=False, indent=2)
    print("Guardado: Data_set/user_ids_by_set.json")

    # Mostrar algunos ejemplos
    print(f"\n=== EJEMPLOS ===")
    if users_only_train:
        print(f"Primeros 10 usuarios SOLO en train: {users_only_train[:10]}")
        print(f"  (Estos son usuarios con 1 solo rating)")

    if users_in_both:
        print(
            f"Primeros 10 usuarios en AMBOS conjuntos: {[u['user_id'] for u in users_in_both[:10]]}"
        )
        print("  Ejemplo de división para usuario", users_in_both[0]["user_id"], ":")
        print(f"    Total: {users_in_both[0]['total_ratings']} ratings")
        print(
            f"    Train: {users_in_both[0]['train_ratings']} ({users_in_both[0]['train_percentage']}%)"
        )
        print(
            f"    Test: {users_in_both[0]['test_ratings']} ({users_in_both[0]['test_percentage']}%)"
        )

    # Distribución de ratings por usuario
    rating_distributions = {}
    for user_info in user_analysis:
        total = user_info["total_ratings"]
        if total not in rating_distributions:
            rating_distributions[total] = 0
        rating_distributions[total] += 1

    print(f"\n=== DISTRIBUCIÓN DE RATINGS POR USUARIO ===")
    sorted_dist = sorted(rating_distributions.items())
    for num_ratings, num_users in sorted_dist[:10]:  # Primeros 10
        print(f"  {num_ratings} ratings: {num_users} usuarios")

    print(f"\n✅ Análisis completado!")
    print("Archivos generados:")
    print("- user_train_test_analysis.json (análisis detallado por usuario)")
    print("- user_ids_by_set.json (IDs de usuarios por conjunto)")


if __name__ == "__main__":
    analyze_train_test_split()

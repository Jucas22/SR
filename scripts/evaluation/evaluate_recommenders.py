#!/usr/bin/env python
"""
Evaluacion offline del SR (Trabajo 7).

Calcula Precision, Recall, F1, MAE y nDCG para los tres recomendadores
(Basado en Contenido, Colaborativo e Hibrido) usando los ratings de test.

Genera:
- Graficas PNG en scripts/evaluation/output/figures/
- Excel con promedios y detalle por usuario en scripts/evaluation/output/

Uso:
    python -m scripts.evaluation.evaluate_recommenders
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project_paths import CLEAN_DATA_DIR, USER_REGISTRY_PATH
from Backend.recommenders import ContentBasedRecommender, ColaborativeRecommender
from Backend.recommenders.hybrid import HybridRecommender

# ---------------------------------------------------------------------------
# Parametros de evaluacion
# ---------------------------------------------------------------------------
SEED = 42
N_USERS = 10          # None = todos los usuarios de test
TOP_K = 50              # tamano de la lista de recomendaciones por usuario
MIN_TEST_RATINGS = 5    # incluir todos los usuarios con al menos 5 ratings en test
UMBRAL_REC = 3.5        # umbral para considerar un item recomendado relevante
UMBRAL_TEST = 3.5       # umbral para considerar un item de test relevante

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Carga de datos y recomendadores
# ---------------------------------------------------------------------------
def load_data():
    movies_path = CLEAN_DATA_DIR / "enhanced_movies.json"
    train_path = CLEAN_DATA_DIR / "train_ratings.json"
    test_path = CLEAN_DATA_DIR / "test_ratings.json"

    with open(movies_path, "r", encoding="utf-8") as f:
        movies_data = json.load(f)
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    with open(USER_REGISTRY_PATH, "r", encoding="utf-8") as f:
        user_registry = json.load(f)

    movies_df = pd.DataFrame(movies_data)
    if "id" in movies_df.columns:
        movies_df = movies_df.rename(columns={"id": "movieId"})

    text_cols = ["overview", "tagline", "keywords", "tags", "cast", "crew", "director"]
    for col in text_cols + ["title", "generos"]:
        if col not in movies_df.columns:
            movies_df[col] = ""
    for col in text_cols:
        movies_df[col] = movies_df[col].fillna("").astype(str)

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # Restringir test a peliculas existentes (slide: "eliminar peliculas que no existan")
    valid_movie_ids = set(movies_df["movieId"].astype(int).tolist())
    test_df = test_df[test_df["movieId"].astype(int).isin(valid_movie_ids)].copy()

    return movies_df, train_df, test_df, user_registry


def build_recommenders(movies_df: pd.DataFrame, train_df: pd.DataFrame, user_registry: dict):
    print("[+] Construyendo recomendador BC...")
    content = ContentBasedRecommender(
        categorical_columns=["generos"],
        feature_text_columns=["overview", "keywords", "tags"],
        profile_strategy="weighted",
        min_reduced_genres=5,
        max_reduced_genres=8,
    )
    content.fit(movies_df, ratings=train_df, user_registry=user_registry)

    print("[+] Construyendo recomendador Colaborativo...")
    collaborative = ColaborativeRecommender(
        user_registry_path=str(USER_REGISTRY_PATH),
        movie_data_path=str(CLEAN_DATA_DIR / "enhanced_movies.json"),
    )

    print("[+] Construyendo recomendador Hibrido...")
    hybrid = HybridRecommender(content_model=content, collaborative_model=collaborative)

    return {"BC": content, "Col": collaborative, "Hibrido": hybrid}


# ---------------------------------------------------------------------------
# Adaptadores: producen una lista uniforme [(movie_id, predicted_rating), ...]
# ---------------------------------------------------------------------------
def _score_to_rating(score: float) -> float:
    """Convierte un score normalizado [0,1] a un rating en [0.5, 5]."""
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return 0.0
    return float(max(0.0, min(1.0, score))) * 5.0


def recommend_bc(model, user_id: int, top_k: int) -> List[Tuple[int, float]]:
    df = model.recommend(user_id=user_id, top_k=top_k, exclude_seen=True, return_df=True)
    if df is None or len(df) == 0:
        return []
    return [(int(r["movie_id"]), _score_to_rating(r["score"])) for _, r in df.iterrows()]


def recommend_col(model, user_id: int, top_k: int) -> List[Tuple[int, float]]:
    raw = model.recommend(str(user_id), top_n=top_k)
    return [(int(mid), float(rating)) for mid, rating in raw]


def recommend_hybrid(model, user_id: int, top_k: int) -> List[Tuple[int, float]]:
    raw = model.recommend(user_id=user_id, top_k=top_k, return_details=False)
    return [(int(mid), _score_to_rating(score)) for mid, score in raw]


RECOMMEND_FNS = {
    "BC": recommend_bc,
    "Col": recommend_col,
    "Hibrido": recommend_hybrid,
}


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------
def precision_recall_f1(recs: List[Tuple[int, float]], test_ratings: Dict[int, float]):
    rec_relevantes = {mid for mid, r in recs if r >= UMBRAL_REC}
    test_relevantes = {mid for mid, r in test_ratings.items() if r >= UMBRAL_TEST}

    inter = rec_relevantes & test_relevantes
    precision = len(inter) / len(rec_relevantes) if rec_relevantes else 0.0
    recall = len(inter) / len(test_relevantes) if test_relevantes else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def mae(recs: List[Tuple[int, float]], test_ratings: Dict[int, float]):
    diffs = []
    for mid, pred in recs:
        if mid in test_ratings:
            diffs.append(abs(pred - test_ratings[mid]))
    if not diffs:
        return None
    return float(np.mean(diffs))


def ndcg(recs: List[Tuple[int, float]], test_ratings: Dict[int, float]) -> float:
    # 1) Filtrar recomendados >= umbral_rec
    rec_filtrados = [(mid, r) for mid, r in recs if r >= UMBRAL_REC]
    # 2) Filtrar test >= umbral_test y asignar score = rating (definicion libre)
    test_filtrados = {mid: r for mid, r in test_ratings.items() if r >= UMBRAL_TEST}

    if not test_filtrados:
        return 0.0

    # 3) Score por item de la lista de recomendados (0 si no esta en test)
    scores_rec = [test_filtrados.get(mid, 0.0) for mid, _ in rec_filtrados]
    # 4) Ideal: lista de test ordenada por score descendente
    scores_ideal = sorted(test_filtrados.values(), reverse=True)

    def _dcg(scores):
        return sum(s / math.log2(i + 2) for i, s in enumerate(scores))

    dcg_rec = _dcg(scores_rec)
    idcg = _dcg(scores_ideal)
    return dcg_rec / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Bucle principal de evaluacion
# ---------------------------------------------------------------------------
def select_users(test_df: pd.DataFrame, registry_user_ids: set, n, seed: int) -> List[int]:
    counts = test_df.groupby("userId").size()
    candidates = [
        int(uid) for uid, c in counts.items()
        if c >= MIN_TEST_RATINGS and str(uid) in registry_user_ids
    ]
    if n is None:
        return sorted(candidates)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return sorted(candidates[:n])


def evaluate():
    movies_df, train_df, test_df, user_registry = load_data()
    registry_user_ids = set(user_registry.get("users", user_registry).keys())

    users = select_users(test_df, registry_user_ids, N_USERS, SEED)
    print(f"[+] Usuarios seleccionados ({len(users)}): {users}")

    recommenders = build_recommenders(movies_df, train_df, user_registry)

    # Agrupar test por usuario
    test_by_user: Dict[int, Dict[int, float]] = {}
    for uid, group in test_df.groupby("userId"):
        test_by_user[int(uid)] = {
            int(row.movieId): float(row.rating) for row in group.itertuples()
        }

    rows = []
    for uid in users:
        test_ratings = test_by_user.get(uid, {})
        if not test_ratings:
            continue
        print(f"  - Evaluando usuario {uid} (test={len(test_ratings)} items)...")
        for sr_name, model in recommenders.items():
            try:
                recs = RECOMMEND_FNS[sr_name](model, uid, TOP_K)
            except Exception as exc:
                print(f"    [!] {sr_name} fallo para usuario {uid}: {exc}")
                recs = []
            p, r, f1 = precision_recall_f1(recs, test_ratings)
            m = mae(recs, test_ratings)
            nd = ndcg(recs, test_ratings)
            rows.append({
                "userId": uid,
                "SR": sr_name,
                "n_recomendados": len(recs),
                "n_test": len(test_ratings),
                "precision": p,
                "recall": r,
                "f1": f1,
                "mae": m,
                "ndcg": nd,
            })

    return pd.DataFrame(rows), users


# ---------------------------------------------------------------------------
# Salida: graficas y Excel
# ---------------------------------------------------------------------------
SR_NAMES = ["BC", "Col", "Hibrido"]
SR_COLORS = {"BC": "#1f77b4", "Col": "#d62728", "Hibrido": "#2ca02c"}


def _pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df.pivot(index="userId", columns="SR", values=metric).reindex(columns=SR_NAMES)


def _setup_user_axis(ax, user_ids):
    n = len(user_ids)
    x = np.arange(n)
    if n <= 30:
        ax.set_xticks(x)
        ax.set_xticklabels([str(u) for u in user_ids], rotation=45, fontsize=8)
    else:
        step = max(1, n // 20)
        idx = list(range(0, n, step))
        ax.set_xticks(idx)
        ax.set_xticklabels([str(user_ids[i]) for i in idx], rotation=45, fontsize=8)
    return x


def plot_per_sr(df: pd.DataFrame):
    """Graficas 1-3: una por SR con lineas para precision, recall, F1."""
    for sr in SR_NAMES:
        sub = df[df["SR"] == sr].sort_values("userId")
        user_ids = sub["userId"].tolist()
        fig, ax = plt.subplots(figsize=(12, 5))
        x = _setup_user_axis(ax, user_ids)
        marker_size = 3 if len(user_ids) > 50 else 5
        ax.plot(x, sub["precision"].values, marker="o", markersize=marker_size, label="Precision", linewidth=1)
        ax.plot(x, sub["recall"].values, marker="s", markersize=marker_size, label="Recall", linewidth=1)
        ax.plot(x, sub["f1"].values, marker="^", markersize=marker_size, label="F1", linewidth=1)
        ax.set_title(f"Precision / Recall / F1 - SR {sr} ({len(user_ids)} usuarios)")
        ax.set_xlabel("Usuario")
        ax.set_ylabel("Valor")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"01_PRF1_{sr}.png", dpi=120)
        plt.close(fig)


def plot_per_metric(df: pd.DataFrame, metric: str, ylabel: str, fname: str, ylim=(0, 1)):
    """Graficas 4-6 (y MAE/nDCG): una por metrica con lineas para cada SR."""
    pivot = _pivot(df, metric)
    user_ids = pivot.index.tolist()
    fig, ax = plt.subplots(figsize=(12, 5))
    x = _setup_user_axis(ax, user_ids)
    marker_size = 3 if len(user_ids) > 50 else 5
    for sr in SR_NAMES:
        if sr in pivot.columns:
            ax.plot(x, pivot[sr].values, marker="o", markersize=marker_size, linewidth=1,
                    label=sr, color=SR_COLORS[sr])
    ax.set_title(f"{ylabel} por usuario ({len(user_ids)} usuarios)")
    ax.set_xlabel("Usuario")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=120)
    plt.close(fig)


def make_plots(df: pd.DataFrame):
    plot_per_sr(df)
    plot_per_metric(df, "precision", "Precision", "02_Precision_comparativa.png")
    plot_per_metric(df, "recall", "Recall", "03_Recall_comparativa.png")
    plot_per_metric(df, "f1", "F1", "04_F1_comparativa.png")
    # MAE: escala dependiente del rating
    mae_pivot = _pivot(df, "mae")
    max_mae = float(np.nanmax(mae_pivot.values)) if mae_pivot.size else 1.0
    plot_per_metric(df, "mae", "MAE", "05_MAE_comparativa.png", ylim=(0, max(1.0, max_mae * 1.1)))
    plot_per_metric(df, "ndcg", "nDCG", "06_nDCG_comparativa.png")


def write_excel(df: pd.DataFrame):
    avg = df.groupby("SR")[["precision", "recall", "f1", "mae", "ndcg"]].mean().reindex(SR_NAMES)
    avg = avg.rename_axis("SR").reset_index()

    excel_path = OUTPUT_DIR / "evaluacion_SR.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        avg.to_excel(writer, sheet_name="Promedios", index=False)
        df.to_excel(writer, sheet_name="Detalle", index=False)
        for metric in ["precision", "recall", "f1", "mae", "ndcg"]:
            _pivot(df, metric).to_excel(writer, sheet_name=f"por_usuario_{metric}")
    print(f"[+] Excel generado: {excel_path}")
    return avg


# ---------------------------------------------------------------------------
def main():
    df, users = evaluate()
    if df.empty:
        print("[!] No se obtuvieron resultados.")
        return
    df.to_csv(OUTPUT_DIR / "evaluacion_detalle.csv", index=False)
    make_plots(df)
    avg = write_excel(df)
    print("\n=== Promedios por SR ===")
    print(avg.to_string(index=False))
    print(f"\n[+] Figuras en: {FIG_DIR}")


if __name__ == "__main__":
    main()

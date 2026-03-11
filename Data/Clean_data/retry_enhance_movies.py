"""
Script para reintentar el enriquecimiento de películas que fallaron en el primer intento.
Lee peliculas_no_actualizadas.json, consulta TMDb y añade los resultados a enhanced_movies.json.
"""

import json
import time
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, Optional, List


# ── Configuración TMDb ──────────────────────────────────────────────
BASE_URL = "https://api.themoviedb.org/3/movie/{id}?language=en-US"
HEADERS = {
    "accept": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI5OTAxYjY1MTI1ZGRjMDVlYTM0YzQ1M2U1MWIzZDE5YiIsIm5iZiI6MTc3Mjk3MDUyOS4yNzQsInN1YiI6IjY5YWQ2MjIxNWUzNDUyMWFiNDBiMjMwZSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.Ct9BoFHz2Ft8RV1pxIkEkkJCdK6sclKAUgV_R2BX5Tk",
}

MAX_CONCURRENT_REQUESTS = 30
RATE_LIMIT_DELAY = 0.25
BATCH_SIZE = 100

# ── Rutas ────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent
INPUT_FILE = DATA_DIR / "peliculas_no_actualizadas.json"
ENHANCED_FILE = DATA_DIR / "enhanced_movies.json"
BACKUP_FILE = DATA_DIR / "enhanced_movies_backup.json"
PROGRESS_FILE = DATA_DIR / "progress_retry.json"


# ── Funciones auxiliares ─────────────────────────────────────────────
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 Guardado: {path} ({len(data)} películas)")


def merge_movie_data(existing: Dict, tmdb: Optional[Dict]) -> Dict:
    """Combina datos existentes con datos de TMDb (misma lógica que el script original)."""
    movie = existing.copy()

    if tmdb is None:
        movie.update(
            {
                "budget": None,
                "revenue": None,
                "runtime": None,
                "overview": None,
                "tagline": None,
                "release_date": None,
                "original_title": None,
                "original_language": None,
                "homepage": None,
                "status": None,
                "popularity": None,
                "vote_average_tmdb": None,
                "vote_count_tmdb": None,
                "backdrop_path": None,
                "production_companies": None,
                "production_countries": None,
                "spoken_languages": None,
                "belongs_to_collection": None,
                "adult": None,
            }
        )
        return movie

    movie.update(
        {
            "budget": tmdb.get("budget"),
            "revenue": tmdb.get("revenue"),
            "runtime": tmdb.get("runtime"),
            "overview": tmdb.get("overview"),
            "tagline": tmdb.get("tagline"),
            "release_date": tmdb.get("release_date"),
            "original_title": tmdb.get("original_title"),
            "original_language": tmdb.get("original_language"),
            "homepage": tmdb.get("homepage"),
            "status": tmdb.get("status"),
            "popularity": tmdb.get("popularity"),
            "vote_average_tmdb": tmdb.get("vote_average"),
            "vote_count_tmdb": tmdb.get("vote_count"),
            "backdrop_path": tmdb.get("backdrop_path"),
            "adult": tmdb.get("adult"),
            "belongs_to_collection": tmdb.get("belongs_to_collection"),
        }
    )

    if tmdb.get("poster_path"):
        movie["poster_path"] = tmdb["poster_path"]
    if tmdb.get("vote_average") is not None:
        movie["puntuacion_media"] = tmdb["vote_average"]
    if tmdb.get("vote_count") is not None:
        movie["votos"] = tmdb["vote_count"]

    if tmdb.get("production_companies"):
        movie["production_companies"] = [
            {
                "id": c.get("id"),
                "name": c.get("name"),
                "origin_country": c.get("origin_country"),
            }
            for c in tmdb["production_companies"]
        ]
    else:
        movie["production_companies"] = None

    if tmdb.get("production_countries"):
        movie["production_countries"] = [
            {"iso_3166_1": c.get("iso_3166_1"), "name": c.get("name")}
            for c in tmdb["production_countries"]
        ]
    else:
        movie["production_countries"] = None

    if tmdb.get("spoken_languages"):
        movie["spoken_languages"] = [
            {
                "iso_639_1": l.get("iso_639_1"),
                "name": l.get("name"),
                "english_name": l.get("english_name"),
            }
            for l in tmdb["spoken_languages"]
        ]
    else:
        movie["spoken_languages"] = None

    return movie


# ── Peticiones async ─────────────────────────────────────────────────
async def fetch_movie(session, movie_id, semaphore, max_retries=2):
    """Obtiene detalles de una película desde TMDb con reintentos."""
    async with semaphore:
        url = BASE_URL.format(id=movie_id)
        for attempt in range(max_retries + 1):
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        return movie_id, await resp.json()
                    elif resp.status == 429:
                        wait = 5 * (attempt + 1)
                        print(f"  ⏳ Rate limit para {movie_id}, esperando {wait}s...")
                        await asyncio.sleep(wait)
                        continue
                    elif resp.status == 404:
                        return movie_id, None
                    else:
                        print(f"  ⚠️ HTTP {resp.status} para {movie_id}")
                        return movie_id, None
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                if attempt < max_retries:
                    await asyncio.sleep(2)
                    continue
                print(f"  ❌ Error para {movie_id}: {e}")
                return movie_id, None
        return movie_id, None


async def process_batch(movies_batch, processed_ids):
    """Procesa un lote de películas en paralelo."""
    to_process = [m for m in movies_batch if m["id"] not in processed_ids]
    if not to_process:
        return [merge_movie_data(m, None) for m in movies_batch]

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    connector = aiohttp.TCPConnector(limit=50, limit_per_host=30)

    async with aiohttp.ClientSession(headers=HEADERS, connector=connector) as session:
        tasks = [fetch_movie(session, m["id"], semaphore) for m in to_process]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    tmdb_map = {}
    for r in results:
        if isinstance(r, tuple):
            tmdb_map[r[0]] = r[1]

    enhanced = []
    for m in movies_batch:
        mid = m["id"]
        if mid in tmdb_map:
            enhanced.append(merge_movie_data(m, tmdb_map[mid]))
            processed_ids.add(mid)
        else:
            enhanced.append(merge_movie_data(m, None))
    return enhanced


# ── Progreso ─────────────────────────────────────────────────────────
def load_progress():
    if PROGRESS_FILE.exists():
        try:
            return json.load(open(PROGRESS_FILE, "r"))
        except Exception:
            pass
    return {"processed": [], "last_index": 0}


def save_progress(processed_ids, idx):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"processed": list(processed_ids), "last_index": idx}, f)


# ── Main ─────────────────────────────────────────────────────────────
async def main():
    print("=== REINTENTO DE ENRIQUECIMIENTO DE PELÍCULAS FALLIDAS ===")
    print(f"Entrada:  {INPUT_FILE}")
    print(f"Salida:   {ENHANCED_FILE}")

    movies = load_json(INPUT_FILE)
    print(f"Películas a reintentar: {len(movies)}")

    # Cargar enhanced_movies existente
    existing_enhanced = load_json(ENHANCED_FILE)
    existing_ids = {m["id"] for m in existing_enhanced}
    print(f"Películas ya en enhanced: {len(existing_enhanced)}")

    # Progreso
    progress = load_progress()
    processed_ids = set(progress.get("processed", []))
    start_index = progress.get("last_index", 0)
    print(f"Ya procesadas en reintento previo: {len(processed_ids)}")

    new_enhanced = []

    try:
        for i in range(start_index, len(movies), BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, len(movies))
            batch = movies[i:batch_end]

            print(
                f"\n🚀 Lote {i // BATCH_SIZE + 1}: películas {i + 1}-{batch_end} de {len(movies)}"
            )

            enhanced_batch = await process_batch(batch, processed_ids)
            new_enhanced.extend(enhanced_batch)

            save_progress(processed_ids, batch_end)
            print(f"  ✅ Lote completado. Progreso: {batch_end}/{len(movies)}")

            # Backup cada 5 lotes
            if (i // BATCH_SIZE + 1) % 5 == 0:
                save_json(existing_enhanced + new_enhanced, BACKUP_FILE)

    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por el usuario. Guardando progreso...")
        save_progress(processed_ids, len(new_enhanced))
        save_json(existing_enhanced + new_enhanced, BACKUP_FILE)
        return

    # Combinar: enhanced existente + nuevas enriquecidas (sin duplicados)
    final = existing_enhanced.copy()
    for m in new_enhanced:
        if m["id"] not in existing_ids:
            final.append(m)
            existing_ids.add(m["id"])

    save_json(final, ENHANCED_FILE)

    # Limpiar progreso
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()

    # Estadísticas
    successful = sum(1 for m in new_enhanced if m.get("overview") is not None)
    failed = len(new_enhanced) - successful

    print(f"\n🎉 REINTENTO COMPLETADO")
    print(f"  Películas reintentadas: {len(new_enhanced)}")
    print(f"  Enriquecidas con éxito: {successful}")
    print(f"  Sin datos de TMDb:      {failed}")
    print(f"  Total en enhanced_movies.json: {len(final)}")


if __name__ == "__main__":
    asyncio.run(main())

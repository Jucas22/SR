import pandas as pd
import json
from pathlib import Path
import numpy as np
import random
import requests
import time
import asyncio
import aiohttp
from typing import Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor


def load_json_data(file_path):
    """Carga datos desde archivo JSON"""
    print(f"Cargando datos desde: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Datos cargados: {len(data)} películas")
        return data
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return []


def save_json_data(data, filename, indent=2):
    """Guarda datos en formato JSON"""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        print(f"Guardado: {filename} ({len(data)} películas)")
    except Exception as e:
        print(f"Error guardando datos: {e}")


# Configuración API TMDb
BASE_URL = "https://api.themoviedb.org/3/movie/{id}?language=en-US"
HEADERS = {
    "accept": "application/json",
    "Authorization": "....",
}

# Archivo de progreso para continuar desde donde se quedó si hay errores
PROGRESS_FILE = "progress_movies_update.json"

# Configuración de paralelización
MAX_CONCURRENT_REQUESTS = 30  # Máximo de peticiones simultáneas
RATE_LIMIT_DELAY = 0.25  # Delay base entre peticiones (4 por segundo)
BATCH_SIZE = 100  # Tamaño de lote para procesamiento


async def get_movie_details_from_tmdb_async(
    session: aiohttp.ClientSession, movie_id: int, semaphore: asyncio.Semaphore
) -> tuple[int, Optional[Dict]]:
    """
    Obtiene detalles de una película desde TMDb API de manera asíncrona
    """
    async with semaphore:  # Limitar concurrencia
        url = BASE_URL.format(id=movie_id)

        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return movie_id, data
                elif response.status == 404:
                    print(f"  Película {movie_id} no encontrada en TMDb")
                    return movie_id, None
                elif response.status == 429:
                    # Rate limit - esperar y reintentar
                    print(
                        f"  Rate limit alcanzado para película {movie_id}, esperando..."
                    )
                    await asyncio.sleep(5)
                    return await get_movie_details_from_tmdb_async(
                        session, movie_id, semaphore
                    )
                else:
                    print(f"  Error {response.status} para película {movie_id}")
                    return movie_id, None

        except asyncio.TimeoutError:
            print(f"  Timeout para película {movie_id}")
            return movie_id, None
        except Exception as e:
            print(f"  Error de conexión para película {movie_id}: {e}")
            return movie_id, None
        finally:
            # Rate limiting - esperar un poco entre peticiones
            await asyncio.sleep(RATE_LIMIT_DELAY / MAX_CONCURRENT_REQUESTS)


def get_movie_details_from_tmdb(movie_id: int) -> Optional[Dict]:
    """
    Función síncrona de respaldo para compatibilidad
    """
    url = BASE_URL.format(id=movie_id)

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            print(f"  Película {movie_id} no encontrada en TMDb")
            return None
        elif response.status_code == 429:
            print(f"  Rate limit alcanzado, esperando 10 segundos...")
            time.sleep(10)
            return get_movie_details_from_tmdb(movie_id)
        else:
            print(f"  Error {response.status_code} para película {movie_id}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"  Error de conexión para película {movie_id}: {e}")
        return None


def merge_movie_data(existing_movie: Dict, tmdb_data: Optional[Dict]) -> Dict:
    """
    Combina datos existentes con datos de TMDb
    """
    # Empezar con los datos existentes
    enhanced_movie = existing_movie.copy()

    if tmdb_data is None:
        # Si no hay datos de TMDb, añadir campos como null
        enhanced_movie.update(
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
        return enhanced_movie

    # Mapear campos de TMDb a nuestro formato
    enhanced_movie.update(
        {
            "budget": tmdb_data.get("budget"),
            "revenue": tmdb_data.get("revenue"),
            "runtime": tmdb_data.get("runtime"),
            "overview": tmdb_data.get("overview"),
            "tagline": tmdb_data.get("tagline"),
            "release_date": tmdb_data.get("release_date"),
            "original_title": tmdb_data.get("original_title"),
            "original_language": tmdb_data.get("original_language"),
            "homepage": tmdb_data.get("homepage"),
            "status": tmdb_data.get("status"),
            "popularity": tmdb_data.get("popularity"),
            "vote_average_tmdb": tmdb_data.get("vote_average"),
            "vote_count_tmdb": tmdb_data.get("vote_count"),
            "backdrop_path": tmdb_data.get("backdrop_path"),
            "adult": tmdb_data.get("adult"),
            "belongs_to_collection": tmdb_data.get("belongs_to_collection"),
        }
    )

    # Actualizar campos existentes con datos más recientes de TMDb
    if tmdb_data.get("poster_path"):
        enhanced_movie["poster_path"] = tmdb_data["poster_path"]

    if tmdb_data.get("vote_average") is not None:
        enhanced_movie["puntuacion_media"] = tmdb_data["vote_average"]

    if tmdb_data.get("vote_count") is not None:
        enhanced_movie["votos"] = tmdb_data["vote_count"]

    # Procesar listas más complejas
    if "production_companies" in tmdb_data and tmdb_data["production_companies"]:
        enhanced_movie["production_companies"] = [
            {
                "id": company.get("id"),
                "name": company.get("name"),
                "origin_country": company.get("origin_country"),
            }
            for company in tmdb_data["production_companies"]
        ]
    else:
        enhanced_movie["production_companies"] = None

    if "production_countries" in tmdb_data and tmdb_data["production_countries"]:
        enhanced_movie["production_countries"] = [
            {"iso_3166_1": country.get("iso_3166_1"), "name": country.get("name")}
            for country in tmdb_data["production_countries"]
        ]
    else:
        enhanced_movie["production_countries"] = None

    if "spoken_languages" in tmdb_data and tmdb_data["spoken_languages"]:
        enhanced_movie["spoken_languages"] = [
            {
                "iso_639_1": lang.get("iso_639_1"),
                "name": lang.get("name"),
                "english_name": lang.get("english_name"),
            }
            for lang in tmdb_data["spoken_languages"]
        ]
    else:
        enhanced_movie["spoken_languages"] = None

    return enhanced_movie


def load_progress() -> Dict:
    """Carga el progreso guardado"""
    try:
        if Path(PROGRESS_FILE).exists():
            with open(PROGRESS_FILE, "r") as f:
                return json.load(f)
    except:
        pass
    return {"processed": [], "last_index": 0}


def save_progress(processed_ids: List[int], current_index: int):
    """Guarda el progreso actual"""
    progress = {
        "processed": processed_ids,
        "last_index": current_index,
        "timestamp": time.time(),
    }
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


async def process_movies_batch_async(
    movies_batch: List[Dict], processed_ids: set
) -> List[Dict]:
    """
    Procesa un lote de películas de manera asíncrona
    """
    # Filtrar películas que ya han sido procesadas
    movies_to_process = [
        movie
        for movie in movies_batch
        if movie.get("id") and movie.get("id") not in processed_ids
    ]

    if not movies_to_process:
        return movies_batch  # Retornar sin cambios si todas están procesadas

    print(f"  Procesando lote de {len(movies_to_process)} películas en paralelo...")

    # Crear semáforo para limitar concurrencia
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Configurar cliente HTTP
    connector = aiohttp.TCPConnector(limit=50, limit_per_host=30)
    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(
        headers=HEADERS, connector=connector, timeout=timeout
    ) as session:

        # Crear tareas para todas las películas del lote
        tasks = [
            get_movie_details_from_tmdb_async(session, movie["id"], semaphore)
            for movie in movies_to_process
        ]

        # Ejecutar todas las tareas en paralelo
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Procesar resultados
    tmdb_data_dict = {}
    for result in results:
        if isinstance(result, tuple) and len(result) == 2:
            movie_id, data = result
            tmdb_data_dict[movie_id] = data
        elif isinstance(result, Exception):
            print(f"  Error en petición: {result}")

    # Combinar datos originales con datos de TMDb
    enhanced_movies = []
    for movie in movies_batch:
        movie_id = movie.get("id")
        if movie_id in tmdb_data_dict:
            tmdb_data = tmdb_data_dict[movie_id]
            enhanced_movie = merge_movie_data(movie, tmdb_data)
            processed_ids.add(movie_id)
        else:
            enhanced_movie = movie  # Mantener datos originales si no se procesó

        enhanced_movies.append(enhanced_movie)

    return enhanced_movies


async def main_async():
    """
    Función principal asíncrona para procesamiento paralelo
    """
    # Rutas de archivos
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent
    input_file = data_dir / "Clean_data" / "cleaned_movies.json"
    output_file = data_dir / "Clean_data" / "enhanced_movies.json"
    backup_file = data_dir / "Clean_data" / "enhanced_movies_backup.json"

    print("=== ENRIQUECIMIENTO PARALELO DE PELÍCULAS CON TMDb ===")
    print(f"Procesamiento paralelo: {MAX_CONCURRENT_REQUESTS} peticiones simultáneas")
    print(f"Archivo de entrada: {input_file}")
    print(f"Archivo de salida: {output_file}")

    # Cargar datos existentes
    movies = load_json_data(input_file)
    if not movies:
        print("Error: No se pudieron cargar las películas.")
        return

    # Cargar progreso previo
    progress = load_progress()
    processed_ids = set(progress.get("processed", []))
    start_index = progress.get("last_index", 0)

    print(f"Películas totales: {len(movies)}")
    print(f"Ya procesadas: {len(processed_ids)}")
    print(f"Comenzando desde índice: {start_index}")

    enhanced_movies = []

    try:
        # Procesar películas en lotes
        for i in range(start_index, len(movies), BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, len(movies))
            movies_batch = movies[i:batch_end]

            print(
                f"\n🚀 Procesando lote {i//BATCH_SIZE + 1}: películas {i+1}-{batch_end} de {len(movies)}"
            )

            # Procesar lote en paralelo
            enhanced_batch = await process_movies_batch_async(
                movies_batch, processed_ids
            )
            enhanced_movies.extend(enhanced_batch)

            # Guardar progreso
            save_progress(list(processed_ids), batch_end)
            print(f"  ✅ Lote completado. Progreso guardado: {batch_end}/{len(movies)}")

            # Guardar backup cada ciertos lotes
            if (i // BATCH_SIZE + 1) % 5 == 0:
                save_json_data(enhanced_movies, backup_file)
                print(f"  💾 Backup guardado: {len(enhanced_movies)} películas")

    except KeyboardInterrupt:
        print("\n🛑 Proceso interrumpido por el usuario")
        save_progress(list(processed_ids), len(enhanced_movies))
        save_json_data(enhanced_movies, backup_file)
        print(f"Progreso guardado. Películas procesadas: {len(enhanced_movies)}")
        return

    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        save_progress(list(processed_ids), len(enhanced_movies))
        save_json_data(enhanced_movies, backup_file)
        return

    # Guardar resultado final
    save_json_data(enhanced_movies, output_file)

    # Limpiar archivo de progreso
    if Path(PROGRESS_FILE).exists():
        Path(PROGRESS_FILE).unlink()

    # Estadísticas finales
    successful_enhancements = sum(
        1 for movie in enhanced_movies if movie.get("budget") is not None
    )

    print("\n🎉 PROCESO PARALELO COMPLETADO")
    print(f"Total de películas procesadas: {len(enhanced_movies)}")
    print(f"Películas enriquecidas con TMDb: {successful_enhancements}")
    print(
        f"Películas sin datos de TMDb: {len(enhanced_movies) - successful_enhancements}"
    )
    print(f"Eficiencia: {(successful_enhancements/len(enhanced_movies)*100):.1f}%")
    print(f"Archivo guardado en: {output_file}")

    # Mostrar ejemplo
    if enhanced_movies:
        sample_movie = next(
            (movie for movie in enhanced_movies if movie.get("overview")),
            enhanced_movies[0],
        )
        print(f"\n📋 Ejemplo de película enriquecida:")
        print(f"Título: {sample_movie.get('titulo', 'N/A')}")
        print(
            f"Puntuación actualizada: {sample_movie.get('puntuacion_media', 'N/A')}/10 (TMDb)"
        )
        print(
            f"Votos actualizados: {sample_movie.get('votos', 'N/A'):,}"
            if sample_movie.get("votos")
            else "Votos: N/A"
        )
        if sample_movie.get("budget"):
            print(f"Presupuesto: ${sample_movie.get('budget', 0):,}")
        if sample_movie.get("revenue"):
            print(f"Ingresos: ${sample_movie.get('revenue', 0):,}")


def main():
    """
    Función principal que ejecuta el procesamiento paralelo
    """
    asyncio.run(main_async())


def main_sequential():
    """
    Función principal secuencial (versión original como respaldo)
    """
    # Rutas de archivos
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent
    input_file = data_dir / "Clean_data" / "cleaned_movies.json"
    output_file = data_dir / "Clean_data" / "enhanced_movies_sequential.json"
    backup_file = data_dir / "Clean_data" / "enhanced_movies_backup_sequential.json"

    print("=== ENRIQUECIMIENTO SECUENCIAL DE PELÍCULAS CON TMDb ===")
    print(f"Archivo de entrada: {input_file}")
    print(f"Archivo de salida: {output_file}")

    # Cargar datos existentes
    movies = load_json_data(input_file)
    if not movies:
        print("Error: No se pudieron cargar las películas.")
        return

    # Cargar progreso previo
    progress = load_progress()
    processed_ids = set(progress.get("processed", []))
    start_index = progress.get("last_index", 0)

    print(f"Películas totales: {len(movies)}")
    print(f"Ya procesadas: {len(processed_ids)}")
    print(f"Comenzando desde índice: {start_index}")

    enhanced_movies = []

    try:
        for i, movie in enumerate(movies):
            if i < start_index:
                continue

            movie_id = movie.get("id")
            if not movie_id:
                print(f"Película {i+1}/{len(movies)}: Sin ID, saltando...")
                enhanced_movies.append(movie)
                continue

            if movie_id in processed_ids:
                print(
                    f"Película {i+1}/{len(movies)}: ID {movie_id} ya procesada, saltando..."
                )
                enhanced_movies.append(movie)
                continue

            print(
                f"Procesando película {i+1}/{len(movies)}: ID {movie_id} - '{movie.get('titulo', 'Sin título')}'"
            )

            # Hacer petición a TMDb
            tmdb_data = get_movie_details_from_tmdb(movie_id)

            # Combinar datos
            enhanced_movie = merge_movie_data(movie, tmdb_data)
            enhanced_movies.append(enhanced_movie)

            # Marcar como procesada
            processed_ids.add(movie_id)

            # Guardar progreso cada 10 películas
            if (i + 1) % 10 == 0:
                save_progress(list(processed_ids), i + 1)
                print(f"  Progreso guardado: {i + 1}/{len(movies)}")

            # Rate limiting - esperar entre peticiones
            time.sleep(0.25)  # 4 calls per second max

            # Guardar backup cada 50 películas
            if (i + 1) % 50 == 0:
                save_json_data(enhanced_movies, backup_file)
                print(f"  Backup guardado: {len(enhanced_movies)} películas")

    except KeyboardInterrupt:
        print("\n🛑 Proceso interrumpido por el usuario")
        save_progress(list(processed_ids), i)
        save_json_data(enhanced_movies, backup_file)
        print(f"Progreso guardado. Películas procesadas: {len(enhanced_movies)}")
        return

    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        save_progress(list(processed_ids), i)
        save_json_data(enhanced_movies, backup_file)
        return

    # Guardar resultado final
    save_json_data(enhanced_movies, output_file)
    print(f"\n🎉 PROCESO SECUENCIAL COMPLETADO")
    print(f"Archivo guardado en: {output_file}")


async def test_main_async(limit: int):
    """
    Versión de prueba asíncrona que procesa solo un número limitado de películas
    """
    # Rutas de archivos
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent
    input_file = data_dir / "Clean_data" / "cleaned_movies.json"
    output_file = (
        data_dir / "Clean_data" / f"enhanced_movies_test_{limit}_parallel.json"
    )

    print(f"=== MODO PRUEBA PARALELO: {limit} PELÍCULAS ===")
    print(f"Concurrencia: {MAX_CONCURRENT_REQUESTS} peticiones simultáneas")

    # Cargar datos existentes
    movies = load_json_data(input_file)
    if not movies:
        print("Error: No se pudieron cargar las películas.")
        return

    # Tomar solo las primeras N películas
    test_movies = movies[:limit]

    # Procesar en paralelo
    processed_ids = set()
    enhanced_movies = await process_movies_batch_async(test_movies, processed_ids)

    # Guardar resultado
    save_json_data(enhanced_movies, output_file)

    print(f"\n🎉 PRUEBA PARALELA COMPLETADA")
    print(f"Películas procesadas: {len(enhanced_movies)}")
    print(
        f"Películas enriquecidas: {sum(1 for m in enhanced_movies if m.get('budget') is not None)}"
    )
    print(f"Archivo guardado en: {output_file}")


def test_main(limit: int):
    """
    Versión de prueba secuencial que procesa solo un número limitado de películas
    """
    # Rutas de archivos
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent
    input_file = data_dir / "Clean_data" / "cleaned_movies.json"
    output_file = (
        data_dir / "Clean_data" / f"enhanced_movies_test_{limit}_sequential.json"
    )

    print(f"=== MODO PRUEBA SECUENCIAL: {limit} PELÍCULAS ===")

    # Cargar datos existentes
    movies = load_json_data(input_file)
    if not movies:
        print("Error: No se pudieron cargar las películas.")
        return

    # Tomar solo las primeras N películas
    test_movies = movies[:limit]

    enhanced_movies = []

    for i, movie in enumerate(test_movies):
        movie_id = movie.get("id")
        print(
            f"Procesando película {i+1}/{len(test_movies)}: ID {movie_id} - '{movie.get('titulo', 'Sin título')}'"
        )

        # Hacer petición a TMDb
        tmdb_data = get_movie_details_from_tmdb(movie_id)

        # Combinar datos
        enhanced_movie = merge_movie_data(movie, tmdb_data)
        enhanced_movies.append(enhanced_movie)

        # Rate limiting
        time.sleep(0.25)

    # Guardar resultado
    save_json_data(enhanced_movies, output_file)

    print(f"\n🎉 PRUEBA SECUENCIAL COMPLETADA")
    print(f"Películas procesadas: {len(enhanced_movies)}")
    print(
        f"Películas enriquecidas: {sum(1 for m in enhanced_movies if m.get('budget') is not None)}"
    )
    print(f"Archivo guardado en: {output_file}")


def show_enhanced_movie_structure():
    """
    Muestra la estructura del JSON enriquecido como referencia
    """
    structure = {
        "campos_existentes_mantenidos": {
            "id": "ID de TMDb (sin cambios)",
            "imdb_id": "ID de IMDb (sin cambios)",
            "titulo": "Título en español (sin cambios)",
            "generos": "Array de IDs de géneros (sin cambios)",
            "keywords": "Array de keywords (sin cambios)",
        },
        "campos_existentes_actualizados": {
            "poster_path": "Ruta del poster (ACTUALIZADO con TMDb más reciente)",
            "puntuacion_media": "Puntuación media (ACTUALIZADA con vote_average TMDb)",
            "votos": "Número de votos (ACTUALIZADO con vote_count TMDb)",
        },
        "campos_nuevos_tmdb": {
            "budget": "Presupuesto de la película",
            "revenue": "Ingresos de la película",
            "runtime": "Duración en minutos",
            "overview": "Sinopsis en inglés",
            "tagline": "Eslogan de la película",
            "release_date": "Fecha de lanzamiento (YYYY-MM-DD)",
            "original_title": "Título original",
            "original_language": "Idioma original (código)",
            "homepage": "Sitio web oficial",
            "status": "Estado (Released, etc.)",
            "popularity": "Puntuación de popularidad TMDb",
            "vote_average_tmdb": "Puntuación media TMDb (copia de referencia)",
            "vote_count_tmdb": "Número de votos TMDb (copia de referencia)",
            "backdrop_path": "Ruta del backdrop",
            "adult": "Contenido para adultos (boolean)",
            "belongs_to_collection": "Información de colección",
            "production_companies": "Empresas productoras",
            "production_countries": "Países de producción",
            "spoken_languages": "Idiomas hablados",
        },
    }

    print("=== ESTRUCTURA DE DATOS ENRIQUECIDOS ===")
    print(json.dumps(structure, indent=2, ensure_ascii=False))
    print("\n📝 NOTA IMPORTANTE:")
    print(
        "• Los campos existentes 'poster_path', 'puntuacion_media' y 'votos' se ACTUALIZAN"
    )
    print(
        "• Se mantienen copias de referencia como 'vote_average_tmdb' y 'vote_count_tmdb'"
    )
    print("• Si TMDb no tiene datos, se mantienen los valores originales")
    print(f"\n⚡ CONFIGURACIÓN DE PARALELIZACIÓN:")
    print(f"• Peticiones concurrentes: {MAX_CONCURRENT_REQUESTS}")
    print(f"• Tamaño de lote: {BATCH_SIZE}")
    print(f"• Rate limiting: {RATE_LIMIT_DELAY}s entre peticiones")


def analyze_current_data():
    """
    Función auxiliar para analizar los datos actuales antes del enriquecimiento
    """
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent
    input_file = data_dir / "Clean_data" / "cleaned_movies.json"

    movies = load_json_data(input_file)
    if not movies:
        return

    print("=== ANÁLISIS DE DATOS ACTUALES ===")
    print(f"Total de películas: {len(movies)}")

    # Contar campos existentes
    fields_present = {}
    for movie in movies[:100]:  # Analizar primeras 100
        for field in movie.keys():
            fields_present[field] = fields_present.get(field, 0) + 1

    print("\nCampos existentes (en primeras 100 películas):")
    for field, count in sorted(fields_present.items()):
        print(f"  {field}: {count}/100")

    # Mostrar ejemplo
    if movies:
        print(f"\nEjemplo de película actual:")
        example = movies[0]
        for key, value in example.items():
            if isinstance(value, str) and len(value) > 50:
                print(f"  {key}: {value[:50]}...")
            else:
                print(f"  {key}: {value}")


# Función para crear un resumen de las mejoras
def create_enhancement_summary(input_file: str, output_file: str):
    """
    Crea un resumen de las mejoras realizadas comparando archivos antes/después
    """
    try:
        # Cargar ambos archivos
        original_movies = load_json_data(input_file)
        enhanced_movies = load_json_data(output_file)

        if not original_movies or not enhanced_movies:
            print("Error: No se pudieron cargar los archivos para comparación.")
            return

        print("=== RESUMEN DE MEJORAS ===")
        print(f"Películas originales: {len(original_movies)}")
        print(f"Películas mejoradas: {len(enhanced_movies)}")

        # Verificar nuevos campos
        original_fields = set(original_movies[0].keys()) if original_movies else set()
        enhanced_fields = set(enhanced_movies[0].keys()) if enhanced_movies else set()
        new_fields = enhanced_fields - original_fields

        print(f"\nNuevos campos añadidos: {len(new_fields)}")
        for field in sorted(new_fields):
            print(f"  + {field}")

        # Contar éxito de enriquecimiento
        successful_enhancements = 0
        for movie in enhanced_movies:
            if any(movie.get(field) is not None for field in new_fields):
                successful_enhancements += 1

        print(
            f"\nPelículas exitosamente enriquecidas: {successful_enhancements}/{len(enhanced_movies)}"
        )
        print(
            f"Porcentaje de éxito: {(successful_enhancements/len(enhanced_movies)*100):.1f}%"
        )

    except Exception as e:
        print(f"Error creando resumen: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Enriquecer datos de películas con información de TMDb"
    )
    parser.add_argument(
        "--show-structure",
        action="store_true",
        help="Mostrar la estructura del JSON enriquecido",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continuar desde donde se quedó el proceso anterior",
    )
    parser.add_argument(
        "--test",
        "-t",
        type=int,
        metavar="N",
        help="Procesar solo las primeras N películas para prueba",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Usar procesamiento secuencial en lugar de paralelo",
    )
    parser.add_argument(
        "--concurrent",
        "-c",
        type=int,
        metavar="N",
        default=MAX_CONCURRENT_REQUESTS,
        help=f"Número de peticiones concurrentes (por defecto: {MAX_CONCURRENT_REQUESTS})",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        metavar="N",
        default=BATCH_SIZE,
        help=f"Tamaño de lote para procesamiento (por defecto: {BATCH_SIZE})",
    )

    args = parser.parse_args()

    if args.show_structure:
        show_enhanced_movie_structure()
    else:
        # Actualizar configuración según argumentos
        concurrent_requests = (
            args.concurrent if args.concurrent else MAX_CONCURRENT_REQUESTS
        )
        batch_size = args.batch_size if args.batch_size else BATCH_SIZE

        # Actualizar variables globales para que las funciones las usen
        globals()["MAX_CONCURRENT_REQUESTS"] = concurrent_requests
        globals()["BATCH_SIZE"] = batch_size

        if args.test:
            print(f"🧪 MODO PRUEBA: Procesando solo las primeras {args.test} películas")
            if args.sequential:
                test_main(args.test)
            else:
                asyncio.run(test_main_async(args.test))
        elif args.sequential:
            print("🐌 MODO SECUENCIAL: Procesamiento uno por uno")
            main_sequential()
        else:
            print(
                f"🚀 MODO PARALELO: {concurrent_requests} peticiones simultáneas, lotes de {batch_size}"
            )
            main()

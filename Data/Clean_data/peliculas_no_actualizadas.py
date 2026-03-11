"""
Script para encontrar películas de cleaned_movies.json que NO están en enhanced_movies.json.
Muestra los IDs de IMDb de las películas no actualizadas.
"""

import json
from pathlib import Path

data_dir = Path(__file__).parent

# Cargar ambos archivos
with open(data_dir / "cleaned_movies.json", "r", encoding="utf-8") as f:
    cleaned = json.load(f)

with open(data_dir / "enhanced_movies.json", "r", encoding="utf-8") as f:
    enhanced = json.load(f)

# IDs presentes en enhanced
enhanced_ids = {movie["id"] for movie in enhanced}

# Películas que están en cleaned pero NO en enhanced
no_actualizadas = [m for m in cleaned if m["id"] not in enhanced_ids]

print(f"Total en cleaned_movies:   {len(cleaned)}")
print(f"Total en enhanced_movies:  {len(enhanced)}")
print(f"Películas NO actualizadas: {len(no_actualizadas)}")
print()

# Mostrar IMDb IDs
imdb_ids = [m.get("imdb_id", "N/A") for m in no_actualizadas]
print("IMDb IDs de películas no actualizadas:")
for imdb_id in imdb_ids:
    print(imdb_id)

# Guardar las películas no actualizadas en un nuevo JSON
output_path = data_dir / "peliculas_no_actualizadas2.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(no_actualizadas, f, ensure_ascii=False, indent=2)

print(f"\nGuardado en: {output_path}")

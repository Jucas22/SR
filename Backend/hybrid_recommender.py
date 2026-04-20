

class HybridRecommender:

    def __init__(self, content_model, collaborative_model):
        self.content_model = content_model
        self.collaborative_model = collaborative_model

    def recommend(self, user_id, top_k=10, alpha=0.5, beta=0.5):

        # 1. Obtener recomendaciones
        content_recs = self.content_model.recommend(user_id, top_k=50, return_df=True)
        collab_recs = self.collaborative_model.recommend(user_id, top_n=50)

        # Convertir colaborativo a dict
        collab_dict = {movie_id: score for movie_id, score in collab_recs}

        # Convertir contenido a dict
        content_dict = {
            row["movie_id"]: row["score"]
            for _, row in content_recs.iterrows()
        }

        # 2. Mezclar items
        all_items = set(content_dict.keys()).union(collab_dict.keys())

        hybrid_scores = {}

        for item in all_items:
            r_content = content_dict.get(item, 0)
            r_collab = collab_dict.get(item, 0)

            # 3. Score híbrido
            hybrid_scores[item] = alpha * r_content + beta * r_collab

        # 4. Ordenar
        ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

        return ranked[:top_k]
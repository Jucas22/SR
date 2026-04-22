APP_STYLES = """
<style>
.main-header {
    text-align: center;
    padding: 1rem;
    background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
    border-radius: 10px;
    margin-bottom: 2rem;
}

[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
    display: flex;
    flex-direction: column;
}

.movie-card-title {
    font-weight: 700;
    font-size: 0.9rem;
    line-height: 1.25;
    margin: 6px 0 2px 0;
    min-height: 2.5em;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}

.movie-card-genres {
    color: #999;
    font-size: 0.78rem;
    margin: 0 0 4px 0;
    min-height: 1.3em;
}

[data-testid="stImage"] img {
    border-radius: 8px;
    object-fit: cover;
    aspect-ratio: 2/3;
    width: 100%;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}

[data-testid="stImage"] img:hover {
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
    transform: translateY(-2px);
}

.stButton > button {
    width: 100%;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

[data-testid="stDialog"] {
    border-radius: 16px;
}

[data-testid="stDialog"] [data-testid="stImage"] img {
    border-radius: 12px;
}

@media (max-width: 768px) {
    .movie-card-title {
        font-size: 0.8rem;
    }

    .movie-card-genres {
        font-size: 0.7rem;
    }
}
</style>
"""

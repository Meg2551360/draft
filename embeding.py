import os
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# Конфигурация
METADATA_CSV = "image_metadata.csv"  # CSV с колонками: file,page,chapter,bbox,image_path
OUTPUT_CSV = "image_database.csv"
VECTOR_DIM = 512  # размер эмбеддинга CLIP ViT-B/32

# Инициализируем CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

def embed_image(image_path: str) -> torch.Tensor:
    """
    Извлекает и нормализует эмбеддинг изображения.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats.squeeze(0)


def main():
    # Читаем метаданные
    df = pd.read_csv(METADATA_CSV)

    # Подготовка записей для CSV
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding images and building CSV"):
        img_path = row['image_path']
        try:
            emb = embed_image(img_path).cpu().numpy()
        except Exception as e:
            print(f"Ошибка при эмбеддинге {img_path}: {e}")
            continue

        # Формируем одну запись: метаданные + эмбеддинг
        record = {
            'file': row['file'],
            'page': int(row['page']),
            'chapter': row.get('chapter', ''),
            'bbox': row['bbox'],
            **{f'emb_{i}': float(emb[i]) for i in range(VECTOR_DIM)}
        }
        records.append(record)

    # Сохраняем в CSV
    out_df = pd.DataFrame(records)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Database saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()





# import os
# import hnswlib
# import numpy as np
# import pandas as pd
# from transformers import CLIPModel, CLIPProcessor
# from PIL import Image
# import torch

# # Конфигурация
# EMBEDDINGS_CSV = "image_database.csv"  # CSV с emb_0..emb_511 и метаданными
# IMAGE_DIR = "extracted_images"
# QUERY_IMAGE = "query.jpg"       # путь к изображению-запросу
# K = 10                           # число ближайших соседей
# DISTANCE_THRESHOLD = 0.3         # порог отбрасывания малопохожих
# INDEX_PATH = "hnsw_index.bin"
# HTML_REPORT = "search_results.md"

# # Загрузка модели CLIP
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# model.eval()

# # Шаг 1. Подготовка данных
# print("Loading embeddings CSV...")
# df = pd.read_csv(EMBEDDINGS_CSV)
# # Вытягиваем векторы из столбцов emb_0..emb_511
# emb_cols = [f'emb_{i}' for i in range(512)]
# vectors = df[emb_cols].to_numpy(dtype=np.float32)

# # Нормировка (L2-норма = 1)
# norms = np.linalg.norm(vectors, axis=1, keepdims=True)
# vectors = vectors / np.maximum(norms, 1e-12)

# # Шаг 2. Построение/загрузка индекса
# if os.path.exists(INDEX_PATH):
#     print("Loading existing HNSW index...")
#     p = hnswlib.Index(space='cosine', dim=512)
#     p.load_index(INDEX_PATH)
# else:
#     print("Building HNSW index...")
#     p = hnswlib.Index(space='cosine', dim=512)
#     p.init_index(max_elements=vectors.shape[0], ef_construction=200, M=16)
#     p.add_items(vectors)
#     p.set_ef(50)
#     p.save_index(INDEX_PATH)
#     print(f"Index saved to {INDEX_PATH}")

# # Шаг 3. Поиск
# print(f"Embedding query image: {QUERY_IMAGE}")
# def embed_image(path):
#     img = Image.open(path).convert('RGB')
#     inputs = processor(images=img, return_tensors='pt')
#     with torch.no_grad():
#         feats = model.get_image_features(**inputs)
#     feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
#     return feats.squeeze(0).cpu().numpy().astype(np.float32)

# query_vec = embed_image(QUERY_IMAGE)
# labels, distances = p.knn_query(query_vec, k=K)

# # Шаг 4. Фильтрация и сбор результатов
# results = []
# for idx, dist in zip(labels[0], distances[0]):
#     if dist > DISTANCE_THRESHOLD:
#         continue
#     row = df.iloc[idx]
#     results.append({
#         'file': row['file'],
#         'page': row['page'],
#         'chapter': row['chapter'],
#         'bbox': row['bbox'],
#         'distance': float(dist),
#         'image_path': row['image_path']
#     })

# # Шаг 5. Генерация Markdown-отчёта
# with open(HTML_REPORT, 'w', encoding='utf-8') as f:
#     f.write(f"# Search Results for {QUERY_IMAGE}\n\n")
#     for i, r in enumerate(results, 1):
#         f.write(f"## Rank {i} - Distance: {r['distance']:.4f}\n")
#         f.write(f"**File:** {r['file']}, **Page:** {r['page']}, **Chapter:** {r['chapter']}\n\n")
#         rel = os.path.relpath(r['image_path'], start=os.getcwd())
#         f.write(f"![]({rel}){{width=200}}\n\n")

# print(f"Report saved to {HTML_REPORT}")


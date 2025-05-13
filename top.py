import os
import hnswlib
import numpy as np
import pandas as pd
import cv2
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch

# Конфигурация
EMBEDDINGS_CSV = "image_database.csv"    # CSV с emb_0..emb_511 и метаданными
QUERY_IMAGE = "query.jpg"                # путь к изображению-запросу
K = 10                                    # число ближайших соседей
DISTANCE_THRESHOLD = 0.3                  # порог отбрасывания малопохожих
INDEX_PATH = "hnsw_index.bin"

# Загрузка модели CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# Шаг 1. Загрузка эмбеддингов и нормировка
print("Loading embeddings CSV...")
df = pd.read_csv(EMBEDDINGS_CSV)
emb_cols = [f'emb_{i}' for i in range(512)]
vectors = df[emb_cols].to_numpy(dtype=np.float32)
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
vectors = vectors / np.maximum(norms, 1e-12)

# Шаг 2. Построение / загрузка индекса HNSW с помощью HNSWLIB
# Если файл индекса существует, грузим его, иначе строим заново
if os.path.exists(INDEX_PATH):
    print("Loading existing HNSW index...")
    # Создаем объект индекса и загружаем сохраненный
    p = hnswlib.Index(space='cosine', dim=512)
    p.load_index(INDEX_PATH)
    # Настраиваем ef (размер вакансии поиска)
    p.set_ef(50)
else:
    print("Building HNSW index...")
    # Инициализация индекса: max_elements = количество векторов, ef_construction и M — параметры качества/памяти
    p = hnswlib.Index(space='cosine', dim=512)
    p.init_index(max_elements=vectors.shape[0], ef_construction=200, M=16)
    # Добавляем векторы в индекс
    p.add_items(vectors)
    # Настраиваем ef для поиска (чем больше, тем точнее и медленнее)
    p.set_ef(50)
    # Сохраняем индекс на диск для последующего быстрого загрузки
    p.save_index(INDEX_PATH)
    print(f"Index built and saved to {INDEX_PATH}")

# Шаг 3. Функция эмбеддинга для запроса. Функция эмбеддинга для запроса

def embed_image(path):
    img = Image.open(path).convert('RGB')
    inputs = processor(images=img, return_tensors='pt')
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats.squeeze(0).cpu().numpy().astype(np.float32)

# Эмбеддинг запроса и поиск
print(f"Embedding query image: {QUERY_IMAGE}")
query_vec = embed_image(QUERY_IMAGE)
labels, distances = p.knn_query(query_vec, k=K)

# Шаг 4. Отображение результатов через OpenCV
for rank, (idx, dist) in enumerate(zip(labels[0], distances[0]), start=1):
    if dist > DISTANCE_THRESHOLD:
        continue
    row = df.iloc[idx]
    img_path = row['image_path']
    img = cv2.imread(img_path)
    if img is None:
        continue
    # Подпись окна: ранк, файл, страница, глава, расстояние
    title = f"#{rank} {row['file']} p{row['page']} | d={dist:.3f}"
    cv2.imshow(title, cv2.resize(img, (400, int(img.shape[0] * 400 / img.shape[1]))))

# Показываем окно с запросом
query_img = cv2.imread(QUERY_IMAGE)
cv2.imshow('Query Image', cv2.resize(query_img, (400, int(query_img.shape[0] * 400 / query_img.shape[1]))))

print("Press any key in an image window to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()
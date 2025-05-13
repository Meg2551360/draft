# k — число выдаваемых результатов
k = 5

# предполагаем, что у вас есть:
# - query_vec: эмбеддинг входного изображения, shape=(D,)
# - index: ваш Faiss-индекс, к которому добавлены все векторы
# - all_images: список метаданных тех же векторов, где each = {file, chapter, page, bbox, vector}

# 1. Ищем k ближайших
distances, indices = index.search(query_vec[np.newaxis, :], k)

# distances: shape=(1, k) — L2-расстояния или другие метрики  
# indices:   shape=(1, k) — индексы соседей в all_images

# 2. Собираем результаты
results = []
for rank, idx in enumerate(indices[0]):
    md = all_images[idx]
    results.append({
        "rank": rank + 1,
        "file": md["file"],
        "chapter": md["chapter"],
        "page": md["page"],
        "distance": float(distances[0][rank]),
        "bbox": md.get("bbox")
    })

# 3. Вернём JSON
return {"results": results}



from fastapi import FastAPI, Query, File, UploadFile
from io import BytesIO
from PIL import Image

app = FastAPI()

@app.post("/search/")
async def search(
    file: UploadFile = File(...),
    k: int = Query(5, ge=1, le=20, description="Число возвращаемых похожих изображений")
):
    img = Image.open(BytesIO(await file.read()))
    qv = embed_image(img)               # ваш CLIP/ResNet→вектор
    D, I = index.search(qv[np.newaxis, :], k)

    results = []
    for rank, idx in enumerate(I[0]):
        md = all_images[idx]
        results.append({
            "rank":    rank + 1,
            "file":    md["file"],
            "chapter": md["chapter"],
            "page":    md["page"],
            "distance": float(D[0][rank])
        })

    return {"query": file.filename, "k": k, "results": results}

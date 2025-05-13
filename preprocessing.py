import fitz  # PyMuPDF
import re
import os

# Папка для сохранения извлеченных изображений
OUTPUT_DIR = "extracted_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Регулярные выражения для заголовков глав
CHAPTER_PATTERNS = [
    re.compile(r"^Глава\s+\d+", re.IGNORECASE),
    re.compile(r"^Chapter\s+\d+", re.IGNORECASE)
]
# Регулярное выражение для подписей рисунков (Figure 1 – caption)
CAPTION_PATTERN = re.compile(r"Figure\s+(\d+)\s*[–-]\s*(.+)", re.IGNORECASE)


def extract_images_and_metadata(pdf_path):
    """
    Извлекает изображения из PDF вместе с метаданными:
    - номер страницы
    - позиция (bbox)
    - xref изображения
    Сохраняет файлы в OUTPUT_DIR и возвращает список метаданных.
    """
    metadata = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b.get("type") == 1:  # блок-изображение
                xref = b.get("image")
                bbox = b.get("bbox")  # [x0, y0, x1, y1]
                img = doc.extract_image(xref)
                img_bytes = img["image"]
                ext = img["ext"]
                img_name = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_p{page_num+1}_xref{xref}.{ext}"
                out_path = os.path.join(OUTPUT_DIR, img_name)
                with open(out_path, "wb") as f:
                    f.write(img_bytes)
                metadata.append({
                    "file": pdf_path,
                    "page": page_num + 1,
                    "bbox": bbox,
                    "xref": xref,
                    "image_path": out_path,
                    # временно без caption, добавим позже
                    "caption": None
                })
    doc.close()
    return metadata


def parse_chapters(pdf_path):
    """
    Парсит текст PDF для построения карты "страница -> глава".
    Ищет в начале каждого текстового блока заголовки глав по CHAPTER_PATTERNS.
    Возвращает dict: {page_number: chapter_title}
    """
    chapter_map = {}
    current_chapter = None
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        lines = text.splitlines()
        for line in lines[:5]:
            stripped = line.strip()
            for pat in CHAPTER_PATTERNS:
                if pat.match(stripped):
                    current_chapter = stripped
                    break
            if current_chapter:
                break
        chapter_map[page_num + 1] = current_chapter
    doc.close()
    return chapter_map


def parse_captions(pdf_path):
    """
    Извлекает подписи рисунков (Figure N – caption) на каждой странице.
    Возвращает dict: {(page_number, figure_number): caption_text}
    """
    captions = {}
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        for line in text.splitlines():
            match = CAPTION_PATTERN.match(line.strip())
            if match:
                fig_num = int(match.group(1))
                caption_text = match.group(2).strip()
                captions[(page_num + 1, fig_num)] = caption_text
    doc.close()
    return captions


def assign_captions_to_images(metadata, captions):
    """
    По метаданным изображений и найденным подписям пытается сопоставить подпись конкретному изображению.
    Предполагаем, что подпись Figure N стоит после N-го изображения на странице.
    """
    # Сгруппируем изображения по странице и сортируем по bbox.y0 (вертикальный порядок)
    from collections import defaultdict
    grouped = defaultdict(list)
    for md in metadata:
        grouped[md['page']].append(md)
    for page, imgs in grouped.items():
        # сортировка по y0 (второй элемент bbox)
        imgs.sort(key=lambda x: x['bbox'][1])
        for idx, md in enumerate(imgs, start=1):
            caption = captions.get((page, idx))
            md['caption'] = caption
    return metadata


if __name__ == "__main__":
    pdf_files = ["agro_technologies.pdf", "general_technologies.pdf"]
    all_metadata = []
    all_chapters = {}

    for pdf in pdf_files:
        print(f"Обрабатываю {pdf}...")
        imgs_md = extract_images_and_metadata(pdf)
        chap_map = parse_chapters(pdf)
        caps = parse_captions(pdf)
        imgs_with_caps = assign_captions_to_images(imgs_md, caps)
        all_metadata.extend(imgs_with_caps)
        all_chapters[pdf] = chap_map

    # Пример вывода
    for md in all_metadata:
        print(md)



# page.get_text("dict"):
#     'type': 1,  # 1 = image block
#     'bbox': (x0, y0, x1, y1),  # координаты в pts
#     'width': W,  # ширина в пикселях
#     'height': H,  # высота в пикселях
#     'colorspace': n,  # 1/3/4
#     'bpc': 8,  # типичное значение
#     'xref': xref_num,  # для extract_image()
#     'ext': 'jpeg',  # или 'png', 'gif', etc.
#     # Доп. данные через extract_image:
#     'image': bytes_data,
#     'cs-name': 'DeviceRGB',
#     'icc': icc_profile_bytes,
#     'dpi': (x_dpi, y_dpi)  # если есть в метаданных




# import fitz  # PyMuPDF
# import re
# import os
# import pandas as pd
# from collections import defaultdict
# import ace_tools as tools

# # Папка для сохранения извлеченных изображений
# OUTPUT_DIR = "extracted_images"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Регулярки для глав и подписей
# CHAPTER_PATTERNS = [
#     re.compile(r"^Глава\s+\d+", re.IGNORECASE),
#     re.compile(r"^Chapter\s+\d+", re.IGNORECASE)
# ]
# CAPTION_PATTERN = re.compile(r"Figure\s+(\d+)\s*[–-]\s*(.+)", re.IGNORECASE)

# def extract_images_and_metadata(pdf_path):
#     metadata = []
#     doc = fitz.open(pdf_path)
#     for page_num in range(len(doc)):
#         page = doc[page_num]
#         blocks = page.get_text("dict")["blocks"]
#         for b in blocks:
#             if b.get("type") == 1:
#                 xref = b.get("image")
#                 bbox = b.get("bbox")
#                 img = doc.extract_image(xref)
#                 ext = img["ext"]
#                 img_name = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_p{page_num+1}_xref{xref}.{ext}"
#                 out_path = os.path.join(OUTPUT_DIR, img_name)
#                 with open(out_path, "wb") as f:
#                     f.write(img["image"])
#                 metadata.append({
#                     "file": os.path.basename(pdf_path),
#                     "page": page_num + 1,
#                     "bbox": bbox,
#                     "xref": xref,
#                     "image_path": out_path,
#                     "caption": None,
#                     "chapter": None
#                 })
#     doc.close()
#     return metadata

# def parse_chapters(pdf_path):
#     chapter_map = {}
#     current_chapter = None
#     doc = fitz.open(pdf_path)
#     for page_num in range(len(doc)):
#         page = doc[page_num]
#         lines = page.get_text("text").splitlines()
#         for line in lines[:5]:
#             stripped = line.strip()
#             for pat in CHAPTER_PATTERNS:
#                 if pat.match(stripped):
#                     current_chapter = stripped
#                     break
#             if current_chapter:
#                 break
#         chapter_map[page_num + 1] = current_chapter
#     doc.close()
#     return chapter_map

# def parse_captions(pdf_path):
#     captions = {}
#     doc = fitz.open(pdf_path)
#     for page_num in range(len(doc)):
#         lines = doc.get_page_text(page_num).splitlines()
#         for line in lines:
#             match = CAPTION_PATTERN.match(line.strip())
#             if match:
#                 fig_num = int(match.group(1))
#                 captions[(page_num + 1, fig_num)] = match.group(2).strip()
#     doc.close()
#     return captions

# def assign_captions_and_chapters(metadata, captions, chapters):
#     grouped = defaultdict(list)
#     for md in metadata:
#         grouped[md['page']].append(md)
#     for page, imgs in grouped.items():
#         imgs.sort(key=lambda x: x['bbox'][1])
#         for idx, md in enumerate(imgs, start=1):
#             md['caption'] = captions.get((page, idx))
#             md['chapter'] = chapters.get(page)
#     return metadata

# # Обработка PDF
# pdf_files = ["agro_technologies.pdf", "general_technologies.pdf"]
# all_records = []
# for pdf in pdf_files:
#     imgs = extract_images_and_metadata(pdf)
#     ch_map = parse_chapters(pdf)
#     caps = parse_captions(pdf)
#     enriched = assign_captions_and_chapters(imgs, caps, ch_map)
#     all_records.extend(enriched)

# # Создание DataFrame
# df = pd.DataFrame(all_records)
# df.to_csv("image_metadata.csv", index=False)


# # Показать результат пользователю
# tools.display_dataframe_to_user("Extracted Images Metadata", df)


# pip install faiss_cpu-1.7.2-cp39-cp39-win_amd64.whl (https://github.com/onfido/faiss-windows)
# pip install annoy
# pip install hnswlib
# pip install nmslib

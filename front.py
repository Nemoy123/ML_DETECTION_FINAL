# pip install streamlit requests pillow
# streamlit run front.py


import streamlit as st
import requests
import time
from PIL import Image
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

API_URL = "http://localhost:9000"  # поменяй при необходимости

MAX_FILE_SIZE = 100 * 1024 * 1024      # 10 MB
MAX_TOTAL_SIZE = 1000 * 1024 * 1024    # 100 MB

CLASS_COLORS = {
    "car": "red",
    "van": "orange",
    "truck": "blue",
    "tricycle": "purple",
    "awning-tricycle": "brown",
    "bus": "cyan",
    "motor": "green",
}

# ------------------------
# UTILS
# ------------------------
def draw_bboxes(image: Image.Image, detections: list) -> Image.Image:
    draw = ImageDraw.Draw(image)

    for det in detections:
        x1 = int(det["x1"])
        y1 = int(det["y1"])
        x2 = int(det["x2"])
        y2 = int(det["y2"])

        class_name = det.get("class_name", "unknown")
        conf = det.get("confidence", 0)

        color = CLASS_COLORS.get(class_name, "white")

        # bbox
        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=color,
            width=3
        )

        label = f"{class_name} {conf:.2f}"

        text_bbox = draw.textbbox((0, 0), label)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        draw.rectangle(
            [(x1, y1 - text_h - 4), (x1 + text_w + 4, y1)],
            fill=color
        )

        draw.text(
            (x1 + 2, y1 - text_h - 2),
            label,
            fill="white"
        )

    return image

def create_session():
    r = requests.post(f"{API_URL}/session")
    r.raise_for_status()
    return r.json()["session_id"]


def upload_file(session_id, file):
    files = {"file": (file.name, file, file.type)}
    r = requests.post(
        f"{API_URL}/detect/async",
        files=files,
        params={"session_id": session_id},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def get_task_status(task_id, session_id):
    r = requests.get(
        f"{API_URL}/task/{task_id}",
        params={"session_id": session_id},
    )
    r.raise_for_status()
    return r.json()


# ------------------------
# UI
# ------------------------

st.set_page_config(
    page_title="Transport Object Detection",
    layout="wide",
)

st.title("Transport Object Detection")
st.write("Загрузите изображения для обнаружения объектов.")

# ------------------------
# FILE UPLOAD
# ------------------------

uploaded_files = st.file_uploader(
    "Выберите изображения",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    total_size = sum(f.size for f in uploaded_files)

    if total_size > MAX_TOTAL_SIZE:
        st.error("❌ Суммарный размер файлов превышает 100 МБ")
        st.stop()

    for f in uploaded_files:
        if f.size > MAX_FILE_SIZE:
            st.error(f"❌ Файл {f.name} больше 10 МБ")
            st.stop()

    st.success(f"✔ Загружено файлов: {len(uploaded_files)}")
    st.info(f"Общий размер: {total_size / 1024 / 1024:.2f} MB")

# ------------------------
# PROCESS BUTTON
# ------------------------

if st.button("🚀 Запустить обработку", disabled=not uploaded_files):
    with st.spinner("Создание сессии..."):
        session_id = create_session()

    st.session_state["session_id"] = session_id
    st.session_state["tasks"] = []

    for file in uploaded_files:
        with st.spinner(f"Отправка {file.name}"):
            task = upload_file(session_id, file)
            st.session_state["tasks"].append(task)

    st.success("Все файлы отправлены в очередь")

# ------------------------
# TASK MONITORING
# ------------------------

if "tasks" in st.session_state:
    st.divider()
    st.subheader("📊 Статус задач")

    for task in st.session_state["tasks"]:
        task_id = task["task_id"]
        filename = task["filename"]

        container = st.container(border=True)
        with container:
            st.markdown(f"### 📄 {filename}")

            progress_bar = st.progress(0)
            status_text = st.empty()
            result_container = st.empty()

            while True:
                status = get_task_status(task_id, st.session_state["session_id"])

                progress = status.get("progress", 0)
                state = status["status"]

                progress_bar.progress(progress)
                status_text.write(f"Статус: **{state}** ({progress}%)")

                if state == "completed":
                    result = status.get("result", {})
                    detections = result.get("detections", [])

                    st.success("✅ Обработка завершена")

                    # показать изображение с bbox
                    image = Image.open(file).convert("RGB")
                    image_with_boxes = draw_bboxes(image, detections)

                    st.image(
                        image_with_boxes,
                        caption=f"{filename} — найдено объектов: {len(detections)}",
                        use_column_width=True
                    )

                    # опционально — JSON под спойлером
                    with st.expander("Показать JSON результата"):
                        st.json(result)

                    break

                if state == "failed":
                    st.error(f"❌ Ошибка: {status.get('error')}")
                    break

                time.sleep(1)

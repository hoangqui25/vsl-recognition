from flask import Flask, render_template, request, send_from_directory
import sys
import os
import json
from inference import predict_video
from flask import Flask
from datetime import datetime
app = Flask(
    __name__,
    template_folder="app/templates",
    static_folder="app/static"
)

now = datetime.now()
# =========================
# CONFIG
# =========================
VIDEO_FOLDER = "app/uploads"
HISTORY_FILE = "app/history.json"


app.config["UPLOAD_FOLDER"] = VIDEO_FOLDER


# tạo folder upload

os.makedirs(VIDEO_FOLDER, exist_ok=True)


# =========================
# SERVE VIDEO
# =========================


@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(VIDEO_FOLDER, filename)


# =========================
# HISTORY
def load_history():

    if not os.path.exists(HISTORY_FILE):

        return []

    try:

        with open(HISTORY_FILE, "r", encoding="utf-8") as f:

            data = json.load(f)

            if isinstance(data, list):

                return data

            return []

    except Exception:

        return []


def save_history(item):

    history = load_history()

    history.insert(0, item)

    # giữ 20 kết quả gần nhất

    history = history[:20]

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:

        json.dump(history, f, ensure_ascii=False, indent=4)


# =========================
# HOME
# =========================


@app.route("/")
def home():

    return render_template("index.html", history=load_history())


# =========================
# PREDICT
# =========================


@app.route("/predict", methods=["POST"])
def predict():

    if "video" not in request.files:

        return "Missing video"

    video = request.files["video"]

    if video.filename == "":

        return "No video selected"

    # lưu video

    filename = video.filename

    save_path = os.path.join(VIDEO_FOLDER, filename)

    video.save(save_path)

    try:

        # =====================
        # RUN MODEL
        # =====================

        result = predict_video(save_path)

        # kiểm tra output

        if not isinstance(result, dict):

            raise Exception("predict_video must return dictionary")

        # thêm default nếu thiếu

        result.setdefault("gloss", "Unknown")

        result.setdefault("confidence", 0)

        result.setdefault("top5", [])

        result.setdefault("skeleton", {"pose": 0, "left_hand": 0, "right_hand": 0})

        # =====================
        # SAVE HISTORY
        # =====================

        history_item = {
            "time": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "video": filename,
            "result": result["gloss"],
            "confidence": float(result["confidence"]),
        }

        save_history(history_item)

        return render_template(
            "result.html", result=result, video=filename, history=load_history()
        )

    except Exception as e:

        return f"""

        <h2>
        Prediction Error
        </h2>

        <pre>
        {e}
        </pre>

        """

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=True)

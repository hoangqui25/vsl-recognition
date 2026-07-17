from flask import Flask, render_template, request, send_from_directory
import os
import json
from datetime import datetime
from pathlib import Path

# Import các hàm model của bạn
from inference import (
    resolve_device,
    extract_skeleton,
    predict,
    detection_statistics
)


app = Flask(
    __name__,
    template_folder="app/templates",
    static_folder="app/static"
)


# =========================
# CONFIG
# =========================

VIDEO_FOLDER = "app/uploads"
HISTORY_FILE = "app/history.json"

app.config["UPLOAD_FOLDER"] = VIDEO_FOLDER


os.makedirs(VIDEO_FOLDER, exist_ok=True)


# =========================
# MODEL PREDICT
# =========================

def predict_video(video_path):
    """
    Predict one video for Flask application

    Return:
    {
        gloss,
        confidence,
        top5,
        skeleton
    }
    """

    video_path = Path(video_path)


    # ==========================
    # Config
    # ==========================

    checkpoint_path = Path(
        "app/checkpoints/best.pt"
    )

    model_asset_path = Path(
        "models/mediapipe/holistic_landmarker.task"
    )

    num_frames = 8


    # ==========================
    # Device
    # ==========================

    device = resolve_device("auto")


    # ==========================
    # Extract skeleton
    # ==========================

    features = extract_skeleton(
        video_path=video_path,
        model_asset_path=model_asset_path,
        num_frames=num_frames
    )


    # ==========================
    # Predict model
    # ==========================

    predictions, metadata = predict(
        checkpoint_path=checkpoint_path,
        features=features,
        device=device,
        top_k=5
    )


    # ==========================
    # Skeleton statistics
    # ==========================

    stats = detection_statistics(features)


    skeleton_result = {

        "pose":
            round(
                stats["pose_detected_ratio"] * 100,
                2
            ),

        "left_hand":
            round(
                stats["left_hand_detected_ratio"] * 100,
                2
            ),

        "right_hand":
            round(
                stats["right_hand_detected_ratio"] * 100,
                2
            )
    }



    # ==========================
    # Top 5 predictions
    # ==========================

    top5 = []

    for item in predictions:

        top5.append({

            "label":
                item["label"],

            "confidence":
                round(
                    item["confidence"] * 100,
                    2
                )
        })


    # ==========================
    # Final result
    # ==========================

    result = {

        "gloss":
            predictions[0]["label"],

        "confidence":
            round(
                predictions[0]["confidence"] * 100,
                2
            ),

        "top5":
            top5,

        "skeleton":
            skeleton_result
    }


    return result



# =========================
# SERVE VIDEO
# =========================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        VIDEO_FOLDER,
        filename
    )



# =========================
# HISTORY
# =========================

def load_history():

    if not os.path.exists(HISTORY_FILE):
        return []


    try:

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

            if isinstance(data, list):
                return data

            return []

    except Exception:

        return []



def save_history(item):

    history = load_history()

    history.insert(0,item)

    history = history[:20]


    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            ensure_ascii=False,
            indent=4
        )



# =========================
# HOME
# =========================

@app.route("/")
def home():

    return render_template(
        "index.html",
        history=load_history()
    )



# =========================
# PREDICT
# =========================

@app.route("/predict", methods=["POST"])
def predict_route():


    if "video" not in request.files:

        return "Missing video"



    video = request.files["video"]


    if video.filename == "":

        return "No video selected"



    filename = video.filename


    save_path = os.path.join(
        VIDEO_FOLDER,
        filename
    )


    video.save(save_path)



    try:

        result = predict_video(save_path)


        history_item = {

            "time":
                datetime.now().strftime(
                    "%d/%m/%Y %H:%M"
                ),

            "video":
                filename,

            "result":
                result["gloss"],

            "confidence":
                float(
                    result["confidence"]
                )
        }


        save_history(history_item)



        return render_template(

            "result.html",

            result=result,

            video=filename,

            history=load_history()

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



# =========================
# RUN
# =========================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
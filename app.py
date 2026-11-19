import os
import sys
import time
import uuid
import shutil
import threading
import subprocess
from typing import Optional, Dict, Any

import cv2
import torch
import torchvision.ops
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from peft import LoraConfig, get_peft_model
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Smoke & Fire Detection Web Studio", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "web")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[SmokeFireAI] Initializing on device: {DEVICE}")

# Model registry & caching
MODEL_PATHS = {
    "sota_5ep": os.path.join(BASE_DIR, "checkpoints", "best_rf_detr_dora.pt"),
    "gray_3ep": os.path.join(BASE_DIR, "checkpoints", "best_rf_detr_dora_3ep.pt"),
}

loaded_models: Dict[str, Any] = {}
model_lock = threading.Lock()
processor_instance = None

def get_processor():
    global processor_instance
    if processor_instance is None:
        processor_instance = AutoImageProcessor.from_pretrained("Roboflow/rf-detr-nano")
    return processor_instance

def load_detection_model(model_key: str = "sota_5ep"):
    global loaded_models
    with model_lock:
        if model_key in loaded_models:
            return loaded_models[model_key]

        ckpt_path = MODEL_PATHS.get(model_key)
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

        print(f"[SmokeFireAI] Loading model '{model_key}' from {ckpt_path} onto {DEVICE}...")
        model_id = "Roboflow/rf-detr-nano"
        model = AutoModelForObjectDetection.from_pretrained(
            model_id, num_labels=2, ignore_mismatched_sizes=True
        )
        model.config.id2label = {0: "smoke", 1: "fire"}
        model.config.label2id = {"smoke": 0, "fire": 1}

        dora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            use_dora=True,
        )
        model.model.backbone = get_peft_model(model.model.backbone, dora_config)

        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE)
        model.eval()

        loaded_models[model_key] = model
        print(f"[SmokeFireAI] Model '{model_key}' successfully loaded and cached in VRAM/RAM.")
        return model

# In-memory Job Tracking
jobs: Dict[str, Dict[str, Any]] = {}

def draw_hud(frame, frame_idx, total_frames, num_smoke, num_fire, conf_thresh, model_label):
    h, w = frame.shape[:2]
    hud_h = max(44, int(h * 0.05))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (12, 14, 18), -1)
    alpha = 0.82
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    font_scale = max(0.42, w / 1700.0)
    font_thick = max(1, int(round(font_scale * 1.8)))

    # Title & model info
    title_text = f"RF-DETR Nano + DoRA ({model_label}) | Conf >= {conf_thresh*100:.0f}%"
    cv2.putText(
        frame,
        title_text,
        (16, int(hud_h * 0.65)),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        (255, 255, 255),
        font_thick,
        cv2.LINE_AA,
    )

    # Frame and object counts
    status_text = f"Frame {frame_idx}/{total_frames} | Smoke: {num_smoke} | Fire: {num_fire}"
    (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, font_thick)
    cv2.putText(
        frame,
        status_text,
        (w - tw - 16, int(hud_h * 0.65)),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        (220, 225, 230),
        font_thick,
        cv2.LINE_AA,
    )

def draw_detections(frame, detections, width, height):
    class_styles = {
        0: {"name": "SMOKE", "bgr": (255, 229, 0), "text_color": (0, 0, 0)},     # Cyan/Yellow
        1: {"name": "FIRE",  "bgr": (48, 59, 255),  "text_color": (255, 255, 255)}, # Vivid Fire Red
    }

    base_thick = max(2, int(round(min(width, height) / 320.0)))
    font_scale = max(0.44, min(width, height) / 850.0)
    font_thick = max(1, int(round(font_scale * 2.0)))
    corner_len = max(14, int(round(min(width, height) / 36.0)))

    for det in detections:
        cls_id = det["class_id"]
        score = det["score"]
        x1, y1, x2, y2 = det["box"]
        style = class_styles.get(cls_id, {"name": "OBJECT", "bgr": (0, 255, 0), "text_color": (0, 0, 0)})
        bgr = style["bgr"]

        # Main bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, base_thick)

        # High-tech corner accents
        c_thick = base_thick + 2
        cv2.line(frame, (x1, y1), (min(x2, x1 + corner_len), y1), bgr, c_thick)
        cv2.line(frame, (x1, y1), (x1, min(y2, y1 + corner_len)), bgr, c_thick)
        cv2.line(frame, (x2, y1), (max(x1, x2 - corner_len), y1), bgr, c_thick)
        cv2.line(frame, (x2, y1), (x2, min(y2, y1 + corner_len)), bgr, c_thick)
        cv2.line(frame, (x1, y2), (min(x2, x1 + corner_len), y2), bgr, c_thick)
        cv2.line(frame, (x1, y2), (x1, max(y1, y2 - corner_len)), bgr, c_thick)
        cv2.line(frame, (x2, y2), (max(x1, x2 - corner_len), y2), bgr, c_thick)
        cv2.line(frame, (x2, y2), (x2, max(y1, y2 - corner_len)), bgr, c_thick)

        # Label badge
        label = f"{style['name']} {score * 100:.1f}%"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale, font_thick)
        badge_y1 = max(0, y1 - lh - 10)
        badge_y2 = y1 if y1 - lh - 10 >= 0 else y1 + lh + 10
        badge_x2 = min(width, x1 + lw + 14)

        cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), bgr, -1)
        text_y = badge_y2 - 6
        cv2.putText(
            frame,
            label,
            (x1 + 6, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            style["text_color"],
            font_thick,
            cv2.LINE_AA,
        )

def process_video_worker(
    job_id: str,
    input_path: str,
    output_path: str,
    confidence_thresh: float,
    model_type: str,
    iou_thresh: float = 0.35,
    batch_size: int = 8,
):
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["message"] = "Initializing model and reading video..."

        model_label = "5-Epoch SOTA Color" if model_type == "sota_5ep" else "3-Epoch Grayscale"
        processor = get_processor()
        model = load_detection_model(model_type)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open input video file: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1

        jobs[job_id]["total_frames"] = total_frames
        jobs[job_id]["width"] = width
        jobs[job_id]["height"] = height
        jobs[job_id]["fps"] = fps

        temp_avi = output_path.replace(".mp4", f"_temp_{job_id}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(temp_avi, fourcc, fps, (width, height))

        frames_buffer = []
        processed_count = 0
        total_smoke_boxes = 0
        total_fire_boxes = 0
        t0 = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_buffer.append(frame)

            if len(frames_buffer) == batch_size:
                pil_images = [
                    Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_buffer
                ]
                inputs = processor(images=pil_images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(DEVICE)

                with torch.no_grad():
                    with torch.amp.autocast(DEVICE, dtype=torch.float16 if DEVICE == "cuda" else torch.bfloat16):
                        outputs = model(pixel_values=pixel_values)

                logits = outputs.logits.sigmoid()
                pred_boxes = outputs.pred_boxes

                for b_idx, f_orig in enumerate(frames_buffer):
                    f_logits = logits[b_idx]
                    f_boxes = pred_boxes[b_idx]
                    scores, labels = f_logits.max(dim=-1)

                    keep = scores >= confidence_thresh
                    k_scores = scores[keep]
                    k_labels = labels[keep]
                    k_boxes = f_boxes[keep]

                    detections = []
                    if len(k_scores) > 0:
                        cx, cy, bw, bh = k_boxes[:, 0], k_boxes[:, 1], k_boxes[:, 2], k_boxes[:, 3]
                        x1 = ((cx - bw / 2.0) * width).clamp(0, width)
                        y1 = ((cy - bh / 2.0) * height).clamp(0, height)
                        x2 = ((cx + bw / 2.0) * width).clamp(0, width)
                        y2 = ((cy + bh / 2.0) * height).clamp(0, height)
                        xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

                        for c_id in [0, 1]:
                            c_mask = (k_labels == c_id)
                            if c_mask.sum() == 0:
                                continue
                            c_xyxy = xyxy[c_mask]
                            c_sc = k_scores[c_mask]
                            nms_idx = torchvision.ops.nms(c_xyxy, c_sc, iou_threshold=iou_thresh)

                            for idx in nms_idx:
                                detections.append({
                                    "class_id": c_id,
                                    "score": float(c_sc[idx].item()),
                                    "box": [int(v.item()) for v in c_xyxy[idx]],
                                })

                    detections.sort(key=lambda d: d["score"])
                    num_smoke = sum(1 for d in detections if d["class_id"] == 0)
                    num_fire = sum(1 for d in detections if d["class_id"] == 1)
                    total_smoke_boxes += num_smoke
                    total_fire_boxes += num_fire

                    draw_detections(f_orig, detections, width, height)
                    draw_hud(f_orig, processed_count + 1, total_frames, num_smoke, num_fire, confidence_thresh, model_label)
                    writer.write(f_orig)
                    processed_count += 1

                frames_buffer.clear()
                elapsed = time.time() - t0
                curr_fps = processed_count / max(elapsed, 0.001)
                progress_pct = min(95.0, round((processed_count / total_frames) * 90.0, 1))
                jobs[job_id]["progress"] = progress_pct
                jobs[job_id]["current_frame"] = processed_count
                jobs[job_id]["process_fps"] = round(curr_fps, 1)
                jobs[job_id]["smoke_detections"] = total_smoke_boxes
                jobs[job_id]["fire_detections"] = total_fire_boxes
                jobs[job_id]["message"] = f"Processed {processed_count}/{total_frames} frames ({curr_fps:.1f} FPS)..."

        # Residual buffer
        if len(frames_buffer) > 0:
            pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_buffer]
            inputs = processor(images=pil_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(DEVICE)
            with torch.no_grad():
                with torch.amp.autocast(DEVICE, dtype=torch.float16 if DEVICE == "cuda" else torch.bfloat16):
                    outputs = model(pixel_values=pixel_values)
            logits = outputs.logits.sigmoid()
            pred_boxes = outputs.pred_boxes

            for b_idx, f_orig in enumerate(frames_buffer):
                f_logits = logits[b_idx]
                f_boxes = pred_boxes[b_idx]
                scores, labels = f_logits.max(dim=-1)
                keep = scores >= confidence_thresh
                k_scores = scores[keep]
                k_labels = labels[keep]
                k_boxes = f_boxes[keep]

                detections = []
                if len(k_scores) > 0:
                    cx, cy, bw, bh = k_boxes[:, 0], k_boxes[:, 1], k_boxes[:, 2], k_boxes[:, 3]
                    x1 = ((cx - bw / 2.0) * width).clamp(0, width)
                    y1 = ((cy - bh / 2.0) * height).clamp(0, height)
                    x2 = ((cx + bw / 2.0) * width).clamp(0, width)
                    y2 = ((cy + bh / 2.0) * height).clamp(0, height)
                    xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

                    for c_id in [0, 1]:
                        c_mask = (k_labels == c_id)
                        if c_mask.sum() == 0:
                            continue
                        c_xyxy = xyxy[c_mask]
                        c_sc = k_scores[c_mask]
                        nms_idx = torchvision.ops.nms(c_xyxy, c_sc, iou_threshold=iou_thresh)
                        for idx in nms_idx:
                            detections.append({
                                "class_id": c_id,
                                "score": float(c_sc[idx].item()),
                                "box": [int(v.item()) for v in c_xyxy[idx]],
                            })

                detections.sort(key=lambda d: d["score"])
                num_smoke = sum(1 for d in detections if d["class_id"] == 0)
                num_fire = sum(1 for d in detections if d["class_id"] == 1)
                total_smoke_boxes += num_smoke
                total_fire_boxes += num_fire

                draw_detections(f_orig, detections, width, height)
                draw_hud(f_orig, processed_count + 1, total_frames, num_smoke, num_fire, confidence_thresh, model_label)
                writer.write(f_orig)
                processed_count += 1

        cap.release()
        writer.release()

        # Encoding with ffmpeg to standard browser-playable H.264
        jobs[job_id]["message"] = "Encoding browser-optimized H.264 video with FFmpeg..."
        jobs[job_id]["progress"] = 96.0

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", temp_avi,
            "-i", input_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "19",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            output_path,
        ]
        res = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            # Fallback to video only if audio mapping failed
            fallback_cmd = [
                "ffmpeg", "-y",
                "-i", temp_avi,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "19",
                "-pix_fmt", "yuv420p",
                output_path,
            ]
            subprocess.run(fallback_cmd, check=True)

        if os.path.exists(temp_avi):
            try:
                os.remove(temp_avi)
            except Exception:
                pass

        total_duration = time.time() - t0
        output_size_mb = os.path.getsize(output_path) / (1024 * 1024)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100.0
        jobs[job_id]["current_frame"] = processed_count
        jobs[job_id]["total_time"] = round(total_duration, 2)
        jobs[job_id]["avg_fps"] = round(processed_count / max(total_duration, 0.001), 1)
        jobs[job_id]["output_size_mb"] = round(output_size_mb, 2)
        jobs[job_id]["result_url"] = f"/api/video/{job_id}"
        jobs[job_id]["message"] = f"Finished! {processed_count} frames processed in {total_duration:.1f}s ({output_size_mb:.2f} MB)."

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["message"] = f"Error processing video: {str(e)}"
        print(f"[SmokeFireAI ERROR] Job {job_id} failed: {e}", file=sys.stderr)

# API Endpoints
@app.post("/api/process")
async def start_processing(
    video: UploadFile = File(...),
    confidence: float = Form(0.40),
    model_type: str = Form("sota_5ep"),
    iou_thresh: float = Form(0.35),
):
    if not (0.01 <= confidence <= 0.99):
        raise HTTPException(status_code=400, detail="Confidence threshold must be between 0.01 and 0.99")

    job_id = str(uuid.uuid4())
    safe_name = os.path.splitext(video.filename)[0].replace(" ", "_")
    input_file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{safe_name}.mp4")
    output_file_path = os.path.join(OUTPUT_DIR, f"detected_{job_id}_{safe_name}.mp4")

    with open(input_file_path, "wb") as f:
        content = await video.read()
        f.write(content)

    jobs[job_id] = {
        "job_id": job_id,
        "filename": video.filename,
        "confidence": confidence,
        "model_type": model_type,
        "status": "pending",
        "progress": 0.0,
        "current_frame": 0,
        "total_frames": 0,
        "smoke_detections": 0,
        "fire_detections": 0,
        "result_path": output_file_path,
        "input_path": input_file_path,
        "message": "Queued for inference...",
    }

    t = threading.Thread(
        target=process_video_worker,
        args=(job_id, input_file_path, output_file_path, confidence, model_type, iou_thresh),
        daemon=True,
    )
    t.start()

    return JSONResponse({"job_id": job_id, "status": "queued", "message": "Video uploaded and processing started."})

@app.post("/api/process_sample")
async def start_processing_sample(
    sample_name: str = Form(...),
    confidence: float = Form(0.40),
    model_type: str = Form("sota_5ep"),
    iou_thresh: float = Form(0.35),
):
    sample_path = os.path.join(VIDEOS_DIR, sample_name)
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample video not found")

    job_id = str(uuid.uuid4())
    safe_name = os.path.splitext(sample_name)[0].replace(" ", "_")
    output_file_path = os.path.join(OUTPUT_DIR, f"detected_{job_id}_{safe_name}.mp4")

    jobs[job_id] = {
        "job_id": job_id,
        "filename": sample_name,
        "confidence": confidence,
        "model_type": model_type,
        "status": "pending",
        "progress": 0.0,
        "current_frame": 0,
        "total_frames": 0,
        "smoke_detections": 0,
        "fire_detections": 0,
        "result_path": output_file_path,
        "input_path": sample_path,
        "message": "Queued sample for inference...",
    }

    t = threading.Thread(
        target=process_video_worker,
        args=(job_id, sample_path, output_file_path, confidence, model_type, iou_thresh),
        daemon=True,
    )
    t.start()

    return JSONResponse({"job_id": job_id, "status": "queued", "message": "Sample processing started."})

@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(jobs[job_id])

@app.api_route("/api/video/{job_id}", methods=["GET", "HEAD"])
async def get_processed_video(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "completed" or not os.path.exists(job["result_path"]):
        raise HTTPException(status_code=400, detail="Video is not ready yet or file is missing")

    filename = os.path.basename(job["result_path"])
    return FileResponse(
        job["result_path"],
        media_type="video/mp4",
        filename=filename,
        headers={"Accept-Ranges": "bytes"}
    )

@app.get("/api/samples")
async def list_sample_videos():
    samples = []
    if os.path.exists(VIDEOS_DIR):
        for f in os.listdir(VIDEOS_DIR):
            if f.lower().endswith((".mp4", ".avi", ".mov")) and "[DETECTED]" not in f and "[3EP_GRAY_DETECTED]" not in f:
                f_path = os.path.join(VIDEOS_DIR, f)
                samples.append({
                    "name": f,
                    "size_mb": round(os.path.getsize(f_path) / (1024 * 1024), 2),
                })
    return JSONResponse({"samples": samples})

# Preload SOTA model at server startup
@app.on_event("startup")
def startup_event():
    threading.Thread(target=load_detection_model, args=("sota_5ep",), daemon=True).start()

# Mount static directory for frontend
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

"""
modules/preprocessor.py
------------------------
Tech 2 responsibilities (วันอาทิตย์ - วันอังคาร):

1. อ่านภาพจาก URL ผ่าน requests และแปลงเป็น NumPy Array ด้วย OpenCV
2. Pre-processing: ปรับขนาดภาพ, ปรับสี RGB, ลด Noise
3. Quality Metrics: blur (Laplacian Variance), brightness, contrast, noise
4. ปรับสเกลค่า Quality Scores ให้อยู่ในช่วง [0.0, 1.0]

ต่อยอดจาก test_cv2.py เดิม (โหลดภาพ + resize + denoise + normalize)
โดยแยกเป็นฟังก์ชัน reusable สำหรับให้ Tech Lead เรียกใช้งานผ่าน main.py
"""

import cv2
import numpy as np
import requests


# ==========================================
# 1. โหลดภาพจาก URL
# ==========================================
def read_image_from_url(url: str, timeout: int = 10) -> np.ndarray:
    """
    ดาวน์โหลดรูปภาพจาก URL และแปลงเป็น NumPy Array (RGB)

    Args:
        url: ลิงก์รูปภาพ
        timeout: เวลารอสูงสุด (วินาที)

    Returns:
        np.ndarray รูปแบบ RGB (H, W, 3)

    Raises:
        ValueError: ถ้าดาวน์โหลดไม่สำเร็จ หรือ decode รูปไม่ได้
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        raise ValueError(f"ไม่สามารถเชื่อมต่อ URL ได้: {e}")

    if response.status_code != 200:
        raise ValueError(f"ดาวน์โหลดรูปภาพไม่สำเร็จ (Status Code: {response.status_code})")

    image_array = np.frombuffer(response.content, np.uint8)
    img_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("ไม่สามารถ decode ข้อมูลเป็นรูปภาพได้ (ไฟล์อาจเสียหายหรือไม่ใช่รูปภาพ)")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def read_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """แปลงข้อมูล bytes (เช่นจาก upload ผ่าน FastAPI UploadFile) เป็น NumPy Array (RGB)"""
    image_array = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("ไม่สามารถ decode ข้อมูลเป็นรูปภาพได้")

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ==========================================
# 2. Pre-processing: Resize + Color + Denoise
# ==========================================
def preprocess_image(
    img_rgb: np.ndarray,
    target_size: tuple = (224, 224),
    denoise: bool = True,
) -> np.ndarray:
    """
    ปรับขนาดภาพ + ลด Noise ให้พร้อมส่งเข้าโมเดล

    Args:
        img_rgb: ภาพต้นฉบับ (RGB)
        target_size: ขนาดเป้าหมาย (W, H) — ค่า default 224x224 ตาม CLIP input size
                     (ใช้ 384x384 ได้ถ้าโมเดลต้องการ)
        denoise: เปิด/ปิดการลด Noise (Non-Local Means)

    Returns:
        np.ndarray ภาพที่ผ่านการ resize/denoise แล้ว (RGB, uint8)
    """
    if img_rgb is None or img_rgb.size == 0:
        raise ValueError("ภาพต้นฉบับว่างเปล่า ไม่สามารถประมวลผลได้")

    # เลือก interpolation ตามทิศทางการปรับขนาด (ย่อ = AREA ให้ผลลัพธ์คมกว่า, ขยาย = LINEAR)
    h, w = img_rgb.shape[:2]
    shrinking = target_size[0] < w or target_size[1] < h
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR

    resized_img = cv2.resize(img_rgb, target_size, interpolation=interpolation)

    if denoise:
        processed_img = cv2.fastNlMeansDenoisingColored(resized_img, None, 10, 10, 7, 21)
    else:
        processed_img = resized_img

    return processed_img


def normalize_for_model(img_rgb: np.ndarray) -> np.ndarray:
    """Min-Max Scaling ภาพให้อยู่ในช่วง 0.0-1.0 (float32) พร้อมส่งเข้าโมเดล/Tensor"""
    return img_rgb.astype("float32") / 255.0


def normalize_for_display(img_rgb: np.ndarray) -> np.ndarray:
    """Contrast Stretching ให้ค่าพิกเซลอยู่ในช่วง 0-255 (uint8) สำหรับแสดงผล"""
    norm_img = np.zeros_like(img_rgb)
    cv2.normalize(img_rgb, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return norm_img


# ==========================================
# 3. Quality Metrics
# ==========================================
def calculate_blur(img_rgb: np.ndarray) -> float:
    """
    วัดความคมชัดของภาพด้วย Laplacian Variance
    ค่ายิ่งสูง = ภาพยิ่งคมชัด (ไม่เบลอ), ค่าต่ำ = ภาพเบลอ
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(laplacian_var)


def calculate_brightness(img_rgb: np.ndarray) -> float:
    """ค่าความสว่างเฉลี่ยของภาพ (0-255) โดยใช้ V channel ของ HSV"""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    return float(np.mean(v_channel))


def calculate_contrast(img_rgb: np.ndarray) -> float:
    """ค่า Contrast ของภาพ วัดจากส่วนเบี่ยงเบนมาตรฐาน (std) ของค่าความสว่าง"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return float(np.std(gray))


def calculate_noise(img_rgb: np.ndarray) -> float:
    """
    ประมาณค่า Noise ของภาพ โดยเปรียบเทียบภาพต้นฉบับกับภาพที่ผ่านการลด Noise แล้ว
    ค่ายิ่งสูง = สัญญาณรบกวน (noise) ยิ่งมาก
    """
    denoised = cv2.fastNlMeansDenoisingColored(img_rgb, None, 10, 10, 7, 21)
    diff = cv2.absdiff(img_rgb, denoised)
    return float(np.mean(diff))


def calculate_quality_metrics(img_rgb: np.ndarray) -> dict:
    """คืนค่า raw metrics ทั้งหมดของภาพในรูป dict เดียว"""
    return {
        "blur": calculate_blur(img_rgb),
        "brightness": calculate_brightness(img_rgb),
        "contrast": calculate_contrast(img_rgb),
        "noise": calculate_noise(img_rgb),
    }


# ==========================================
# 4. ปรับสเกล Quality Scores ให้อยู่ในช่วง [0.0, 1.0]
# ==========================================
# ค่าอ้างอิง (reference range) สำหรับแต่ละ metric — ปรับได้ตามชุดข้อมูลจริงภายหลัง
_METRIC_RANGES = {
    "blur": (0, 1000),         # Laplacian variance: ยิ่งสูงยิ่งคมชัด
    "brightness": (0, 255),    # V channel เฉลี่ย
    "contrast": (0, 128),      # std ของค่าความสว่าง
    "noise": (0, 30),          # ค่าเฉลี่ยผลต่างก่อน/หลัง denoise
}


def _min_max_scale(value: float, min_val: float, max_val: float) -> float:
    """Min-Max scale ค่าเดียวให้อยู่ในช่วง [0.0, 1.0] พร้อม clip ขอบเขต"""
    if max_val == min_val:
        return 0.0
    scaled = (value - min_val) / (max_val - min_val)
    return float(np.clip(scaled, 0.0, 1.0))


def normalize_quality_scores(raw_metrics: dict) -> dict:
    """
    แปลง raw quality metrics ให้อยู่ในช่วง [0.0, 1.0] ทุกตัว

    หมายเหตุ:
    - blur, contrast: ยิ่งค่าสูง = คุณภาพยิ่งดี -> scale ตรงไปตรงมา
    - noise: ยิ่งค่าสูง = คุณภาพยิ่งแย่ -> กลับค่า (1 - scaled) ให้ "1.0 หมายถึงดี" เสมอ
    """
    blur_min, blur_max = _METRIC_RANGES["blur"]
    bright_min, bright_max = _METRIC_RANGES["brightness"]
    contrast_min, contrast_max = _METRIC_RANGES["contrast"]
    noise_min, noise_max = _METRIC_RANGES["noise"]

    blur_score = _min_max_scale(raw_metrics["blur"], blur_min, blur_max)
    brightness_score = _min_max_scale(raw_metrics["brightness"], bright_min, bright_max)
    contrast_score = _min_max_scale(raw_metrics["contrast"], contrast_min, contrast_max)

    # noise: กลับทิศทาง เพราะ noise สูง = แย่
    noise_scaled = _min_max_scale(raw_metrics["noise"], noise_min, noise_max)
    noise_score = 1.0 - noise_scaled

    return {
        "blur": round(blur_score, 4),
        "brightness": round(brightness_score, 4),
        "contrast": round(contrast_score, 4),
        "noise": round(noise_score, 4),
    }


def get_quality_scores(img_rgb: np.ndarray) -> dict:
    """Convenience function: คำนวณ raw metrics แล้ว normalize ให้อยู่ใน [0,1] ในขั้นตอนเดียว"""
    raw = calculate_quality_metrics(img_rgb)
    normalized = normalize_quality_scores(raw)
    return {"raw": raw, "normalized": normalized}

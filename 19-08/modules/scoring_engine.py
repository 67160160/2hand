"""
modules/scoring_engine.py
--------------------------
Tech 2 responsibilities (วันจันทร์ - วันอังคาร):

- เขียนฟังก์ชัน Cosine Similarity: คำนวณความเหมือนระหว่างเวกเตอร์ภาพใหม่กับภาพอ้างอิง
- ทดสอบเทียบกับคู่ภาพ 3 กรณี (เหมือน 100%, ตัดขอบ/ปรับแสง, คนละสินค้า)

รับ input เป็นเวกเตอร์ที่ผ่านการ L2 Normalize มาจาก Tech Lead (main.py)
หรือจะรับเวกเตอร์ raw แล้ว normalize เองในนี้ก็ได้ (safe-guard)
"""

import numpy as np


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """
    ปรับความยาวเวกเตอร์ให้เป็น 1.0 (v / ||v||_2)
    ใส่ไว้เป็น safeguard เผื่อเวกเตอร์ที่รับเข้ามายังไม่ได้ normalize
    """
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray, assume_normalized: bool = False) -> float:
    """
    คำนวณ Cosine Similarity ระหว่างเวกเตอร์สองตัว

    Args:
        vec_a, vec_b: เวกเตอร์ภาพ (เช่น CLIP embedding 512 มิติ)
        assume_normalized: ถ้า True จะข้ามขั้นตอน normalize (เร็วกว่า, ใช้เมื่อมั่นใจว่า
                            เวกเตอร์ผ่าน L2 normalize มาจาก Tech Lead แล้ว)

    Returns:
        ค่า similarity ในช่วง [-1.0, 1.0] โดย 1.0 = เหมือนกันทุกประการ
    """
    vec_a = np.asarray(vec_a, dtype=np.float64)
    vec_b = np.asarray(vec_b, dtype=np.float64)

    if vec_a.shape != vec_b.shape:
        raise ValueError(f"ขนาดเวกเตอร์ไม่ตรงกัน: {vec_a.shape} vs {vec_b.shape}")

    if not assume_normalized:
        vec_a = l2_normalize(vec_a)
        vec_b = l2_normalize(vec_b)

    similarity = float(np.dot(vec_a, vec_b))
    # กันค่า floating point เกินขอบเขตเล็กน้อย เช่น 1.0000000002
    return float(np.clip(similarity, -1.0, 1.0))


def batch_cosine_similarity(query_vec: np.ndarray, reference_vecs: dict) -> dict:
    """
    เทียบเวกเตอร์ภาพใหม่ (query) กับเวกเตอร์อ้างอิงหลายภาพในคราวเดียว

    Args:
        query_vec: เวกเตอร์ภาพใหม่ที่ต้องการตรวจสอบ
        reference_vecs: dict {image_id: vector} ของภาพอ้างอิงทั้งหมดใน Mock DB

    Returns:
        dict {image_id: similarity_score} เรียงจากคะแนนสูงไปต่ำ
    """
    scores = {
        image_id: cosine_similarity(query_vec, ref_vec)
        for image_id, ref_vec in reference_vecs.items()
    }
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


def find_best_match(query_vec: np.ndarray, reference_vecs: dict) -> tuple:
    """
    หาภาพอ้างอิงที่คล้ายกับ query มากที่สุด

    Returns:
        (image_id, similarity_score) ของภาพที่คล้ายที่สุด หรือ (None, 0.0) ถ้าไม่มีภาพอ้างอิง
    """
    if not reference_vecs:
        return None, 0.0

    scores = batch_cosine_similarity(query_vec, reference_vecs)
    best_id = next(iter(scores))
    return best_id, scores[best_id]

import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt

# 1. ระบุ URL ของรูปภาพ
# (สามารถเปลี่ยนเป็นลิงก์ภาพตัวอย่างอื่นๆ ที่ต้องการทดสอบได้)
url = "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg?utm_source=th.wikipedia.org&utm_campaign=index&utm_content=original"

# เพิ่ม headers เพื่อจำลองว่าเป็นเว็บเบราว์เซอร์
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

print("กำลังดาวน์โหลดรูปภาพจาก URL...")
# ใส่ headers เข้าไปใน requests.get
response = requests.get(url, headers=headers)
# ตรวจสอบว่าดาวน์โหลดสำเร็จหรื
if response.status_code == 200:
    # 2. แปลงข้อมูลไบต์เป็นอาร์เรย์และถอดรหัสเป็นรูปภาพด้วย OpenCV
    image_array = np.frombuffer(response.content, np.uint8)
    img_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # แปลงสี BGR เป็น RGB ให้แสดงผลได้ถูกต้อง
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    print("ดาวน์โหลดสำเร็จ! กำลังประมวลผลภาพ...")

    # ==========================================
    # ขั้นตอนที่ 1: Resize (ปรับขนาดภาพ)
    # ==========================================
    # ปรับขนาดภาพให้เป็น 640x640 ซึ่งเป็นขนาดมาตรฐานสำหรับโมเดล Object Detection
    target_size = (640, 640)
    resized_img = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)

    # ==========================================
    # ขั้นตอนที่ 2: Denoise (ลดสัญญาณรบกวน)
    # ==========================================
    # ใช้ Non-Local Means Denoising เพื่อลดเกรนภาพโดยยังคงขอบวัตถุไว้
    denoised_img = cv2.fastNlMeansDenoisingColored(resized_img, None, 10, 10, 7, 21)

    # ==========================================
    # ขั้นตอนที่ 3: Color Normalization (ปรับบรรทัดฐานของค่าสี)
    # ==========================================
    # แบบที่ A: สำหรับแสดงผล (Contrast Stretching ให้อยู่ในช่วง 0-255)
    norm_img_cv2 = np.zeros_like(denoised_img)
    cv2.normalize(denoised_img, norm_img_cv2, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # แบบที่ B: สำหรับส่งเข้าเทรนโมเดล (Min-Max Scaling ให้อยู่ในช่วง 0.0 - 1.0)
    # ตัวแปรนี้พร้อมนำไปจัดรูปเป็น Tensor หรือ Array เข้าโมเดลได้เลย
    norm_img_model = denoised_img.astype('float32') / 255.0

    # ==========================================
    # ขั้นตอนที่ 4: แสดงผลเปรียบเทียบ
    # ==========================================
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    plt.title("1. Original RGB")
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title(f"2. Resized {target_size}")
    plt.imshow(resized_img)
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("3. Denoised")
    plt.imshow(denoised_img)
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("4. Normalized (0-255)")
    plt.imshow(norm_img_cv2)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("ประมวลผลเสร็จสิ้น!")

else:
    print(f"เกิดข้อผิดพลาด: ไม่สามารถโหลดรูปภาพได้ (Status Code: {response.status_code})")

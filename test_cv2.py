import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. โหลดภาพ (เปลี่ยน 'sample.jpg' เป็นพาทไฟล์ภาพของคุณ)
# ใช้ cv2.cvtColor เพื่อสลับสีจาก BGR (ค่าเริ่มต้นของ OpenCV) เป็น RGB ให้แสดงผลได้ถูกต้อง
img_bgr = cv2.imread('sample.jpg')
if img_bgr is None:
    print("ไม่พบไฟล์ภาพ กรุณาตรวจสอบพาท")
else:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ==========================================
    # 1. Resize (ปรับขนาดภาพ)
    # ==========================================
    # ปรับขนาดภาพให้เป็น 640x640 (ขนาดที่นิยมใช้เป็น Input ของโมเดลต่างๆ)
    target_size = (640, 640)
    resized_img = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)

    # ==========================================
    # 2. Denoise (ลดสัญญาณรบกวน)
    # ==========================================
    # ใช้ฟังก์ชัน Non-Local Means Denoising สำหรับภาพสี
    # ช่วยกำจัดเกรนหรือ Noise ในภาพถ่ายสภาพแวดล้อมจริง โดยยังคงความคมชัดของขอบวัตถุ (Edges)
    # พารามิเตอร์: src, dst, h, hColor, templateWindowSize, searchWindowSize
    denoised_img = cv2.fastNlMeansDenoisingColored(resized_img, None, 10, 10, 7, 21)
    
    # ทางเลือกเพิ่มเติม: ถ้าต้องการแค่เบลอภาพเพื่อลด Noise แบบง่ายๆ สามารถใช้ GaussianBlur ได้
    # blurred_img = cv2.GaussianBlur(resized_img, (5, 5), 0)

    # ==========================================
    # 3. Color Normalization (ปรับบรรทัดฐานของค่าสี)
    # ==========================================
    # แบบที่ 1: การใช้ cv2.normalize (ยืด-หด การกระจายตัวของพิกเซล)
    # มีประโยชน์เวลาเจอภาพที่มืดหรือสว่างเกินไป (Contrast Stretching)
    norm_img_cv2 = np.zeros_like(denoised_img)
    cv2.normalize(denoised_img, norm_img_cv2, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # แบบที่ 2: Min-Max Scaling ด้วย NumPy (นิยมใช้ก่อนโยนเข้าโมเดล Machine Learning)
    # แปลงช่วงพิกเซลจาก 0-255 ให้กลายเป็น 0.0 - 1.0
    norm_img_model = denoised_img.astype('float32') / 255.0

    # ==========================================
    # แสดงผลเปรียบเทียบ (ใช้ Matplotlib เพื่อความสะดวกหากรันใน Colab/Jupyter)
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
    plt.title("3. Denoised (NLMeans)")
    plt.imshow(denoised_img)
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("4. Normalized (0-255)")
    plt.imshow(norm_img_cv2)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
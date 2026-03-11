import cv2
import os

def apply_enhanced_adaptive_filter(img_dir):
    if not os.path.exists(img_dir):
        print(f"⚠️ 경로를 찾을 수 없습니다: {img_dir}")
        return

    # 1. CLAHE 객체 생성 (어두운 곳 글씨 대비 강화)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    file_list = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"📂 {img_dir} 변환 시작 (640px, CLAHE + Adaptive 21/3)...")

    for filename in file_list:
        img_path = os.path.join(img_dir, filename)
        src = cv2.imread(img_path)
        if src is None: continue

        # 2. 640x640 해상도로 고정
        img_640 = cv2.resize(src, (640, 640))

        # 3. 그레이스케일 변환
        gray = cv2.cvtColor(img_640, cv2.COLOR_BGR2GRAY)

        # 4. [핵심] CLAHE 적용 (카메라 설정과 동일)
        enhanced_gray = clahe.apply(gray)

        # 5. Adaptive Threshold 적용
        # blockSize=21, C=3, INV 필터 적용
        binary_inv = cv2.adaptiveThreshold(
            enhanced_gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            21, 
            3
        )

        # 6. 저장 (원본 덮어쓰기)
        cv2.imwrite(img_path, binary_inv)

    print(f"✅ {img_dir} 변환 완료!")

# 경로 설정
base_path = "/home/limdoyeon/realsense_apriltag_1/carplate.v1i.yolov11"

# 실행 (Train, Valid, Test 모든 데이터에 동일 필터 적용)
for split in ["train", "valid", "test"]:
    apply_enhanced_adaptive_filter(os.path.join(base_path, f"{split}/images"))
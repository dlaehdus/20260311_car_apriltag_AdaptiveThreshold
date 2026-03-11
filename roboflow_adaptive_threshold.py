# 이미지 전처리 라이브러리
import cv2
# 폴더 경로 이동 파일 목록읽기
import os


# 폴더 경로를 입력받아 필터를 적용하는 함수를 정의
def apply_enhanced_adaptive_filter(img_dir):
    # 입력된 경로가 실제로 존재하는지 확인하여 에러를 방지합니다.
    if not os.path.exists(img_dir):
        print(f"경로를 찾을 수 없습니다: {img_dir}")
        return
    # CLAHE는 이미지의 일부분씩 대비를 높여주는 알고리즘입니다.
    # clipLimit=3.0: 대비가 너무 과해져 노이즈가 생기는 것을 방지하는 한계치입니다.
    # tileGridSize=(8, 8): 이미지를 8x8 구역으로 나누어 각 구역별로 대비를 조절합니다. 어두운 곳에 있는 번호판 숫자를 선명하게 만드는 핵심 도구
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    # 폴더 내 파일 중 이미지 확장자(jpg, png 등)를 가진 파일들만 골라 리스트로 만듭니다.
    file_list = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"{img_dir} 변환 시작")
    # 찾은 이미지들을 하나씩 꺼내어 아래 과정을 반복합니다.
    for filename in file_list:
        img_path = os.path.join(img_dir, filename)
        # 이미지를 읽어옵니다.
        src = cv2.imread(img_path)
        if src is None: continue
        # YOLO 학습 설정인 imgsz: 640에 맞춰 모든 이미지를 640x640 크기로 강제 고정합니다.
        img_640 = cv2.resize(src, (640, 640))
        # 컬러 이미지를 흑백(Grayscale)으로 바꿉니다. (이진화를 위한 필수 단계)
        gray = cv2.cvtColor(img_640, cv2.COLOR_BGR2GRAY)
        # 앞서 설정한 CLAHE를 적용하여 흐릿하거나 어두운 부분의 숫자 윤곽을 뚜렷하게 만듭니다.
        enhanced_gray = clahe.apply(gray)
        # Adaptive Threshold: 이미지 전체에 하나의 기준을 두지 않고, 주변 픽셀값들을 보고 구역마다 최적의 문턱값(Threshold)을 정해 흑백(이진화)으로 만듭니다.
        # 255: 흰색의 최대값입니다.
        # ADAPTIVE_THRESH_GAUSSIAN_C: 주변 픽셀에 가중치를 두어 더 부드럽게 이진화합니다.
        # THRESH_BINARY_INV: 글자를 흰색, 배경을 검은색으로 반전시킵니다. (YOLO가 글자 형태에 집중하게 함)
        # 21: 주변 영역의 크기(Block Size)입니다. 숫자가 작을수록 세밀한 선을 찾습니다.
        # 3: 계산된 평균값에서 뺄 상수입니다. 노이즈 제거 역할을 합니다.
        binary_inv = cv2.adaptiveThreshold(
            enhanced_gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            19, 
            9
        )
        # 필터가 적용된 결과물로 기존 원본 파일을 덮어씁니다.
        cv2.imwrite(img_path, binary_inv)
    print(f"{img_dir} 변환 완료")
# 데이터셋이 들어있는 바탕 경로
base_path = "/home/pfcheon/Desktop/limdoyeon/carplate.v1i.yolov11"

# 실행 (Train, Valid, Test 모든 데이터에 동일 필터 적용)
for split in ["train", "valid", "test"]:
    apply_enhanced_adaptive_filter(os.path.join(base_path, f"{split}/images"))

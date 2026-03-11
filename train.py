# YOLO 시리즈의 최신 라이브러리를 가져옵니다.
from ultralytics import YOLO

# 미리 학습된 YOLO11 Medium 모델을 불러옵니다.
# 'm'은 복잡한 패턴(다양한 숫자 클래스)을 잘 학습하며, 번호판 내 글자/숫자 인식에 적합합니다.
model = YOLO('/home/limdoyeon/realsense_apriltag_1/runs/detect/EV_Plate_Character_MultiClass_v1/weights/last.pt')

# 데이터셋의 경로(학습/검증 이미지 위치)와 클래스 정보가 적힌 설정 파일의 경로입니다.
yaml_path = "/home/limdoyeon/realsense_apriltag_1/carplate.v1i.yolov11/data.yaml"

if __name__ == '__main__':
    model.train(
        
        resume=True,

        data=yaml_path,
        # 전체 데이터를 250번 반복 학습합니다. 46개 클래스의 복잡성을 고려해 과적합 방지 위해 에폭 줄임.
        epochs=150,
        # 성능향상이 80에폭동안 없으면 학습 종료. 멀티 클래스 학습의 안정성을 위해 조정.
        patience=80,
        # 입력 이미지 크기. 고해상도 유지로 작은 글자나 비스듬한 왜곡을 세밀하게 포착.
        imgsz=640,
        # 배치 크기. 메모리 안정성을 위해 4 유지.
        batch=1,
        # Automatic Mixed Precision. 속도와 메모리 최적화.
        amp=True,
        # 워커 수. 균형 유지.
        workers=4,
        # GPU 지정.
        device=0,
        # 멀티 클래스(46개 숫자/코드 클래스) 학습을 위해 single_cls 제거.
        # 이미지를 40도까지 회전. 비스듬한 차량 각도(번호판 왜곡) 대응.
        degrees=40.0,
        # 이동 증강. 글자가 프레임 구석에 있을 때 대비.
        translate=0.25,
        # 스케일 증강. 거리/크기 변화에 강함.
        scale=0.75,
        # 전단 증강. 기울어진 플레이트로 인한 글자 왜곡 대응.
        shear=18.0,
        # 투영 증강. 3D 관점 변화(비스듬한 각도)로 어떤 방향에서도 글자 인식 강화.
        perspective=0.0009,
        # 상하 반전. 번호판이 뒤집힌 듯한 극단 시나리오 대비.
        flipud=0.4,
        # 좌우 반전. 차량 방향 변화 대응.
        fliplr=0.5,
        # 모자이크 증강. 복잡한 배경(차량, 도로)에서 글자만 선별 학습.
        mosaic=0.9,
        # Mixup. 다른 이미지 글자 블렌딩으로 일반화 향상, 과적합 방지.
        mixup=0.25,
        # Copy-paste. 희귀 숫자/코드 조합(10-45) 패턴을 증가시켜 균형 학습.
        copy_paste=0.35,
        # 멀티 스케일. 이미지 크기 변동으로 다양한 해상도/각도 글자 인식.
        multi_scale=True,
        # 색상 증강. 조명 변화(낮/밤, 그림자) 대응, 글자 선명도 유지.
        hsv_h=0.04,
        hsv_s=0.8,
        hsv_v=0.5,
        # AdamW 최적화기. 멀티 클래스에서 안정적 수렴, 과적합 방지.
        optimizer='AdamW',
        # 학습률. 낮은 시작으로 세밀한 글자 특징 학습.
        lr0=0.0008,
        lrf=0.008,
        # 가중치 감쇠 강화. 46개 클래스 과적합 방지 핵심.
        weight_decay=0.001,
        # 모멘텀. 학습 안정화.
        momentum=0.937,
        # 모자이크 종료 시점. 마지막 15에폭에서 증강 끄고 정밀 조정.
        close_mosaic=15,
        # 코사인 LR 스케줄러. 부드러운 학습률 변화로 과적합 최소화.
        cos_lr=True,
        # 드롭아웃. 네트워크 과적합 방지 추가 (YOLO 지원).
        dropout=0.15,
        # 라벨 스무딩. 클래스 불균형(희귀 코드) 대응, 일반화 향상.
        label_smoothing=0.1,
        # 학습 결과 저장 폴더. 버전 관리.
        name='EV_Plate_Character_MultiClass_v1'
    )
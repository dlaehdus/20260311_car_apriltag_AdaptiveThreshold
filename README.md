<img width="971" height="241" alt="image" src="https://github.com/user-attachments/assets/6fa90c24-69db-4fbd-a0a0-f62c0433c4e3" />기존의 Car_apriltag의 방법으로는 어두운곳에서나 조도가 바뀌면 HSV의 필터를 이용해서 번호판이 가려지는 문제가 발생함 이에 따라 HSV필터를 제거한 버전으로 사용하려 했으나 이도 조도가 낮으면 인식률이 저하됨을 확인함

따라서 이번에는 Adaptive Threshold필터를 이용해서 새롭게 데이터를 학습시키고 해당 데이터를 기반으로 인식률문제를 해결할것임
https://charlezz.com/?p=45322


1. 번호판검출에 필요한 데이터셋을 다운로드
       https://universe.roboflow.com/multimedia2024-ychar/carplate-hoowb/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true
       <img width="713" height="160" alt="image" src="https://github.com/user-attachments/assets/7305e7a7-cb95-40d1-93ac-17f02df8938d" />
2. 해당 이미지를 Adaptive Threshold필터를 적용한 이미지로 모두 변경, roboflow_adaptive_threshold.py 참조
       <img width="1259" height="658" alt="image" src="https://github.com/user-attachments/assets/db273c29-38ca-4bd0-bef8-4f310ba02361" />
3. Yolo11m기반으로 train을 시킴 train.py파일 참조
       NoMachine을 사용해 학습중인 모습
       <img width="1709" height="976" alt="image" src="https://github.com/user-attachments/assets/1a2ab38a-89f3-464a-bf96-68ab95b492ee" />
       학습 결과 세부 파라미터는 runs파일의 args.yaml을 참조
       정확도와 재현율, mAP50등등
       <img width="971" height="241" alt="image" src="https://github.com/user-attachments/assets/9f82dd8c-7d82-45df-97f9-00d8e7c78246" />
       본인은 A6000 GPU4장으로 학습을 진행함




















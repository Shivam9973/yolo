from ultralytics import YOLO

model = YOLO('yolov8n.yaml')
model = YOLO('yolov8n.pt')
model = YOLO('yolov8n.yaml').load('yolov8n.pt')

model.train(data='coco128.yaml', epochs=1, imgsz=640)
source = "C://Users//shant//Videos//Valorant//Valorant 2023.08.09 - 16.53.55.01.mp4"

results = model(source, stream=True)

model.predict(source, save=True, imgsz=320, conf=0.5)
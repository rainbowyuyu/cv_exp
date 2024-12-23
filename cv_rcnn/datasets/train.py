from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.train(data="data.yaml", epochs=500, imgsz=640, batch=100, device='cuda')

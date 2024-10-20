from ultralytics import YOLO
model = YOLO("yolo11n-seg.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="pastis.yaml", epochs=100, imgsz=640)
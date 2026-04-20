from ultralytics import YOLO

# 1. Load a pre-trained "Nano" model (smallest and fastest for drones)
model = YOLO('yolov8n.pt')

# 2. Train the model
# 'data.yaml' contains the paths to your tunnel images and labels
results = model.train(data='tunnel_data.yaml', epochs=50, imgsz=640)

# 3. Export the model to use in your drone script
model.export(format='onnx')

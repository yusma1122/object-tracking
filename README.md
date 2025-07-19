# 📷 Real-Time Object Detection from Screen Capture using YOLOv5

This project performs **real-time object detection** on **protected video streams** (e.g. CCTV web player that requires cookies or tokens) by using **screen capture** and **YOLOv5**.

Instead of directly accessing the stream with tools like ffmpeg or cv2.VideoCapture, this project uses `mss` to **capture a region of the screen** where the video is displayed, then runs object detection frame-by-frame.

---

## 🎯 Background

I started this project while learning YOLOv5 with my webcam. After getting comfortable with the basics, I wanted to apply it to public CCTV streams. However, I discovered that many CCTV feeds use one-time tokens or cookie-based authentication that prevent direct programmatic access.

Rather than trying to bypass these protections, I developed a creative workaround using screen capture!

---

## 🔧 Features

- ✅ Works even when video streams are token-protected or restricted
- ✅ Uses `mss` to grab a specific screen area  
- ✅ Real-time object detection with YOLOv5
- ✅ Bounding box visualization with OpenCV
- ✅ Lightweight and easy to customize

---

## 📋 Requirements

Make sure you have Python **3.8+** installed, then install the dependencies:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
torch>=1.7
torchvision
opencv-python
mss
numpy
```

You also need the **YOLOv5 repository** (local clone):

```bash
git clone https://github.com/ultralytics/yolov5.git yolov5-master
```

And place your model file (`yolov5s.pt`, or any custom `.pt` file) in the project folder.

---

## ▶️ How to Use

1. **Play your CCTV video** on screen (e.g. from a browser window)

2. **Update the screen capture region** in the script to match your video position:

```python
monitor = {
    "top": 200,    # Adjust based on video position on screen
    "left": 100,   # Distance from left edge of screen
    "width": 640,  # Video width
    "height": 360  # Video height
}
```

3. **Run the script:**

```bash
python detect_from_screen.py
```

4. **Press `q` to quit**

---

## 📁 Main Script: `detect_from_screen.py`

```python
import torch
import cv2
import numpy as np
import mss

# ==== CONFIGURE SCREEN CAPTURE COORDINATES ====
monitor = {
    "top": 200,    # Top position of video on your screen
    "left": 100,   # Left position of video on your screen  
    "width": 640,  # Video width
    "height": 360  # Video height
}

# Load YOLOv5 model
print("Loading YOLOv5 model...")
model = torch.hub.load('yolov5-master', 'custom', path='yolov5s.pt', source='local')
model.conf = 0.5  # confidence threshold
print("Model loaded successfully!")

# Start screen capture
sct = mss.mss()
print("Starting screen capture... Press 'q' to quit")

while True:
    # Capture screen region
    frame = np.array(sct.grab(monitor))
    frame = frame[:, :, :3]  # Remove alpha channel (BGRA -> BGR)
    
    # Run YOLO detection
    results = model(frame)
    annotated = results.render()[0]
    
    # Display result
    cv2.imshow("YOLOv5 on Screen Capture", annotated)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("Detection stopped.")
```

---

## 🔧 Customization

### Using Your Own Model
Replace `yolov5s.pt` with your custom trained model:

```python
model = torch.hub.load('yolov5-master', 'custom', path='your_custom_model.pt', source='local')
```

### Adjusting Detection Settings
```python
model.conf = 0.4   # Lower = more detections (less strict)
model.iou = 0.45   # IoU threshold for NMS
```

### Finding Screen Coordinates
To help you find the right coordinates, run this helper script:

```python
import mss
import cv2
import numpy as np

# This will show your current mouse position
def show_coordinates():
    sct = mss.mss()
    monitor = {"top": 0, "left": 0, "width": 400, "height": 300}
    
    while True:
        frame = np.array(sct.grab(monitor))
        frame = frame[:, :, :3]
        cv2.imshow("Position Finder - Move this window", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

show_coordinates()
```

---

## 🎥 Example Use Cases

- **City Traffic CCTV**: Monitor vehicle counts and types
- **Security Monitoring**: Detect people or objects in protected camera feeds  
- **Wildlife Cameras**: Analyze animal behavior from web-based camera streams
- **Retail Analytics**: Count customers from store camera displays
- **Industrial Monitoring**: Quality control from protected machinery cameras

---

## 📝 Notes

- This method works with **any video displayed on screen** (YouTube, embedded players, protected streams, etc.)
- Performance depends on your hardware - expect 15-30 FPS on modern systems
- For better accuracy, ensure good lighting and video quality
- The model runs locally, so no internet required after setup

---

## 🐛 Troubleshooting

**Q: The detection window is black**
A: Check your monitor coordinates. Make sure they match where your video is displayed.

**Q: Low FPS performance**
A: Try using a smaller YOLOv5 model like `yolov5n.pt` or reduce the capture area.

**Q: Model not loading**
A: Ensure you've cloned the YOLOv5 repository and the model file is in the correct location.

---

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Credits

- [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics
- [mss](https://github.com/BoboTiG/python-mss) - Python cross-platform screenshot tool
- OpenCV for computer vision operations

---

⭐ **If this project helped you, please give it a star!** ⭐
import torch
import cv2
import numpy as np
import mss

# === Load YOLOv5 Model ===
model = torch.hub.load('yolov5-master', 'custom', path='yolov5s.pt', source='local')
model.conf = 0.5  # Confidence threshold

# === Atur area screen capture (koordinat video CCTV di layar) ===
monitor = {
    "top": 200,      # ganti ini sesuai posisi atas video di layar kamu
    "left": 100,     # ganti ini sesuai posisi kiri video di layar kamu
    "width": 640,    # lebar area video
    "height": 360    # tinggi area video
}

sct = mss.mss()

print("✅ Deteksi dimulai. Tekan 'q' untuk keluar.")

while True:
    # Ambil frame dari layar
    frame = np.array(sct.grab(monitor))
    frame = frame[:, :, :3]  # Hilangkan alpha channel (BGRA -> BGR)

    # Deteksi objek dengan YOLOv5
    results = model(frame)

    # Render bounding box dan label
    annotated_frame = results.render()[0]

    # Tampilkan hasil ke layar
    cv2.imshow("YOLOv5 - CCTV Screen Capture", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup semua jendela
cv2.destroyAllWindows()

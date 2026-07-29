import os
import sys
import time
import cv2
import math
import mediapipe as mp

from Actuator import connect_dobot

# --- KOORDINAT AMAN AWAL (SAFE HOME) ---
current_x = 200 
current_y = 0
current_z = 0

def kode_untuk_dobot(jari_terbuka, petak_kolom, petak_baris, device):
    global current_x, current_y, current_z
    
    if jari_terbuka < 3:
        # Mode 1: Depan-belakang (X) dan Kiri-kanan (Y)
        A = 0
        B = 0
        C = 0
        D = 0
        current_y = A * petak_kolom + B
        current_x = C * petak_baris + D
        
        device.move_to(current_x, current_y, current_z, 0, wait=False) 
    else:
        # Mode 2: Atas-bawah (Z) dan Kiri-kanan (Y)
        A = 0
        B = 0
        E = 0
        F = 0
        current_y = A * petak_kolom + B
        current_z = E * petak_baris + F
        
        device.move_to(current_x, current_y, current_z, 0, wait=False)


# --- INISIALISASI DOBOT ---
print("[INFO] Menghubungkan ke Dobot...")
device = connect_dobot()
if device is None:
    print("[ERROR] Dobot tidak terhubung. Keluar.")
    sys.exit(1)

print("[INFO] Memulai proses Homing (sekitar 30 detik)...")
device.home()
time.sleep(30)


# --- 1. Sembunyikan Log Warning TensorFlow ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- 2. Fix Path File Model ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")

if not os.path.exists(MODEL_PATH):
    print(f"\n[ERROR] File 'hand_landmarker.task' TIDAK DITEMUKAN di: {MODEL_PATH}")
    sys.exit(1)

# --- 3. Inisialisasi MediaPipe Tasks ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None

def print_result(result, output_image, timestamp_ms: int):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
    result_callback=print_result
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

UKURAN_PETAK = 5 

# --- PENGATURAN COOLDOWN DOBOT (DIPERCEPAT) ---
waktu_terakhir_kirim = 0
COOLDOWN_DOBOT = 0.1  # Respon robot dipercepat (kirim perintah setiap 0.1 detik)

# --- PENGATURAN KAMERA & FPS ---
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FPS, 24) # Mencoba meminta hardware untuk set ke 24 FPS

TARGET_FPS = 24
FRAME_DELAY = 1.0 / TARGET_FPS # Durasi minimal untuk 1 frame (detik)

with HandLandmarker.create_from_options(options) as landmarker:
    while video.isOpened():
        loop_start_time = time.time() # Catat waktu awal frame

        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape 

        # Proses MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        if latest_result and latest_result.hand_landmarks:
            for hand_landmarks, handedness in zip(latest_result.hand_landmarks, latest_result.handedness):
                sum_x = 0
                sum_y = 0
                
                for connection in HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    pt1 = (int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h))
                    pt2 = (int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    sum_x += cx
                    sum_y += cy
                
                titik_berat_x = int(sum_x / 21) + 15
                titik_berat_y = int(sum_y / 21) + 50
                petak_baris = titik_berat_y // UKURAN_PETAK
                petak_kolom = titik_berat_x // UKURAN_PETAK
                cv2.circle(frame, (titik_berat_x, titik_berat_y), 10, (0, 255, 255), -1)
                
                jari_terbuka = 0
                hand_label = handedness[0].category_name
                
                def get_dist(idx1, idx2):
                    x1, y1 = hand_landmarks[idx1].x * w, hand_landmarks[idx1].y * h
                    x2, y2 = hand_landmarks[idx2].x * w, hand_landmarks[idx2].y * h
                    return math.hypot(x1 - x2, y1 - y2)

                if get_dist(4, 17) > get_dist(3, 17):
                    jari_terbuka += 1

                jari_lainnya = [(8, 6), (12, 10), (16, 14), (20, 18)]
                for tip, pip in jari_lainnya:
                    if get_dist(tip, 0) > get_dist(pip, 0):
                        jari_terbuka += 1

                teks_lokasi = f"Lokasi: ({petak_baris}, {petak_kolom})"
                cv2.putText(frame, teks_lokasi, (titik_berat_x - 40, titik_berat_y - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            
                teks_jari = f"Jari Terbuka: {jari_terbuka}"
                cv2.putText(frame, teks_jari, (titik_berat_x - 40, titik_berat_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                print(f"[{hand_label}] Jari: {jari_terbuka} | Petak: ({petak_kolom}, {petak_baris})")

                waktu_sekarang = time.time()
                if (waktu_sekarang - waktu_terakhir_kirim) > COOLDOWN_DOBOT:
                    kode_untuk_dobot(jari_terbuka, petak_kolom, petak_baris, device)
                    waktu_terakhir_kirim = waktu_sekarang 

        cv2.imshow("Hand Control - Grid System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # --- FITUR BARU: SOFTWARE FPS LIMITER ---
        # Memaksa loop untuk menunggu (delay) jika prosesnya lebih cepat dari target 24 FPS
        loop_time = time.time() - loop_start_time
        if loop_time < FRAME_DELAY:
            time.sleep(FRAME_DELAY - loop_time)

video.release()
cv2.destroyAllWindows()
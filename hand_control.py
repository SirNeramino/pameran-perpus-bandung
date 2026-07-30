import os
import sys
import time
import cv2
import math
import mediapipe as mp

from Actuator import connect_dobot

# --- KOORDINAT AMAN AWAL (SAFE HOME) ---
current_x = 250 
current_y = 0
current_z = 0

# --- LIMIT FISIK DOBOT (JANGAN DIUBAH TANPA CEK MANUAL DOBOT) ---
X_MIN, X_MAX = 155, 245
Y_MIN, Y_MAX = -125, 130
Z_MIN, Z_MAX = -12, 130

# --- RENTANG GRID HASIL DETEKSI TANGAN ---
# NOTE: asumsi resolusi kamera 640x480 (default OpenCV jika tidak di-set).
# Jika resolusi kamera kamu berbeda, ukur ulang dengan print(w, h) lalu
# hitung ulang KOLOM_MIN/MAX dan BARIS_MIN/MAX:
#   titik_berat_x = mean_x + 15   -> range: 15 .. (w+15)
#   titik_berat_y = mean_y + 50   -> range: 50 .. (h+50)
#   petak_kolom = titik_berat_x // UKURAN_PETAK
#   petak_baris = titik_berat_y // UKURAN_PETAK
KOLOM_MIN, KOLOM_MAX = 3, 131     # dari w=640
BARIS_MIN, BARIS_MAX = 10, 106    # dari h=480

# --- KOEFISIEN LINEAR (grid -> koordinat Dobot) ---
# Y = A * petak_kolom + B  (dipakai di Mode 1 & Mode 2)
A = (Y_MAX - Y_MIN) / (KOLOM_MAX - KOLOM_MIN)
B = Y_MIN - A * KOLOM_MIN

# X = C * petak_baris + D  (Mode 1)
C = (X_MAX - X_MIN) / (BARIS_MAX - BARIS_MIN)
D = X_MIN - C * BARIS_MIN

# Z = E * petak_baris + F  (Mode 2)
E = (Z_MIN - Z_MAX) / (BARIS_MAX - BARIS_MIN)
F = Z_MAX - E * BARIS_MIN               

def clamp(value, lo, hi):
    return max(lo, min(hi, value))

def clear_dobot_queue(device):
    """
    Coba kosongkan antrian command Dobot supaya command lama yang belum
    tereksekusi tidak menumpuk di belakang command baru (penyebab utama
    robot 'telat' merespons gerakan tangan terkini).
    CATATAN: nama method tergantung versi pydobot/pydobotplus, coba beberapa.
    """
    for method_name in ("clear_queue", "set_queued_cmd_clear", "stop_queue"):
        method = getattr(device, method_name, None)
        if callable(method):
            try:
                method()
                return
            except Exception:
                continue
    # Kalau tidak ada method yang cocok, cukup diam (fallback ke behavior lama).
    # Cek Actuator.py / dokumentasi pydobotplus untuk nama method yang benar.

def kode_untuk_dobot(jari_terbuka, petak_kolom, petak_baris, device):
    global current_x, current_y, current_z
    
    clear_dobot_queue(device)
    
    if jari_terbuka < 3:
        # Mode 1: Depan-belakang (X) dan Kiri-kanan (Y)
        current_y = clamp(A * petak_kolom + B, Y_MIN, Y_MAX)
        current_x = clamp(C * petak_baris + D, X_MIN, X_MAX)
        
        device.move_to(current_x, current_y, current_z, 0, wait=False) 
    else:
        # Mode 2: Atas-bawah (Z) dan Kiri-kanan (Y)
        current_y = clamp(A * petak_kolom + B, Y_MIN, Y_MAX)
        current_z = clamp(E * petak_baris + F, Z_MIN, Z_MAX)
        
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

# --- PERCEPAT KECEPATAN GERAK DOBOT ---
# CATATAN: nama method berbeda-beda tergantung versi pydobot/pydobotplus.
# Ini mencoba beberapa nama method yang umum dipakai. Kalau semuanya gagal,
# cek dokumentasi/isi Actuator.py kamu untuk nama method speed yang benar,
# lalu ganti baris di bawah sesuai API yang tersedia.
try:
    device.speed(velocity=100, acceleration=100)
    print("[INFO] Speed Dobot di-set via device.speed().")
except AttributeError:
    try:
        device.set_ptp_joint_params(velocity=100, acceleration=100)
        print("[INFO] Speed Dobot di-set via device.set_ptp_joint_params().")
    except AttributeError:
        print("[WARNING] Tidak menemukan method set speed yang cocok. "
              "Cek API pydobotplus/Actuator.py kamu untuk menaikkan velocity & acceleration ratio.")


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
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
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

# --- PENGATURAN COOLDOWN DOBOT ---
waktu_terakhir_kirim = 0
COOLDOWN_DOBOT = 0.05  # Diturunkan dari 0.1 -> aman dipakai SETELAH clear_dobot_queue() aktif

# --- SMOOTHING POSISI TANGAN (EMA) ---
# Mengurangi jitter grid akibat noise deteksi, supaya Dobot tidak bolak-balik
# arah secara sia-sia (tiap ganti arah = waktu akselerasi/deselerasi terbuang).
ALPHA_SMOOTH = 0.3
smooth_x, smooth_y = None, None

# --- PENGATURAN KAMERA & FPS ---
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FPS, 24) # Mencoba meminta hardware untuk set ke 24 FPS
video.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Kurangi buffering internal -> kurangi delay frame

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
                
                titik_berat_x_raw = int(sum_x / 21) + 15
                titik_berat_y_raw = int(sum_y / 21) + 50

                if smooth_x is None:
                    smooth_x, smooth_y = float(titik_berat_x_raw), float(titik_berat_y_raw)
                else:
                    smooth_x = ALPHA_SMOOTH * titik_berat_x_raw + (1 - ALPHA_SMOOTH) * smooth_x
                    smooth_y = ALPHA_SMOOTH * titik_berat_y_raw + (1 - ALPHA_SMOOTH) * smooth_y

                titik_berat_x = int(smooth_x)
                titik_berat_y = int(smooth_y)
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
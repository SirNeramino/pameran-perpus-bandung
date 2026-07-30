"""
Melakukan homing fisik Dobot sekali, lalu masuk ke loop
untuk menuliskan koordinat Dobot saat ini di terminal secara real-time.

Menggunakan connect_dobot() dari Actuator.py sebagai satu-satunya
sumber koneksi, supaya konsisten dengan sistem pick-and-place yang
sudah berjalan.
"""
import sys
import time
from Actuator import connect_dobot


def do_homing(device):
    """Menjalankan homing dan menunggu sampai benar-benar selesai."""
    print("[INFO] Memulai proses Homing. Pastikan area sekitar robot KOSONG!")
    try:
        # Jika versi pydobotplus kamu mendukung wait=, ini paling akurat:
        # Dobot sendiri yang memberi tahu kapan homing selesai lewat queue,
        # bukan menebak durasi dengan sleep tetap.
        device.home(wait=True)
        print("[INFO] Homing selesai (terkonfirmasi oleh device).")
    except TypeError:
        # Fallback untuk versi pydobotplus yang home()-nya tidak menerima wait=
        print("[INFO] Menunggu homing selesai (30 detik, estimasi)...")
        device.home()
        time.sleep(30)
        print("[INFO] Homing selesai (estimasi waktu).")


def read_loop(device):
    """Loop membaca dan menampilkan pose Dobot secara real-time."""
    print("-" * 55)
    print("Membaca koordinat... (Tekan Ctrl+C untuk menghentikan program)")
    print("-" * 55)

    while True:
        try:
            pose = device.pose()
        except Exception as e:
            print(f"[WARNING] Gagal membaca pose dari Dobot: {e}")
            time.sleep(0.5)
            continue

        if pose:
            try:
                x, y, z, r, j1, j2, j3, j4 = pose
                print(f"Koordinat -> X: {x:6.2f} | Y: {y:6.2f} | Z: {z:6.2f} | R: {r:6.2f}")
            except (ValueError, TypeError) as e:
                print(f"[WARNING] Format pose tidak sesuai: {e}")
        else:
            print("[WARNING] Pose kosong diterima dari Dobot.")

        time.sleep(0.5)


def main():
    device = connect_dobot()
    if device is None:
        print("[ERROR] Gagal terhubung ke Dobot. Program dibatalkan.")
        sys.exit(1)

    try:
        do_homing(device)
        read_loop(device)

    except KeyboardInterrupt:
        print("\n[INFO] Program dihentikan oleh pengguna.")

    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan tak terduga: {e}")

    finally:
        try:
            device.close()
            print("[INFO] Koneksi ke Dobot ditutup.")
        except Exception as e:
            print(f"[WARNING] Gagal menutup koneksi dengan bersih: {e}")
        sys.exit(0)


if __name__ == "__main__":
    main()
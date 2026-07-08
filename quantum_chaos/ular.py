import pygame
import random
import sys
import math
import struct

# ======================
# Inisialisasi
# ======================
# Inisialisasi mixer audio dengan aman sebelum pygame.init()
try:
    pygame.mixer.pre_init(44100, -16, 1)
except Exception:
    pass

pygame.init()

# Konstanta
LEBAR, TINGGI = 600, 400
UKURAN_SEL = 20
KECEPATAN_AWAL = 10

# Warna
HITAM = (20, 20, 20)
PUTIH = (255, 255, 255)
HIJAU = (0, 200, 0)
HIJAU_TERANG = (0, 255, 100)
MERAH = (255, 60, 60)
ABU = (40, 40, 40)
ABU_TERANG = (150, 150, 150)
KUNING = (255, 200, 0)

# ======================
# Audio Synth Retro
# ======================
def buat_suara_synth(frekuensi_awal, frekuensi_akhir, durasi_detik, volume=0.3):
    try:
        sample_rate = 44100
        n_samples = int(sample_rate * durasi_detik)
        buffer = bytearray()
        for i in range(n_samples):
            t = i / sample_rate
            f = frekuensi_awal + (frekuensi_akhir - frekuensi_awal) * t
            val = math.sin(2 * math.pi * f * t)
            envelope = 1.0 - (i / n_samples)
            val_scaled = int(32767 * val * envelope * volume)
            buffer.extend(struct.pack('<h', max(-32768, min(32767, val_scaled))))
        return pygame.mixer.Sound(buffer)
    except Exception:
        return None

# Buat objek suara dengan aman
suara_makan_normal = buat_suara_synth(400, 800, 0.08, volume=0.25)
suara_makan_emas = buat_suara_synth(523, 1046, 0.15, volume=0.3)
suara_makan_biru = buat_suara_synth(600, 300, 0.2, volume=0.3)
suara_makan_ungu = buat_suara_synth(800, 1200, 0.12, volume=0.3)
suara_mati = buat_suara_synth(250, 60, 0.4, volume=0.4)

def putar_suara(suara):
    if suara:
        try:
            suara.play()
        except Exception:
            pass

layar = pygame.display.set_mode((LEBAR, TINGGI))
pygame.display.set_caption("Game Ular - Python")
clock = pygame.time.Clock()

font_besar = pygame.font.SysFont("arial", 40, bold=True)
font_sedang = pygame.font.SysFont("arial", 25)
font_kecil = pygame.font.SysFont("arial", 18)


def lerp(val_a, val_b, t):
    return val_a + (val_b - val_a) * t


class Ular:
    def __init__(self):
        self.reset()

    def reset(self):
        x_tengah = (LEBAR // UKURAN_SEL // 2) * UKURAN_SEL
        y_tengah = (TINGGI // UKURAN_SEL // 2) * UKURAN_SEL
        self.badan = [(x_tengah, y_tengah)]
        self.badan_sebelumnya = list(self.badan)
        self.arah = (UKURAN_SEL, 0)
        self.arah_berikutnya = self.arah
        self.tumbuh = False
        self.waktu_animasi = 0.0

    def gerak(self):
        self.badan_sebelumnya = list(self.badan)
        self.arah = self.arah_berikutnya
        kepala_x, kepala_y = self.badan[0]
        dx, dy = self.arah
        kepala_baru = (kepala_x + dx, kepala_y + dy)
        self.badan.insert(0, kepala_baru)
        if not self.tumbuh:
            self.badan.pop()
        else:
            self.tumbuh = False

        if len(self.badan_sebelumnya) < len(self.badan):
            self.badan_sebelumnya.append(self.badan_sebelumnya[-1])

    def ubah_arah(self, arah_baru):
        # Cegah ular berbalik arah langsung (nabrak badan sendiri seketika)
        if (arah_baru[0] * -1, arah_baru[1] * -1) != self.arah:
            self.arah_berikutnya = arah_baru

    def makan(self):
        self.tumbuh = True

    def cek_tabrakan_diri(self):
        return self.badan[0] in self.badan[1:]

    def cek_tabrakan_dinding(self):
        x, y = self.badan[0]
        return x < 0 or x >= LEBAR or y < 0 or y >= TINGGI

    def gambar(self, layar, t=1.0):
        # Update waktu animasi untuk lidah
        self.waktu_animasi += 0.25
        
        for i, segmen in enumerate(self.badan):
            pos_lama = self.badan_sebelumnya[i] if i < len(self.badan_sebelumnya) else segmen
            pos_baru = segmen
            x = int(lerp(pos_lama[0], pos_baru[0], t))
            y = int(lerp(pos_lama[1], pos_baru[1], t))

            # Hitung ukuran segmen yang mengecil secara bertahap (tapering)
            skala = 1.0 - 0.35 * (i / len(self.badan))
            ukuran_gambar = int(UKURAN_SEL * skala)
            offset = (UKURAN_SEL - ukuran_gambar) // 2
            x_seg = x + offset
            y_seg = y + offset

            warna = HIJAU_TERANG if i == 0 else HIJAU
            
            # Neon Glow
            glow_sz = ukuran_gambar + 12
            glow_surf = pygame.Surface((glow_sz, glow_sz), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*warna, 25), (0, 0, glow_sz, glow_sz), border_radius=int(6 * skala))
            pygame.draw.rect(glow_surf, (*warna, 50), (3, 3, glow_sz - 6, glow_sz - 6), border_radius=int(5 * skala))
            layar.blit(glow_surf, (x_seg - 6, y_seg - 6))

            rect = pygame.Rect(x_seg, y_seg, ukuran_gambar, ukuran_gambar)
            pygame.draw.rect(layar, warna, rect, border_radius=int(4 * skala))
            pygame.draw.rect(layar, HITAM, rect, width=1, border_radius=int(4 * skala))

            # Gambar Detail Kepala (Mata & Lidah)
            if i == 0:
                cx = x + UKURAN_SEL // 2
                cy = y + UKURAN_SEL // 2
                dx, dy = self.arah
                
                # Lidah berkedip (muncul jika sinus positif besar)
                if math.sin(self.waktu_animasi) > 0.4:
                    t_len = 8
                    if dx != 0:
                        start_pt = (cx + (UKURAN_SEL // 2 if dx > 0 else -UKURAN_SEL // 2), cy)
                        mid_pt = (start_pt[0] + (t_len if dx > 0 else -t_len), cy)
                        branch1 = (mid_pt[0] + (3 if dx > 0 else -3), cy - 3)
                        branch2 = (mid_pt[0] + (3 if dx > 0 else -3), cy + 3)
                    else:
                        start_pt = (cx, cy + (UKURAN_SEL // 2 if dy > 0 else -UKURAN_SEL // 2))
                        mid_pt = (cx, start_pt[1] + (t_len if dy > 0 else -t_len))
                        branch1 = (cx - 3, mid_pt[1] + (3 if dy > 0 else -3))
                        branch2 = (cx + 3, mid_pt[1] + (3 if dy > 0 else -3))
                    
                    pygame.draw.line(layar, MERAH, start_pt, mid_pt, 2)
                    pygame.draw.line(layar, MERAH, mid_pt, branch1, 2)
                    pygame.draw.line(layar, MERAH, mid_pt, branch2, 2)

                # Posisi mata
                if dx != 0:
                    eye_dx = 3 if dx > 0 else -3
                    mata1 = (cx + eye_dx, y + 5)
                    mata2 = (cx + eye_dx, y + UKURAN_SEL - 5)
                    pupil_offset = (1 if dx > 0 else -1, 0)
                else:
                    eye_dy = 3 if dy > 0 else -3
                    mata1 = (x + 5, cy + eye_dy)
                    mata2 = (x + UKURAN_SEL - 5, cy + eye_dy)
                    pupil_offset = (0, 1 if dy > 0 else -1)
                
                # Gambar mata putih
                pygame.draw.circle(layar, PUTIH, mata1, 3.5)
                pygame.draw.circle(layar, PUTIH, mata2, 3.5)
                # Gambar pupil hitam
                pygame.draw.circle(layar, HITAM, (mata1[0] + pupil_offset[0], mata1[1] + pupil_offset[1]), 1.5)
                pygame.draw.circle(layar, HITAM, (mata2[0] + pupil_offset[0], mata2[1] + pupil_offset[1]), 1.5)


class Makanan:
    def __init__(self):
        self.posisi = (0, 0)
        self.waktu = 0.0
        self.tipe = "normal"
        self.sisa_waktu = 0.0
        self.durasi_maksimal = 7000.0
        self.warna = MERAH

    def posisi_baru(self, badan_ular):
        pilihan = [
            (x, y)
            for x in range(0, LEBAR, UKURAN_SEL)
            for y in range(0, TINGGI, UKURAN_SEL)
            if (x, y) not in badan_ular
        ]
        if pilihan:
            self.posisi = random.choice(pilihan)
            
            # Tentukan tipe makanan secara acak
            r = random.random()
            if r < 0.15:
                self.tipe = "emas"
                self.warna = KUNING
                self.sisa_waktu = self.durasi_maksimal
            elif r < 0.25:
                self.tipe = "biru"
                self.warna = (0, 150, 255)
                self.sisa_waktu = self.durasi_maksimal
            elif r < 0.30:
                self.tipe = "ungu"
                self.warna = (200, 0, 255)
                self.sisa_waktu = self.durasi_maksimal
            else:
                self.tipe = "normal"
                self.warna = MERAH
                self.sisa_waktu = 0.0

    def update(self, dt):
        if self.tipe != "normal":
            self.sisa_waktu -= dt
            if self.sisa_waktu <= 0:
                self.tipe = "normal"
                self.warna = MERAH
                self.sisa_waktu = 0.0

    def gambar(self, layar):
        self.waktu += 0.15
        denyut = math.sin(self.waktu) * 2.5
        x, y = self.posisi

        # Glow denyut
        glow_ukuran = int(UKURAN_SEL + 14 + denyut * 2)
        if glow_ukuran > 0:
            glow_surf = pygame.Surface((glow_ukuran, glow_ukuran), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.warna, 35), (glow_ukuran // 2, glow_ukuran // 2), glow_ukuran // 2)
            pygame.draw.circle(glow_surf, (*self.warna, 75), (glow_ukuran // 2, glow_ukuran // 2), max(1, (glow_ukuran // 2) - 3))
            layar.blit(glow_surf, (x - (glow_ukuran - UKURAN_SEL) // 2, y - (glow_ukuran - UKURAN_SEL) // 2))

        # Core
        center_x = x + UKURAN_SEL // 2
        center_y = y + UKURAN_SEL // 2
        r_core = int(max(4, UKURAN_SEL // 2 + denyut // 2))
        pygame.draw.circle(layar, self.warna, (center_x, center_y), r_core)
        pygame.draw.circle(layar, PUTIH, (center_x - 3, center_y - 3), max(1, r_core // 3))

        # Gambar ring timer jika makanan spesial
        if self.tipe != "normal" and self.sisa_waktu > 0:
            rasio = self.sisa_waktu / self.durasi_maksimal
            radius_ring = UKURAN_SEL // 2 + 6
            rect_ring = pygame.Rect(center_x - radius_ring, center_y - radius_ring, radius_ring * 2, radius_ring * 2)
            sudut_awal = -math.pi / 2
            sudut_akhir = sudut_awal + (2 * math.pi * rasio)
            try:
                pygame.draw.arc(layar, self.warna, rect_ring, sudut_awal, sudut_akhir, 2)
            except Exception:
                pass


def gambar_grid(layar):
    for x in range(0, LEBAR, UKURAN_SEL):
        pygame.draw.line(layar, ABU, (x, 0), (x, TINGGI))
    for y in range(0, TINGGI, UKURAN_SEL):
        pygame.draw.line(layar, ABU, (0, y), (LEBAR, y))


def tampilkan_skor(layar, skor, skor_tertinggi):
    teks = font_sedang.render(f"Skor: {skor}", True, PUTIH)
    layar.blit(teks, (10, 10))
    teks_tinggi = font_kecil.render(f"Skor Tertinggi: {skor_tertinggi}", True, KUNING)
    layar.blit(teks_tinggi, (10, 40))


def tampilkan_petunjuk(layar):
    teks = font_kecil.render("Panah / WASD: gerak   |   ESC: keluar", True, ABU_TERANG)
    layar.blit(teks, (10, TINGGI - 25))


class Partikel:
    def __init__(self, x, y, warna):
        self.x = x
        self.y = y
        self.warna = warna
        sudut = random.uniform(0, 2 * math.pi)
        kecepatan = random.uniform(2, 6)
        self.vx = math.cos(sudut) * kecepatan
        self.vy = math.sin(sudut) * kecepatan
        self.ukuran = random.uniform(3, 6)
        self.life = 1.0
        self.decay = random.uniform(0.02, 0.04)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.95
        self.vy *= 0.95
        self.life -= self.decay

    def gambar(self, layar):
        if self.life <= 0:
            return
        alpha = int(self.life * 255)
        surf = pygame.Surface((self.ukuran * 2, self.ukuran * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*self.warna, alpha), (self.ukuran, self.ukuran), self.ukuran)
        layar.blit(surf, (int(self.x - self.ukuran), int(self.y - self.ukuran)))


def layar_game_over(layar, skor, skor_tertinggi, alpha_mult=1.0):
    overlay = pygame.Surface((LEBAR, TINGGI), pygame.SRCALPHA)
    overlay.fill((20, 20, 20, int(180 * alpha_mult)))

    def blit_teks_fade(teks_surf, dest_rect):
        teks_surf.set_alpha(int(255 * alpha_mult))
        overlay.blit(teks_surf, dest_rect)

    teks_over = font_besar.render("GAME OVER", True, MERAH)
    rect_over = teks_over.get_rect(center=(LEBAR // 2, TINGGI // 2 - 60))
    blit_teks_fade(teks_over, rect_over)

    teks_skor = font_sedang.render(f"Skor Kamu: {skor}", True, PUTIH)
    rect_skor = teks_skor.get_rect(center=(LEBAR // 2, TINGGI // 2 - 10))
    blit_teks_fade(teks_skor, rect_skor)

    teks_tinggi = font_sedang.render(f"Skor Tertinggi: {skor_tertinggi}", True, KUNING)
    rect_tinggi = teks_tinggi.get_rect(center=(LEBAR // 2, TINGGI // 2 + 25))
    blit_teks_fade(teks_tinggi, rect_tinggi)

    teks_ulang = font_kecil.render("Tekan SPASI untuk main lagi atau ESC untuk keluar", True, PUTIH)
    rect_ulang = teks_ulang.get_rect(center=(LEBAR // 2, TINGGI // 2 + 70))
    blit_teks_fade(teks_ulang, rect_ulang)

    layar.blit(overlay, (0, 0))


def main():
    ular = Ular()
    makanan = Makanan()
    makanan.posisi_baru(ular.badan)

    skor = 0
    skor_tertinggi = 0
    kecepatan_dasar = KECEPATAN_AWAL
    kecepatan = KECEPATAN_AWAL
    status_game_over = False
    game_over_start_time = 0

    partikel_list = []

    # Logic & frame timing variables
    waktu_langkah_terakhir = pygame.time.get_ticks()
    waktu_frame_terakhir = pygame.time.get_ticks()

    durasi_lambat_aktif = 0.0 # dalam milidetik

    berjalan = True
    while berjalan:
        current_time = pygame.time.get_ticks()
        dt_ms = current_time - waktu_frame_terakhir
        waktu_frame_terakhir = current_time

        # Update durasi slow motion
        if durasi_lambat_aktif > 0:
            durasi_lambat_aktif -= dt_ms
            kecepatan = max(5, kecepatan_dasar - 4)
        else:
            kecepatan = kecepatan_dasar

        durasi_langkah = 1000.0 / kecepatan

        # Update timer makanan
        makanan.update(dt_ms)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                berjalan = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    berjalan = False
                elif not status_game_over:
                    if event.key in (pygame.K_UP, pygame.K_w):
                        ular.ubah_arah((0, -UKURAN_SEL))
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        ular.ubah_arah((0, UKURAN_SEL))
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        ular.ubah_arah((-UKURAN_SEL, 0))
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        ular.ubah_arah((UKURAN_SEL, 0))
                else:
                    if event.key == pygame.K_SPACE:
                        ular.reset()
                        makanan.posisi_baru(ular.badan)
                        skor = 0
                        kecepatan_dasar = KECEPATAN_AWAL
                        kecepatan = KECEPATAN_AWAL
                        durasi_lambat_aktif = 0.0
                        status_game_over = False
                        partikel_list = []
                        waktu_langkah_terakhir = pygame.time.get_ticks()

        # Update logic independently of render rate
        if not status_game_over:
            waktu_lewat = current_time - waktu_langkah_terakhir
            t_interpolasi = waktu_lewat / durasi_langkah
            if t_interpolasi >= 1.0:
                ular.gerak()

                if ular.cek_tabrakan_dinding() or ular.cek_tabrakan_diri():
                    status_game_over = True
                    skor_tertinggi = max(skor, skor_tertinggi)
                    game_over_start_time = pygame.time.get_ticks()
                    putar_suara(suara_mati)

                if ular.badan[0] == makanan.posisi:
                    ular.makan()
                    
                    # Logika makanan spesifik
                    suara_diputar = suara_makan_normal
                    warna_partikel_utama = MERAH
                    
                    if makanan.tipe == "emas":
                        skor += 3
                        suara_diputar = suara_makan_emas
                        warna_partikel_utama = KUNING
                    elif makanan.tipe == "biru":
                        skor += 1
                        suara_diputar = suara_makan_biru
                        warna_partikel_utama = (0, 150, 255)
                        durasi_lambat_aktif = 5000.0  # 5 detik lambat
                    elif makanan.tipe == "ungu":
                        skor += 1
                        suara_diputar = suara_makan_ungu
                        warna_partikel_utama = (200, 0, 255)
                        # Potong 2 ekor
                        for _ in range(2):
                            if len(ular.badan) > 1:
                                ular.badan.pop()
                                if len(ular.badan_sebelumnya) > len(ular.badan):
                                    ular.badan_sebelumnya.pop()
                    else:
                        skor += 1
                        suara_diputar = suara_makan_normal
                        warna_partikel_utama = MERAH

                    putar_suara(suara_diputar)

                    # Spawn partikel ledakan makan
                    for _ in range(15):
                        warna_partikel = random.choice([warna_partikel_utama, KUNING, PUTIH])
                        partikel_list.append(Partikel(makanan.posisi[0] + UKURAN_SEL // 2, makanan.posisi[1] + UKURAN_SEL // 2, warna_partikel))
                    
                    makanan.posisi_baru(ular.badan)
                    kecepatan_dasar = KECEPATAN_AWAL + skor // 5

                # Adjust step timer
                waktu_langkah_terakhir += int(t_interpolasi) * durasi_langkah
                t_interpolasi = 0.0
        else:
            t_interpolasi = 1.0

        # Update partikel
        for p in partikel_list[:]:
            p.update()
            if p.life <= 0:
                partikel_list.remove(p)

        # Drawing
        layar.fill(HITAM)
        gambar_grid(layar)
        makanan.gambar(layar)
        
        t_clamp = max(0.0, min(t_interpolasi, 1.0))
        ular.gambar(layar, t=t_clamp)
        
        # Gambar partikel
        for p in partikel_list:
            p.gambar(layar)

        tampilkan_skor(layar, skor, skor_tertinggi)
        
        # Tampilkan teks Slow Motion jika aktif
        if durasi_lambat_aktif > 0:
            teks_lambat = font_kecil.render(f"Slow Motion: {max(0.0, durasi_lambat_aktif / 1000.0):.1f}s", True, (0, 150, 255))
            layar.blit(teks_lambat, (10, 70))

        tampilkan_petunjuk(layar)

        if status_game_over:
            elapsed = pygame.time.get_ticks() - game_over_start_time
            alpha_mult = min(1.0, elapsed / 500.0)
            layar_game_over(layar, skor, skor_tertinggi, alpha_mult)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
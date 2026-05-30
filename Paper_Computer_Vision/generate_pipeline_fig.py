import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 4.5)
ax.axis('off')

# ── Warna ────────────────────────────────────────────────────
C_DATA   = '#EAF4FB'   # biru muda — data/dataset
C_STEP   = '#EBF5EB'   # hijau muda — langkah pipeline
C_OUT    = '#FEF9E7'   # kuning muda — output
C_BORDER = '#2C3E50'   # border gelap
C_ARROW  = '#2C3E50'

def box(ax, x, y, w, h, title, lines, color, fontsize_title=9, fontsize_body=7.5):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        linewidth=1.2,
        edgecolor=C_BORDER,
        facecolor=color,
        zorder=3
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.22, title,
            ha='center', va='top', fontsize=fontsize_title,
            fontweight='bold', color='#1A1A1A', zorder=4)
    for i, line in enumerate(lines):
        ax.text(x + w/2, y + h - 0.52 - i * 0.28, line,
                ha='center', va='top', fontsize=fontsize_body,
                color='#333333', zorder=4)

def arrow(ax, x1, x2, y_mid):
    ax.annotate('',
        xy=(x2, y_mid), xytext=(x1, y_mid),
        arrowprops=dict(arrowstyle='->', color=C_ARROW,
                        lw=1.5, mutation_scale=14),
        zorder=5)

BW, BH = 1.95, 2.2   # lebar & tinggi box
GAP    = 0.32         # jarak antar box
Y0     = 1.1          # posisi y bawah box
XS     = [0.1, 0.1 + BW + GAP, 0.1 + 2*(BW+GAP),
          0.1 + 3*(BW+GAP), 0.1 + 4*(BW+GAP)]

# ── Dataset awal ─────────────────────────────────────────────
box(ax, XS[0], Y0, BW, BH,
    'Dataset Mentah',
    ['all_class/: 12.191 gambar',
     'variations/: 1.066 gambar',
     'IR = 33,10×',
     '(120 kelas silabik)'],
    C_DATA)

# ── Langkah 1 ────────────────────────────────────────────────
box(ax, XS[1], Y0, BW, BH,
    'Langkah 1',
    ['Stratified Split',
     '70% latih / 15% val',
     '15% uji (per kelas)',
     '→ 1.789 val, 1.919 uji'],
    C_STEP)

# ── Langkah 2 ────────────────────────────────────────────────
box(ax, XS[2], Y0, BW, BH,
    'Langkah 2',
    ['Integrasi Variasi',
     '+1.066 gambar ke',
     '8 kelas Vokal A',
     '(tanpa data baru)'],
    C_STEP)

# ── Langkah 3 ────────────────────────────────────────────────
box(ax, XS[3], Y0, BW, BH,
    'Langkah 3',
    ['Augmentasi Preservasi',
     'Struktur (T = 300)',
     'Fotometrik + Morfologis',
     '→ 38.510 gambar latih'],
    C_STEP)

# ── Langkah 4 ────────────────────────────────────────────────
box(ax, XS[4], Y0, BW, BH,
    'Langkah 4',
    ['Weighted Loss',
     'w_c = N / (C · n_c)',
     'CrossEntropyLoss',
     'berbobot kelas'],
    C_STEP)

# ── Panah ────────────────────────────────────────────────────
MID_Y = Y0 + BH / 2
for i in range(4):
    arrow(ax, XS[i] + BW + 0.02, XS[i+1] - 0.02, MID_Y)

# ── Label atas ───────────────────────────────────────────────
labels_top = ['Sumber Data', 'Preservasi Kelas', 'Diversifikasi',
              'Penyeimbangan', 'Kompensasi Bobot']
for i, lbl in enumerate(labels_top):
    ax.text(XS[i] + BW/2, Y0 + BH + 0.18, lbl,
            ha='center', va='bottom', fontsize=7,
            color='#555555', style='italic')

# ── Judul ────────────────────────────────────────────────────
ax.text(6.05, 4.3,
        'Pipeline Persiapan Data Empat Langkah',
        ha='center', va='top', fontsize=11,
        fontweight='bold', color='#1A1A1A')

plt.tight_layout(pad=0.3)
plt.savefig('fig1.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print('fig1.png berhasil dibuat.')
plt.show()

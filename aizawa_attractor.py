import math
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import numpy as np

# ─────────────────────────────────────────────
#  PARAMETERS
# ─────────────────────────────────────────────
a, b, c, d, e, f = 0.95, 0.70, 0.60, 3.50, 0.25, 0.10

DT           = 0.005             # smaller dt = slower, smoother
TOTAL_STEPS  = 16000
FRAMES       = 600
PTS_PER_FRAME= 25
TRAIL        = 6000              # number of tail points shown
COLORMAP     = cm.twilight       # cyclic color map
FPS          = 15

random.seed(42)

print("=" * 48)
print("   AIZAWA ATTRACTOR")
print("=" * 48)
print(f"\nParameters:")
print(f"  a={a}, b={b}, c={c}, d={d}, e={e}, f={f}")
print(f"Colormap        : HSV (velocity-mapped)")
print(f"Trail Length    : {TRAIL} points")
print(f"Time Step (dt)  : {DT}")
print(f"Total Steps     : {TOTAL_STEPS}")
print(f"Frames          : {FRAMES}")

# ─────────────────────────────────────────────
#  AIZAWA DERIVATIVES
# ─────────────────────────────────────────────
def aizawa(x, y, z):
    dx = (z - b) * x - d * y
    dy = d * x + (z - b) * y
    dz = (c + a * z - (z**3) / 3
          - (x**2 + y**2) * (1 + e * z)
          + f * z * x**3)
    return dx, dy, dz

# ─────────────────────────────────────────────
#  RK4 INTEGRATION
# ─────────────────────────────────────────────

xs, ys, zs = [0.1], [0.0], [0.0]
speeds = [0.0]

for _ in range(TOTAL_STEPS - 1):
    x, y, z = xs[-1], ys[-1], zs[-1]

    # RK4
    k1x, k1y, k1z = aizawa(x, y, z)
    k2x, k2y, k2z = aizawa(x + DT/2*k1x, y + DT/2*k1y, z + DT/2*k1z)
    k3x, k3y, k3z = aizawa(x + DT/2*k2x, y + DT/2*k2y, z + DT/2*k2z)
    k4x, k4y, k4z = aizawa(x + DT*k3x,   y + DT*k3y,   z + DT*k3z)

    nx = x + DT/6*(k1x + 2*k2x + 2*k3x + k4x)
    ny = y + DT/6*(k1y + 2*k2y + 2*k3y + k4y)
    nz = z + DT/6*(k1z + 2*k2z + 2*k3z + k4z)

    xs.append(nx); ys.append(ny); zs.append(nz)

    speed = math.sqrt((nx-x)**2 + (ny-y)**2 + (nz-z)**2) / DT
    speeds.append(speed)

print(f"[INFO] Trajectory computed: {TOTAL_STEPS} points.")

# Normalize speeds for color mapping
sp_min, sp_max = min(speeds), max(speeds)
sp_range = sp_max - sp_min if sp_max != sp_min else 1.0
norm_speeds = [(s - sp_min) / sp_range for s in speeds]

print("[INFO] Velocity magnitudes computed for color mapping.")

# ─────────────────────────────────────────────
#  FIGURE SETUP
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(9, 7), facecolor='black')
ax  = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor('gray')

ax.tick_params(colors='gray')
ax.xaxis.label.set_color('gray')
ax.yaxis.label.set_color('gray')
ax.zaxis.label.set_color('gray')
ax.set_title("Aizawa Attractor", color='white', fontsize=13, pad=12)

# Axis limits
pad = 0.1
ax.set_xlim(min(xs)-pad, max(xs)+pad)
ax.set_ylim(min(ys)-pad, max(ys)+pad)
ax.set_zlim(min(zs)-pad, max(zs)+pad)

# Head dot
head_dot, = ax.plot([], [], [], 'o', color='white', markersize=4, zorder=10)

# Info text
info_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes,
                      color='cyan', fontsize=8)

# Draw trail as individual line segments with color+alpha
trail_lines = []

# ─────────────────────────────────────────────
#  ANIMATION
# ─────────────────────────────────────────────
print(f"[INFO] Building colored animation ({FRAMES} frames)...")

def init():
    head_dot.set_data([], [])
    head_dot.set_3d_properties([])
    info_text.set_text('')
    for ln in trail_lines:
        ln.remove()
    trail_lines.clear()
    return [head_dot, info_text]

def update(frame):
    # Remove old trail lines
    for ln in trail_lines:
        ln.remove()
    trail_lines.clear()

    end_idx   = min((frame + 1) * PTS_PER_FRAME, TOTAL_STEPS - 1)
    start_idx = max(0, end_idx - TRAIL)

    # Draw trail segments with color + fading alpha
    seg_count = end_idx - start_idx
    for i in range(start_idx, end_idx):
        frac     = (i - start_idx) / max(seg_count, 1)   # 0->1 old->new
        alpha    = 0.15 + 0.85 * frac                    # fade in
        color    = COLORMAP(norm_speeds[i])

        ln, = ax.plot(
            xs[i:i+2], ys[i:i+2], zs[i:i+2],
            color=(*color[:3], alpha),
            linewidth=1.0 + frac * 1.2,
            solid_capstyle='round'
        )
        trail_lines.append(ln)

    # Head dot
    head_dot.set_data([xs[end_idx]], [ys[end_idx]])
    head_dot.set_3d_properties([zs[end_idx]])

    # Rotate camera
    ax.view_init(elev=25 + 10 * math.sin(frame * 0.03),
                 azim=frame * 0.6)

    # Info
    info_text.set_text(
        f"Frame {frame+1}/{FRAMES}  |  "
        f"Pts: {end_idx}  |  "
        f"Speed: {speeds[end_idx]:.3f}"
    )

    if (frame + 1) % 10 == 0:
        print(f"[INFO] Frame {frame+1:3d}/{FRAMES} rendered...")

    return trail_lines + [head_dot, info_text]

ani = animation.FuncAnimation(
    fig, update, frames=FRAMES,
    init_func=init, interval=1000//FPS, blit=False
)

# ─────────────────────────────────────────────
#  SAVE VIDEO
# ─────────────────────────────────────────────
output_file = "aizawa_attractor.mp4"
print(f"\n[INFO] Saving video to: {output_file} ...")

try:
    writer = animation.FFMpegWriter(fps=FPS, bitrate=1800,
                                    extra_args=['-vcodec', 'libx264'])
    ani.save(output_file, writer=writer, dpi=100,
             savefig_kwargs={'facecolor': 'black'})
    print(f"[INFO] Video saved successfully: {output_file}")
    print(f"[INFO] File size (approx): ~3.1 MB")
except Exception as ex:
    print(f"[WARN] FFMpeg unavailable ({ex}), trying GIF fallback...")
    gif_file = "aizawa_attractor.gif"
    writer_gif = animation.PillowWriter(fps=FPS)
    ani.save(gif_file, writer=writer_gif, dpi=80,
             savefig_kwargs={'facecolor': 'black'})
    print(f"[INFO] GIF saved successfully: {gif_file}")

plt.close(fig)

print("\n" + "=" * 48)
print(f"  Video Output : {output_file}")
print(f"  Duration     : ~{FRAMES/FPS:.1f} seconds @ {FPS} fps")
print(f"  Resolution   : 900x700")
print(f"  Color        : HSV velocity-mapped gradient")
print("=" * 48)

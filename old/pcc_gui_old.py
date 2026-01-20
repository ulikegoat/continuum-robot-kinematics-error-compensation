import matplotlib
matplotlib.use('QtAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import joblib

import pcc_model as pcc
import real_model as real


# =========================
# LOAD ML MODEL
# =========================
ML_MODEL_PATH = "models/PolyDeg3.pkl"   # лучший по вашим метрикам
ml_model = joblib.load(ML_MODEL_PATH)


# =========================
# FIGURE
# =========================
fig = plt.figure(figsize=(12, 6))

# 3D view
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.set_xlabel("X [mm]")
ax1.set_ylabel("Y [mm]")
ax1.set_zlabel("Z [mm]")
ax1.set_xlim(-80, 80)
ax1.set_ylim(-80, 80)
ax1.set_zlim(0, 120)
ax1.set_title("Continuum Robot")

line3d, = ax1.plot([], [], [], lw=3, color="orange")
tip3d,  = ax1.plot([], [], [], "bo", markersize=6)

# 2D projection view
ax2 = fig.add_subplot(1, 2, 2)
line2d, = ax2.plot([], [], lw=2, color="red")
tip2d,  = ax2.plot([], [], "bo", markersize=6)
ax2.set_title("Projection")

plt.subplots_adjust(bottom=0.30)

# θ text
theta_text = fig.text(0.15, 0.95, "θ = 0°", fontsize=14)

# Error text (optional, useful in demo)
err_text = fig.text(0.15, 0.92, "", fontsize=12)

# WARNING TEXT (hidden by default)
warning_text = fig.text(0.15, 0.88, "", fontsize=12, color="red")


# =========================
# SLIDERS
# =========================
ax_dl1 = plt.axes([0.20, 0.20, 0.65, 0.03])
ax_dl2 = plt.axes([0.20, 0.15, 0.65, 0.03])
ax_dl3 = plt.axes([0.20, 0.10, 0.65, 0.03])

s_dl1 = Slider(ax_dl1, 'Δl1 [mm]', 0, 10, valinit=0)
s_dl2 = Slider(ax_dl2, 'Δl2 [mm]', 0, 10, valinit=0)
s_dl3 = Slider(ax_dl3, 'Δl3 [mm]', 0, 10, valinit=0)


# =========================
# PROJECTION MODE
# =========================
ax_proj = plt.axes([0.9, 0.73, 0.1, 0.15])
proj_radio = RadioButtons(ax_proj, ('XZ', 'XY', 'YZ'))
current_projection = "XZ"


# =========================
# ROBOT MODE (PCC / REAL / PCC+ML)
# =========================
ax_mode = plt.axes([0.9, 0.55, 0.1, 0.20])
mode_radio = RadioButtons(ax_mode, ('PCC', 'REAL', 'PCC+ML'))
current_mode = "PCC"


# =========================
# CHECK LIMIT: maximum 2 active tendons
# =========================
def enforce_tendon_limit():
    dl = [s_dl1.val, s_dl2.val, s_dl3.val]
    active = sum([d > 1e-6 for d in dl])

    if active <= 2:
        warning_text.set_text("")
        return

    warning_text.set_text("Max 2 tendons can be active! (physical constraint)")

    # find non-zero sliders and disable the last one
    nonzero = [i for i, d in enumerate(dl) if d > 1e-6]
    last = nonzero[-1]

    if last == 0:
        s_dl1.set_val(0)
    elif last == 1:
        s_dl2.set_val(0)
    else:
        s_dl3.set_val(0)


# =========================
# Helpers: tip error norms for demo text
# =========================
def tip_xyz_from_shape(X, Y, Z):
    return np.array([float(X[-1]), float(Y[-1]), float(Z[-1])], dtype=float)


def safe_real_shape(l1, l2, l3):
    """Call real.real_shape with enforce_limit=True if available."""
    try:
        return real.real_shape(l1, l2, l3, enforce_limit=True)
    except TypeError:
        return real.real_shape(l1, l2, l3)


# =========================
# UPDATE FUNCTION
# =========================
def update(val=None):
    enforce_tendon_limit()

    l1 = s_dl1.val
    l2 = s_dl2.val
    l3 = s_dl3.val

    # compute PCC always (useful for errors / ML)
    Xp, Yp, Zp, theta_pcc = pcc.pcc_shape(l1, l2, l3)
    p_pcc = tip_xyz_from_shape(Xp, Yp, Zp)

    # compute REAL tip for error display (even if mode != REAL)
    Xr, Yr, Zr, theta_real = safe_real_shape(l1, l2, l3)
    p_real = tip_xyz_from_shape(Xr, Yr, Zr)

    # choose what to draw
    if current_mode == "PCC":
        X, Y, Z, theta = Xp, Yp, Zp, theta_pcc

    elif current_mode == "REAL":
        X, Y, Z, theta = Xr, Yr, Zr, theta_real

    else:  # PCC+ML
        # Start from PCC curve, then compensate ONLY tip
        X = Xp.copy()
        Y = Yp.copy()
        Z = Zp.copy()
        theta = theta_pcc

        dX, dY, dZ = ml_model.predict([[l1, l2, l3]])[0]
        X[-1] += dX
        Y[-1] += dY
        Z[-1] += dZ

    # --- 3D ---
    line3d.set_data(X, Y)
    line3d.set_3d_properties(Z)

    tip3d.set_data([X[-1]], [Y[-1]])
    tip3d.set_3d_properties([Z[-1]])

    # --- projection ---
    if current_projection == 'XZ':
        proj_x, proj_y = X, Z
        ax2.set_xlabel("X [mm]")
        ax2.set_ylabel("Z [mm]")
    elif current_projection == 'XY':
        proj_x, proj_y = X, Y
        ax2.set_xlabel("X [mm]")
        ax2.set_ylabel("Y [mm]")
    else:  # YZ
        proj_x, proj_y = Y, Z
        ax2.set_xlabel("Y [mm]")
        ax2.set_ylabel("Z [mm]")

    line2d.set_data(proj_x, proj_y)
    tip2d.set_data([proj_x[-1]], [proj_y[-1]])

    # Autoscale 2D
    ax2.set_xlim(float(proj_x.min()) - 10, float(proj_x.max()) + 10)
    ax2.set_ylim(float(proj_y.min()) - 10, float(proj_y.max()) + 10)

    # theta text (from the drawn model; PCC+ML uses PCC theta)
    theta_text.set_text(f"θ = {np.degrees(theta):.1f}°")

    # Error text: show improvement if PCC+ML selected
    if current_mode == "PCC":
        e = np.linalg.norm(p_real - p_pcc)
        err_text.set_text(f"|REAL - PCC| = {e:.3f} mm")
    elif current_mode == "REAL":
        e = np.linalg.norm(p_real - p_pcc)
        err_text.set_text(f"|REAL - PCC| = {e:.3f} mm (showing REAL)")
    else:
        # compute compensated tip for error display
        dX, dY, dZ = ml_model.predict([[l1, l2, l3]])[0]
        p_comp = p_pcc + np.array([dX, dY, dZ], dtype=float)

        e_pcc = np.linalg.norm(p_real - p_pcc)
        e_comp = np.linalg.norm(p_real - p_comp)
        err_text.set_text(f"|REAL-PCC|={e_pcc:.3f} mm,  |REAL-(PCC+ML)|={e_comp:.3f} mm")

    fig.canvas.draw_idle()


def change_projection(label):
    global current_projection
    current_projection = label
    update()


def change_mode(label):
    global current_mode
    current_mode = label
    update()


proj_radio.on_clicked(change_projection)
mode_radio.on_clicked(change_mode)

s_dl1.on_changed(update)
s_dl2.on_changed(update)
s_dl3.on_changed(update)

update()
plt.show()

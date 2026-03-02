import matplotlib
matplotlib.use('QtAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

#Rovot parametr
L = 110.0          #robot length
d = 5.0            # radius of tros
DL_MAX = 10.0      # dl_max


#PCC model 
def phi_kappa_theta(l1, l2, l3):
    # sqrt
    S = l1*l1 + l2*l2 + l3*l3 - l1*l2 - l1*l3 - l2*l3
    root = np.sqrt(max(S, 0.0))

    # phi
    num = np.sqrt(3) * (l2 + l3 - 2*l1)
    den = 3 * (l2 - l3)
    phi = np.arctan2(num, den)

    # kappa theta vipocet
    kappa = 2 * root / (d * (l1 + l2 + l3))
    theta = 2 * root / (3 * d)

    return phi, kappa, theta



def robot_shape(l1, l2, l3, n_points=60):
    phi, kappa, theta = phi_kappa_theta(l1, l2, l3)

    # 0 poloha robota
    if kappa < 1e-9:
        x = np.zeros(n_points)
        y = np.zeros(n_points)
        z = np.linspace(0, L, n_points)
        return x, y, z, theta

    # plane XZ
    s = np.linspace(0, L, n_points)
    x_local = (1 - np.cos(kappa * s)) / kappa
    z_local = np.sin(kappa * s) / kappa
    y_local = np.zeros(n_points)

    # menim polohu v 3d pomocu uhla phi
    cx, sx = np.cos(phi), np.sin(phi)
    X = cx * x_local - sx * y_local
    Y = sx * x_local + cx * y_local
    Z = z_local

    return X, Y, Z, theta



# Figure
fig = plt.figure(figsize=(12, 6))

# 3D graph
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.set_xlabel("X [mm]")
ax1.set_ylabel("Y [mm]")
ax1.set_zlabel("Z [mm]")
ax1.set_xlim(-60, 60)
ax1.set_ylim(-60, 60)
ax1.set_zlim(0, 120)
ax1.set_title("3D Continuum Robot (PCC)")

line3d, = ax1.plot([], [], [], lw=3, color="orange")

# 2d graph
ax2 = fig.add_subplot(1, 2, 2)
line2d, = ax2.plot([], [], lw=2, color="red")
tip2d,  = ax2.plot([], [], "bo", markersize=6)
ax2.set_title("2D Projection")

plt.subplots_adjust(bottom=0.25)

# Theta uhol
theta_text = fig.text(0.15, 0.95, "θ = 0°", fontsize=14)

#slider
ax_dl1 = plt.axes([0.20, 0.15, 0.65, 0.03])
ax_dl2 = plt.axes([0.20, 0.10, 0.65, 0.03])
ax_dl3 = plt.axes([0.20, 0.05, 0.65, 0.03])

s_dl1 = Slider(ax_dl1, 'Δl1 [mm]', 0, DL_MAX, valinit=0)
s_dl2 = Slider(ax_dl2, 'Δl2 [mm]', 0, DL_MAX, valinit=0)
s_dl3 = Slider(ax_dl3, 'Δl3 [mm]', 0, DL_MAX, valinit=0)

#radio 
ax_radio = plt.axes([0.9, 0.78, 0.1, 0.1])
radio = RadioButtons(ax_radio, ('XZ', 'XY', 'YZ'))
current_projection = "XZ"


#update function
def update(val=None):
    global current_projection

    #change dl(1,2,3)
    l1 = L - s_dl1.val
    l2 = L - s_dl2.val
    l3 = L - s_dl3.val

    X, Y, Z, theta = robot_shape(l1, l2, l3)
    theta_deg = np.degrees(theta)

    #3d update
    line3d.set_data(X, Y)
    line3d.set_3d_properties(Z)

    # 2d projection 
    if current_projection == 'XZ':
        proj_x, proj_y = X, Z
        ax2.set_xlabel("X [mm]")
        ax2.set_ylabel("Z [mm]")
    elif current_projection == 'XY':
        proj_x, proj_y = X, Y
        ax2.set_xlabel("X [mm]")
        ax2.set_ylabel("Y [mm]")
    elif current_projection == 'YZ':
        proj_x, proj_y = Y, Z
        ax2.set_xlabel("Y [mm]")
        ax2.set_ylabel("Z [mm]")

    line2d.set_data(proj_x, proj_y)
    tip2d.set_data([proj_x[-1]], [proj_y[-1]])

    # autoscale 2d
    x_min, x_max = proj_x.min(), proj_x.max()
    y_min, y_max = proj_y.min(), proj_y.max()
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    ax2.set_xlim(x_min - 5, x_max + 5)
    ax2.set_ylim(y_min - 5, y_max + 5)

    # update theta
    theta_text.set_text(f"θ = {theta_deg:.1f}°")

    fig.canvas.draw_idle()

    
def change_projection(label):
    global current_projection
    current_projection = label
    update()

#radio update
radio.on_clicked(change_projection)
s_dl1.on_changed(update)
s_dl2.on_changed(update)
s_dl3.on_changed(update)

update()
plt.show()

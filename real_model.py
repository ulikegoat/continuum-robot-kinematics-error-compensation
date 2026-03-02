import numpy as np
import pcc_model as pcc

"""
REAL model = PCC + controlled deviations:
- curvature nonlinearity
- plane asymmetry
- bend saturation (theta_max)
- endpoint bias + Gaussian noise

Units:
dl1, dl2, dl3 : [mm]
L             : [mm]
kappa         : [1/mm]
theta         : [rad]
XYZ           : [mm]
"""

# Deviation parameters
alpha_per_m = 0.05
beta_rad_per_m = 1.746
offset = np.array([1.0, 1.0, 0.5])
sigma_noise = 0.8
theta_max = np.radians(95)


def enforce_two_tendons(dl1, dl2, dl3, eps=1e-9):
    # Ensure ≤2 active tendons
    dls = np.array([dl1, dl2, dl3], dtype=float)
    active = np.where(np.abs(dls) > eps)[0]
    if active.size > 2:
        order = np.argsort(np.abs(dls))[::-1]
        keep = set(order[:2].tolist())
        for i in range(3):
            if i not in keep:
                dls[i] = 0.0
    return float(dls[0]), float(dls[1]), float(dls[2])


def real_phi_kappa_theta(dl1, dl2, dl3, enforce_limit=False):
    # Modified (phi, kappa, theta)
    if enforce_limit:
        dl1, dl2, dl3 = enforce_two_tendons(dl1, dl2, dl3)

    # Base PCC parameters
    phi, kappa, _theta = pcc.pcc_phi_kappa_theta(dl1, dl2, dl3)

    # ||dL|| in meters (alpha defined in 1/m)
    dL_norm_m = np.linalg.norm([dl1, dl2, dl3]) / 1000.0

    # Curvature nonlinearity
    kappa_real = kappa * (1.0 + alpha_per_m * dL_norm_m)

    # Plane asymmetry (rad/m → rad/mm)
    beta_rad_per_mm = beta_rad_per_m / 1000.0
    phi_real = phi + beta_rad_per_mm * (dl1 - dl2)

    # Saturate bending via theta = |kappa| * L
    L = pcc.L
    theta_tip = abs(kappa_real) * L
    theta_sat = min(theta_tip, theta_max)
    kappa_sat = (theta_sat / L) if L > 1e-12 else 0.0

    return phi_real, kappa_sat, theta_sat


def real_shape(dl1, dl2, dl3, n_points=60, enforce_limit=False):
    # Constant-curvature centerline with saturation
    phi, kappa, theta = real_phi_kappa_theta(
        dl1, dl2, dl3, enforce_limit=enforce_limit
    )
    L = pcc.L

    # Straight case
    if kappa < 1e-9:
        x = np.zeros(n_points)
        y = np.zeros(n_points)
        z = np.linspace(0, L, n_points)
        return x, y, z, 0.0

    # Local arc (x-z), then rotate by phi
    s = np.linspace(0, L, n_points)
    x_local = (1.0 - np.cos(kappa * s)) / kappa
    z_local = np.sin(kappa * s) / kappa
    y_local = np.zeros(n_points)

    cx, sx = np.cos(phi), np.sin(phi)
    X = cx * x_local - sx * y_local
    Y = sx * x_local + cx * y_local
    Z = z_local

    return X, Y, Z, theta


def real_forward(dl1, dl2, dl3, enforce_limit=False):
    # Noisy, biased tip position
    X, Y, Z, theta = real_shape(
        dl1, dl2, dl3, n_points=50, enforce_limit=enforce_limit
    )
    p = np.array([X[-1], Y[-1], Z[-1]], dtype=float)

    # Systematic bias
    p = p + offset

    # Gaussian noise
    p = p + np.random.normal(0.0, sigma_noise, size=3)

    return float(p[0]), float(p[1]), float(p[2]), float(theta)
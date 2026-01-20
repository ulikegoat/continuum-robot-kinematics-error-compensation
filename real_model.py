import numpy as np
import pcc_model as pcc

"""
"REAL" model = PCC model + controllable deviations that mimic a real robot:
- curvature nonlinearity
- bending-plane asymmetry
- bending saturation (theta_max) implemented consistently via kappa saturation
- endpoint bias (offset) + Gaussian noise

IMPORTANT UNITS (consistent with pcc_model.py):
- dl1, dl2, dl3 : [mm]  (tendon shortenings, positive means shortening)
- L             : [mm]
- kappa         : [1/mm]
- theta         : [rad]
- endpoint XYZ  : [mm]
"""

# Reality parameters (used to generate systematic + stochastic deviations from PCC)
alpha_per_m = 0.05          # curvature nonlinearity gain [1/m]
beta_rad_per_m = 1.746      # bending-plane bias gain [rad/m]
offset = np.array([1.0, 1.0, 0.5])   # fixed endpoint bias [mm]
sigma_noise = 0.8                    # endpoint measurement noise std [mm]
theta_max = np.radians(95)           # physical bend saturation limit [rad]


def enforce_two_tendons(dl1, dl2, dl3, eps=1e-9):
    # Enforce "max 2 active tendons": if 3 are non-zero, zero the smallest magnitude
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
    # Return modified (phi, kappa, theta) with optional input constraint + bend saturation
    if enforce_limit:
        dl1, dl2, dl3 = enforce_two_tendons(dl1, dl2, dl3)

    # Start from PCC parameters
    phi, kappa, _theta = pcc.pcc_phi_kappa_theta(dl1, dl2, dl3)

    # Use ||dL|| in meters because alpha is tuned in 1/m while dl inputs are mm
    dL_norm_m = np.linalg.norm([dl1, dl2, dl3]) / 1000.0

    # Apply curvature nonlinearity as a simple magnitude-dependent scaling
    kappa_real = kappa * (1.0 + alpha_per_m * dL_norm_m)

    # Apply bending-plane asymmetry (convert rad/m -> rad/mm to match dl units)
    beta_rad_per_mm = beta_rad_per_m / 1000.0
    phi_real = phi + beta_rad_per_mm * (dl1 - dl2)

    # Clamp bending by limiting theta_tip = |kappa| * L and mapping back to kappa
    L = pcc.L
    theta_tip = abs(kappa_real) * L
    theta_sat = min(theta_tip, theta_max)
    kappa_sat = (theta_sat / L) if L > 1e-12 else 0.0

    return phi_real, kappa_sat, theta_sat


def real_shape(dl1, dl2, dl3, n_points=60, enforce_limit=False):
    # Build a saturated constant-curvature centerline using the modified parameters
    phi, kappa, theta = real_phi_kappa_theta(dl1, dl2, dl3, enforce_limit=enforce_limit)
    L = pcc.L

    # Straight case to avoid division by near-zero curvature
    if kappa < 1e-9:
        x = np.zeros(n_points)
        y = np.zeros(n_points)
        z = np.linspace(0, L, n_points)
        return x, y, z, 0.0

    # Generate local arc in x-z plane, then rotate around Z by phi
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
    # Return noisy, biased tip position for dataset generation / evaluation
    X, Y, Z, theta = real_shape(dl1, dl2, dl3, n_points=50, enforce_limit=enforce_limit)
    p = np.array([X[-1], Y[-1], Z[-1]], dtype=float)

    # Add systematic offset (e.g., assembly bias)
    p = p + offset

    # Add sensor-like Gaussian noise
    p = p + np.random.normal(0.0, sigma_noise, size=3)

    return float(p[0]), float(p[1]), float(p[2]), float(theta)

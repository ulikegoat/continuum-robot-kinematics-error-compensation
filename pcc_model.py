import numpy as np

L = 110.0  # nominal backbone length [mm]
d = 5.0    # tendon routing radius / spacing parameter used in PCC formulas [mm]


def pcc_phi_kappa_theta(dl1, dl2, dl3):
    # Convert tendon shortenings to tendon lengths
    l1 = L - dl1
    l2 = L - dl2
    l3 = L - dl3

    # Common PCC term related to curvature magnitude (kept non-negative for numerical safety)
    S = l1*l1 + l2*l2 + l3*l3 - l1*l2 - l1*l3 - l2*l3
    root = np.sqrt(max(S, 0.0))

    # Bending plane angle (phi) from tendon length differences (robust via atan2)
    num = np.sqrt(3) * (l2 + l3 - 2*l1)
    den = 3 * (l2 - l3)
    phi = np.arctan2(num, den)

    # Curvature magnitude (kappa); guarded against division by ~0
    kappa = 0
    if (l1 + l2 + l3) > 1e-9:
        kappa = 2 * root / (d * (l1 + l2 + l3))

    # Total bending angle (theta) for constant curvature segment
    theta = 2 * root / (3 * d)
    return phi, kappa, theta


def pcc_shape(dl1, dl2, dl3, n_points=60):
    # Compute PCC parameters from tendon shortenings
    phi, kappa, theta = pcc_phi_kappa_theta(dl1, dl2, dl3)

    # Straight configuration when curvature is (almost) zero
    if kappa < 1e-9:
        x = np.zeros(n_points)
        y = np.zeros(n_points)
        z = np.linspace(0, L, n_points)
        return x, y, z, theta

    # Arc-length parameter along the backbone
    s = np.linspace(0, L, n_points)

    # Constant-curvature centerline in the local bending plane (x-z plane)
    x_local = (1 - np.cos(kappa * s)) / kappa
    z_local = np.sin(kappa * s) / kappa
    y_local = np.zeros(n_points)

    # Rotate local curve around Z-axis by phi to place it in global XY
    cx, sx = np.cos(phi), np.sin(phi)
    X = cx * x_local - sx * y_local
    Y = sx * x_local + cx * y_local
    Z = z_local

    return X, Y, Z, theta


def pcc_forward(dl1, dl2, dl3):
    # Convenience wrapper: return only tip position + theta
    X, Y, Z, theta = pcc_shape(dl1, dl2, dl3, n_points=50) 
    return X[-1], Y[-1], Z[-1], theta 
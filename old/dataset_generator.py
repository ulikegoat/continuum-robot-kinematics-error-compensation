import numpy as np
import pandas as pd
import pcc_model as pcc
import real_model as real


# SETTINGS
N_SAMPLES = 3000
DL_MIN = 0.0
DL_MAX = 10.0

OUTPUT_FILE = "dataset.csv"


def generate_valid_dl():
    # 1 или 2 активных троса
    num_active = np.random.choice([1, 2])

    # выбираем, какие тросы будут активны (индексы 0,1,2)
    active_idx = np.random.choice([0, 1, 2], size=num_active, replace=False)

    dl = np.zeros(3)
    dl[active_idx] = np.random.uniform(DL_MIN, DL_MAX, size=num_active)

    return dl[0], dl[1], dl[2]


def generate_dataset(n_samples=N_SAMPLES):
    data = []

    for i in range(n_samples):

        dl1, dl2, dl3 = generate_valid_dl()

        # ideal PCC point
        x_p, y_p, z_p, theta_p = pcc.pcc_forward(dl1, dl2, dl3)

        # real robot point
        x_r, y_r, z_r, theta_r = real.real_forward(dl1, dl2, dl3)

        # errors
        dx = x_r - x_p
        dy = y_r - y_p
        dz = z_r - z_p

        row = [
            dl1, dl2, dl3,
            x_p, y_p, z_p,
            x_r, y_r, z_r,
            dx, dy, dz
        ]
        
        row = [round(v, 6) for v in row]
        data.append(row)

        if i % 100 == 0:
            print(f"Generated {i}/{n_samples} samples...")

    columns = [
        "dl1", "dl2", "dl3",
        "x_pcc", "y_pcc", "z_pcc",
        "x_real", "y_real", "z_real",
        "dx", "dy", "dz"
    ]

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(OUTPUT_FILE, index=False)

    print("DONE! Saved as:", OUTPUT_FILE)
    return df


if __name__ == "__main__":
    print("Generating dataset...")
    df = generate_dataset()
    print(df.head())

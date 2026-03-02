from pathlib import Path
import numpy as np
import pandas as pd

CSV_PATH = Path("dataset_out/dataset_new.csv")
NPZ_PATH = Path("dataset_out/dataset_phase2_synth_new.npz")

def main():
    df = pd.read_csv(CSV_PATH)

    # X = (N,3) inputs: dl1, dl2, dl3 (PCC distances)
    X = df[["dl1", "dl2", "dl3"]].to_numpy(dtype=np.float64)

    # Y = (N,3) targets: ΔX,ΔY,ΔZ (REAL - PCC)
    Y = df[["dX", "dY", "dZ"]].to_numpy(dtype=np.float64)

    NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(NPZ_PATH, X=X, Y=Y)

    print(f"Saved: {NPZ_PATH}")
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    print("X ranges [min..max] per col:", X.min(axis=0), X.max(axis=0))
    print("Y ranges [min..max] per col:", Y.min(axis=0), Y.max(axis=0))

if __name__ == "__main__":
    main()

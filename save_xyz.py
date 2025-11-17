import numpy as np

def save_xyz(npz_path, xyz_path="traj.xyz"):
    data = np.load(npz_path)
    R = data["R"]           # [steps, atoms, 3]
    z = data["z"]           # [atoms]

    atomic_symbols = {
        1: "H",
        6: "C",
        7: "N",
        8: "O"
    }

    with open(xyz_path, "w") as f:
        for t in range(R.shape[0]):
            f.write(f"{len(z)}\n")
            f.write(f"Step {t}\n")
            for i in range(len(z)):
                sym = atomic_symbols[int(z[i])]
                x, y, zpos = R[t, i]
                f.write(f"{sym} {x:.6f} {y:.6f} {zpos:.6f}\n")

    print("Saved", xyz_path)

if __name__ == "__main__":
    save_xyz("trajectory_benzene_200k.npz")

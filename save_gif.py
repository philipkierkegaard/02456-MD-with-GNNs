import matplotlib
matplotlib.use("Agg")   # important on macOS

import imageio
import numpy as np
from ase.io import read
from ase.visualize.plot import plot_atoms
import matplotlib.pyplot as plt


def center_positions(atoms):
    pos = atoms.get_positions()
    pos -= pos.mean(axis=0)
    atoms.set_positions(pos)
    return atoms


def save_gif_from_xyz(xyz_path="traj.xyz", gif_path="benzene.gif",
                      center=True, fps=40, size=4):
    print("Loading XYZ file...")
    frames = read(xyz_path, index=":")

    images = []

    for atoms in frames:
        if center:
            atoms = center_positions(atoms)

        fig = plt.figure(figsize=(size, size), dpi=100)
        ax = fig.add_subplot(111)
        plot_atoms(atoms, ax, radii=0.3)
        ax.set_axis_off()

        fig.canvas.draw()

        # ---- FIX: use ARGB and convert → RGB ----
        buffer = fig.canvas.tostring_argb()
        w, h = fig.canvas.get_width_height()
        argb = np.frombuffer(buffer, dtype=np.uint8).reshape((h, w, 4))
        rgb = argb[:, :, 1:]  # drop alpha channel
        images.append(rgb)
        # -----------------------------------------

        plt.close(fig)

    print("Saving GIF...")
    imageio.mimsave(gif_path, images, fps=fps)
    print("Saved:", gif_path)


if __name__ == "__main__":
    save_gif_from_xyz("traj.xyz", "benzene.gif")

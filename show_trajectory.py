import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

# --------------------------------------------------------
# Load trajectory
# trajectory shape: [num_steps+1, num_atoms, 3]
# --------------------------------------------------------
trajectory = np.load("trajectory.npy")
num_steps, num_atoms, _ = trajectory.shape
print(f"Loaded trajectory: {num_steps} steps, {num_atoms} atoms")

# --------------------------------------------------------
# Atom types for coloring (example: benzene C6H6)
# --------------------------------------------------------
# 6 carbons, 6 hydrogens
atom_colors = ["black"]*6 + ["red"]*6  # C=black, H=red



# --------------------------------------------------------
# Setup 3D plot
# --------------------------------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Set axes limits based on trajectory extents
xyz_min = trajectory.min(axis=(0,1)) - 0.5
xyz_max = trajectory.max(axis=(0,1)) + 0.5
ax.set_xlim(xyz_min[0], xyz_max[0])
ax.set_ylim(xyz_min[1], xyz_max[1])
ax.set_zlim(xyz_min[2], xyz_max[2])

# --------------------------------------------------------
# Initialize scatter plot with first frame
# --------------------------------------------------------
scat = ax.scatter(
    trajectory[0,:,0], trajectory[0,:,1], trajectory[0,:,2],
    s=100, c=atom_colors
)

# --------------------------------------------------------
# Animation update function
# --------------------------------------------------------
scale = 10
def update(frame):
    pos = trajectory[frame]
    # exaggerate movement relative to first frame
    pos_vis = trajectory[0] + (pos - trajectory[0]) * scale
    scat._offsets3d = (pos_vis[:,0], pos_vis[:,1], pos_vis[:,2])
    ax.set_title(f"Step {frame}")
    return scat,




# --------------------------------------------------------
# Create animation
# --------------------------------------------------------
ani = FuncAnimation(fig, update, frames=num_steps, interval=50, blit=False)

ani.save("trajectory.gif", writer=PillowWriter(fps=20))
print("Saved trajectory.gif")

plt.show()

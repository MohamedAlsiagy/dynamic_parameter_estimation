import numpy as np
import matplotlib.pyplot as plt

# Parameters
R = 1.0
num_points = 100000  # More points = smoother sphere

# Generate random points inside the sphere
phi = np.random.uniform(0, 2*np.pi, num_points)
costheta = np.random.uniform(-1, 1, num_points)
u = np.random.uniform(0, 1, num_points)

theta = np.arccos(costheta)
r = R * (u ** (1/3))  # Uniform distribution in sphere

x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

# Normalize distance for color mapping
frac = r / R  # 0 = center, 1 = edge

# Create colors: Red → Green, and alpha decreases outward
colors = np.zeros((num_points, 4))
colors[:, 0] = 1 - frac   # Red
colors[:, 1] = frac       # Green
colors[:, 2] = 0          # Blue
colors[:, 3] = (1 - frac)**2   # Alpha (center opaque, edge transparent)

# Plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, c=colors, s=1, marker='o', depthshade=False)

ax.set_axis_off()
ax.set_box_aspect([1, 1, 1])

# Save image
output_file = "sphere_pointcloud_gradient.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0)
plt.close(fig)

print(f"Sphere image saved as {output_file}")

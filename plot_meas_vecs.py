import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from plot_pattern import plot_pattern

# Load measurement vectors
data = sio.loadmat("meas_vecs_M_4.mat")
BS_meas_vecs = data['BS_meas_vecs']

# Create a single figure for all patterns
fig = plt.figure(1, figsize=(10, 8))
ax = fig.add_subplot(111, polar=True)

# Configure plot
ax.grid(True, alpha=0.25)
ax.set_rlabel_position(90)

# Plot each measurement vector on the same plot
for i in range(BS_meas_vecs.shape[1]):
    fig, ax = plot_pattern(BS_meas_vecs[:, i], fig=fig, ax=ax, label=f'Vector {i+1}')

plt.tight_layout()
plt.show()

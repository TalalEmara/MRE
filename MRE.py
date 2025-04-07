import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants and parameters
n_protons = 7
duration = 0.03  # 30 ms
fs = 10000  # sampling rate
t = np.linspace(0, duration, int(duration * fs))
wave_freq = 60  # Hz
wave_speed = 3  # m/s
gamma = 42.58e6  # Hz/T
G_amp = 0.02  # T/m

velocity = np.ones(n_protons) * 3

# Initialize Protons
xPositions = np.arange(0, n_protons, 1)
vDisplacement = np.zeros((n_protons, len(t)))

# For MEG
G = np.zeros((len(t), 11))  # 11 vertical levels for each frame
for i in range(len(t)):
    if i % 2 == 0:
        G[i, :] = np.linspace(-2, 2, 11)  # Gradients for protons on even frames
    else:
        G[i, :] = -np.linspace(-2, 2, 11)  # Gradients for protons on odd frames

# Initialize phase array to store accumulated phase values for each proton at each frame
phase = np.zeros((n_protons, len(t)))  # shape: (n_protons, len(t))

# Loop through each proton and accumulate phase for each frame
for i in range(n_protons):
    for frame in range(len(t)):
        # Accumulate phase for this proton at this frame
        phase[i, frame] = gamma * np.sum(G[frame, :] * velocity[i]) * (t[1] - t[0])

# Raw MRI signal
signal = np.sum(np.cos(phase), axis=0)

# Plot setup
colors = plt.cm.viridis(np.linspace(0, 1, n_protons))  # Generate distinct colors for each proton
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))

ax1.set_title('Proton Movements')
ax1.set_xlim([0, n_protons + 1])  # Number of protons
ax1.set_ylim([-1, 1])  # To scale the movement
ax1.set_xlabel('Protons')
ax1.set_ylabel('Displacement (m)')
protons_dots = [ax1.plot([], [], 'o', color=colors[i], label=f'Proton {i + 1}')[0] for i in range(n_protons)]

# Plot 2: Displacement graph of each proton
ax2.set_title('Displacement of Each Proton Over Time')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Displacement (m)')
displacement_lines = [ax2.plot(t, vDisplacement[i, :], label=f'Proton {i + 1}', color=colors[i])[0] for i in
                      range(n_protons)]

# Plot 3: MEG
ax3.set_title('Magnetic Gradient (MEG)')
ax3.set_xlabel('Vertical Levels of Proton')
ax3.set_ylabel('MEG Amplitude (T/m)')

# Initialize lines for each proton (we will update these lines)
lines = [ax3.plot([], [], label=f'Proton {i + 1}', color=colors[i])[0] for i in range(n_protons)]

# Plot 4: Phase graph
ax4.set_title('Phase of Each Proton Over Time')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Phase (Radians)')
bars = [ax4.bar(t[0], phase[i, 0], label=f'Proton {i + 1}', color=colors[i])[0] for i in range(n_protons)]


# Function to update the plots
def update(frame):
    # Update displacement for each proton
    for i in range(n_protons):
        vDisplacement[i, frame] = velocity[i] * t[frame]

    # Update proton movement plot (Protons as moving dots)
    for i in range(n_protons):
        protons_dots[i].set_data([i], [np.sin(2 * np.pi * wave_freq * t[frame])])  # Fix x as a sequence

    # Update displacement lines
    for i in range(n_protons):
        displacement_lines[i].set_ydata(vDisplacement[i, :])

    # Update phase bars (update their heights)
    for i in range(n_protons):
        bars[i].set_height(phase[i, frame])  # Update bar height for current frame

    # Update MEG plot (11 vertical levels for this frame)
    for i in range(n_protons):
        lines[i].set_data(np.arange(11), G[frame, :])  # Update with current frame's MEG values

    return protons_dots + displacement_lines + bars + lines


# Create animation
ani = FuncAnimation(fig, update, frames=len(t), interval=1000 / fs, blit=True)

# Final layout adjustments
plt.tight_layout()
plt.show()

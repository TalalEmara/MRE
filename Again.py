import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

# Parameters
freq = 30  # Hz
omega = 2 * np.pi * freq
wavelength = 5
amplitude = 1
n_protons = 10
xPositions = np.arange(0, n_protons, 1)

# Proton class
class Proton:
    def __init__(self, speed=300, phase=0, color="red", yPosition=0):
        self.speed = speed
        self.phase = phase
        self.color = color
        self.yPosition = yPosition

    def calculatePosition(self, x, time):
        if self.speed == 0:
            self.yPosition = 0.5
        else:
            k = 2 * np.pi * freq / self.speed
            self.yPosition = amplitude * np.sin(k * x - omega * time)
        return self.yPosition

    def updatePhase(self, meg_area):
        gamma = 2 * np.pi * 42.58e6  # rad/s/T for hydrogen
        self.phase += gamma * meg_area * self.yPosition

# Create protons with different colors
colors = plt.cm.rainbow(np.linspace(0, 1, n_protons))
protons = [Proton(color="red") for i in range(n_protons - 4)]
protons += [Proton(speed=2000, color='blue') for _ in range(3)]
protons.append(Proton(speed=0, color='black'))

timePoint = 0
megCounter = 0

# Data buffers for real-time waveform plot
time_values = []
signal_values = []

# Set up figure and axes
fig = plt.figure(figsize=(12, 10))
gs = GridSpec(3, 2, figure=fig, width_ratios=[0.9, 0.1])

# Proton position plot
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_xlim(min(xPositions) - 1, max(xPositions) + 1)
ax1.set_ylim(-1.5, 1.5)
ax1.set_title('Proton Positions')
proton_dots = [ax1.plot([], [], 'o', color=proton.color)[0] for proton in protons]

# MEG Gradient plot
ax2 = fig.add_subplot(gs[0, 1])
y_positions = np.linspace(-1.5, 1.5, 30)
bars = ax2.barh(y=y_positions, width=np.zeros_like(y_positions), height=0.2, color='red')
ax2.set_title('MEG Gradient')
ax2.set_xlim(-1.8, 1.8)
ax2.set_ylim(-1.8, 1.8)

# Phase shift plot
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_title('Phase Shift')
ax3.set_xlim(min(xPositions) - 1, max(xPositions) + 1)
ax3.set_ylim(-1e9, 4e9)
ax3.set_xlabel('Proton Index')
ax3.set_ylabel('Phase')
ax3.grid(True)
phase_dots = [ax3.plot([], [], 'o', color=proton.color)[0] for proton in protons]

# Combined signal plot
ax4 = fig.add_subplot(gs[2, 0])
combined_line, = ax4.plot([], [], 'm-')
ax4.set_title('Raw MRE Signal | Combined Cosine Signal from All Protons')
ax4.set_xlim(0, 10)
ax4.set_ylim(-n_protons, n_protons)
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Signal")

# Animation update function
def update_plot(frame):
    global timePoint, megCounter

    # Proton positions
    for i, proton in enumerate(protons):
        x = xPositions[i]
        y = proton.calculatePosition(x, timePoint)
        proton_dots[i].set_data([x], [y])

    # MEG Gradient
    if megCounter < 15:
        widths = np.linspace(-1.5, 1.5, 30)
    elif 23 <= megCounter <= 37:
        widths = np.linspace(1.5, -1.5, 30)
    else:
        widths = np.zeros(30)

    for bar, width in zip(bars, widths):
        bar.set_width(width)

    # Update phases & dots
    for i, proton in enumerate(protons):
        closest_index = np.argmin(np.abs(y_positions - proton.yPosition))
        corresponding_meg = widths[closest_index]
        proton.updatePhase(corresponding_meg)
        phase_dots[i].set_data([xPositions[i]], [proton.phase])

    # Only update the combined signal when MEG is zero
    if np.all(widths == 0) and megCounter >37:  # Check if the MEG gradient is zero
        combined_signal = sum(np.cos(omega * timePoint + proton.phase) for proton in protons)
        time_values.append(timePoint)
        signal_values.append(combined_signal)

        # Trim buffer to maintain window size
        window_size = 1000
        if len(time_values) > window_size:
            time_values.pop(0)
            signal_values.pop(0)

        combined_line.set_data(time_values, signal_values)
        ax4.set_xlim(max(0, time_values[0]), time_values[-1])

    # Advance time and MEG counter
    timePoint += 0.005
    megCounter = (megCounter + 1) % 60

    return proton_dots + list(bars) + phase_dots + [combined_line]

# Run animation
ani = FuncAnimation(fig, update_plot, frames=1000, interval=100, blit=True)
plt.tight_layout()
plt.show()

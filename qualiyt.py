import streamlit as st
st.title("我的 Python 小工具")

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.patches import Circle

class ParticleSystem:
    def __init__(self, n=10000, dt=0.03, box_size=15, collision_prob=0.08, mass=1.0):
        self.n = n
        self.dt = dt
        self.box_size = box_size
        self.collision_prob = collision_prob
        self.m = mass
        self.pos = np.random.rand(n, 2) * box_size
        self._initialize_velocities()

    def _initialize_velocities(self):
        scale = np.sqrt(1 / self.m)
        speed = maxwell.rvs(scale=scale, size=self.n)
        angle = np.random.rand(self.n) * 2 * np.pi
        self.vel = np.column_stack([speed * np.cos(angle), speed * np.sin(angle)])

    def update_mass(self, new_mass):
        self.m = new_mass
        self._initialize_velocities()

    def update_particle_count(self, new_n):
        old_n = self.n
        self.n = new_n
        if new_n > old_n:
            new_pos = np.random.rand(new_n - old_n, 2) * self.box_size
            scale = np.sqrt(1 / self.m)
            new_speed = maxwell.rvs(scale=scale, size=new_n - old_n)
            new_angle = np.random.rand(new_n - old_n) * 2 * np.pi
            new_vel = np.column_stack([new_speed * np.cos(new_angle),
                                     new_speed * np.sin(new_angle)])
            self.pos = np.vstack([self.pos, new_pos])
            self.vel = np.vstack([self.vel, new_vel])
        elif new_n < old_n:
            self.pos = self.pos[:new_n]
            self.vel = self.vel[:new_n]

    def update(self):
        self.pos += self.vel * self.dt
        self._handle_boundary()
        self._handle_collisions()

    def _handle_boundary(self):
        self.vel[:, 0] = np.where(
            (self.pos[:, 0] < 0) | (self.pos[:, 0] > self.box_size),
            -self.vel[:, 0], self.vel[:, 0]
        )
        self.vel[:, 1] = np.where(
            (self.pos[:, 1] < 0) | (self.pos[:, 1] > self.box_size),
            -self.vel[:, 1], self.vel[:, 1]
        )

    def _handle_collisions(self):
        collision_mask = np.random.rand(self.n) < self.collision_prob
        num_collisions = np.sum(collision_mask)
        if num_collisions > 0:
            scale = np.sqrt(1 / self.m)
            speed = maxwell.rvs(scale=scale, size=num_collisions)
            angle = np.random.rand(num_collisions) * 2 * np.pi
            self.vel[collision_mask] = np.column_stack([
                speed * np.cos(angle),
                speed * np.sin(angle)
            ])

    @property
    def speeds(self):
        return np.linalg.norm(self.vel, axis=1)

    def get_characteristic_speeds(self):
        scale = np.sqrt(1 / self.m)
        v_p = np.sqrt(2) * scale
        v_mean = 2 * np.sqrt(2 / np.pi) * scale
        v_rms = np.sqrt(3) * scale
        return v_p, v_mean, v_rms

# Create system
system = ParticleSystem()

# Create figure
fig = plt.figure(figsize=(16, 9))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# Adjust layout
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.25, top=0.9, wspace=0.2)

# Particle motion plot
scatter = ax1.scatter([], [], s=2, alpha=0.4)
ax1.set_xlim(0, system.box_size)
ax1.set_ylim(0, system.box_size)
ax1.set_title('Particle Motion', fontsize=12)
ax1.set_xticks([])
ax1.set_yticks([])

# Speed distribution plot
current_max = 5
speed_range = np.linspace(0, current_max, 300)
v_p, v_mean, v_rms = system.get_characteristic_speeds()

# Initial plot
ax2.plot(speed_range, maxwell.pdf(speed_range, scale=np.sqrt(1/system.m)),
         'r-', lw=2, label='Theoretical')
hist = ax2.hist([], bins=50, density=True, alpha=0.6,
                range=(0, current_max), label='Simulated')

# Characteristic speeds
v_p_line = ax2.axvline(v_p, color='blue', linestyle='--', alpha=0.7, label='Most Probable')
v_mean_line = ax2.axvline(v_mean, color='green', linestyle='--', alpha=0.7, label='Mean')
v_rms_line = ax2.axvline(v_rms, color='purple', linestyle='--', alpha=0.7, label='RMS')

ax2.set_title(f'Speed Distribution (m = {system.m:.1f})', fontsize=12)
ax2.set_xlim(0, current_max)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Dynamic scaling for y-axis
def auto_scale_yaxis():
    scale = np.sqrt(1/system.m)
    peak = maxwell.pdf(0, scale=scale)
    y_max = max(1.0, peak * 1.1)
    ax2.set_ylim(0, y_max)

auto_scale_yaxis()

# Controls
ax_mass = plt.axes((0.3, 0.15, 0.4, 0.03))
mass_slider = Slider(ax_mass, 'Particle Mass', 0.1, 10.0, valinit=system.m)

# Particle count radio buttons - Updated implementation
ax_particle = plt.axes((0.3, 0.05, 0.4, 0.08))
choices = ['1k (10³)', '10k (10⁴)', '100k (10⁵)', '1M (10⁶)']
particle_buttons = RadioButtons(ax_particle, choices, active=1)

# Style the radio buttons
plt.setp(particle_buttons.labels, fontsize=11, fontfamily='sans-serif',
         fontweight='medium', verticalalignment='center')

# Style the radio button circles using findobj
for circle in ax_particle.findobj(match=Circle):
    circle.set_edgecolor('#555555')
    circle.set_facecolor('#eeeeee')
    circle.set_alpha(0.8)
    circle.set_linewidth(1.5)
    circle.set_radius(0.04)

# Style the radio button container
ax_particle.set_facecolor('#f5f5f5')
for spine in ax_particle.spines.values():
    spine.set_visible(True)
    spine.set_color('#cccccc')
    spine.set_linewidth(1)

# Add title
ax_particle.text(0.5, 1.15, 'Particle Count', transform=ax_particle.transAxes,
                ha='center', va='center', fontsize=10, fontweight='bold')

# Adjust label positions
for label in particle_buttons.labels:
    label.set_y(label.get_position()[1] + 0.02)

# Reset button
ax_reset = plt.axes((0.72, 0.05, 0.15, 0.04))
reset_button = Button(ax_reset, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')

def update_mass(val):
    system.update_mass(val)

    global current_max
    scale = np.sqrt(1/val)
    current_max = min(15, max(5, 4*scale))  # Increased max range

    # Update characteristic speeds
    v_p, v_mean, v_rms = system.get_characteristic_speeds()
    v_p_line.set_xdata([v_p, v_p])
    v_mean_line.set_xdata([v_mean, v_mean])
    v_rms_line.set_xdata([v_rms, v_rms])

    ax2.set_title(f'Speed Distribution (m = {val:.1f})')
    auto_scale_yaxis()

def update_particle_count(label):
    n_dict = {'1k (10³)': 1000, '10k (10⁴)': 10000,
              '100k (10⁵)': 100000, '1M (10⁶)': 1000000}
    system.update_particle_count(n_dict[label])

def reset(event):
    mass_slider.reset()
    particle_buttons.set_active(1)
    system.__init__(n=10000, mass=1.0)
    update_mass(1.0)

mass_slider.on_changed(update_mass)
particle_buttons.on_clicked(update_particle_count)
reset_button.on_clicked(reset)

def init():
    scatter.set_offsets(system.pos)
    return scatter,

def update(frame):
    system.update()
    scatter.set_offsets(system.pos)

    speeds = system.speeds
    ax2.clear()

    # Adaptive binning
    bins = max(30, min(100, int(system.n/1000)))

    # Plot with dynamic range
    speed_range = np.linspace(0, current_max, 300)
    ax2.plot(speed_range, maxwell.pdf(speed_range, scale=np.sqrt(1/system.m)),
             'r-', lw=2, label='Theoretical')
    ax2.hist(speeds, bins=bins, density=True, alpha=0.6,
             range=(0, current_max), label='Simulated')

    # Update characteristic speeds
    v_p, v_mean, v_rms = system.get_characteristic_speeds()
    ax2.axvline(v_p, color='blue', linestyle='--', alpha=0.7, label='Most Probable')
    ax2.axvline(v_mean, color='green', linestyle='--', alpha=0.7, label='Mean')
    ax2.axvline(v_rms, color='purple', linestyle='--', alpha=0.7, label='RMS')

    ax2.set_title(f'Speed Distribution (m = {system.m:.1f})')
    ax2.set_xlim(0, current_max)
    auto_scale_yaxis()
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    return scatter,

ani = animation.FuncAnimation(
    fig, update, frames=200, init_func=init,
    blit=False, interval=50, cache_frame_data=False
)

plt.show()

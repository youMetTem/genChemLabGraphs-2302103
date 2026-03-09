import matplotlib.pyplot as plt
import numpy as np

# measured data
data = [(436, 1.85), (546, 4.55), (580, 5.45)] 

x = np.array([e[0] for e in data])
y = np.array([e[1] for e in data])
    
slope, intercept = np.polyfit(x, y, 1)

x_line = np.linspace(400, 700, 100)
y_line = slope * x_line + intercept

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(x_line, y_line, color='darkblue', linestyle='-', label='Regression Line', zorder=1)
ax.scatter(x, y, facecolors='none', edgecolors='red', marker='o', s=20, linewidths=2, label='Data Points', zorder=2)

ax.set_title("Calibration Plot of the Simple Homemade Spectroscope using Fluorescent Light", fontsize=13, pad=15, fontweight='bold')
ax.set_xlabel("Wavelength (nm)", fontsize=11)
ax.set_ylabel("Position on Spectroscope (cm)", fontsize=11)

ax.grid(True, which='major', color='gray', linestyle='-', alpha=0.5)
ax.grid(True, which='minor', color='lightgray', linestyle='--', alpha=0.5)
ax.minorticks_on()

ax.margins(0.15) 

sign = "+" if intercept >= 0 else "-"
equation_text = f"Model Equation: y = {slope:.5f}x {sign} {abs(intercept):.5f}"

ax.text(0.95, 0.05, equation_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

ax.legend(loc='upper left')
plt.tight_layout()
plt.show()

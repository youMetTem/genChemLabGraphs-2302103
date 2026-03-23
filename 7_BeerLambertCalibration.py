import matplotlib.pyplot as plt
import numpy as np

# measured data points
data = [
    (0*10**(-4), 0.00),   # blank value
    (1.96*10**(-4), 0.33),
    (3.93*10**(-4), 0.67),
    (5.89*10**(-4), 0.98),
    (7.86*10**(-4), 1.21),
    (9.82*10**(-4), 1.40)
]

x = np.array([e[0] for e in data])
y = np.array([e[1] for e in data])

slope, intercept = np.polyfit(x, y, 1)
x_line = np.linspace(0, 0.0011, 100)
y_line = slope * x_line + intercept

y_pred = slope * x + intercept
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(x_line, y_line,
        color='darkblue',
        linestyle='-',
        label='Regression Line',
        zorder=1)

ax.scatter(x, y,
           facecolors='none',
           edgecolors='red',
           marker='o',
           s=20,
           linewidths=2,
           label='Data Points',
           zorder=2)

ax.set_title(
    "Calibration Curve for Spectrophotometric Analysis of Aspirin",
    fontsize=13,
    pad=15,
    fontweight='bold'
)

ax.set_xlabel("Aspirin Concentration (M)", fontsize=11)
ax.set_ylabel("Absorbance at 530 nm", fontsize=11)

ax.grid(True, which='major', color='gray', linestyle='-', alpha=0.5)
ax.grid(True, which='minor', color='lightgray', linestyle='--', alpha=0.5)
ax.minorticks_on()

ax.margins(0.15)

sign = "+" if intercept >= 0 else "-"
equation_text = (
    f"Model Equation: A = {slope:.5f}C {sign} {abs(intercept):.5f}\n"
    f"$R^2$ = {r_squared:.5f}"
)

ax.text(
    0.95,
    0.05,
    equation_text,
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment='bottom',
    horizontalalignment='right',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
)

ax.legend(loc='upper left')

plt.tight_layout()
plt.show()

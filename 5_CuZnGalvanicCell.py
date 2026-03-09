import numpy as np
import matplotlib.pyplot as plt

cu_conc = np.array([0.0001, 0.001, 0.01, 0.1, 1.0])
log_cu = np.log10(cu_conc)
E_calc = 1.1296 + 0.0296 * log_cu

# Experimental Values
E_exp = np.array([0.83, 0.85, 0.86, 0.90, 0.95])

m_exp, b_exp = np.polyfit(log_cu, E_exp, 1)
E_fit = m_exp * log_cu + b_exp
m_calc, b_calc = np.polyfit(log_cu, E_calc, 1)

x_line = np.linspace(min(log_cu), max(log_cu), 100)
y_line_exp = m_exp * x_line + b_exp
y_line_calc = m_calc * x_line + b_calc

fig, ax = plt.subplots(figsize=(9,6))

ax.plot(x_line, y_line_calc, color='darkblue', linestyle='-', label='Calculated', zorder=1)
ax.plot(x_line, y_line_exp, color='black', linestyle='--', label='Experimental Fit', zorder=1)

ax.scatter(log_cu, E_exp,
           facecolors='none',
           edgecolors='red',
           marker='o',
           s=20,
           linewidths=1.5,
           label='Experimental Data',
           zorder=2)

ax.set_title("Ecell - log[Cu²⁺]: Cu–Zn Galvanic Cell", fontsize=13, pad=15, fontweight='bold')
ax.set_xlabel("log [Cu²⁺] (M)", fontsize=11)
ax.set_ylabel("Ecell (V)", fontsize=11)

ax.grid(True, which='major', color='gray', linestyle='-', alpha=0.5)
ax.grid(True, which='minor', color='lightgray', linestyle='--', alpha=0.5)
ax.minorticks_on()

ax.margins(0.15)

sign = "+" if b_exp >= 0 else "-"
equation_text = f"Experimental Fit: E = {m_exp:.5f}x {sign} {abs(b_exp):.5f}"

ax.text(0.95, 0.05,
        equation_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5',
                  facecolor='white',
                  alpha=0.9,
                  edgecolor='gray'))

ax.legend(loc='upper left')

plt.tight_layout()
plt.show()

print(f"Calculated line: slope = {m_calc:.4f}, intercept = {b_calc:.4f}")
print(f"Experimental best-fit line: slope = {m_exp:.4f}, intercept = {b_exp:.4f}")
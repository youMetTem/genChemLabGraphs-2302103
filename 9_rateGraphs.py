import matplotlib.pyplot as plt
import numpy as np

def make_plot(x, y, x_line, y_line, title, xlabel, ylabel, equation_text, filename,
              line_color='darkblue', point_edge='red'):
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        x_line, y_line,
        color=line_color,
        linestyle='-',
        label='Regression Line',
        zorder=1
    )

    ax.scatter(
        x, y,
        facecolors='none',
        edgecolors=point_edge,
        marker='o',
        s=20,
        linewidths=2,
        label='Data Points',
        zorder=2
    )

    ax.set_title(title, fontsize=13, pad=15, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    ax.grid(True, which='major', color='gray', linestyle='-', alpha=0.5)
    ax.grid(True, which='minor', color='lightgray', linestyle='--', alpha=0.5)
    ax.minorticks_on()

    ax.margins(0.15)

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
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def linreg_with_r2(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return slope, intercept, r_squared

x1 = np.array([-1.09691, -1.25181, -1.39794, -1.61979])
y1 = np.array([-4.69037, -4.82391, -5.00307, -5.23459])
slope1, intercept1, r21 = linreg_with_r2(x1, y1)
x1_line = np.linspace(min(x1) - 0.05, max(x1) + 0.05, 100)
y1_line = slope1 * x1_line + intercept1
sign1 = "+" if intercept1 >= 0 else "-"
eq1 = (
    f"Model Equation: y = {slope1:.5f}x {sign1} {abs(intercept1):.5f}\n"
    f"$R^2$ = {r21:.5f}"
)
make_plot(
    x1, y1, x1_line, y1_line,
    "log(rate) vs. log[I⁻]",
    "log[I⁻] (M)",
    "log(rate) (M/s)",
    eq1,
    "graph_1_lograte_vs_logI.png"
)

x2 = np.array([-1.39794, -1.55284, -1.69897, -1.92082])
y2 = np.array([-4.69037, -4.79588, -4.96257, -5.20115])
slope2, intercept2, r22 = linreg_with_r2(x2, y2)
x2_line = np.linspace(min(x2) - 0.05, max(x2) + 0.05, 100)
y2_line = slope2 * x2_line + intercept2
sign2 = "+" if intercept2 >= 0 else "-"
eq2 = (
    f"Model Equation: y = {slope2:.5f}x {sign2} {abs(intercept2):.5f}\n"
    f"$R^2$ = {r22:.5f}"
)
make_plot(
    x2, y2, x2_line, y2_line,
    "log(rate) vs. log[S₂O₈²⁻]",
    "log[S₂O₈²⁻] (M)",
    "log(rate) (M/s)",
    eq2,
    "graph_2_lograte_vs_logS2O8.png"
)

x3 = np.array([1/303.15, 1/313.15, 1/293.15, 1/283.15])
y3 = np.log(np.array([31.33, 23.07, 96.66, 172.87]))
slope3, intercept3, r23 = linreg_with_r2(x3, y3)
x3_line = np.linspace(min(x3) - 0.00003, max(x3) + 0.00003, 100)
y3_line = slope3 * x3_line + intercept3
sign3 = "+" if intercept3 >= 0 else "-"
eq3 = (
    f"Model Equation: y = {slope3:.2f}x {sign3} {abs(intercept3):.2f}\n"
    f"$R^2$ = {r23:.5f}"
)
make_plot(
    x3, y3, x3_line, y3_line,
    "ln(Δt) vs. 1/T for Uncatalyzed Reaction",
    "1/T (K⁻¹)",
    "ln(Δt)",
    eq3,
    "graph_3_lndt_vs_invT_uncatalyzed.png"
)

x4 = np.array([1/303.15, 1/313.15, 1/293.15, 1/283.15])
y4 = np.log(np.array([10.15, 2.50, 17.43, 21.12]))
slope4, intercept4, r24 = linreg_with_r2(x4, y4)
x4_line = np.linspace(min(x4) - 0.00003, max(x4) + 0.00003, 100)
y4_line = slope4 * x4_line + intercept4
sign4 = "+" if intercept4 >= 0 else "-"
eq4 = (
    f"Model Equation: y = {slope4:.2f}x {sign4} {abs(intercept4):.2f}\n"
    f"$R^2$ = {r24:.5f}"
)
make_plot(
    x4, y4, x4_line, y4_line,
    "ln(Δt) vs. 1/T for Catalyzed Reaction",
    "1/T (K⁻¹)",
    "ln(Δt)",
    eq4,
    "graph_4_lndt_vs_invT_catalyzed.png"
)

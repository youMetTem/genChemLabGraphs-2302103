import matplotlib.pyplot as plt
import numpy as np

# Measured Data Points

# Before mixing
t_before = np.array([-60, -45, -30, -15, 0])
T_before = np.array([30, 30, 30, 30, 30])

# After mixing
t_after = np.array([15, 30, 45, 60, 75, 90, 105, 120, 150, 180, 210, 240, 270, 300])
T_after = np.array([38.7, 38.6, 38.5, 38.5, 38.4, 38.3, 38.2, 38.2, 38.1, 38.1, 38.0, 37.9, 37.8, 37.8])

def fit_line(x, y):
    return np.polyfit(x, y, 1)

def eval_line(p, x):
    return np.polyval(p, x)

def intersection(p1, p2):
    m1, b1 = p1
    m2, b2 = p2
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return x, y

def sse(x, y):
    p = fit_line(x, y)
    return np.sum((y - eval_line(p, x)) ** 2)

def detect_case(t_after, T_after):
    peak_idx = np.argmax(T_after)

    if 0 < peak_idx < len(T_after) - 1:
        left = np.diff(T_after[:peak_idx + 1])
        right = np.diff(T_after[peak_idx:])
        if np.sum(left > 0.02) >= 1 and np.sum(right < -0.02) >= 2:
            return "A", peak_idx

    n = len(t_after)
    p_all = fit_line(t_after, T_after)
    sse_all = np.sum((T_after - eval_line(p_all, t_after)) ** 2)

    best_split = None
    best_score = np.inf
    best_p1 = None
    best_p2 = None

    for split in range(3, n - 2):
        x1, y1 = t_after[:split], T_after[:split]
        x2, y2 = t_after[split:], T_after[split:]

        p1 = fit_line(x1, y1)
        p2 = fit_line(x2, y2)

        score = np.sum((y1 - eval_line(p1, x1)) ** 2) + np.sum((y2 - eval_line(p2, x2)) ** 2)

        if score < best_score:
            best_score = score
            best_split = split
            best_p1 = p1
            best_p2 = p2

    m1 = best_p1[0]
    m2 = best_p2[0]

    if best_score < 0.65 * sse_all and m1 < 0 and m2 < 0 and abs(m1 - m2) > 0.002:
        return "C", best_split
    else:
        return "B", None


case_type, info = detect_case(t_after, T_after)

p_before = fit_line(t_before, T_before)
T_cold = eval_line(p_before, 0)

fig, ax = plt.subplots(figsize=(9, 6))

x_before = np.linspace(t_before.min(), 0, 100)
ax.plot(x_before, eval_line(p_before, x_before),
        color='darkblue',
        linestyle='-',
        zorder=1)

if case_type == "A":
    peak_idx = info

    rise_end = max(2, peak_idx)
    t_rise = t_after[:rise_end]
    T_rise = T_after[:rise_end]
    p_rise = fit_line(t_rise, T_rise)

    t_cool = t_after[peak_idx:]
    T_cool = T_after[peak_idx:]
    p_cool = fit_line(t_cool, T_cool)

    t_final, T_final = intersection(p_rise, p_cool)

    m_yellow = (T_final - T_cold) / t_final
    b_yellow = T_cold
    p_yellow = np.array([m_yellow, b_yellow])

    x_yellow = np.linspace(0, t_final, 100)
    ax.plot(x_yellow, eval_line(p_yellow, x_yellow),
            color='goldenrod',
            linestyle='-',
            zorder=1)

    x_cool = np.linspace(t_final, t_after.max(), 200)
    ax.plot(x_cool, eval_line(p_cool, x_cool),
            color='darkblue',
            linestyle='-',
            zorder=1)

elif case_type == "B":
    p_after = fit_line(t_after, T_after)
    T_final = eval_line(p_after, 0)

    x_after = np.linspace(0, t_after.max(), 200)
    ax.plot(x_after, eval_line(p_after, x_after),
            color='darkblue',
            linestyle='-',
            zorder=1)

    t_final = 0.0

elif case_type == "C":
    split = info

    t_first = t_after[:split]
    T_first = T_after[:split]
    p_first = fit_line(t_first, T_first)

    t_second = t_after[split:]
    T_second = T_after[split:]
    p_second = fit_line(t_second, T_second)

    T_final = eval_line(p_first, 0)

    x_first = np.linspace(0, t_first.max(), 200)
    ax.plot(x_first, eval_line(p_first, x_first),
            color='darkblue',
            linestyle='-',
            zorder=1)

    x_second = np.linspace(t_second.min(), t_second.max(), 200)
    ax.plot(x_second, eval_line(p_second, x_second),
            color='goldenrod',
            linestyle='-',
            zorder=1)

    t_final = 0.0

ax.scatter(t_before, T_before,
           facecolors='none',
           edgecolors='red',
           marker='o',
           s=45,
           linewidths=2,
           label='Data Points',
           zorder=2)

ax.scatter(t_after, T_after,
           facecolors='none',
           edgecolors='red',
           marker='o',
           s=45,
           linewidths=2,
           zorder=2)

ax.scatter([t_final], [T_final],
           color='green',
           s=60,
           label='Final Temperature',
           zorder=3)

ax.set_title("NH3-HCl Enthalpy Change Determination",
             fontsize=13,
             pad=15,
             fontweight='bold')
ax.set_xlabel("Time (s)", fontsize=11)
ax.set_ylabel("Temperature (°C)", fontsize=11)

ax.grid(True, which='major', color='gray', linestyle='-', alpha=0.5)
ax.grid(True, which='minor', color='lightgray', linestyle='--', alpha=0.5)
ax.minorticks_on()

ax.margins(0.12)
ax.set_xlim(-80, 320)

all_y = np.concatenate([T_before, T_after, np.array([T_final])])
ax.set_ylim(all_y.min() - 0.5, all_y.max() + 0.7)

equation_text = (
    f"Detected Case = {case_type}\n"
    f"Final Temperature = {T_final:.2f} °C at t = {t_final:.2f} s"
)

ax.text(0.95,
        0.05,
        equation_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5',
                  facecolor='white',
                  alpha=0.9,
                  edgecolor='gray'))

ax.legend(loc='upper left')
plt.tight_layout()
plt.show()
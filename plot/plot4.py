import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

plt.rc("font", size=15.5)
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

r = np.array([0, 1, 2])
x = ["0.01", "0.05", "0.1"]
y1 = np.array([178.42, 48.95, 39.03])

y2 = 196.61 / y1
ax.bar(r - 0.075, y2, width=0.15, color="tab:green")
for i in range(len(r)):
    ax.text(
        r[i] - 0.3,
        y2[i] + 0.1,
        f"{y2[i]:.2f}",
        color="tab:green",
        fontsize=16,
    )

ax.grid(axis="y", linestyle="dotted", alpha=0.3, zorder=0)
ax.set_ylim(0, 6)
ax.set_xlim(min(r) - 0.5, max(r) + 0.5)
ax.set_xticks(r)
ax.set_xticklabels(x)
ax.tick_params(axis='y', labelcolor="tab:green")
ax.set_xlabel(r"$\eta$-Knob Ratio")
ax.set_ylabel(r'$\gamma$-Efficiency', color="tab:green")

ax2 = ax.twinx()
ax2.set_ylim(0, 1.1)
ax2.set_ylabel(r"$\delta$-Accuracy", color="tab:orange")
ax2.tick_params(axis='y', labelcolor="tab:orange")
y3 = np.array([0.450, 0.380, 0.360])
y3 = y3 / 0.44
ax2.bar(r + 0.075, y3, width=0.15, color="tab:orange")
for i in range(len(r)):
    ax2.text(
        r[i],
        y3[i] + 0.01,
        f"{y3[i]:.2f}",
        color="tab:orange",
        fontsize=16,
    )

# plt.tight_layout()
plt.savefig("pic/eta.png")

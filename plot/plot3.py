import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

plt.rc("font", size=16)
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

r = np.array([0, 1, 2, 3, 4])
x = ["0.1", "0.3", "0.5", "0.7", "0.9"]
y1 = np.array([21.08, 32.41, 52.31, 160.50, 199.25])

y2 = 196.61 / y1
ax.bar(r, y2, width=0.15, color="tab:cyan")
for i in range(len(r)):
    ax.text(
        r[i] - 0.1,
        y2[i] + 0.5,
        f"{y2[i]:.2f}",
        color="tab:cyan",
        fontsize=16,
    )


ax.grid(axis="y", linestyle="dotted", alpha=0.3, zorder=0)
ax.set_ylim(0, 20)
ax.set_xlim(min(r) - 0.5, max(r) + 0.5)
ax.set_xticks(r)
ax.set_xticklabels(x)
ax.set_xlabel("Private Passage Ratio")
ax.set_ylabel(r"$\gamma$-Efficiency")

ax2 = ax.twinx()
ax2.set_ylim(-400, 300)
ax2.plot(y1, color="darkgoldenrod")
ax2.scatter(r, y1, marker='x', color="darkgoldenrod")
ax2.set_ylabel("Inference Latency (sec/10 batch)", color="darkgoldenrod")
ax2.set_yticks([])
for i in range(len(r)):
    ax2.text(
        r[i] - 0.2,
        y1[i] + 25,
        f"{y1[i]:.2f}",
        color="darkgoldenrod",
        fontsize=16,
    )

plt.savefig("pic/efficiency_comparison_under_different_privacy_ratios.png")

# libraries
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# y-axis in bold
rc("font", size="14")

# Values of each group
bars2 = [21.0133, 8.6451, 13.42, 6.08]
bars1 = [8.0187, 6.41, 4.25, 6.08]

# Heights of bars1 + bars2
bars = np.add(bars1, bars2).tolist()

# The position of the bars on the x-axis
r = [0, 1, 2, 3]

plt.figure(figsize=(6, 6), dpi=300)
# Names of group and bar width
names = ["RAG-Seq\n(TEE-GPU)", "PRAG-Seq", "FiD\n(TEE-GPU)", "PFiD"]
barWidth = 0.8


b1 = plt.bar(r, bars1, color="#A9C6F9", edgecolor="white", width=barWidth, zorder=1)
b2 = plt.bar(r, bars2, bottom=bars1, color="#FFD29B", edgecolor="white", width=barWidth, zorder=1)

plt.plot(r[:2], list(map(lambda x, y: x + y, bars1[:2], bars2[:2])), color="black")
plt.plot(r[2:], list(map(lambda x, y: x + y, bars1[2:], bars2[2:])), color="black")

plt.scatter(
    r, list(map(lambda x, y: x + y, bars1, bars2)), color="black", s=80, marker="^"
)


def annotate_bars(x_positions, enc_vals, dec_vals):
    for i in range(len(x_positions)):
        plt.text(
            x_positions[i],
            enc_vals[i] / 2,
            f"{enc_vals[i]:.2f}",
            ha="center",
            va="center",
            color="black",
            fontsize=14,
        )
        plt.text(
            x_positions[i],
            enc_vals[i] + dec_vals[i] / 2,
            f"{dec_vals[i]:.2f}",
            ha="center",
            va="center",
            color="black",
            fontsize=14,
        )
        plt.text(
            x_positions[i],
            enc_vals[i] + dec_vals[i] - 1,
            f"{enc_vals[i] + dec_vals[i]:.2f}",
            ha="center",
            va="center",
            color="black",
            fontsize=14,
        )


plt.text(0.75, 22, "1.93X", ha="center", va="center", color="black", fontsize=14)
plt.text(2.75, 15, "1.45X", ha="center", va="center", color="black", fontsize=14)


annotate_bars(r[:2], bars1, bars2)
annotate_bars(r[2:], bars1[2:], bars2[2:])

plt.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
# for spine in plt.gca().spines.values():
#     spine.set_visible(False)

# Custom X axis
plt.xticks(r, names)
plt.xlabel("Model")
plt.ylabel("Inference Latency (sec/batch)")

plt.legend(
    [b2, b1],
    [
        "Encoder",
        "Decoder",
    ],
    frameon=False,
    fontsize=14,
)

plt.tight_layout()

# Show graphic
plt.savefig("tmp")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

plt.rc("font", size=16)
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

r = np.array([0, 1, 2, 3, 4])
x = ["0.1", "0.3", "0.5", "0.7", "0.9"]
y1 = np.array([0.420, 0.398, 0.399, 0.395, 0.419])
# y1_ = (y1 - 0.3) * 6
# ax.plot(r, y1_, color="darkgoldenrod", zorder=9)
# ax.scatter(r, y1_, marker="^", color="darkgoldenrod", s=20, zorder=9)
# for i in range(len(r)):
#     ax.text(
#         r[i],
#         y1_[i] + 0.05,
#         f"{y1[i]:.3f}",
#         ha="center",
#         va="center",
#         color="darkgoldenrod",
#         fontsize=12,
#     )

y2 = y1 / 0.418
ax.bar(r - 0.1, y2, width=0.15, color="tab:blue", label="FiD")
y3 = y1 / 0.409
ax.bar(r + 0.1, y3, width=0.15, color="tab:red", label="RAG-Seq")
for i in range(len(r)):
    ax.text(
        r[i] - 0.5,
        y2[i] + 0.015,
        f"{y2[i]:.2f}",
        color="tab:blue",
        fontsize=16,
    )
    ax.text(
        r[i],
        y3[i] + 0.015,
        f"{y3[i]:.2f}",
        color="tab:red",
        fontsize=16,
    )

# 主图绘制
# ax.plot(r, bars2, color="red", zorder=1, label="FiD")
# ax.scatter(r, bars2, marker="o", color="red", s=20, zorder=5)
# ax.plot(r, bars3, color="blue", zorder=1, label="RAG-Seq")
# ax.scatter(r, bars3, marker="x", color="blue", s=20, zorder=5)
# ax.bar([it - 0.025 for it in r], h1, width=0.05, color="#ff7f0e", zorder=2, label="FiD")
# ax.bar(
#     [it + 0.025 for it in r], h2, width=0.05, color="#1f77b4", zorder=2, label="RAG-Seq"
# )

# ax.axhline(y=0.418, color="#b22222", linestyle="-", linewidth=1, label="FiD")
# ax.axhline(
# y=0.409, color="#6a0dad", linestyle="-", linewidth=1, label="RAG-Seq"
# )

# annotate_bars([-0.9], [0.418])
# annotate_bars([-0.9], [0.409])

# for i in range(len(r)):
#     ax.text(
#         r[i],
#         h1[i] + 0.002,
#         f"{[it / 0.418 for it in bars1][i]:.2f} {[it / 0.409 for it in bars1][i]:.2f}",
#         ha="center",
#         va="center",
#         color="black",
#         fontsize=12,
#     )

ax.grid(axis="y", linestyle="dotted", alpha=0.3, zorder=0)
ax.set_ylim(0, 1.1)
ax.set_xlim(min(r) - 0.5, max(r) + 0.5)
ax.set_xticks(r)
ax.set_xticklabels(x)
ax.set_xlabel("Private Passage Ratio")
ax.set_ylabel(r"$\delta$-Accuracy")

ax.legend(loc="lower right", fontsize=16)

# 添加右侧 y 轴
# ax2 = ax.twinx()
# ax2.tick_params(axis="y", colors="darkgoldenrod")
# ax2.set_ylabel("Exact Match Scores", color="darkgoldenrod")
# ax2.set_ylim(0.3, 0.45)

# plt.tight_layout()
plt.savefig("pic/accuracy_comparison_under_different_privacy_ratios.png")

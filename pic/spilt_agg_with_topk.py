import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plotfig(
    k_values,
    eta_values,
    d_values,
    ex,
    f1,
    time,
    ex_std=[1, 1, 1],
    f1_std=[0.5286666666666666, 0.542],  # k = 5, 10, 15
    time_std=[
        144.40443420410156,
        253.6190755367279,
    ],  # k = 5, 10, 15
    f1_pub=[],
    f1_pri=[],
    ex_pub=[],
    ex_pri=[],
):
    # ------------------------
    # 参数设置
    # ------------------------
    siz = [len(k_values), len(eta_values), len(d_values)]
    accuracy = np.array(f1).reshape(siz) / np.array(f1_std)[:, np.newaxis, np.newaxis]
    accuracy_pub = (
        []
        if len(f1_pub) == 0
        else np.array(f1_pub).reshape(siz) / np.array(f1_std)[:, np.newaxis, np.newaxis]
    )
    accuracy_pri = (
        []
        if len(f1_pri) == 0
        else np.array(f1_pri).reshape(siz) / np.array(f1_std)[:, np.newaxis, np.newaxis]
    )
    efficiency = (
        1 / np.array(time).reshape(siz) * np.array(time_std)[:, np.newaxis, np.newaxis]
    )
    # ------------------------
    # 统一坐标轴范围
    # ------------------------
    acc_min, acc_max = 0, 1.15
    eff_min, eff_max = 0, 15

    # ------------------------
    # 绘图
    # ------------------------
    fig, axes = plt.subplots(
        len(k_values),
        len(eta_values),
        figsize=(4.5 * len(eta_values), 3.2 * len(k_values)),
        sharex=True,
        sharey=True,
        dpi=300,
    )

    blue = "tab:blue"
    red = "tab:red"

    # 用于计算 Case 编号
    case_num = 1

    for i, k_idx in enumerate(range(len(k_values))):
        for j, eta_idx in enumerate(range(len(eta_values))):
            ax1 = axes[i, j]
            ax1.xaxis.set_major_locator(MaxNLocator(integer=False))
            ax1.grid(True, which="major", linestyle=":", color="gray", alpha=0.3)
            ax1.set_ylim(acc_min, acc_max)

            # Accuracy
            label_acc = f"Case {case_num} Accuracy"
            (l1,) = ax1.plot(
                d_values,
                accuracy[k_idx, eta_idx, :],
                marker="o",
                markersize=6,
                color=blue,
                linewidth=2,
                label=label_acc,
            )
            for x, y in zip(d_values, accuracy[k_idx, eta_idx, :]):
                ax1.text(
                    x,
                    y + 0.02,
                    f"{y:.2f}",
                    ha="center",
                    va="bottom",
                    color=blue,
                    fontweight="bold",
                )
            ax1.set_xlabel("d (Private Document Ratio)")
            ax1.set_ylabel("Accuracy", color=blue)
            ax1.tick_params(axis="y", labelcolor=blue)

            if len(accuracy_pub) != 0:
                ax1.plot(
                    d_values,
                    accuracy_pub[k_idx, eta_idx, :],
                    marker="x",
                    markersize=6,
                    color="green",
                    linewidth=2,
                    label=label_acc,
                )
            if len(accuracy_pri) != 0:
                ax1.plot(
                    d_values,
                    accuracy_pri[k_idx, eta_idx, :],
                    marker="^",
                    markersize=6,
                    color="magenta",
                    linewidth=2,
                    label=label_acc,
                )

            # Efficiency
            label_eff = f"Case {case_num} Efficiency"
            ax2 = ax1.twinx()
            # (l2,) = ax2.plot(
            #     d_values,
            #     efficiency[k_idx, eta_idx, :],
            #     marker="s",
            #     markersize=5,
            #     linestyle="--",
            #     color=red,
            #     linewidth=2,
            #     label=label_eff,
            # )
            l2 = ax2.bar(
                d_values,
                efficiency[k_idx, eta_idx, :],
                color=red,
                label=label_eff,
                width=0.05,
            )
            ax2.bar_label(l2, fmt="%.2f", padding=3, color=red, fontweight="bold")
            ax2.set_ylim(eff_min, eff_max)
            ax2.set_ylabel("Efficiency", color=red)
            ax2.tick_params(axis="y", labelcolor=red)

            ax1.set_title(
                f"case-{case_num}: k = {k_values[k_idx]}, η = {eta_values[eta_idx]}"
            )

            # 可见性
            ax1.set_zorder(ax2.get_zorder() + 1)
            ax1.patch.set_visible(False)

            case_num += 1  # 下一子图 Case 编号加 1

    # ------------------------
    # 全局图例
    # ------------------------
    fig.legend(
        [l1, l2],
        ["Accuracy", "Efficiency"],
        loc="upper right",
        ncol=2,
        #    bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        fancybox=True,
        framealpha=0.8,
        edgecolor="gray",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("tmp.png")


if __name__ == "__main__":
    k_values = np.linspace(5, 15, 3, dtype=int)
    eta_values = np.round(np.linspace(0.3, 0.9, 3), 1)
    d_values = np.round(np.linspace(0.1, 0.9, 5), 1)
    accuracy = np.array(
        [
            0.5061428571428571,
            0.524,
            0.5065714285714286,
            0.5149047619047619,
            0.5149047619047619,
            0.5545714285714286,
            0.5192380952380952,
            0.5362380952380952,
            0.5362380952380952,
            0.5362380952380952,
            0.5359047619047619,
            0.5359047619047619,
            0.5359047619047619,
            0.5359047619047619,
            0.5359047619047619,
            0.5206666666666666,
            0.5409047619047619,
            0.5629047619047618,
            0.5362380952380952,
            0.5362380952380952,
            0.5486666666666666,
            0.5342380952380953,
            0.5342380952380953,
            0.5342380952380953,
            0.5342380952380953,
            0.522,
            0.522,
            0.522,
            0.522,
            0.522,
            0.532,
            0.5423333333333333,
            0.5359047619047619,
            0.5279047619047619,
            0.5359047619047619,
            0.532,
            0.522,
            0.522,
            0.522,
            0.522,
            0.542,
            0.542,
            0.542,
            0.542,
            0.542,
        ]
    ).reshape(3, 3, 5)
    f1_pub = np.array(
        [
            0.5061428571428571,
            0.4701904761904762,
            0.4193333333333334,
            0.3974285714285714,
            0.28719047619047616,
            0.5512380952380952,
            0.517095238095238,
            0.40323809523809523,
            0.3227619047619048,
            0.23457142857142854,
            0.5062380952380952,
            0.48257142857142854,
            0.4382857142857142,
            0.3962380952380953,
            0.2480952380952381,
            0.5006666666666666,
            0.48242857142857143,
            0.4169047619047619,
            0.3521904761904762,
            0.1967142857142857,
            0.5286666666666666,
            0.4713333333333334,
            0.42423809523809525,
            0.3855238095238096,
            0.19704761904761903,
            0.4965,
            0.4805714285714286,
            0.42733333333333334,
            0.3163809523809524,
            0.24466666666666664,
            0.517,
            0.5115,
            0.4897380952380952,
            0.31466666666666665,
            0.283,
            0.5153333333333333,
            0.5506666666666667,
            0.48900000000000005,
            0.2853809523809524,
            0.237,
            0.5386666666666666,
            0.5153333333333333,
            0.401,
            0.3303809523809524,
            0.18690476190476185,
        ]
    ).reshape(3, 3, 5)[1:, :, :]
    f1_pri = np.array(
        [
            0.5149047619047619,
            0.5149047619047619,
            0.5149047619047619,
            0.5149047619047619,
            0.5149047619047619,
            0.5362380952380952,
            0.5362380952380952,
            0.5362380952380952,
            0.5362380952380952,
            0.5362380952380952,
            0.5359047619047619,
            0.5359047619047619,
            0.5359047619047619,
            0.5359047619047619,
            0.5359047619047619,
            0.5362380952380952,
            0.5362380952380952,
            0.5362380952380952,
            0.5362380952380952,
            0.5362380952380952,
            0.5342380952380953,
            0.5342380952380953,
            0.5342380952380953,
            0.5342380952380953,
            0.5342380952380953,
            0.522,
            0.522,
            0.522,
            0.522,
            0.522,
            0.5359047619047619,
            0.5359047619047619,
            0.5359047619047619,
            0.5359047619047619,
            0.5359047619047619,
            0.522,
            0.522,
            0.522,
            0.522,
            0.522,
            0.542,
            0.542,
            0.542,
            0.542,
            0.542,
        ]
    ).reshape(3, 3, 5)[1:, :, :]
    efficiency = np.array(
        [
            27.033034086227417,
            26.069616317749023,
            26.753559350967407,
            25.18136167526245,
            25.660566091537476,
            38.08126640319824,
            37.646239042282104,
            37.761775970458984,
            37.95325303077698,
            37.11277413368225,
            48.2907292842865,
            44.76395225524902,
            38.59632587432861,
            33.941041231155396,
            36.459094762802124,
            30.760765314102173,
            29.918830394744873,
            28.932377338409424,
            29.021787405014038,
            28.041388273239136,
            105.75593662261963,
            104.98419976234436,
            104.09012079238892,
            101.62205815315247,
            98.32439637184143,
            145.86375331878662,
            152.74777388572693,
            155.93184232711792,
            143.81999373435974,
            154.94513177871704,
            47.232481718063354,
            45.179280042648315,
            45.809569120407104,
            43.46003532409668,
            39.254724740982056,
            155.46671223640442,
            156.0874936580658,
            154.3441412448883,
            152.05152583122253,
            158.1869535446167,
            241.4185528755188,
            239.4584345817566,
            234.67131900787354,
            234.8232901096344,
            241.0881745815277,
        ]
    ).reshape(3, 3, 5)[1:, :, :]
    plotfig(
        k_values,
        eta_values,
        d_values,
        [],
        accuracy,
        efficiency,
        f1_pub=f1_pub,
        f1_pri=f1_pri,
    )

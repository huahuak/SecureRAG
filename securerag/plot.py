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
    f1_std=[0.5475714285714286, 0.5286666666666666, 0.542],  # k = 5, 10, 15
    time_std=[
        45.480968713760376,
        144.40443420410156,
        253.6190755367279,
    ],  # k = 5, 10, 15
):
    # ------------------------
    # 参数设置
    # ------------------------
    siz = [len(k_values), len(eta_values), len(d_values)]
    accuracy = np.array(f1).reshape(siz) / np.array(f1_std)[:, np.newaxis, np.newaxis]
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
                )
            ax1.set_xlabel("d (Private Document Ratio)")
            ax1.set_ylabel("Accuracy", color=blue)
            ax1.tick_params(axis="y", labelcolor=blue)

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
            ax2.bar_label(l2, fmt="%.2f", padding=3, color=red)
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
    k_values = np.arange(5, 20, 5)
    eta_values = np.arange(0.3, 1.0, 0.3)
    d_values = np.arange(0.1, 1, 0.2)
    accuracy = np.array(
        [
            0.47888095238095235,
            0.4824047619047619,
            0.4656031746031745,
            0.44461111111111107,
            0.4242142857142857,
            0.49018253968253966,
            0.4911825396825397,
            0.48974603174603176,
            0.48684920634920625,
            0.4810158730158729,
            0.49057936507936506,
            0.49057936507936506,
            0.4969603174603174,
            0.49419047619047624,
            0.4814126984126984,
            0.48407142857142865,
            0.47042063492063496,
            0.4808571428571428,
            0.49184920634920626,
            0.4855714285714286,
            0.48907142857142866,
            0.4776111111111111,
            0.4845555555555556,
            0.4888015873015873,
            0.48713492063492064,
            0.4866587301587302,
            0.4823253968253968,
            0.4799920634920634,
            0.4822142857142857,
            0.4833253968253968,
            0.4973253968253967,
            0.5010873015873016,
            0.49184920634920637,
            0.4928015873015873,
            0.49057936507936506,
            0.500547619047619,
            0.4779920634920635,
            0.4924365079365079,
            0.4799920634920634,
            0.48554761904761906,
            0.4899920634920634,
            0.49388095238095236,
            0.494436507936508,
            0.4822142857142857,
            0.4877698412698413,
        ]
    ).reshape(3, 3, 5)

    efficiency = np.array(
        [
            25.52489185333252,
            25.428996086120605,
            24.242544412612915,
            23.13689136505127,
            22.37240242958069,
            27.258346796035767,
            27.668561935424805,
            26.664417028427124,
            25.888784408569336,
            24.43326473236084,
            28.94894528388977,
            28.825067043304443,
            28.317713499069214,
            26.8024160861969,
            25.675918102264404,
            36.86022472381592,
            34.54542779922485,
            32.281485080718994,
            29.399068355560303,
            28.32910394668579,
            39.853190898895264,
            38.28965902328491,
            34.79051232337952,
            35.37290668487549,
            32.734477043151855,
            44.03308176994324,
            40.90989637374878,
            39.12001085281372,
            36.9361469745636,
            34.32242250442505,
            46.76331853866577,
            43.764511823654175,
            39.099568605422974,
            36.23124885559082,
            33.09775972366333,
            52.916040658950806,
            48.947365045547485,
            45.971346855163574,
            41.984426498413086,
            39.61722254753113,
            55.23153233528137,
            53.355167388916016,
            48.960525035858154,
            46.01782774925232,
            44.17535948753357,
        ]
    ).reshape(3, 3, 5)
    plotfig(k_values, eta_values, d_values, [], accuracy, efficiency)

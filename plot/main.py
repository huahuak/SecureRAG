import math
import matplotlib.pyplot as plt

def compute_and_plot(k, a):
    n_values = []
    ratio_values = []
    tmp_values = []

    sum_numerator = 0
    sum_denominator = 0

    for n in range(1, k + 1):
        numerator = math.comb(a, n)
        denominator = math.comb(k, n)
        sum_numerator += numerator
        sum_denominator += denominator

        ratio = numerator / denominator * 100
        tmp = sum_numerator / sum_denominator * 100

        n_values.append(n)
        ratio_values.append(ratio)
        tmp_values.append(tmp)

        print(f"n={n}, ratio={ratio:.2f}%, tmp={tmp:.2f}%")

    S = sum_numerator / sum_denominator * 100
    print(f"S = {S:.2f}%")

    # 画图
    plt.figure(figsize=(8, 5))
    plt.plot(n_values, ratio_values, 'o-', label="ratio (each n)")
    plt.plot(n_values, tmp_values, 's--', label="cumulative tmp (up to n)")

    plt.title(f"Proportion of Red Combinations (k={k}, a={a})")
    plt.xlabel("n")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 110)
    plt.grid(True)
    plt.legend()
    plt.savefig("./log/proportion_plot.png")

# 示例
k = 10
a = 7
compute_and_plot(k, a)

import time  # Wall-clock timing

import matplotlib.pyplot as plt
import numpy as np
import torch  # PyTorch for tensor operations

# ----- Configuration and Data Collection -----
Ns = [768, 1024, 2048]
Ks = [5, 10, 15]
benchmark_results = []

# Constants for robust timing
WARMUP_RUNS = 5  # Runs to discard for stability
N_MEASUREMENTS = 100  # Times to measure and average

# Initialize CUDA Events for accurate GPU timing (outside the loop)
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

for K in Ks:
    N = 768
    # ----- Data Setup -----
    A_cpu = torch.rand(16 * K, 200, N, dtype=torch.float32)
    B_cpu = torch.rand(N, N, dtype=torch.float32)

    # Pre-transfer B to GPU once (static matrix)
    B_gpu = B_cpu.to("cuda")

    # ----- 1. CPU Matmul Timing (Averaging over N_MEASUREMENTS) -----
    cpu_times = []
    for _ in range(N_MEASUREMENTS):
        # Use perf_counter for reliable CPU duration measurement
        start = time.perf_counter()
        C_cpu = torch.matmul(A_cpu, B_cpu)
        cpu_times.append(time.perf_counter() - start)

    cpu_time = np.mean(cpu_times)

    # ----- 2. GPU Warm-up Phase (Excluding Cold Start Influence) -----
    print(f"Warming up for K={K}...")
    for _ in range(WARMUP_RUNS):
        A_gpu_dummy = A_cpu.to("cuda")  # H2D transfer
        C_gpu_dummy = torch.matmul(A_gpu_dummy, B_gpu)  # Compute
        C_gpu_dummy.to("cpu")  # D2H transfer
        torch.cuda.synchronize()  # Wait for everything to finish

    # ----- 3. GPU Matmul and Transfer Timing (Measuring steady-state average) -----

    h2d_times, gpu_compute_times, d2h_times = [], [], []

    for _ in range(N_MEASUREMENTS):
        # Time H2D Transfer
        start_event.record()
        A_gpu = A_cpu.to("cuda")  # Transfer A
        end_event.record()
        end_event.synchronize()
        h2d_times.append(
            start_event.elapsed_time(end_event) / 1000.0
        )  # Convert ms to s

        # Time GPU Compute
        start_event.record()
        C_gpu = torch.matmul(A_gpu, B_gpu)
        end_event.record()
        end_event.synchronize()
        gpu_compute_times.append(start_event.elapsed_time(end_event) / 1000.0)

        # Time D2H Transfer
        start_event.record()
        C_result = C_gpu.to("cpu")
        end_event.record()
        end_event.synchronize()
        d2h_times.append(start_event.elapsed_time(end_event) / 1000.0)

    # Calculate robust average times
    h2d_time = np.mean(h2d_times)
    gpu_compute_time = np.mean(gpu_compute_times)
    d2h_time = np.mean(d2h_times)

    gpu_total_time = h2d_time + gpu_compute_time + d2h_time

    # Store (CPU_Time, H2D, Compute, D2H, GPU_Total)
    benchmark_results.append(
        (cpu_time, h2d_time, gpu_compute_time, d2h_time, gpu_total_time)
    )

# -----------------------------------------------
# --- PLOTTING (Consistent with the previous style) ---
# -----------------------------------------------
plt.figure(figsize=(7, 5), dpi=300)

# ----- X positions and bar width -----
x_indices = np.arange(len(Ns))  # Indices for N=768, 1024, 2048
width = 0.35
group_spacing = 0.5  # Space between N groups

# Positions for CPU bars (left) and GPU bars (right)
x_cpu = x_indices - width / 2
x_gpu = x_indices + width / 2

# ----- Bar Labels and Colors (Matching First Example Style) -----
gpu_colors = ["tab:green", "tab:olive", "tab:orange"]  # H2D, Compute, D2H
gpu_labels = ["GPU Compute", "Host to Device", "Device to Host"]

# We use the list to track which legend items have been drawn
legend_handles = []
legend_labels = []

# ----- Plotting Loop for each N size -----
for i, K in enumerate(Ks):
    cpu_t, h2d_t, gpu_c, d2h_t, gpu_t = benchmark_results[i]
    current_x_gpu = x_gpu[i]

    # --- 1. CPU Bar ---
    # Draw only the first CPU bar to get the legend handle
    if i == 0:
        cpu_bar = plt.bar(
            x_cpu[i],
            cpu_t,
            width=width,
            color="tab:green",
            label="CPU Compute",
            edgecolor="black",
        )
        legend_handles.append(cpu_bar)
        legend_labels.append("CPU Compute")
    else:
        plt.bar(x_cpu[i], cpu_t, width=width, color="tab:green", edgecolor="black")

    # Add text label for CPU time
    plt.text(x_cpu[i], cpu_t + 0.001, f"{cpu_t:.3f}", ha="center", fontsize=14)

    # --- 2. GPU Stacked Bar ---
    gpu_values = [
        gpu_c,
        h2d_t,
        d2h_t,
    ]
    # Bottoms calculation must be cumulative
    gpu_bottoms = [0, gpu_c, h2d_t + gpu_c]

    for j, (val, bottom, color, label) in enumerate(
        zip(gpu_values, gpu_bottoms, gpu_colors, gpu_labels)
    ):
        # Draw the bar segment
        bar_segment = plt.bar(
            current_x_gpu,
            val,
            bottom=bottom,
            width=width,
            color=color,
            hatch="/",
            edgecolor="black",
        )

        # Draw the legend handle only on the first N (i=0)
        if i == 0:
            legend_handles.append(bar_segment)
            legend_labels.append(label)

        # Add percentage label to the segment
        plt.text(
            current_x_gpu,
            bottom + val / 2,
            f"{val/gpu_t*100:.1f}%",
            ha="center",
            fontsize=14,
            color="white",
        )

    # Add text label for GPU total time
    plt.text(current_x_gpu, gpu_t + 0.001, f"{gpu_t:.3f}", ha="center", fontsize=14)

# ----- X-axis labels -----
plt.xticks(x_indices, [str(K) for K in Ks])

# ----- Labels and title -----
plt.ylabel("Runtime (seconds)", fontsize=14)
plt.xlabel("The number of retrieved passages K", fontsize=14)

# ----- Legend outside the plot to the right -----
plt.legend(legend_handles, legend_labels, fontsize=14, loc="upper left")
plt.grid(True, which="major", linestyle=":", color="gray", alpha=0.3)

plt.tight_layout()
plt.savefig("tmp.png")

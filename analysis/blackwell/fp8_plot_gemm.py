import json
import matplotlib.pyplot as plt
import numpy as np

colors = ["#8E69B8", "#E59952", "#68AC5A", "#7CB9BC", "#DE836B", "#55555A"]


blackwell_baselines = {
    "cublaslt": {
        "1024": 347.08,
        "2048": 1667.20,
        "4096": 3046.83,
        "8192": 3134.89,
        "16384": 2875.66,
    },
    "cutlass": {
        "1024": 209.509,
        "2048": 984.925,
        "4096": 2500.09,
        "8192": 2617.16,
        "16384": 2038.03,
    },
}

def process_data(data_list):
    """Separate numeric values and OOM indices"""
    values = []
    oom_indices = []
    for i, val in enumerate(data_list):
        if val == "OOM":
            values.append(0)
            oom_indices.append(i)
        else:
            values.append(val)
    return values, oom_indices


for device in ['mi355x']:

    # Read data
    try:
        with open(f'{device}_fp8_gemm.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {device}_fp8_gemm.json: {e}")
        continue

    # Extract data for plotting
    matrix_sizes = sorted([int(size) for size in data.keys()])
    hipblaslt_tflops = [data[str(size)]['tflops_hipblaslt'] for size in matrix_sizes]
    tk_tflops = [data[str(size)]['tflops'] for size in matrix_sizes]
    cublaslt_tflops = [blackwell_baselines['cublaslt'][str(size)] for size in matrix_sizes]
    cutlass_tflops = [blackwell_baselines['cutlass'][str(size)] for size in matrix_sizes]

    # Process data to separate OOM values
    hipblaslt_vals, hipblaslt_oom = process_data(hipblaslt_tflops)
    tk_vals, tk_oom = process_data(tk_tflops)
    cublaslt_vals, cublaslt_oom = process_data(cublaslt_tflops)
    cutlass_vals, cutlass_oom = process_data(cutlass_tflops)


    max_tflops = max(max(hipblaslt_vals), max(tk_vals), max(cublaslt_vals), max(cutlass_vals))

    # Create bar chart
    x = np.arange(len(matrix_sizes))
    width = 0.145

    fig, ax = plt.subplots(figsize=(15, 6))
    first_bar = x - 2*width
    second_bar = x - width
    third_bar = x
    fourth_bar = x + width
    bars1 = ax.bar(first_bar, hipblaslt_vals, width, label='HipBLAS LT', color=colors[0])
    bars3 = ax.bar(second_bar, tk_vals, width, label='HipKittens', color=colors[3])
    bars2 = ax.bar(third_bar, cublaslt_vals, width, label='CUBLAS LT', color=colors[1])
    bars5 = ax.bar(fourth_bar, cutlass_vals, width, label='Cutlass', color=colors[4])

    # Plot X markers for OOM
    oom_height = max_tflops * 0.95

    for idx in hipblaslt_oom:
        ax.plot(x[idx] - 2*width, oom_height, 'x', color=colors[1], markersize=15, markeredgewidth=3)
        ax.text(x[idx] - 2*width, oom_height + max_tflops * 0.03, 'OOM', ha='center', va='bottom', fontsize=10, color=colors[1])

    for idx in tk_oom:
        ax.plot(x[idx], oom_height, 'x', color=colors[3], markersize=15, markeredgewidth=3)
        ax.text(x[idx], oom_height + max_tflops * 0.03, 'OOM', ha='center', va='bottom', fontsize=10, color=colors[3])

    for idx in cublaslt_oom:
        ax.plot(x[idx] - width, oom_height, 'x', color=colors[1], markersize=15, markeredgewidth=3)
        ax.text(x[idx] - width, oom_height + max_tflops * 0.03, 'OOM', ha='center', va='bottom', fontsize=10, color=colors[1])

    for idx in cutlass_oom:
        ax.plot(x[idx] + 2*width, oom_height, 'x', color=colors[4], markersize=15, markeredgewidth=3)
        ax.text(x[idx] + 2*width, oom_height + max_tflops * 0.03, 'OOM', ha='center', va='bottom', fontsize=10, color=colors[4])

    # Add value labels on bars
    for bar, value in zip(bars1, hipblaslt_vals):
        if value > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=10)

    for bar, value in zip(bars3, tk_vals):
        if value > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=10)

    for bar, value in zip(bars2, cublaslt_vals):
        if value > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=10)

    for bar, value in zip(bars5, cutlass_vals):
        if value > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=10)

    # add some padding to the top of the y-axis to prevent label overlap
    ax.set_ylim(0, max_tflops * 1.15)
    ax.set_xlabel('Matrix Size (N×N)', fontsize=16)
    ax.set_ylabel('Performance (TFLOPS)', fontsize=16)
    ax.set_title(f'FP8 GEMM Performance Comparison across Hardware', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(matrix_sizes, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=14)

    plt.tight_layout()
    plt.show()

    output_file = f'blackwell_fp8_gemm_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    # Print summary
    print(f"Matrix sizes tested: {matrix_sizes}")
    print(f"HipBLAS LT TFLOPS: {[f'{t:.2f}' if t > 0 else 'OOM' for t in hipblaslt_vals]}")
    print(f"HipKittens TFLOPS: {[f'{t:.2f}' if t > 0 else 'OOM' for t in tk_vals]}")
    print(f"CUBLAS LT TFLOPS: {[f'{t:.2f}' if t > 0 else 'OOM' for t in cublaslt_vals]}")
    print(f"Cutlass TFLOPS: {[f'{t:.2f}' if t > 0 else 'OOM' for t in cutlass_vals]}")

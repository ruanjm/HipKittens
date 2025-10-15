import json
import matplotlib.pyplot as plt
import numpy as np

colors = ["#8E69B8", "#E59952", "#68AC5A", "#7CB9BC", "#4E7AE3"]


for device in ['mi350x']:

    # Read data
    try:
        with open(f'{device}_data_to_log.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {device}_data_to_log.json: {e}")
        continue

    # Extract data for plotting
    aiter_tflops = [data['aiter_bkwd']['tflops_ref']]
    tk_4warps_tflops = [data['GQA_bkwd_4warps']['tflops_tk']]
    tk_8warps_tflops = [data['GQA_bkwd_8warps']['tflops_tk']]
    tk_asm_interleaved_tflops = [data['GQA_bkwd_asm_interleaved']['tflops_tk']]

    # Create bar chart
    x = np.arange(4)
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 6))
    bars0 = ax.bar(x[0], tk_8warps_tflops, width, label='HipKittens (8 warps)', color=colors[0])
    bars1 = ax.bar(x[1], tk_4warps_tflops, width, label='HipKittens (4 warps)', color=colors[1])
    bars2 = ax.bar(x[2], tk_asm_interleaved_tflops, width, label='HipKittens (ASM Interleaved)', color=colors[2])
    bars3 = ax.bar(x[3], aiter_tflops, width, label='AITER (AMD)', color=colors[3])

    max_tflops = max(max(aiter_tflops), max(tk_4warps_tflops), max(tk_8warps_tflops), max(tk_asm_interleaved_tflops))


    # TK 8 warps
    for bar, value in zip(bars0, tk_8warps_tflops):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                f'{value:.0f}', ha='center', va='bottom', fontsize=14)

    # TK 4 warps
    for bar, value in zip(bars1, tk_4warps_tflops):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                f'{value:.0f}', ha='center', va='bottom', fontsize=14)

    # TK ASM Interleaved
    for bar, value in zip(bars2, tk_asm_interleaved_tflops):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                f'{value:.0f}', ha='center', va='bottom', fontsize=14)

    # AITER
    for bar, value in zip(bars3, aiter_tflops):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                f'{value:.0f}', ha='center', va='bottom', fontsize=14)

    # add some padding to the top of the y-axis to prevent label overlap
    ax.set_ylim(0, max_tflops * 1.15)
    ax.set_xlabel('Kernel', fontsize=16)
    ax.set_ylabel('Performance (TFLOPS)', fontsize=16)
    ax.set_title('Attention Backwards Performance Comparison MI350X', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(['HK 8-waves', 'HK 4-waves', 'HK ASM 4-waves', 'Assembly'], fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    # ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    output_file = f'{device}_attn_bkwd_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    # Print summary
    print(f"AITER (AMD) TFLOPS: {[f'{t:.2f}' for t in aiter_tflops]}")
    print(f"TK 4 warps TFLOPS: {[f'{t:.2f}' for t in tk_4warps_tflops]}")
    print(f"TK 8 warps TFLOPS: {[f'{t:.2f}' for t in tk_8warps_tflops]}")
    print(f"TK ASM Interleaved TFLOPS: {[f'{t:.2f}' for t in tk_asm_interleaved_tflops]}")


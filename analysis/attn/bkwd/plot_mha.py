import json
import matplotlib.pyplot as plt
import numpy as np



mi355x_baselines_causal = {
    "triton": {
        "1024": 147,
        "2048": 179,
        "4096": 215,
        "8192": 237,
        "16384": 251,
    },
    "ck": {
        "1024": 372.36,
        "2048": 622.46,
        "4096": 800.12,
        "8192": 870.58,
        "16384": 986.59,
    },
    "torch": {
        "1024": 34,
        "2048": 36,
        "4096": 38,
        "8192": 37,
        "16384": "OOM",
    },
    "aiter": {
        "1024": 375.72,
        "2048": 654.68,
        "4096": 902.16,
        "8192": 1006.03,
        "16384": 1110.15,
    },
    "hk": {
        "1024": 376.26,
        "2048": 627.68,
        "4096": 796.35,
        "8192": 916.25,
        "16384": 1015.41,
    }
}

mi350x_baselines_causal = {
    "triton": {
        "1024": 139.080476,
        "2048": 163.599388,
        "4096": 188.696534,
        "8192": 200.207510,
        "16384": 206.924580,
    },
    "ck": {
        "1024": 350.65,
        "2048": 579.85,
        "4096": 728.33,
        "8192": 813.47,
        "16384": 886.17,
    },
    "torch": {
        "1024": 31.582108,
        "2048": 33.775558,
        "4096": 35.817826,
        "8192": 34.693869,
        "16384": "OOM",
    },
    "aiter": {
        "1024": 416.30,
        "2048": 587.63,
        "4096": 735.77,
        "8192": 799.96,
        "16384": 848.04,
    },
    "hk": {
        "1024": 349.09,
        "2048": 520.36,
        "4096": 698.59,
        "8192": 767.11,
        "16384": 814.23,
    }
}


mi355x_baselines_non_causal = { 
    # triton not available for non-causal bwd attn
    "ck": {
        "1024": 348.60,
        "2048": 415.37,
        "4096": 437.66,
        "8192": 465.44,
        "16384": 465.83,
    },
    "torch": {
        "1024": 71,
        "2048": 78,
        "4096": 84,
        "8192": 81,
        "16384": "OOM",
    },
    "aiter": {
        "1024": 651.94,
        "2048": 911.98,
        "4096": 1043.71,
        "8192": 1115.40,
        "16384": 1164.79,
    },
    "hk": {
        "1024": 686.53,
        "2048": 889.74,
        "4096": 994.30,
        "8192": 1073.31,
        "16384": 1119.40,
    }
}

mi350x_baselines_non_causal = { 
    # triton not available for non-causal bwd attn
    "ck": {
        "1024": 331.73,
        "2048": 391.71,
        "4096": 412.22,
        "8192": 434.40,
        "16384": 443.33,
    },
    "torch": {
        "1024": 70.188240,
        "2048": 76.365602,
        "4096": 81.948592,
        "8192": 79.742189,
        "16384": "OOM",
    },
    "aiter": {
        "1024": 596.54,
        "2048": 776.44,
        "4096": 824.74,
        "8192": 892.54,
        "16384": 919.44,
    },
    "hk": {
        "1024": 611.45,
        "2048": 776.34,
        "4096": 814.93,
        "8192": 850.11,
        "16384": 877.71,
    }
}





colors = ["#8E69B8", "#E59952", "#68AC5A", "#7CB9BC", "#DE836B"]


def process_data(data_list):
    """Separate numeric values and OOM indices"""
    values = []
    oom_indices = []
    for i, val in enumerate(data_list):
        if val == "OOM":
            values.append(0)  # Use 0 for bar height
            oom_indices.append(i)
        else:
            values.append(val)
    return values, oom_indices


for device in ['mi350x', 'mi355x']:

    # Get baseline data
    if device == 'mi355x':
        baselines_causal = mi355x_baselines_causal
        baselines_non_causal = mi355x_baselines_non_causal
    else:  # mi350x
        baselines_causal = mi350x_baselines_causal
        baselines_non_causal = mi350x_baselines_non_causal

    # For MHA backward, we have both causal and non-causal data
    # Create separate plots for each setting
    for setting in ['causal', 'non_causal']:
        if setting == 'causal':
            baselines = baselines_causal
            title_suffix = 'Causal'
        else:
            baselines = baselines_non_causal
            title_suffix = 'Non-Causal'

        # Get sequence lengths from the baseline data
        n_values = sorted([int(n) for n in baselines['torch'].keys()])

        # Extract baseline data for these sequence lengths
        torch_tflops = [baselines['torch'][str(n)] for n in n_values]
        ck_tflops = [baselines['ck'][str(n)] for n in n_values]
        aiter_tflops = [baselines['aiter'][str(n)] for n in n_values]
        tk_tflops = [baselines['hk'][str(n)] for n in n_values]

        # Triton is only available for causal
        if setting == 'causal' and 'triton' in baselines:
            triton_tflops = [baselines['triton'][str(n)] for n in n_values]
        else:
            triton_tflops = []

        # Process data to separate OOM values
        torch_vals, torch_oom = process_data(torch_tflops)
        ck_vals, ck_oom = process_data(ck_tflops)
        if triton_tflops:
            triton_vals, triton_oom = process_data(triton_tflops)
        else:
            triton_vals, triton_oom = [], []

        # Calculate max for numeric values only
        numeric_vals = aiter_tflops + tk_tflops
        numeric_vals.extend([v for v in torch_vals if v != 0])
        numeric_vals.extend([v for v in ck_vals if v != 0])
        if triton_vals:
            numeric_vals.extend([v for v in triton_vals if v != 0])
        max_tflops = max(numeric_vals) if numeric_vals else 100

        # Create bar chart
        x = np.arange(len(n_values))
        width = 0.17

        fig, ax = plt.subplots(figsize=(16, 6))

        if triton_tflops:
            # 5 bars: PyTorch, Triton, CK, AITER, HipKittens
            first_bar_start = x - 2*width
            second_bar_start = x - width
            third_bar_start = x
            fourth_bar_start = x + width
            fifth_bar_start = x + 2*width

            bars1 = ax.bar(first_bar_start, torch_vals, width, label='PyTorch SDPA', color=colors[4])
            bars2 = ax.bar(second_bar_start, triton_vals, width, label='Triton', color=colors[2])
            bars3 = ax.bar(third_bar_start, ck_vals, width, label='Composable Kernel', color=colors[1])
            bars4 = ax.bar(fourth_bar_start, aiter_tflops, width, label='AITER', color=colors[0])
            bars5 = ax.bar(fifth_bar_start, tk_tflops, width, label='HipKittens', color=colors[3])
        else:
            # 4 bars: PyTorch, CK, AITER, HipKittens
            first_bar_start = x - 1.5*width
            second_bar_start = x - 0.5*width
            third_bar_start = x + 0.5*width
            fourth_bar_start = x + 1.5*width

            bars1 = ax.bar(first_bar_start, torch_vals, width, label='PyTorch SDPA', color=colors[4])
            bars3 = ax.bar(second_bar_start, ck_vals, width, label='Composable Kernel', color=colors[1])
            bars4 = ax.bar(third_bar_start, aiter_tflops, width, label='AITER', color=colors[0])
            bars5 = ax.bar(fourth_bar_start, tk_tflops, width, label='HipKittens', color=colors[3])

        # Plot X markers for OOM
        oom_height = 50  # Position X near top of chart
        if torch_oom:
            for idx in torch_oom:
                offset = -2*width if triton_tflops else -1.5*width
                ax.plot(x[idx] + offset, oom_height, 'x', color=colors[4],
                       markersize=15, markeredgewidth=3)
                ax.text(x[idx] + offset, oom_height + max_tflops * 0.03,
                       'OOM', ha='center', va='bottom', fontsize=10, color=colors[4])

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars1, torch_vals)):
            if value > 0:  # Only label non-OOM bars
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                        f'{value:.0f}', ha='center', va='bottom', fontsize=14)

        if triton_tflops:
            for bar, value in zip(bars2, triton_vals):
                if value > 0:  # Only label non-OOM bars
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                            f'{value:.0f}', ha='center', va='bottom', fontsize=14)

        for bar, value in zip(bars3, ck_vals):
            if value > 0:  # Only label non-OOM bars
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                        f'{value:.0f}', ha='center', va='bottom', fontsize=14)

        for bar, value in zip(bars4, aiter_tflops):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=14)

        for bar, value in zip(bars5, tk_tflops):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=14)

        # add some padding to the top of the y-axis to prevent label overlap
        ax.set_ylim(0, max_tflops * 1.15)
        ax.set_xlabel('Sequence Length (N)', fontsize=16)
        ax.set_ylabel('Performance (TFLOPS)', fontsize=16)
        ax.set_title(f'MHA {title_suffix} Backward Performance Comparison {device.upper()}', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in n_values], fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.legend(fontsize=16)

        plt.tight_layout()
        plt.show()

        output_file = f'{device}_mha_{setting}_bkwd_plot.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")

        # Print summary
        print(f"Sequence lengths tested: {n_values}")
        print(f"AITER TFLOPS: {[f'{t:.2f}' for t in aiter_tflops]}")
        print(f"HipKittens TFLOPS: {[f'{t:.2f}' for t in tk_tflops]}")

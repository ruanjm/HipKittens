import json
import matplotlib.pyplot as plt
import numpy as np



mi355x_baselines_causal = {
    # triton not available for gqa bwd attn
    "ck": {
        "1024": 449.97,
        "2048": 707.80,
        "4096": 829.07,
        "8192": 894.56,
        "16384": 991.03,
    },
    "torch": {
        "1024": 34,
        "2048": 36,
        "4096": 37,
        "8192": "OOM",
        "16384": "OOM",
    },
    "aiter": {
        "1024": 156.00,
        "2048": 181.95,
        "4096": 201.98,
        "8192": 281.81,
        "16384": 343.50,
    },
    "hk": {
        "1024": 450.71,
        "2048": 570.52,
        "4096": 735.79,
        "8192": 912.28,
        "16384": 986.10, # 15 batch size
    }
}

mi350x_baselines_causal = {
    # triton not available for gqa bwd attn
    "ck": {
        "1024": 420.40,
        "2048": 661.89,
        "4096": 754.86,
        "8192": 829.72,
        "16384": 905.81,
    },
    "torch": {
        "1024": 31.600038,
        "2048": 34.332461,
        "4096": 35.813629,
        "8192": "OOM",
        "16384": "OOM",
    },
    "aiter": {
        "1024": 137.76,
        "2048": 164.12,
        "4096": 187.95,
        "8192": 257.95,
        "16384": 313.90,
    },
    "hk": {
        "1024": 401.01,
        "2048": 483.26,
        "4096": 622.12,
        "8192": 737.69,
        "16384": 780.95, # 15 batch size
    }
}


mi355x_baselines_non_causal = {
    # triton not available for gqa bwd attn
    "ck": {
        "1024": 352.54,
        "2048": 418.05,
        "4096": 438.59,
        "8192": 458.98,
        "16384": 462.01,
    },
    "torch": {
        "1024": 72,
        "2048": 79,
        "4096": 83,
        "8192": "OOM",
        "16384": "OOM",
    },
    "aiter": {
        "1024": 297.21,
        "2048": 353.43,
        "4096": 379.22,
        "8192": 392.32,
        "16384": 398.99,
    },
    "hk": {
        "1024": 832.11,
        "2048": 962.46,
        "4096": 1072.95,
        "8192": 1135.57,
        "16384": 1167.41,
    }
}

mi350x_baselines_non_causal = {
    # triton not available for gqa bwd attn
    "ck": {
        "1024": 334.65,
        "2048": 391.00,
        "4096": 416.55,
        "8192": 431.70,
        "16384": 437.35,
    },
    "torch": {
        "1024": 69.704672,
        "2048": 76.491021,
        "4096": 80.135474,
        "8192": "OOM",
        "16384": "OOM",
    },
    "aiter": {
        "1024": 267.64,
        "2048": 320.77,
        "4096": 349.91,
        "8192": 363.39,
        "16384": 372.18,
    },
    "hk": {
        "1024": 730.33,
        "2048": 774.57,
        "4096": 865.53,
        "8192": 899.39,
        "16384": 901.42, # 15 batch size
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

    # For GQA backward, we have both causal and non-causal data
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

        # Process data to separate OOM values
        torch_vals, torch_oom = process_data(torch_tflops)
        ck_vals, ck_oom = process_data(ck_tflops)

        # Calculate max for numeric values only
        numeric_vals = aiter_tflops + tk_tflops
        numeric_vals.extend([v for v in torch_vals if v != 0])
        numeric_vals.extend([v for v in ck_vals if v != 0])
        max_tflops = max(numeric_vals) if numeric_vals else 100

        # Create bar chart
        x = np.arange(len(n_values))
        width = 0.17

        fig, ax = plt.subplots(figsize=(16, 6))
        first_bar_start = x - 1.5*width
        second_bar_start = x - 0.5*width
        third_bar_start = x + 0.5*width
        fourth_bar_start = x + 1.5*width

        bars1 = ax.bar(first_bar_start, torch_vals, width, label='PyTorch SDPA', color=colors[4])
        bars2 = ax.bar(second_bar_start, ck_vals, width, label='Composable Kernel', color=colors[1])
        bars3 = ax.bar(third_bar_start, aiter_tflops, width, label='AITER', color=colors[0])
        bars4 = ax.bar(fourth_bar_start, tk_tflops, width, label='HipKittens', color=colors[3])

        # Plot X markers for OOM
        oom_height = 50  # Position X near top of chart
        if torch_oom:
            for idx in torch_oom:
                ax.plot(x[idx] - 1.5*width, oom_height, 'x', color=colors[4],
                       markersize=15, markeredgewidth=3)
                ax.text(x[idx] - 1.5*width, oom_height + max_tflops * 0.03,
                       'OOM', ha='center', va='bottom', fontsize=10, color=colors[4])

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars1, torch_vals)):
            if value > 0:  # Only label non-OOM bars
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                        f'{value:.0f}', ha='center', va='bottom', fontsize=14)

        for bar, value in zip(bars2, ck_vals):
            if value > 0:  # Only label non-OOM bars
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                        f'{value:.0f}', ha='center', va='bottom', fontsize=14)

        for bar, value in zip(bars3, aiter_tflops):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=14)

        for bar, value in zip(bars4, tk_tflops):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=14)

        # add some padding to the top of the y-axis to prevent label overlap
        ax.set_ylim(0, max_tflops * 1.15)
        ax.set_xlabel('Sequence Length (N)', fontsize=16)
        ax.set_ylabel('Performance (TFLOPS)', fontsize=16)
        ax.set_title(f'GQA {title_suffix} Backward Performance Comparison {device.upper()}', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in n_values], fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.legend(fontsize=16)

        plt.tight_layout()
        plt.show()

        output_file = f'{device}_gqa_{setting}_bkwd_plot.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")

        # Print summary
        print(f"Sequence lengths tested: {n_values}")
        print(f"AITER TFLOPS: {[f'{t:.2f}' for t in aiter_tflops]}")
        print(f"HipKittens TFLOPS: {[f'{t:.2f}' for t in tk_tflops]}")

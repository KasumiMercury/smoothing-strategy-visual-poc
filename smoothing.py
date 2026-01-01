import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from dataclasses import dataclass
    from typing import Optional, Tuple
    import matplotlib
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

    def generate_test_data(length: int = 30, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
    
        baseline = np.random.poisson(lam=3, size=length)
    
        num_spikes = np.random.randint(3, 6)
        spike_positions = np.random.choice(length, size=num_spikes, replace=False)
    
        for pos in spike_positions:
            spike_value = np.random.randint(20, 81)
            baseline[pos] += spike_value
    
        return baseline

    # データ生成
    test_data = generate_test_data(length=30, seed=42)
    print(f"Generated test data (length={len(test_data)}, sum={np.sum(test_data)}):")
    print(test_data)
    return np, plt, test_data


@app.cell
def _(np):
    def triangular_weights(radius: int) -> np.ndarray:
        if radius < 0:
            radius = 0
        size = 2 * radius + 1
        weights = np.zeros(size)
        for i in range(size):
            dist_from_center = abs(i - radius)
            weights[i] = radius + 1 - dist_from_center
        return weights

    def greville_edge_kernel(position: int, length: int, radius: int) -> tuple[np.ndarray, int]:
        full_kernel = triangular_weights(radius)
        full_sum = np.sum(full_kernel)
    
        left_bound = max(0, position - radius)
        right_bound = min(length - 1, position + radius)
    
        kernel_start = radius - (position - left_bound)
        kernel_end = radius + (right_bound - position)
    
        kernel_start = max(0, kernel_start)
        kernel_end = min(len(full_kernel) - 1, kernel_end)
    
        truncated = full_kernel[kernel_start:kernel_end + 1]
        truncated_sum = np.sum(truncated)
    
        if truncated_sum > 0:
            scale_factor = full_sum / truncated_sum
            scaled = truncated * scale_factor
        else:
            scaled = np.ones_like(truncated)
    
        return scaled, left_bound

    def convolve_with_greville(values: np.ndarray, radius: int) -> np.ndarray:
        values = np.array(values, dtype=float)
        length = len(values)
        if length == 0:
            return np.array([])
    
        result = np.zeros(length)
        full_kernel = triangular_weights(radius)
    
        for i in range(length):
            weighted_sum = 0.0
            weight_sum = 0.0
        
            if i < radius or i >= length - radius:
                weights, start_idx = greville_edge_kernel(i, length, radius)
                for j, w in enumerate(weights):
                    if start_idx + j < length:
                        weighted_sum += values[start_idx + j] * w
                        weight_sum += w
            else:
                for k in range(-radius, radius + 1):
                    idx = i + k
                    kernel_idx = k + radius
                    weighted_sum += values[idx] * full_kernel[kernel_idx]
                    weight_sum += full_kernel[kernel_idx]
        
            if weight_sum > 0:
                result[i] = weighted_sum / weight_sum
    
        input_sum = np.sum(values)
        output_sum = np.sum(result)
        if output_sum > 0 and input_sum > 0:
            scale_factor = input_sum / output_sum
            result *= scale_factor
    
        return result

    def normalize_to_sum(values: np.ndarray, target_sum: int) -> np.ndarray:
        values = np.array(values, dtype=float)
    
        if len(values) == 0 or target_sum <= 0:
            return np.zeros(len(values), dtype=int)
    
        total_value = np.sum(values)
    
        if total_value <= 0:
            result = np.zeros(len(values), dtype=int)
            if len(values) > 0:
                base = target_sum // len(values)
                remainder = target_sum % len(values)
                result[:] = base
                result[:remainder] += 1
            return result
    
        scaled = values * target_sum / total_value
        floored = np.floor(scaled).astype(int)
        remainders = scaled - floored
    
        current_sum = np.sum(floored)
        to_distribute = target_sum - current_sum
    
        indices_by_remainder = np.argsort(-remainders)
    
        result = floored.copy()
        for i in range(to_distribute):
            if i < len(indices_by_remainder):
                result[indices_by_remainder[i]] += 1
    
        return result
    return convolve_with_greville, normalize_to_sum


@app.cell
def _(normalize_to_sum, np, test_data):
    def cubic_weights(radius: int) -> np.ndarray:
        if radius < 0:
            radius = 0
        size = 2 * radius + 1
        weights = np.zeros(size)
    
        for i in range(size):
            dist_from_center = abs(i - radius)
            # 正規化された距離 [0, 1]
            u = dist_from_center / (radius if radius > 0 else 1)
            if u <= 1:
                # 3次多項式カーネル
                weights[i] = (1 - u**3)
    
        return weights

    def convolve_with_cubic_greville(values: np.ndarray, radius: int) -> np.ndarray:
        values = np.array(values, dtype=float)
        length = len(values)
        if length == 0:
            return np.array([])
    
        result = np.zeros(length)
        full_kernel = cubic_weights(radius)
        full_sum = np.sum(full_kernel)
    
        for i in range(length):
            weighted_sum = 0.0
            weight_sum = 0.0
        
            left_bound = max(0, i - radius)
            right_bound = min(length - 1, i + radius)
        
            kernel_start = radius - (i - left_bound)
            kernel_end = radius + (right_bound - i)
        
            truncated = full_kernel[kernel_start:kernel_end + 1]
            truncated_sum = np.sum(truncated)
        
            if truncated_sum > 0:
                scale_factor = full_sum / truncated_sum
                scaled_kernel = truncated * scale_factor
            else:
                scaled_kernel = truncated
        
            for j, w in enumerate(scaled_kernel):
                data_idx = left_bound + j
                if data_idx < length:
                    weighted_sum += values[data_idx] * w
                    weight_sum += w
        
            if weight_sum > 0:
                result[i] = weighted_sum / weight_sum
    
        input_sum = np.sum(values)
        output_sum = np.sum(result)
        if output_sum > 0 and input_sum > 0:
            scale_factor = input_sum / output_sum
            result *= scale_factor
    
        return result

    def smooth_cubic_greville(values: np.ndarray, radius: int = 6) -> np.ndarray:
        target_sum = int(np.sum(values))
        smoothed = convolve_with_cubic_greville(values, radius)
        normalized = normalize_to_sum(smoothed, target_sum)
        return normalized

    cubic_result = smooth_cubic_greville(test_data, radius=6)
    print(f"\nCubic Greville (13-term) result:")
    print(f"Original sum: {np.sum(test_data)}, Smoothed sum: {np.sum(cubic_result)}")
    return (cubic_result,)


@app.cell
def _(normalize_to_sum, np, test_data):
    def spike_redistribution_smooth(
        values: np.ndarray,
        window_radius: int = 5,
        redistribution_radius: int = 5,
        anchor_left: bool = True
    ) -> np.ndarray:
        values = np.array(values, dtype=float)
        length = len(values)
        result = values.copy()

        baseline = np.zeros(length)
        for i in range(length):
            left = max(0, i - window_radius)
            right = min(length, i + window_radius + 1)
            baseline[i] = np.median(values[left:right])

        R = int(redistribution_radius)

        for i in range(length):
            if anchor_left and i == 0:
                continue

            excess = max(0.0, values[i] - baseline[i])
            if excess <= 0:
                continue

            left_bound = max(0, i - R)
            right_bound = min(length, i + R + 1)

            targets = []
            weights = []

            for j in range(left_bound, right_bound):
                if anchor_left and j == 0:
                    continue

                dist = abs(j - i)

                w = (R + 1 - dist)
                if w > 0:
                    w = w * w
                    targets.append(j)
                    weights.append(w)

            if not weights:
                continue

            wsum = float(np.sum(weights))
            if wsum <= 0:
                continue

            result[i] -= excess

            weights = np.array(weights, dtype=float) / wsum
            for j, w in zip(targets, weights):
                result[j] += excess * w

        target_sum = int(np.sum(values))
        normalized = normalize_to_sum(result, target_sum)
        return normalized

    spike_result = spike_redistribution_smooth(
        test_data,
        window_radius=5,
        redistribution_radius=5,
        anchor_left=True
    )
    print(f"\nSpike Redistribution result:")
    print(f"Original sum: {np.sum(test_data)}, Smoothed sum: {np.sum(spike_result)}")
    return (spike_result,)


@app.cell
def _(np, plt):
    def plot_individual_comparison(
        original: np.ndarray,
        smoothed: np.ndarray,
        title: str,
        method_name: str
    ) -> plt.Figure:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
        x = np.arange(len(original))
    
        ax2.plot(x, original, 'o-', label='Original', linewidth=2, markersize=6, color='steelblue')
        ax2.plot(x, smoothed, 's-', label=method_name, linewidth=2, markersize=6, color='coral')
        ax2.set_xlabel('Time Bucket (minute)')
        ax2.set_ylabel('Notification Count')
        ax2.set_title(f'{title} - Line Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
        original_sum = np.sum(original)
        smoothed_sum = np.sum(smoothed)
        original_std = np.std(original)
        smoothed_std = np.std(smoothed)
    
        stats_text = f'Sum: {original_sum:.0f} → {smoothed_sum:.0f}\n'
        stats_text += f'Std: {original_std:.2f} → {smoothed_std:.2f}'
    
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
        plt.tight_layout()
        return fig
    return (plot_individual_comparison,)


@app.cell
def _(
    convolve_with_greville,
    normalize_to_sum,
    np,
    plot_individual_comparison,
    plt,
    test_data,
):
    triangular_result = normalize_to_sum(
        convolve_with_greville(test_data, radius=5),
        int(np.sum(test_data))
    )

    fig1 = plot_individual_comparison(
        test_data,
        triangular_result,
        'Triangular Kernel + Greville Edge Handling',
        'Triangular Greville'
    )
    plt.show()
    return (triangular_result,)


@app.cell
def _(cubic_result, plot_individual_comparison, plt, test_data):
    fig2 = plot_individual_comparison(
        test_data,
        cubic_result,
        'Cubic Kernel (13-term) + Greville Edge Handling',
        'Cubic Greville'
    )
    plt.show()
    return


@app.cell
def _(plot_individual_comparison, plt, spike_result, test_data):
    fig3 = plot_individual_comparison(
        test_data,
        spike_result,
        'Spike Detection and Redistribution',
        'Spike Redistribution'
    )
    plt.show()
    return


@app.cell
def _(cubic_result, np, plt, spike_result, test_data, triangular_result):
    def plot_all_approaches_comparison(
        original: np.ndarray,
        triangular: np.ndarray,
        cubic: np.ndarray,
        spike: np.ndarray
    ) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
        x = np.arange(len(original))
    
        methods = [
            ('Original', original, 'steelblue'),
            ('Triangular + Greville', triangular, 'coral'),
            ('Cubic (13-term) + Greville', cubic, 'green'),
            ('Spike Redistribution', spike, 'purple')
        ]
    
        ax1 = axes[0, 0]
        for name, data, color in methods:
            ax1.plot(x, data, 'o-', label=name, linewidth=2, markersize=5, alpha=0.8)
        ax1.set_xlabel('Time Bucket (minute)')
        ax1.set_ylabel('Notification Count')
        ax1.set_title('All Methods - Line Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
        ax2 = axes[0, 1]
        method_names = [m[0] for m in methods]
        stds = [np.std(m[1]) for m in methods]
        colors = [m[2] for m in methods]
        bars = ax2.bar(range(len(method_names)), stds, color=colors, alpha=0.7)
        ax2.set_xticks(range(len(method_names)))
        ax2.set_xticklabels(method_names, rotation=15, ha='right')
        ax2.set_ylabel('Standard Deviation')
        ax2.set_title('Smoothness Comparison (Lower is Smoother)')
        ax2.grid(True, alpha=0.3, axis='y')
    
        for bar, std in zip(bars, stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{std:.2f}', ha='center', va='bottom')
    
        ax3 = axes[1, 0]
        data_matrix = np.array([m[1] for m in methods])
        im = ax3.imshow(data_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax3.set_yticks(range(len(method_names)))
        ax3.set_yticklabels(method_names)
        ax3.set_xlabel('Time Bucket (minute)')
        ax3.set_title('Heatmap Comparison')
        plt.colorbar(im, ax=ax3, label='Notification Count')
    
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
    
        stats_data = []
        for name, data, _ in methods:
            stats_data.append([
                name,
                f'{np.sum(data):.0f}',
                f'{np.mean(data):.2f}',
                f'{np.std(data):.2f}',
                f'{np.max(data):.0f}',
                f'{np.min(data):.0f}'
            ])
    
        table = ax4.table(
            cellText=stats_data,
            colLabels=['Method', 'Sum', 'Mean', 'Std', 'Max', 'Min'],
            cellLoc='center',
            loc='center',
            colWidths=[0.35, 0.13, 0.13, 0.13, 0.13, 0.13]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
    
        for i in range(6):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
    
        ax4.set_title('Statistical Comparison', pad=20, fontsize=12, fontweight='bold')
    
        plt.tight_layout()
        return fig

    fig_all = plot_all_approaches_comparison(
        test_data,
        triangular_result,
        cubic_result,
        spike_result
    )
    plt.show()
    return


@app.cell
def _(cubic_result, np, spike_result, test_data, triangular_result):
    def analyze_smoothing_quality(
        original: np.ndarray,
        smoothed: np.ndarray,
        method_name: str
    ) -> dict:
        # スパイク削減率
        original_max = np.max(original)
        smoothed_max = np.max(smoothed)
        spike_reduction = (original_max - smoothed_max) / original_max * 100
    
        # 標準偏差削減率
        original_std = np.std(original)
        smoothed_std = np.std(smoothed)
        std_reduction = (original_std - smoothed_std) / original_std * 100
    
        # 平均絶対偏差
        mad = np.mean(np.abs(smoothed - original))
    
        # 総和保存性
        sum_preservation = np.sum(smoothed) / np.sum(original) * 100
    
        return {
            'method': method_name,
            'spike_reduction_%': spike_reduction,
            'std_reduction_%': std_reduction,
            'mean_absolute_deviation': mad,
            'sum_preservation_%': sum_preservation,
            'max_original': original_max,
            'max_smoothed': smoothed_max,
            'std_original': original_std,
            'std_smoothed': smoothed_std
        }

    print("\n" + "="*70)
    print("SMOOTHING QUALITY ANALYSIS")
    print("="*70)

    methods_to_analyze = [
        ('Triangular + Greville', triangular_result),
        ('Cubic + Greville', cubic_result),
        ('Spike Redistribution', spike_result)
    ]

    for name, result in methods_to_analyze:
        analysis = analyze_smoothing_quality(test_data, result, name)
        print(f"\n{name}:")
        print(f"  Spike Reduction: {analysis['spike_reduction_%']:.2f}%")
        print(f"  Std Reduction: {analysis['std_reduction_%']:.2f}%")
        print(f"  Mean Absolute Deviation: {analysis['mean_absolute_deviation']:.2f}")
        print(f"  Sum Preservation: {analysis['sum_preservation_%']:.2f}%")
        print(f"  Max: {analysis['max_original']:.0f} → {analysis['max_smoothed']:.0f}")
        print(f"  Std: {analysis['std_original']:.2f} → {analysis['std_smoothed']:.2f}")
    return


if __name__ == "__main__":
    app.run()

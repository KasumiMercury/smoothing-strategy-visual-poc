import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
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
    return Optional, Tuple, np, plt, test_data


@app.cell(hide_code=True)
def _(np, plt):
    def plot_individual_comparison(
        original: np.ndarray,
        smoothed: np.ndarray,
        title: str,
        method_name: str
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7, 5))
        x = np.arange(len(original))

        ax.plot(x, original, 'o-', label='Original', linewidth=2, markersize=6, color='steelblue')
        ax.plot(x, smoothed, 's-', label=method_name, linewidth=2, markersize=6, color='coral')
        ax.set_xlabel('Time Bucket (minute)')
        ax.set_ylabel('Notification Count')
        ax.set_title(f'{title} - Line Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        original_sum = np.sum(original)
        smoothed_sum = np.sum(smoothed)
        original_std = np.std(original)
        smoothed_std = np.std(smoothed)

        stats_text = f'Sum: {original_sum:.0f} → {smoothed_sum:.0f}\n'
        stats_text += f'Std: {original_std:.2f} → {smoothed_std:.2f}'

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        return fig
    return (plot_individual_comparison,)


@app.cell(hide_code=True)
def _(np, plot_individual_comparison, plt, test_data):
    def localized_kernel_regression(data: np.ndarray, window: int = 10, kernel: str = 'triangular') -> np.ndarray:
        n = len(data)
        result = data.copy()

        if n <= 1:
            return result

        def get_kernel_weight(distance: float, bandwidth: float, kernel_type: str) -> float:
            u = distance / bandwidth
            if abs(u) > 1:
                return 0.0

            if kernel_type == 'triangular':
                return 1 - abs(u)
            elif kernel_type == 'epanechnikov':
                return 0.75 * (1 - u**2)
            elif kernel_type == 'gaussian':
                return np.exp(-0.5 * u**2)
            else:
                return 1.0

        smoothed = np.zeros(n)
        smoothed[0] = data[0]

        bandwidth = window

        for i in range(1, n):
            start = max(1, i - window)
            end = min(n, i + window + 1)

            weights = np.zeros(end - start)
            values = data[start:end]

            for j, idx in enumerate(range(start, end)):
                distance = abs(idx - i)
                weights[j] = get_kernel_weight(distance, bandwidth, kernel)

            if np.sum(weights) > 0:
                smoothed[i] = np.sum(weights * values) / np.sum(weights)
            else:
                smoothed[i] = data[i]

        original_sum = np.sum(data)
        smoothed_sum = np.sum(smoothed)

        if smoothed_sum > 0 and abs(smoothed_sum - smoothed[0]) > 1e-10:
            adjustment_factor = (original_sum - smoothed[0]) / (smoothed_sum - smoothed[0])
            result[1:] = smoothed[1:] * adjustment_factor
            result[0] = data[0]

        return result

    lkr_result = localized_kernel_regression(test_data)
    print(f"\nLocalized Kernel Regression result:")
    print(f"Original sum: {np.sum(test_data)}, Smoothed sum: {np.sum(lkr_result)}")

    fig_lkr = plot_individual_comparison(
        test_data,
        lkr_result,
        'Localized Kernel Regression (Triangular Kernel)',
        'Localized Kernel Regression'
    )
    plt.show()
    return (lkr_result,)


@app.cell(hide_code=True)
def _(np, plot_individual_comparison, plt, test_data):
    def lkr_sum_preserve(data: np.ndarray, window: int = 5, kernel: str = 'triangular') -> np.ndarray:
        n = len(data)

        if n <= 1:
            return data.copy().astype(np.int64)

        def get_kernel_weight(distance: float, bandwidth: float, kernel_type: str) -> float:
            if bandwidth == 0:
                return 1.0 if distance == 0 else 0.0

            u = distance / bandwidth
            if abs(u) > 1:
                return 0.0

            if kernel_type == 'triangular':
                return 1 - abs(u)
            elif kernel_type == 'epanechnikov':
                return 0.75 * (1 - u**2)
            elif kernel_type == 'gaussian':
                return np.exp(-0.5 * u**2)
            else:
                return 1.0

        smoothed = np.zeros(n, dtype=np.float64)
        smoothed[0] = float(data[0])

        bandwidth = float(window)

        for i in range(1, n):
            start = max(1, i - window)
            end = min(n, i + window + 1)

            weights = []
            values = []

            for idx in range(start, end):
                distance = abs(idx - i)
                weight = get_kernel_weight(float(distance), bandwidth, kernel)
                if weight > 0:
                    weights.append(weight)
                    values.append(float(data[idx]))

            if len(weights) > 0 and sum(weights) > 0:
                smoothed[i] = sum(w * v for w, v in zip(weights, values)) / sum(weights)
            else:
                smoothed[i] = float(data[i])

        original_sum = int(np.sum(data))
        smoothed_sum = np.sum(smoothed)

        adjusted = smoothed.copy()

        if abs(smoothed_sum - smoothed[0]) > 1e-10:
            target_sum_without_first = original_sum - smoothed[0]
            current_sum_without_first = smoothed_sum - smoothed[0]

            if current_sum_without_first > 0:
                scale_factor = target_sum_without_first / current_sum_without_first
                adjusted[1:] = smoothed[1:] * scale_factor

        result = np.round(adjusted).astype(np.int64)
        result[0] = int(data[0]) 

        current_total = np.sum(result)
        diff = original_sum - current_total

        if diff != 0:
            # 差分を最も値が大きい要素から順に配分
            indices = np.argsort(result[1:])[::-1] + 1  # 降順、左端除外

            # 差分を1ずつ配分
            for i in range(abs(diff)):
                idx = indices[i % len(indices)]
                if diff > 0:
                    result[idx] += 1
                else:
                    result[idx] -= 1
                    if result[idx] < 0:
                        result[idx] = 0

        return result

    lkr_sp_result = lkr_sum_preserve(test_data)
    print(f"\nLKR Sum-Preserving Integer result:")
    print(f"Original sum: {np.sum(test_data)}, Smoothed sum: {np.sum(lkr_sp_result)}")
    fig_lkr_sp = plot_individual_comparison(
        test_data,
        lkr_sp_result,
        'Localized Kernel Regression (Sum-Preserving Integer)',
        'LKR Sum-Preserving Integer'
    )
    plt.show()
    return lkr_sp_result, lkr_sum_preserve


@app.cell(hide_code=True)
def _(lkr_sum_preserve, np, plot_individual_comparison, plt, test_data):
    lkr_sp_epanechnikov_result = lkr_sum_preserve(test_data, kernel='epanechnikov')
    print(f"\nLKR Sum-Preserving Integer (Epanechnikov Kernel) result:")
    print(f"Original sum: {np.sum(test_data)}, Smoothed sum: {np.sum(lkr_sp_epanechnikov_result)}")
    fig_lkr_sp_epanechnikov = plot_individual_comparison(
        test_data,
        lkr_sp_epanechnikov_result,
        'Localized Kernel Regression (Sum-Preserving Integer, Epanechnikov Kernel)',
        'LKR Sum-Preserving Integer (Epanechnikov)'
    )
    plt.show()
    return (lkr_sp_epanechnikov_result,)


@app.cell(hide_code=True)
def _(lkr_sum_preserve, np, plot_individual_comparison, plt, test_data):
    lkr_sp_gaussian_result = lkr_sum_preserve(test_data, kernel='gaussian')
    print(f"\nLKR Sum-Preserving Integer (Gaussian Kernel) result:")
    print(f"Original sum: {np.sum(test_data)}, Smoothed sum: {np.sum(lkr_sp_gaussian_result)}")
    fig_lkr_sp_gaussian = plot_individual_comparison(
        test_data,
        lkr_sp_gaussian_result,
        'Localized Kernel Regression (Sum-Preserving Integer, Gaussian Kernel)',
        'LKR Sum-Preserving Integer (Gaussian)'
    )
    plt.show()
    return (lkr_sp_gaussian_result,)


@app.cell(hide_code=True)
def _(np, plot_individual_comparison, plt, test_data):
    def kernel_regression_with_greville_linear(
        data: np.ndarray, 
        window: int = 10, 
        kernel: str = 'triangular',
        extrapolation_points: int = None
    ) -> np.ndarray:
        n = len(data)

        if n <= 1:
            return data.copy().astype(np.int64)

        if extrapolation_points is None:
            extrapolation_points = window

        extended_data = np.zeros(n + extrapolation_points, dtype=np.float64)
        extended_data[:n] = data.astype(np.float64)

        if n >= 2:
            fit_start = max(1, n - window)  # 左端を除外
            fit_end = n

            x_fit = np.arange(fit_start, fit_end)
            y_fit = data[fit_start:fit_end].astype(np.float64)

            if len(x_fit) > 1:
                coeffs = np.polyfit(x_fit, y_fit, 1)
                slope, intercept = coeffs[0], coeffs[1]

                for i in range(extrapolation_points):
                    x_extrap = n + i
                    extended_data[n + i] = slope * x_extrap + intercept
            else:
                extended_data[n:] = data[-1]

        def kernel_weight(distance: float, bandwidth: float) -> float:
            if bandwidth == 0:
                return 1.0 if distance == 0 else 0.0

            u = distance / bandwidth
            if abs(u) > 1:
                return 0.0

            if kernel == 'triangular':
                return 1 - abs(u)
            elif kernel == 'epanechnikov':
                return 0.75 * (1 - u**2)
            elif kernel == 'gaussian':
                return np.exp(-0.5 * u**2)
            return 1.0

        smoothed = np.zeros(n, dtype=np.float64)
        smoothed[0] = float(data[0])

        for i in range(1, n):
            start = max(1, i - window)
            end = min(len(extended_data), i + window + 1)

            total_weight = 0.0
            weighted_sum = 0.0

            for idx in range(start, end):
                w = kernel_weight(float(abs(idx - i)), float(window))
                if w > 0:
                    total_weight += w
                    weighted_sum += w * extended_data[idx]

            smoothed[i] = weighted_sum / total_weight if total_weight > 0 else float(data[i])

        original_sum = int(np.sum(data))
        if abs(np.sum(smoothed[1:])) > 1e-10:
            scale = (original_sum - smoothed[0]) / np.sum(smoothed[1:])
            smoothed[1:] *= scale

        result = np.zeros(n, dtype=np.int64)
        result[0] = int(data[0])

        floors = np.floor(smoothed[1:]).astype(np.int64)
        result[1:] = floors

        remainders = smoothed[1:] - floors
        diff = original_sum - np.sum(result)

        if diff != 0:
            indices = np.argsort(remainders)[::-1] if diff > 0 else np.argsort(remainders)

            for i in range(min(abs(diff), len(indices))):
                idx = indices[i] + 1
                result[idx] += 1 if diff > 0 else -1
                if result[idx] < 0:
                    result[idx] = 0

        return result

    lkr_greville_result = kernel_regression_with_greville_linear(test_data)
    print(f"\nLKR with Greville Linear Extrapolation result:")
    print(f"Original sum: {np.sum(test_data)}, Smoothed sum: {np.sum(lkr_greville_result)}")
    fig_lkr_greville = plot_individual_comparison(
        test_data,
        lkr_greville_result,
        'Localized Kernel Regression with Greville Linear Extrapolation',
        'LKR with Greville Linear'
    )
    plt.show()
    return kernel_regression_with_greville_linear, lkr_greville_result


@app.cell(hide_code=True)
def _(np, plot_individual_comparison, plt, test_data):
    def kernel_regression_with_greville_quadratic(
        data: np.ndarray, 
        window: int = 10, 
        kernel: str = 'triangular',
        extrapolation_points: int = None
    ) -> np.ndarray:
        """
        Greville補正付き局所カーネル回帰（2次多項式外挿版）
        """
        n = len(data)

        if n <= 1:
            return data.copy().astype(np.int64)

        if extrapolation_points is None:
            extrapolation_points = window

        # 拡張データ配列
        extended_data = np.zeros(n + extrapolation_points, dtype=np.float64)
        extended_data[:n] = data.astype(np.float64)

        # 右端の2次トレンド推定
        if n >= 3:
            fit_start = max(1, n - min(window * 2, n - 1))
            fit_end = n

            x_fit = np.arange(fit_start, fit_end)
            y_fit = data[fit_start:fit_end].astype(np.float64)

            # 2次多項式フィッティング
            if len(x_fit) >= 3:
                coeffs = np.polyfit(x_fit, y_fit, 2)

                # 外挿
                for i in range(extrapolation_points):
                    x_extrap = n + i
                    extended_data[n + i] = np.polyval(coeffs, x_extrap)
            elif len(x_fit) >= 2:
                # フォールバック: 線形
                coeffs = np.polyfit(x_fit, y_fit, 1)
                for i in range(extrapolation_points):
                    x_extrap = n + i
                    extended_data[n + i] = np.polyval(coeffs, x_extrap)
            else:
                extended_data[n:] = data[-1]
        elif n >= 2:
            # 線形フォールバック
            slope = (data[-1] - data[-2])
            for i in range(extrapolation_points):
                extended_data[n + i] = data[-1] + slope * (i + 1)
        else:
            extended_data[n:] = data[-1]

        # カーネル関数
        def kernel_weight(distance: float, bandwidth: float) -> float:
            if bandwidth == 0:
                return 1.0 if distance == 0 else 0.0

            u = distance / bandwidth
            if abs(u) > 1:
                return 0.0

            if kernel == 'triangular':
                return 1 - abs(u)
            elif kernel == 'epanechnikov':
                return 0.75 * (1 - u**2)
            elif kernel == 'gaussian':
                return np.exp(-0.5 * u**2)
            return 1.0

        # スムージング
        smoothed = np.zeros(n, dtype=np.float64)
        smoothed[0] = float(data[0])

        for i in range(1, n):
            start = max(1, i - window)
            end = min(len(extended_data), i + window + 1)

            total_weight = 0.0
            weighted_sum = 0.0

            for idx in range(start, end):
                w = kernel_weight(float(abs(idx - i)), float(window))
                if w > 0:
                    total_weight += w
                    weighted_sum += w * extended_data[idx]

            smoothed[i] = weighted_sum / total_weight if total_weight > 0 else float(data[i])

        # 総計保存・整数化
        original_sum = int(np.sum(data))
        if abs(np.sum(smoothed[1:])) > 1e-10:
            scale = (original_sum - smoothed[0]) / np.sum(smoothed[1:])
            smoothed[1:] *= scale

        result = np.zeros(n, dtype=np.int64)
        result[0] = int(data[0])

        floors = np.floor(smoothed[1:]).astype(np.int64)
        result[1:] = floors

        remainders = smoothed[1:] - floors
        diff = original_sum - np.sum(result)

        if diff != 0:
            indices = np.argsort(remainders)[::-1] if diff > 0 else np.argsort(remainders)
            for i in range(min(abs(diff), len(indices))):
                idx = indices[i] + 1
                result[idx] += 1 if diff > 0 else -1
                if result[idx] < 0:
                    result[idx] = 0

        return result

    lkr_greville_quadratic_result = kernel_regression_with_greville_quadratic(test_data)
    print(f"\nLKR with Greville Quadratic Extrapolation result:")
    print(f"Original sum: {np.sum(test_data)}, Smoothed sum: {np.sum(lkr_greville_quadratic_result)}")
    fig_lkr_greville_quadratic = plot_individual_comparison(
        test_data,
        lkr_greville_quadratic_result,
        'Localized Kernel Regression with Greville Quadratic Extrapolation',
        'LKR with Greville Quadratic'
    )
    plt.show()
    return (
        kernel_regression_with_greville_quadratic,
        lkr_greville_quadratic_result,
    )


@app.cell(hide_code=True)
def _(np, plot_individual_comparison, plt, test_data):
    def kernel_regression_with_greville_mirror(
        data: np.ndarray, 
        window: int = 10, 
        kernel: str = 'triangular',
        mirror_length: int = None
    ) -> np.ndarray:
        n = len(data)

        if n <= 1:
            return data.copy().astype(np.int64)

        if mirror_length is None:
            mirror_length = min(window, n - 1)

        # 拡張データ配列（ミラーリング）
        extended_data = np.zeros(n + mirror_length, dtype=np.float64)
        extended_data[:n] = data.astype(np.float64)

        # 右端をミラーリング（最後の値を軸に反転）
        if n >= 2 and mirror_length > 0:
            anchor = data[-1]
            mirror_source = data[max(1, n - mirror_length - 1):n-1][::-1]  # 左端除外

            # アンカーからの差分を反転
            for i in range(min(mirror_length, len(mirror_source))):
                diff = mirror_source[i] - anchor
                extended_data[n + i] = anchor - diff

        def kernel_weight(distance: float, bandwidth: float) -> float:
            if bandwidth == 0:
                return 1.0 if distance == 0 else 0.0

            u = distance / bandwidth
            if abs(u) > 1:
                return 0.0

            if kernel == 'triangular':
                return 1 - abs(u)
            elif kernel == 'epanechnikov':
                return 0.75 * (1 - u**2)
            elif kernel == 'gaussian':
                return np.exp(-0.5 * u**2)
            return 1.0

        smoothed = np.zeros(n, dtype=np.float64)
        smoothed[0] = float(data[0])

        for i in range(1, n):
            start = max(1, i - window)
            end = min(len(extended_data), i + window + 1)

            total_weight = 0.0
            weighted_sum = 0.0

            for idx in range(start, end):
                w = kernel_weight(float(abs(idx - i)), float(window))
                if w > 0:
                    total_weight += w
                    weighted_sum += w * extended_data[idx]

            smoothed[i] = weighted_sum / total_weight if total_weight > 0 else float(data[i])

        original_sum = int(np.sum(data))
        if abs(np.sum(smoothed[1:])) > 1e-10:
            scale = (original_sum - smoothed[0]) / np.sum(smoothed[1:])
            smoothed[1:] *= scale

        result = np.zeros(n, dtype=np.int64)
        result[0] = int(data[0])

        floors = np.floor(smoothed[1:]).astype(np.int64)
        result[1:] = floors

        remainders = smoothed[1:] - floors
        diff = original_sum - np.sum(result)

        if diff != 0:
            indices = np.argsort(remainders)[::-1] if diff > 0 else np.argsort(remainders)
            for i in range(min(abs(diff), len(indices))):
                idx = indices[i] + 1
                result[idx] += 1 if diff > 0 else -1
                if result[idx] < 0:
                    result[idx] = 0

        return result

    lkr_greville_mirror_result = kernel_regression_with_greville_mirror(test_data)
    print(f"\nLKR with Greville Mirror Extrapolation result:")
    print(f"Original sum: {np.sum(test_data)}, Smoothed sum: {np.sum(lkr_greville_mirror_result)}")
    fig_lkr_greville_mirror = plot_individual_comparison(
        test_data,
        lkr_greville_mirror_result,
        'Localized Kernel Regression with Greville Mirror Extrapolation',
        'LKR with Greville Mirror'
    )
    plt.show()
    return kernel_regression_with_greville_mirror, lkr_greville_mirror_result


@app.cell(hide_code=True)
def _(
    kernel_regression_with_greville_linear,
    kernel_regression_with_greville_mirror,
    kernel_regression_with_greville_quadratic,
    lkr_result,
    lkr_sp_epanechnikov_result,
    lkr_sp_gaussian_result,
    lkr_sp_result,
    np,
    plt,
    test_data,
):
    def plot_all_approaches_comparison(
        original: np.ndarray,
        lkr: np.ndarray,
        lkr_sp: np.ndarray,
        lkr_sp_epanechnikov: np.ndarray,
        lkr_sp_gaussian: np.ndarray,
    ) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        x = np.arange(len(original))

        methods = [
            ('Original', original, 'steelblue'),
            ('Localized Kernel Regression', lkr, 'coral'),
            ('LKR Sum-Preserving Integer (Triangular)', lkr_sp, 'mediumseagreen'),
            ('LKR Sum-Preserving Integer (Epanechnikov)', lkr_sp_epanechnikov, 'orange'),
            ('LKR Sum-Preserving Integer (Gaussian)', lkr_sp_gaussian, 'purple'),
            ('LKR with Greville Linear', kernel_regression_with_greville_linear(original), 'red'),
            ('LKR with Greville Quadratic', kernel_regression_with_greville_quadratic(original), 'brown'),
            ('LKR with Greville Mirror', kernel_regression_with_greville_mirror(original), 'pink'),
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
        lkr_result,
        lkr_sp_result,
        lkr_sp_epanechnikov_result,
        lkr_sp_gaussian_result
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _(
    lkr_greville_mirror_result,
    lkr_greville_quadratic_result,
    lkr_greville_result,
    lkr_result,
    lkr_sp_epanechnikov_result,
    lkr_sp_gaussian_result,
    lkr_sp_result,
    np,
    test_data,
):
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
        ('Localized Kernel Regression', lkr_result),
        ('LKR Sum-Preserving Integer (Triangular)', lkr_sp_result),
        ('LKR Sum-Preserving Integer (Epanechnikov)', lkr_sp_epanechnikov_result),
        ('LKR Sum-Preserving Integer (Gaussian)', lkr_sp_gaussian_result),
        ('LKR with Greville Linear', lkr_greville_result),
        ('LKR with Greville Quadratic', lkr_greville_quadratic_result),
        ('LKR with Greville Mirror', lkr_greville_mirror_result)
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


@app.cell(hide_code=True)
def _(
    kernel_regression_with_greville_linear,
    kernel_regression_with_greville_mirror,
    kernel_regression_with_greville_quadratic,
    lkr_sum_preserve,
    np,
    plt,
):
    test_data_edge = np.array(
        [5, 7, 6, 8, 7, 9, 6, 8, 7, 10,
         6, 7, 8, 7, 9, 6, 8, 7, 10, 6,
         7, 8, 7, 9, 6, 8, 7, 10, 6, 7,
         8, 7, 9, 6, 8, 7, 10, 6, 7, 8,
         300, 350, 800, 400, 200, 150, 120, 110, 105, 100],
        dtype=np.int64
    )

    results = {
        'Original': test_data_edge,
        'No Greville': lkr_sum_preserve(test_data_edge, window=5),
        'Linear Extrap': kernel_regression_with_greville_linear(test_data_edge, window=5),
        'Quad Extrap': kernel_regression_with_greville_quadratic(test_data_edge, window=5),
        'Mirror': kernel_regression_with_greville_mirror(test_data_edge, window=5)
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    end_range = 15
    start_idx = len(test_data_edge) - end_range

    colors = ['black', 'blue', 'green', 'red', 'purple']
    markers = ['o', 's', '^', 'D', 'v']

    for i, (ed_name, ed_result) in enumerate(results.items()):
        ax.plot(
            range(start_idx, len(test_data_edge)),
            ed_result[start_idx:],
            marker=markers[i],
            linestyle='-',
            alpha=0.7,
            label=ed_name,
            markersize=6,
            linewidth=2,
            color=colors[i]
        )

    ax.axvline(x=42, color='red', linestyle='--', alpha=0.3, label='Spike Position')
    ax.set_title('Right-End Behavior Comparison (Last 15 Points)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('greville_right_end_detail.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nRight-End Values Comparison (Last 5 Points):")
    print(f"{'Method':<20s} | {'Values':<40s} | Sum")
    print("-" * 70)

    for ed_name, ed_result in results.items():
        vals = ed_result[-5:]
        print(f"{ed_name:<20s} | {str(list(vals)):<40s} | {np.sum(vals)}")
    return


@app.cell(hide_code=True)
def _(
    kernel_regression_with_greville_linear,
    kernel_regression_with_greville_mirror,
    kernel_regression_with_greville_quadratic,
    lkr_sum_preserve,
    np,
):
    def evaluate_smoothing_quality(original, smoothed, window):
        n = len(original)

        sum_error = abs(np.sum(smoothed) - np.sum(original))
        left_error = abs(smoothed[0] - original[0])

        if n > 2:
            smoothness = np.sum(np.abs(np.diff(smoothed, n=2)))
        else:
            smoothness = 0

        right_end_variance = np.var(smoothed[max(0, n-window):])

        rmse = np.sqrt(np.mean((smoothed - original)**2))

        return {
            'sum_error': sum_error,
            'left_error': left_error,
            'smoothness': smoothness,
            'right_end_var': right_end_variance,
            'rmse': rmse
        }

    # テスト
    test_data_evaluate = np.array([100] * 30 + [100, 120, 500, 130, 110] + [100] * 15, dtype=np.int64)

    methods = {
        'Standard': lambda d: lkr_sum_preserve(d, window=10),
        'Greville Linear': lambda d: kernel_regression_with_greville_linear(d, window=10),
        'Greville Quad': lambda d: kernel_regression_with_greville_quadratic(d, window=10),
        'Greville Mirror': lambda d: kernel_regression_with_greville_mirror(d, window=10)
    }

    print(f"{'Method':<20s} | {'Sum Err':>8s} | {'Left Err':>9s} | {'Smoothness':>11s} | {'RE Var':>8s} | {'RMSE':>8s}")
    print("-" * 90)

    for ev_name, func in methods.items():
        ev_result = func(test_data_evaluate)
        metrics = evaluate_smoothing_quality(test_data_evaluate, ev_result, 10)

        print(f"{ev_name:<20s} | {metrics['sum_error']:8d} | {metrics['left_error']:9d} | "
              f"{metrics['smoothness']:11.2f} | {metrics['right_end_var']:8.2f} | {metrics['rmse']:8.2f}")
    return


@app.cell
def _(mo, test_data):
    start_value_slider = mo.ui.slider(0, 200, value=test_data[0], label='Start Value')
    start_value_slider
    return (start_value_slider,)


@app.cell
def _(mo):
    window_size_slider = mo.ui.slider(1, 50, value=5, label='Window Size')
    window_size_slider
    return (window_size_slider,)


@app.cell(hide_code=True)
def _(start_value_slider, window_size_slider):
    start_value = start_value_slider.value
    window_size = window_size_slider.value
    return start_value, window_size


@app.cell(hide_code=True)
def _(
    Optional,
    np,
    plot_individual_comparison,
    plt,
    start_value,
    test_data,
    window_size,
):
    from typing import Literal

    def kernel_regression_greville_flexible_start(
        data: np.ndarray, 
        window: int = 10, 
        kernel: str = 'triangular',
        extrapolation_points: Optional[int] = None,
        start_value: Optional[int] = None,
        total_preservation: Literal['original', 'proportional'] = 'proportional'
    ) -> np.ndarray:
        n = len(data)

        if n <= 1:
            result = data.copy().astype(np.int64)
            if start_value is not None and n == 1:
                result[0] = int(start_value)
            return result

        if extrapolation_points is None:
            extrapolation_points = window

        if start_value is not None:
            left_value = float(start_value)
        else:
            left_value = float(data[0])

        extended_data = np.zeros(n + extrapolation_points, dtype=np.float64)
        extended_data[:n] = data.astype(np.float64)

        if n >= 2:
            fit_start = max(1, n - window)
            fit_end = n

            x_fit = np.arange(fit_start, fit_end)
            y_fit = data[fit_start:fit_end].astype(np.float64)

            if len(x_fit) > 1:
                coeffs = np.polyfit(x_fit, y_fit, 1)
                slope, intercept = coeffs[0], coeffs[1]

                for i in range(extrapolation_points):
                    x_extrap = n + i
                    extended_data[n + i] = slope * x_extrap + intercept
            else:
                extended_data[n:] = data[-1]

        def kernel_weight(distance: float, bandwidth: float) -> float:
            if bandwidth == 0:
                return 1.0 if distance == 0 else 0.0

            u = distance / bandwidth
            if abs(u) > 1:
                return 0.0

            if kernel == 'triangular':
                return 1 - abs(u)
            elif kernel == 'epanechnikov':
                return 0.75 * (1 - u**2)
            elif kernel == 'gaussian':
                return np.exp(-0.5 * u**2)
            return 1.0

        smoothed = np.zeros(n, dtype=np.float64)
        smoothed[0] = left_value

        for i in range(1, n):
            start = max(0, i - window)
            end = min(len(extended_data), i + window + 1)

            total_weight = 0.0
            weighted_sum = 0.0

            for idx in range(start, end):
                w = kernel_weight(float(abs(idx - i)), float(window))
                if w > 0:
                    value = left_value if idx == 0 else extended_data[idx]
                    total_weight += w
                    weighted_sum += w * value

            smoothed[i] = weighted_sum / total_weight if total_weight > 0 else float(data[i])

        original_sum = int(np.sum(data))


        if total_preservation == 'original':
            target_sum_without_first = original_sum - int(left_value)
            current_sum_without_first = np.sum(smoothed[1:])

            if abs(current_sum_without_first) > 1e-10:
                scale = target_sum_without_first / current_sum_without_first
                smoothed[1:] *= scale

        elif total_preservation == 'proportional':
            if start_value is not None:
                left_change = left_value - float(data[0])
                target_sum_rest = original_sum - int(left_value)
                current_sum_rest = np.sum(smoothed[1:])

                if abs(current_sum_rest) > 1e-10:
                    scale = target_sum_rest / current_sum_rest
                    smoothed[1:] *= scale
            else:
                if abs(np.sum(smoothed[1:])) > 1e-10:
                    scale = (original_sum - smoothed[0]) / np.sum(smoothed[1:])
                    smoothed[1:] *= scale

        result = np.zeros(n, dtype=np.int64)
        result[0] = int(left_value)

        floors = np.floor(smoothed[1:]).astype(np.int64)
        result[1:] = floors

        remainders = smoothed[1:] - floors

        current_sum = np.sum(result)
        diff = original_sum - current_sum

        if diff != 0:
            indices = np.argsort(remainders)[::-1] if diff > 0 else np.argsort(remainders)

            for i in range(min(abs(diff), len(indices))):
                idx = indices[i] + 1
                result[idx] += 1 if diff > 0 else -1
                if result[idx] < 0:
                    result[idx] = 0

        return result

    lkr_flexible_start_result_proportional = kernel_regression_greville_flexible_start(
        test_data,
        window=window_size,
        start_value=start_value,
        total_preservation='proportional'
    )

    print(f"\nLKR with Flexible Start Value (Proportional Total Preservation) result:")
    print(f"Original sum: {np.sum(test_data)}, Smoothed sum: {np.sum(lkr_flexible_start_result_proportional)}")

    p1 = plot_individual_comparison(
        test_data,
        lkr_flexible_start_result_proportional,
        'Localized Kernel Regression (Flexible Start, Proportional Total Preservation)',
        'LKR Flexible Start (Proportional)'
    )
    plt.show()

    lkr_flexible_start_result = kernel_regression_greville_flexible_start(
        test_data,
        window=window_size,
        start_value=start_value,
        total_preservation='original'
    )

    print(f"\nLKR with Flexible Start Value (Original Total Preservation) result:")
    print(f"Original sum: {np.sum(test_data)}, Smoothed sum: {np.sum(lkr_flexible_start_result)}")

    p2 = plot_individual_comparison(
        test_data,
        lkr_flexible_start_result,
        'Localized Kernel Regression (Flexible Start, Original Total Preservation)',
        'LKR Flexible Start (Original)'
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _(
    Optional,
    np,
    plot_individual_comparison,
    plt,
    start_value,
    test_data,
    window_size,
):
    def kernel_regression_gradual_transition(
        data: np.ndarray,
        window: int = 10,
        start_value: Optional[int] = None,
        transition_length: int = 3
    ) -> np.ndarray:
        n = len(data)

        if n <= 1:
            result = data.copy().astype(np.int64)
            if start_value is not None and n == 1:
                result[0] = int(start_value)
            return result

        if start_value is not None:
            left_value = float(start_value)
        else:
            left_value = float(data[0])

        actual_transition_length = min(transition_length, n - 1)

        # 右端のGreville外挿
        extrapolation_points = window
        extended_data = np.zeros(n + extrapolation_points, dtype=np.float64)
        extended_data[:n] = data.astype(np.float64)

        if n >= 2:
            fit_start = max(1, n - window)
            fit_end = n

            x_fit = np.arange(fit_start, fit_end)
            y_fit = data[fit_start:fit_end].astype(np.float64)

            if len(x_fit) > 1:
                coeffs = np.polyfit(x_fit, y_fit, 1)
                slope, intercept = coeffs[0], coeffs[1]

                for i in range(extrapolation_points):
                    x_extrap = n + i
                    extended_data[n + i] = slope * x_extrap + intercept
            else:
                extended_data[n:] = data[-1]

        def kernel_weight(distance: float, bandwidth: float) -> float:
            if bandwidth == 0:
                return 1.0 if distance == 0 else 0.0

            u = distance / bandwidth
            if abs(u) > 1:
                return 0.0

            return 1 - abs(u)

        smoothed = np.zeros(n, dtype=np.float64)
        smoothed[0] = left_value

        for i in range(1, min(actual_transition_length + 1, n)):
            # 遷移比率（0から1へ線形に変化）
            transition_ratio = i / (actual_transition_length + 1)

            start_idx = max(0, i - window)
            end_idx = min(len(extended_data), i + window + 1)

            total_weight = 0.0
            weighted_sum = 0.0

            for idx in range(start_idx, end_idx):
                w = kernel_weight(float(abs(idx - i)), float(window))
                if w > 0:
                    value = left_value if idx == 0 else extended_data[idx]
                    total_weight += w
                    weighted_sum += w * value

            kernel_estimate = weighted_sum / total_weight if total_weight > 0 else float(data[i])

            # 遷移: start_value → kernel_estimate
            # 非線形遷移 ease-out
            ease_ratio = 1 - (1 - transition_ratio) ** 2
            smoothed[i] = left_value * (1 - ease_ratio) + kernel_estimate * ease_ratio

        # 遷移区間以降は通常のカーネル回帰
        for i in range(actual_transition_length + 1, n):
            start_idx = max(0, i - window)
            end_idx = min(len(extended_data), i + window + 1)

            total_weight = 0.0
            weighted_sum = 0.0

            for idx in range(start_idx, end_idx):
                w = kernel_weight(float(abs(idx - i)), float(window))
                if w > 0:
                    value = smoothed[idx] if idx < i and idx <= actual_transition_length else extended_data[idx]
                    total_weight += w
                    weighted_sum += w * value

            smoothed[i] = weighted_sum / total_weight if total_weight > 0 else float(data[i])

        original_sum = int(np.sum(data))
        target_sum_rest = original_sum - int(left_value)
        current_sum_rest = np.sum(smoothed[1:])

        if abs(current_sum_rest) > 1e-10:
            scale = target_sum_rest / current_sum_rest
            smoothed[1:] *= scale

        result = np.zeros(n, dtype=np.int64)
        result[0] = int(left_value)

        floors = np.floor(smoothed[1:]).astype(np.int64)
        result[1:] = floors

        remainders = smoothed[1:] - floors
        diff = original_sum - np.sum(result)

        if diff != 0:
            indices = np.argsort(remainders)[::-1] if diff > 0 else np.argsort(remainders)

            for i in range(min(abs(diff), len(indices))):
                idx = indices[i] + 1
                result[idx] += 1 if diff > 0 else -1
                if result[idx] < 0:
                    result[idx] = 0

        return result

    lkr_gradual_transition_result = kernel_regression_gradual_transition(
        test_data,
        window=window_size,
        start_value=start_value,
        transition_length=5
    )
    print(f"\nLKR with Gradual Transition result:")
    print(f"Original sum: {np.sum(test_data)}, Smoothed sum: {np.sum(lkr_gradual_transition_result)}")
    fig_lkr_gradual_transition = plot_individual_comparison(
        test_data,
        lkr_gradual_transition_result,
        'Localized Kernel Regression with Gradual Transition',
        'LKR Gradual Transition'
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _(
    Optional,
    np,
    plot_individual_comparison,
    plt,
    start_value,
    test_data,
    window_size,
):
    def kernel_regression_gradual_transition_improve(
        data: np.ndarray,
        window: int = 10,
        start_value: Optional[int] = None,
        transition_length: int = 3,
        clamp_start_value: bool = True,
    ) -> np.ndarray:
        x = np.asarray(data)
        if x.ndim != 1:
            raise ValueError("data must be 1-D array")
        n = x.size
        if n == 0:
            return np.array([], dtype=np.int64)

        if np.any(x < 0):
            raise ValueError("data must be non-negative")

        x_int = x.astype(np.int64)
        total = int(x_int.sum())

        if n == 1:
            out0 = int(x_int[0]) if start_value is None else int(start_value)
            if clamp_start_value:
                out0 = int(np.clip(out0, 0, total))
            else:
                if not (0 <= out0 <= total):
                    raise ValueError("start_value out of feasible range")
            return np.array([out0], dtype=np.int64)

        if start_value is None:
            left = int(x_int[0])
        else:
            left = int(start_value)

        if clamp_start_value:
            left = int(np.clip(left, 0, total))
        else:
            if not (0 <= left <= total):
                raise ValueError(f"start_value={left} is out of feasible range [0, {total}]")

        # 端ケース：left==totalなら残りは全部0で確定
        if left == total:
            out = np.zeros(n, dtype=np.int64)
            out[0] = left
            return out

        # 右端外挿
        w = int(window)
        if w < 0:
            raise ValueError("window must be >= 0")

        extrap = w
        ext = np.zeros(n + extrap, dtype=np.float64)
        ext[:n] = x_int.astype(np.float64)

        if n >= 2 and extrap > 0:
            fit_start = max(1, n - w)
            fit_end = n
            xs = np.arange(fit_start, fit_end, dtype=np.float64)
            ys = x_int[fit_start:fit_end].astype(np.float64)
            if xs.size >= 2 and np.any(xs != xs[0]):
                slope, intercept = np.polyfit(xs, ys, 1)
                for i in range(extrap):
                    xx = float(n + i)
                    ext[n + i] = slope * xx + intercept
            else:
                ext[n:] = float(x_int[-1])
        elif extrap > 0:
            ext[n:] = float(x_int[-1])

        def tri_weight(dist: int, bandwidth: int) -> float:
            if bandwidth <= 0:
                return 1.0 if dist == 0 else 0.0
            if dist > bandwidth:
                return 0.0
            return 1.0 - (dist / float(bandwidth))

        sm = np.zeros(n, dtype=np.float64)
        sm[0] = float(left)

        trans_len = min(int(transition_length), n - 1)
        if trans_len < 0:
            trans_len = 0

        # ease-out
        for i in range(1, trans_len + 1):
            ratio = i / float(trans_len + 1)  # 0..1
            ease = 1.0 - (1.0 - ratio) ** 2   # ease-out

            start_idx = max(0, i - w)
            end_idx = min(ext.size, i + w + 1)

            tw = 0.0
            ws = 0.0
            for j in range(start_idx, end_idx):
                ww = tri_weight(abs(j - i), w)
                if ww <= 0:
                    continue
                val = float(left) if j == 0 else float(ext[j])
                tw += ww
                ws += ww * val

            ker = ws / tw if tw > 0 else float(x_int[i])
            sm[i] = float(left) * (1.0 - ease) + ker * ease

        # 通常のカーネル回帰
        for i in range(trans_len + 1, n):
            start_idx = max(0, i - w)
            end_idx = min(ext.size, i + w + 1)

            tw = 0.0
            ws = 0.0
            for j in range(start_idx, end_idx):
                ww = tri_weight(abs(j - i), w)
                if ww <= 0:
                    continue
                if j < i and j <= trans_len:
                    val = float(sm[j])
                else:
                    val = float(ext[j])
                tw += ww
                ws += ww * val

            sm[i] = ws / tw if tw > 0 else float(x_int[i])

        target_rest = float(total - left)
        rest = sm[1:].copy()
        rest_sum = float(rest.sum())

        if rest_sum > 1e-12:
            rest *= (target_rest / rest_sum)
        else:
            rest[:] = target_rest / float(n - 1)

        rest = np.maximum(rest, 0.0)

        out = np.zeros(n, dtype=np.int64)
        out[0] = left

        floors = np.floor(rest).astype(np.int64)
        floors = np.maximum(floors, 0)
        out[1:] = floors

        diff = total - int(out.sum())

        frac = rest - floors.astype(np.float64)

        if diff > 0:
            order = np.argsort(frac)[::-1]
            k = 0
            while diff > 0 and k < order.size:
                idx = int(order[k]) + 1
                out[idx] += 1
                diff -= 1
                k += 1
            k = 0
            while diff > 0:
                idx = int(order[k % order.size]) + 1
                out[idx] += 1
                diff -= 1
                k += 1

        elif diff < 0:
            order = np.argsort(frac)  # 小さい順
            k = 0
            need = -diff
            while need > 0:
                if k >= order.size:
                    raise RuntimeError("Cannot subtract further without going negative")
                idx = int(order[k]) + 1
                if out[idx] > 0:
                    out[idx] -= 1
                    need -= 1
                else:
                    k += 1
            diff = 0

        s = int(out.sum())
        if s != total:
            d = total - s
            if d > 0:
                for i in range(1, n):
                    if d == 0:
                        break
                    out[i] += 1
                    d -= 1
            elif d < 0:
                d = -d
                for i in range(1, n):
                    while d > 0 and out[i] > 0:
                        out[i] -= 1
                        d -= 1
            if int(out.sum()) != total:
                raise RuntimeError(f"sum mismatch: got {int(out.sum())}, expected {total}")

        return out

    lkr_gradual_transition_improved_result = kernel_regression_gradual_transition_improve(
        test_data,
        window=window_size,
        start_value=start_value,
        transition_length=5,
        clamp_start_value=True
    )
    print(f"\nLKR with Gradual Transition Improved result:")
    print(f"Original sum: {np.sum(test_data)}, Smoothed sum: {np.sum(lkr_gradual_transition_improved_result)}")
    fig_lkr_gradual_transition_improved = plot_individual_comparison(
        test_data,
        lkr_gradual_transition_improved_result,
        'Localized Kernel Regression with Gradual Transition (Improved)',
        'LKR Gradual Transition Improved'
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _(
    Optional,
    np,
    plot_individual_comparison,
    plt,
    start_value,
    test_data,
    window_size,
):
    import heapq

    def kernel_regression_gradual_transition_step1(
        data: np.ndarray,
        window: int = 10,
        start_value: Optional[int] = None,
        transition_length: int = 3,
        clamp_start_value: bool = True,
        rounding_smooth_weight: float = 0.25,  # Step1: diff配分時に局所ギザギザを嫌う重み
    ) -> np.ndarray:
        """
        kernel_regression_gradual_transition_improve の Step1 改良版:
          - 入力・start_value は整数のみ前提
          - window==0 は不正
          - 整数化(diff配分)を「局所コスト最小」で行い、形を壊しにくくする

        それ以外（右端外挿、カーネル回帰、総量保存の連続値スケール）は現状のまま維持。
        """

        x = np.asarray(data)
        if x.ndim != 1:
            raise ValueError("data must be 1-D array")
        n = x.size
        if n == 0:
            return np.array([], dtype=np.int64)

        # 整数前提（型が浮動でも内容が整数であることを要求する場合はここを厳密に）
        if not np.issubdtype(x.dtype, np.integer):
            # 「整数のみ前提」なので、浮動が来たらエラーにする（丸めない）
            if np.any(np.floor(x) != x):
                raise ValueError("data must contain integers only")
        if np.any(x < 0):
            raise ValueError("data must be non-negative")

        w = int(window)
        if w <= 0:
            raise ValueError("window must be >= 1")

        x_int = x.astype(np.int64)
        total = int(x_int.sum())

        if n == 1:
            out0 = int(x_int[0]) if start_value is None else int(start_value)
            if clamp_start_value:
                out0 = int(np.clip(out0, 0, total))
            else:
                if not (0 <= out0 <= total):
                    raise ValueError("start_value out of feasible range")
            return np.array([out0], dtype=np.int64)

        # ---- 左端値（必ず int） ----
        if start_value is None:
            left = int(x_int[0])
        else:
            left = int(start_value)

        if clamp_start_value:
            left = int(np.clip(left, 0, total))
        else:
            if not (0 <= left <= total):
                raise ValueError(f"start_value={left} is out of feasible range [0, {total}]")

        if left == total:
            out = np.zeros(n, dtype=np.int64)
            out[0] = left
            return out

        # ---- 右端外挿（現状維持：線形回帰で window 点ぶん外へ） ----
        extrap = w
        ext = np.zeros(n + extrap, dtype=np.float64)
        ext[:n] = x_int.astype(np.float64)

        if n >= 2 and extrap > 0:
            fit_start = max(1, n - w)
            fit_end = n
            xs = np.arange(fit_start, fit_end, dtype=np.float64)
            ys = x_int[fit_start:fit_end].astype(np.float64)
            if xs.size >= 2 and np.any(xs != xs[0]):
                slope, intercept = np.polyfit(xs, ys, 1)
                for i in range(extrap):
                    xx = float(n + i)
                    ext[n + i] = slope * xx + intercept
            else:
                ext[n:] = float(x_int[-1])
        elif extrap > 0:
            ext[n:] = float(x_int[-1])

        # ---- 三角カーネル ----
        def tri_weight(dist: int, bandwidth: int) -> float:
            # window>=1前提なので bandwidth<=0は使わないが、保険で残す
            if bandwidth <= 0:
                return 1.0 if dist == 0 else 0.0
            if dist > bandwidth:
                return 0.0
            return 1.0 - (dist / float(bandwidth))

        # ---- スムージング（連続値） ----
        sm = np.zeros(n, dtype=np.float64)
        sm[0] = float(left)

        trans_len = min(int(transition_length), n - 1)
        if trans_len < 0:
            trans_len = 0

        # i <= trans_len は start -> kernel の遷移（ease-out）
        for i in range(1, trans_len + 1):
            ratio = i / float(trans_len + 1)  # 0..1
            ease = 1.0 - (1.0 - ratio) ** 2   # ease-out

            start_idx = max(0, i - w)
            end_idx = min(ext.size, i + w + 1)

            tw = 0.0
            ws = 0.0
            for j in range(start_idx, end_idx):
                ww = tri_weight(abs(j - i), w)
                if ww <= 0:
                    continue
                val = float(left) if j == 0 else float(ext[j])
                tw += ww
                ws += ww * val

            ker = ws / tw if tw > 0 else float(x_int[i])
            sm[i] = float(left) * (1.0 - ease) + ker * ease

        # trans_len 以降は通常のカーネル回帰
        for i in range(trans_len + 1, n):
            start_idx = max(0, i - w)
            end_idx = min(ext.size, i + w + 1)

            tw = 0.0
            ws = 0.0
            for j in range(start_idx, end_idx):
                ww = tri_weight(abs(j - i), w)
                if ww <= 0:
                    continue
                if j < i and j <= trans_len:
                    val = float(sm[j])
                else:
                    val = float(ext[j])
                tw += ww
                ws += ww * val

            sm[i] = ws / tw if tw > 0 else float(x_int[i])

        # ---- 総計保存（連続値段階） ----
        target_rest = float(total - left)
        rest = sm[1:].copy()
        rest_sum = float(rest.sum())

        if rest_sum > 1e-12:
            rest *= (target_rest / rest_sum)
        else:
            rest[:] = target_rest / float(n - 1)

        rest = np.maximum(rest, 0.0)

        # ---- 整数化（Step1改良：局所コスト最小でdiff配分） ----
        out = np.zeros(n, dtype=np.int64)
        out[0] = left

        floors = np.floor(rest).astype(np.int64)
        floors = np.maximum(floors, 0)
        out[1:] = floors

        diff = total - int(out.sum())
        if diff == 0:
            return out

        # 目的： (out[i]-rest[i-1])^2 + rounding_smooth_weight * 局所差分二乗
        # をできるだけ増やさずに、合計を合わせる（+1/-1 を diff 回繰り返す）

        rest_full = np.zeros(n, dtype=np.float64)
        rest_full[0] = float(left)
        rest_full[1:] = rest

        lam = float(rounding_smooth_weight)

        def fit_delta(i: int, new_val: int) -> float:
            # i>=1のみ（0は固定）
            old = int(out[i])
            r_i = float(rest_full[i])
            # (new-r)^2 - (old-r)^2
            return (float(new_val) - r_i) ** 2 - (float(old) - r_i) ** 2

        def smooth_local_cost(i: int, val_i: int) -> float:
            # 差分二乗の局所寄与（i-1,i,i+1の境界分だけ）
            # sum (out[k]-out[k-1])^2 のうち、iが関係するのは
            # (i - (i-1)) と ((i+1) - i) の2つ（端は1つ）
            c = 0.0
            if i - 1 >= 0:
                c += float(val_i - int(out[i - 1])) ** 2
            if i + 1 < n:
                c += float(int(out[i + 1]) - val_i) ** 2
            return c

        def smooth_delta(i: int, new_val: int) -> float:
            old = int(out[i])
            before = smooth_local_cost(i, old)
            after = smooth_local_cost(i, new_val)
            # ただし、隣接ノード側の寄与も変わるので、
            # i-1 と i+1 のローカルコスト差分も加える（重複しないように注意）
            # ここでは「境界の二乗」を局所コストとしているので、
            # i の変更で影響する境界は (i-1,i) と (i,i+1) のみ → before/after で十分。
            return after - before

        # lazy heap: (marginal_cost, idx, version)
        ver = np.zeros(n, dtype=np.int64)

        def marginal_cost_add(i: int) -> float:
            new_val = int(out[i]) + 1
            return fit_delta(i, new_val) + lam * smooth_delta(i, new_val)

        def marginal_cost_sub(i: int) -> float:
            new_val = int(out[i]) - 1
            return fit_delta(i, new_val) + lam * smooth_delta(i, new_val)

        def push_candidate(heap, i: int, mode: str):
            # mode: "add" or "sub"
            if i <= 0:
                return  # out[0]固定
            if mode == "sub" and out[i] <= 0:
                return
            ver[i] += 1
            mc = marginal_cost_add(i) if mode == "add" else marginal_cost_sub(i)
            heapq.heappush(heap, (mc, i, ver[i]))

        def rebuild_heap(mode: str):
            heap = []
            for i in range(1, n):
                if mode == "sub" and out[i] <= 0:
                    continue
                push_candidate(heap, i, mode)
            return heap

        def pop_valid(heap, mode: str):
            while heap:
                mc, i, vv = heapq.heappop(heap)
                if vv != ver[i]:
                    continue
                if mode == "sub" and out[i] <= 0:
                    continue
                return mc, i
            return None

        if diff > 0:
            heap = rebuild_heap("add")
            for _ in range(diff):
                picked = pop_valid(heap, "add")
                if picked is None:
                    # 理論上ここには来ない（加算は常に可能）
                    break
                _, i = picked
                out[i] += 1
                # 影響するのは i と隣 (i-1,i+1)
                for j in (i - 1, i, i + 1):
                    if 1 <= j < n:
                        push_candidate(heap, j, "add")

        else:
            need = -diff
            heap = rebuild_heap("sub")
            for _ in range(need):
                picked = pop_valid(heap, "sub")
                if picked is None:
                    raise RuntimeError("Cannot subtract further without going negative")
                _, i = picked
                out[i] -= 1
                for j in (i - 1, i, i + 1):
                    if 1 <= j < n:
                        push_candidate(heap, j, "sub")

        # 最終チェック（必ず一致させる）
        s = int(out.sum())
        if s != total:
            raise RuntimeError(f"sum mismatch: got {s}, expected {total}")

        return out


    lkr_kernel_regression_gradual_transition_step1_result = kernel_regression_gradual_transition_step1(
        test_data,
        window=window_size,
        start_value=start_value,
        transition_length=5,
        clamp_start_value=True
    )
    print(f"\nLKR with Gradual Transition Step1 result:")
    print(f"Original sum: {np.sum(test_data)}, Smoothed sum: {np.sum(lkr_kernel_regression_gradual_transition_step1_result)}")
    fig_lkr_kernel_regression_gradual_transition_step1 = plot_individual_comparison(
        test_data,
        lkr_kernel_regression_gradual_transition_step1_result,
        'Localized Kernel Regression with Gradual Transition (Step1)',
        'LKR Gradual Transition Step1'
    )
    plt.show()
    return (heapq,)


@app.cell(hide_code=True)
def _(
    Optional,
    heapq,
    np,
    plot_individual_comparison,
    plt,
    start_value,
    test_data,
    window_size,
):
    # ---------- 共通：Step1の「局所コスト最小」diff配分（整数化） ----------
    def _round_with_local_cost(
        left: int,
        rest: np.ndarray,              # shape (n-1,) float, >=0 が望ましい
        total: int,
        rounding_smooth_weight: float,
    ) -> np.ndarray:
        n = rest.size + 1
        out = np.zeros(n, dtype=np.int64)
        out[0] = int(left)

        floors = np.floor(rest).astype(np.int64)
        floors = np.maximum(floors, 0)
        out[1:] = floors

        diff = int(total - out.sum())
        if diff == 0:
            return out

        rest_full = np.zeros(n, dtype=np.float64)
        rest_full[0] = float(left)
        rest_full[1:] = rest.astype(np.float64)

        lam = float(rounding_smooth_weight)

        def fit_delta(i: int, new_val: int) -> float:
            old = int(out[i])
            r_i = float(rest_full[i])
            return (float(new_val) - r_i) ** 2 - (float(old) - r_i) ** 2

        def smooth_local_cost(i: int, val_i: int) -> float:
            c = 0.0
            if i - 1 >= 0:
                c += float(val_i - int(out[i - 1])) ** 2
            if i + 1 < n:
                c += float(int(out[i + 1]) - val_i) ** 2
            return c

        def smooth_delta(i: int, new_val: int) -> float:
            old = int(out[i])
            before = smooth_local_cost(i, old)
            after = smooth_local_cost(i, new_val)
            return after - before

        ver = np.zeros(n, dtype=np.int64)

        def marginal_cost_add(i: int) -> float:
            new_val = int(out[i]) + 1
            return fit_delta(i, new_val) + lam * smooth_delta(i, new_val)

        def marginal_cost_sub(i: int) -> float:
            new_val = int(out[i]) - 1
            return fit_delta(i, new_val) + lam * smooth_delta(i, new_val)

        def push_candidate(heap, i: int, mode: str):
            if i <= 0:
                return  # out[0]固定
            if mode == "sub" and out[i] <= 0:
                return
            ver[i] += 1
            mc = marginal_cost_add(i) if mode == "add" else marginal_cost_sub(i)
            heapq.heappush(heap, (mc, i, ver[i]))

        def rebuild_heap(mode: str):
            heap = []
            for i in range(1, n):
                if mode == "sub" and out[i] <= 0:
                    continue
                push_candidate(heap, i, mode)
            return heap

        def pop_valid(heap, mode: str):
            while heap:
                mc, i, vv = heapq.heappop(heap)
                if vv != ver[i]:
                    continue
                if mode == "sub" and out[i] <= 0:
                    continue
                return mc, i
            return None

        if diff > 0:
            heap = rebuild_heap("add")
            for _ in range(diff):
                picked = pop_valid(heap, "add")
                if picked is None:
                    break
                _, i = picked
                out[i] += 1
                for j in (i - 1, i, i + 1):
                    if 1 <= j < n:
                        push_candidate(heap, j, "add")
        else:
            need = -diff
            heap = rebuild_heap("sub")
            for _ in range(need):
                picked = pop_valid(heap, "sub")
                if picked is None:
                    raise RuntimeError("Cannot subtract further without going negative")
                _, i = picked
                out[i] -= 1
                for j in (i - 1, i, i + 1):
                    if 1 <= j < n:
                        push_candidate(heap, j, "sub")

        s = int(out.sum())
        if s != int(total):
            raise RuntimeError(f"sum mismatch: got {s}, expected {int(total)}")
        return out


    # ---------- 共通：三角カーネル ----------
    def _tri_weight(dist: int, bandwidth: int) -> float:
        if bandwidth <= 0:
            return 1.0 if dist == 0 else 0.0
        if dist > bandwidth:
            return 0.0
        return 1.0 - (dist / float(bandwidth))


    # ---------- 共通：スムージング本体（外挿 ext を受け取る） ----------
    def _kernel_regression_with_transition(
        x_int: np.ndarray,
        ext: np.ndarray,
        w: int,
        left: int,
        transition_length: int,
    ) -> np.ndarray:
        n = x_int.size
        sm = np.zeros(n, dtype=np.float64)
        sm[0] = float(left)

        trans_len = min(int(transition_length), n - 1)
        if trans_len < 0:
            trans_len = 0

        for i in range(1, trans_len + 1):
            ratio = i / float(trans_len + 1)
            ease = 1.0 - (1.0 - ratio) ** 2  # ease-out

            start_idx = max(0, i - w)
            end_idx = min(ext.size, i + w + 1)

            tw = 0.0
            ws = 0.0
            for j in range(start_idx, end_idx):
                ww = _tri_weight(abs(j - i), w)
                if ww <= 0:
                    continue
                val = float(left) if j == 0 else float(ext[j])
                tw += ww
                ws += ww * val

            ker = ws / tw if tw > 0 else float(x_int[i])
            sm[i] = float(left) * (1.0 - ease) + ker * ease

        for i in range(trans_len + 1, n):
            start_idx = max(0, i - w)
            end_idx = min(ext.size, i + w + 1)

            tw = 0.0
            ws = 0.0
            for j in range(start_idx, end_idx):
                ww = _tri_weight(abs(j - i), w)
                if ww <= 0:
                    continue
                if j < i and j <= trans_len:
                    val = float(sm[j])
                else:
                    val = float(ext[j])
                tw += ww
                ws += ww * val

            sm[i] = ws / tw if tw > 0 else float(x_int[i])

        return sm


    # ============================================================
    # Step2：総量保存（スケール）を「時間重みつき」にして谷を抑える
    # ============================================================
    def kernel_regression_gradual_transition_step2(
        data: np.ndarray,
        window: int = 10,
        start_value: Optional[int] = None,
        transition_length: int = 3,
        clamp_start_value: bool = True,
        rounding_smooth_weight: float = 0.25,
        # Step2追加：残り総量の合わせ方（前半は変えにくく、後半で帳尻を合わせる）
        scale_tau: float = 3.0,          # 小さいほど「早めにスケールを効かせる」。大きいほど後半寄り。
        scale_power: float = 1.0,        # alphaの形状調整（>1でより後半寄り）
    ) -> np.ndarray:
        """
        Step2まで：
          - 整数・総量一致（保証）
          - window==0は不正
          - Step1：diff配分を局所コスト最小化（形を壊しにくい）
          - Step2：総量保存のスケールを時間重みつきに（start_valueがズレても急激な谷を作りにくい）
        """
        x = np.asarray(data)
        if x.ndim != 1:
            raise ValueError("data must be 1-D array")
        n = x.size
        if n == 0:
            return np.array([], dtype=np.int64)

        # 整数のみ前提：値が整数でなければエラー（丸めない）
        if not np.issubdtype(x.dtype, np.integer):
            if np.any(np.floor(x) != x):
                raise ValueError("data must contain integers only")
        if np.any(x < 0):
            raise ValueError("data must be non-negative")

        w = int(window)
        if w <= 0:
            raise ValueError("window must be >= 1")

        x_int = x.astype(np.int64)
        total = int(x_int.sum())

        if n == 1:
            out0 = int(x_int[0]) if start_value is None else int(start_value)
            if clamp_start_value:
                out0 = int(np.clip(out0, 0, total))
            else:
                if not (0 <= out0 <= total):
                    raise ValueError("start_value out of feasible range")
            return np.array([out0], dtype=np.int64)

        left = int(x_int[0]) if start_value is None else int(start_value)
        if clamp_start_value:
            left = int(np.clip(left, 0, total))
        else:
            if not (0 <= left <= total):
                raise ValueError(f"start_value={left} is out of feasible range [0, {total}]")

        if left == total:
            out = np.zeros(n, dtype=np.int64)
            out[0] = left
            return out

        # ---- 右端外挿（Step2では現状維持：polyfit線形）----
        extrap = w
        ext = np.zeros(n + extrap, dtype=np.float64)
        ext[:n] = x_int.astype(np.float64)

        if n >= 2 and extrap > 0:
            fit_start = max(1, n - w)
            xs = np.arange(fit_start, n, dtype=np.float64)
            ys = x_int[fit_start:n].astype(np.float64)
            if xs.size >= 2 and np.any(xs != xs[0]):
                slope, intercept = np.polyfit(xs, ys, 1)
                for i in range(extrap):
                    ext[n + i] = slope * float(n + i) + intercept
            else:
                ext[n:] = float(x_int[-1])
        elif extrap > 0:
            ext[n:] = float(x_int[-1])

        # ---- スムージング（連続値）----
        sm = _kernel_regression_with_transition(
            x_int=x_int,
            ext=ext,
            w=w,
            left=left,
            transition_length=transition_length,
        )

        # ---- Step2：総量保存（時間重みつきスケール）----
        target_rest = float(total - left)
        rest_raw = sm[1:].astype(np.float64)

        # 基本は非負を期待するが、念のため
        rest_raw = np.maximum(rest_raw, 0.0)

        # alpha(i): 前半は小さく（=ほぼ維持）、後半ほど1に近づく（=帳尻を強く反映）
        # i は 1..n-1 の位置
        idx = np.arange(1, n, dtype=np.float64)
        tau = float(max(scale_tau, 1e-6))
        alpha = 1.0 - np.exp(-idx / tau)
        alpha = np.clip(alpha, 0.0, 1.0)
        if scale_power != 1.0:
            alpha = alpha ** float(scale_power)

        # rest_adj = rest_raw * ((1-alpha) + alpha*scale)
        # ここで「合計が target_rest になるscale」を線形に解く：
        base = float(np.sum(rest_raw * (1.0 - alpha)))
        den = float(np.sum(rest_raw * alpha))

        if den > 1e-12:
            scale = (target_rest - base) / den
        else:
            # alphaがほぼ0（=全部維持）なら、普通の一括スケールにフォールバック
            rs = float(rest_raw.sum())
            scale = (target_rest / rs) if rs > 1e-12 else 0.0

        rest_adj = rest_raw * ((1.0 - alpha) + alpha * scale)

        # 非負化（これで合計がズレる可能性があるので、後で補正）
        rest_adj = np.maximum(rest_adj, 0.0)

        # 合計補正：残った誤差を「後半（alpha大）ほど優先」で配る（ロバスト重視）
        cur = float(rest_adj.sum())
        err = target_rest - cur
        if abs(err) > 1e-9:
            weights = alpha.copy()
            ws = float(weights.sum())
            if ws <= 1e-12:
                # 万一：均等
                weights[:] = 1.0
                ws = float(weights.sum())
            # 正なら足す、負なら引く
            delta = (err / ws) * weights
            rest_adj = rest_adj + delta
            # 引き過ぎで負になるのを防ぐ：0でクリップし、再度残差を回収（最大数回で収束）
            for _ in range(3):
                rest_adj = np.maximum(rest_adj, 0.0)
                cur2 = float(rest_adj.sum())
                err2 = target_rest - cur2
                if abs(err2) <= 1e-9:
                    break
                # 足す側のみで回収（負にしたくないので）
                pos = rest_adj > 0.0
                if not np.any(pos):
                    # 全部0なら均等
                    rest_adj[:] = target_rest / float(n - 1)
                    break
                w2 = weights * pos.astype(np.float64)
                ws2 = float(w2.sum())
                if ws2 <= 1e-12:
                    # 均等に
                    rest_adj[pos] += err2 / float(np.sum(pos))
                else:
                    rest_adj += (err2 / ws2) * w2

        # ---- Step1：局所コスト最小の整数化で合計一致 ----
        out = _round_with_local_cost(
            left=left,
            rest=rest_adj,
            total=total,
            rounding_smooth_weight=rounding_smooth_weight,
        )
        return out


    # ============================================================
    # Step3：右端外挿をロバスト化（暴れを抑える）
    # ============================================================
    def kernel_regression_gradual_transition_step3(
        data: np.ndarray,
        window: int = 10,
        start_value: Optional[int] = None,
        transition_length: int = 3,
        clamp_start_value: bool = True,
        rounding_smooth_weight: float = 0.25,
        scale_tau: float = 3.0,
        scale_power: float = 1.0,
        # Step3追加：外挿のロバスト化
        extrap_max: int = 5,            # 外挿点数の上限（windowより小さくする）
        slope_clip_factor: float = 2.0, # 傾きのクリップ（最近の差分レンジに対する倍率）
        slope_window: int = 5,          # 傾き推定に使う末尾点数（Theil-Sen近似）
    ) -> np.ndarray:
        """
        Step3まで：
          - Step2の時間重みつき総量保存
          - Step1の局所コスト最小の整数化
          - Step3：右端外挿をロバスト化（末尾の暴れで形が崩れるのを抑える）
        """
        x = np.asarray(data)
        if x.ndim != 1:
            raise ValueError("data must be 1-D array")
        n = x.size
        if n == 0:
            return np.array([], dtype=np.int64)

        if not np.issubdtype(x.dtype, np.integer):
            if np.any(np.floor(x) != x):
                raise ValueError("data must contain integers only")
        if np.any(x < 0):
            raise ValueError("data must be non-negative")

        w = int(window)
        if w <= 0:
            raise ValueError("window must be >= 1")

        x_int = x.astype(np.int64)
        total = int(x_int.sum())

        if n == 1:
            out0 = int(x_int[0]) if start_value is None else int(start_value)
            if clamp_start_value:
                out0 = int(np.clip(out0, 0, total))
            else:
                if not (0 <= out0 <= total):
                    raise ValueError("start_value out of feasible range")
            return np.array([out0], dtype=np.int64)

        left = int(x_int[0]) if start_value is None else int(start_value)
        if clamp_start_value:
            left = int(np.clip(left, 0, total))
        else:
            if not (0 <= left <= total):
                raise ValueError(f"start_value={left} is out of feasible range [0, {total}]")

        if left == total:
            out = np.zeros(n, dtype=np.int64)
            out[0] = left
            return out

        # ---- Step3：右端外挿（ロバスト）----
        extrap = int(min(w, max(int(extrap_max), 0)))
        ext = np.zeros(n + extrap, dtype=np.float64)
        ext[:n] = x_int.astype(np.float64)

        if extrap > 0 and n >= 2:
            k = int(min(max(int(slope_window), 2), n))
            # 末尾k点（インデックスは n-k .. n-1）
            xs = np.arange(n - k, n, dtype=np.float64)
            ys = x_int[n - k:n].astype(np.float64)

            # Theil–Sen 近似：ペア差分の中央値傾き（ロバスト）
            slopes = []
            for i in range(k):
                for j in range(i + 1, k):
                    dx = xs[j] - xs[i]
                    if dx != 0:
                        slopes.append((ys[j] - ys[i]) / dx)
            if len(slopes) > 0:
                slope = float(np.median(np.array(slopes, dtype=np.float64)))
            else:
                slope = 0.0

            # intercept は median(y - slope*x)
            intercept = float(np.median(ys - slope * xs))

            # 傾きクリップ：最近の差分のレンジに基づく
            diffs = np.diff(ys)
            if diffs.size > 0:
                ref = float(np.median(np.abs(diffs)))  # 代表的な1ステップ変化量
                max_slope = float(slope_clip_factor) * max(ref, 1e-9)
                slope = float(np.clip(slope, -max_slope, max_slope))

            for i in range(extrap):
                xx = float(n + i)
                ext[n + i] = slope * xx + intercept
        elif extrap > 0:
            ext[n:] = float(x_int[-1])

        # ---- スムージング（連続値）----
        sm = _kernel_regression_with_transition(
            x_int=x_int,
            ext=ext,
            w=w,
            left=left,
            transition_length=transition_length,
        )

        # ---- Step2：総量保存（時間重みつきスケール）----
        target_rest = float(total - left)
        rest_raw = np.maximum(sm[1:].astype(np.float64), 0.0)

        idx = np.arange(1, n, dtype=np.float64)
        tau = float(max(scale_tau, 1e-6))
        alpha = 1.0 - np.exp(-idx / tau)
        alpha = np.clip(alpha, 0.0, 1.0)
        if scale_power != 1.0:
            alpha = alpha ** float(scale_power)

        base = float(np.sum(rest_raw * (1.0 - alpha)))
        den = float(np.sum(rest_raw * alpha))
        if den > 1e-12:
            scale = (target_rest - base) / den
        else:
            rs = float(rest_raw.sum())
            scale = (target_rest / rs) if rs > 1e-12 else 0.0

        rest_adj = rest_raw * ((1.0 - alpha) + alpha * scale)
        rest_adj = np.maximum(rest_adj, 0.0)

        cur = float(rest_adj.sum())
        err = target_rest - cur
        if abs(err) > 1e-9:
            weights = alpha.copy()
            ws = float(weights.sum())
            if ws <= 1e-12:
                weights[:] = 1.0
                ws = float(weights.sum())
            delta = (err / ws) * weights
            rest_adj = rest_adj + delta
            for _ in range(3):
                rest_adj = np.maximum(rest_adj, 0.0)
                cur2 = float(rest_adj.sum())
                err2 = target_rest - cur2
                if abs(err2) <= 1e-9:
                    break
                pos = rest_adj > 0.0
                if not np.any(pos):
                    rest_adj[:] = target_rest / float(n - 1)
                    break
                w2 = weights * pos.astype(np.float64)
                ws2 = float(w2.sum())
                if ws2 <= 1e-12:
                    rest_adj[pos] += err2 / float(np.sum(pos))
                else:
                    rest_adj += (err2 / ws2) * w2

        # ---- Step1：局所コスト最小の整数化で合計一致 ----
        out = _round_with_local_cost(
            left=left,
            rest=rest_adj,
            total=total,
            rounding_smooth_weight=rounding_smooth_weight,
        )
        return out

    lkr_gradual_transition_step2_result = kernel_regression_gradual_transition_step2(
        test_data,
        window=window_size,
        start_value=start_value,
        transition_length=5,
        clamp_start_value=True,
        rounding_smooth_weight=0.25,
        scale_tau=3.0,
        scale_power=1.0
    )
    print(f"\nLKR with Gradual Transition Step2 result:")
    print(f"Original sum: {np.sum(test_data)}, Smoothed sum: {np.sum(lkr_gradual_transition_step2_result)}")
    fig_lkr_gradual_transition_step2 = plot_individual_comparison(
        test_data,
        lkr_gradual_transition_step2_result,
        'Localized Kernel Regression with Gradual Transition (Step2)',
        'LKR Gradual Transition Step2'
    )
    plt.show()

    lkr_gradual_transition_step3_result = kernel_regression_gradual_transition_step3(
        test_data,
        window=window_size,
        start_value=start_value,
        transition_length=5,
        extrap_max=5,
        slope_clip_factor=2.0,
        slope_window=5
    )
    print(f"\nLKR with Gradual Transition Step3 result:")
    print(f"Original sum: {np.sum(test_data)}, Smoothed sum: {np.sum(lkr_gradual_transition_step3_result)}")
    fig_lkr_gradual_transition_step3 = plot_individual_comparison(
        test_data,
        lkr_gradual_transition_step3_result,
        'Localized Kernel Regression with Gradual Transition (Step3)',
        'LKR Gradual Transition Step3'
    )
    plt.show()
    return (kernel_regression_gradual_transition_step3,)


@app.cell(hide_code=True)
def _(
    Optional,
    Tuple,
    kernel_regression_gradual_transition_step3,
    np,
    plot_individual_comparison,
    plt,
    start_value,
    test_data,
    window_size,
):
    import warnings
    def _check_feasible_prefix_shift(values: np.ndarray, target: np.ndarray, r: int) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        ±r 分のスライドで target が実現可能か（必要十分条件：prefix上下限）をチェック。
        戻り値: (feasible, L, U)
          L[t] = V[t-r], U[t] = V[t+r]  (端はクリップ)
        """
        v = np.asarray(values, dtype=np.int64)
        y = np.asarray(target, dtype=np.int64)
        if v.ndim != 1 or y.ndim != 1 or v.size != y.size:
            raise ValueError("values and target must be 1-D arrays with same length")

        if np.any(v < 0) or np.any(y < 0):
            return False, np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        n = v.size
        V = np.cumsum(v)
        Y = np.cumsum(y)

        def V_at(idx: int) -> int:
            if idx < 0:
                return 0
            if idx >= n - 1:
                return int(V[-1])
            return int(V[idx])

        rr = int(r)
        L = np.zeros(n, dtype=np.int64)
        U = np.zeros(n, dtype=np.int64)
        for t in range(n):
            L[t] = V_at(t - rr)
            U[t] = V_at(t + rr)

        feasible = (int(y.sum()) == int(v.sum())) and np.all(Y >= L) and np.all(Y <= U)
        return feasible, L, U


    def kernel_regression_gradual_transition_step4(
        data: np.ndarray,
        window: int = 10,
        start_value: Optional[int] = None,
        transition_length: int = 3,
        clamp_start_value: bool = True,
        rounding_smooth_weight: float = 0.25,
        scale_tau: float = 3.0,
        scale_power: float = 1.0,
        extrap_max: int = 5,
        slope_clip_factor: float = 2.0,
        slope_window: int = 5,
        # Step4追加：±r実現可能性チェック（警告のみ）
        slide_radius: int = 5,
        warn_on_infeasible: bool = True,
    ) -> np.ndarray:
        """
        Step4まで：
          - Step3までのスムージング（整数・総量一致）
          - Step4：±slide_radius のスライドで実現可能かを prefix 必要十分条件でチェック
                  実現不可能でも例外は投げず、警告に留める

        注意:
          - ここでの「実現可能性」は、"各通知が元の分から±r以内に移動できる" を
            ヒストグラムとして満たせるか、という意味での必要十分条件（prefix制約）です。
          - 実際の割当（どの通知をどこへ動かすか）は別途構築が必要ですが、
            この条件を満たすなら必ず割当可能です。
        """
        out = kernel_regression_gradual_transition_step3(
            data=data,
            window=window,
            start_value=start_value,
            transition_length=transition_length,
            clamp_start_value=clamp_start_value,
            rounding_smooth_weight=rounding_smooth_weight,
            scale_tau=scale_tau,
            scale_power=scale_power,
            extrap_max=extrap_max,
            slope_clip_factor=slope_clip_factor,
            slope_window=slope_window,
        )

        feasible, L, U = _check_feasible_prefix_shift(
            values=np.asarray(data, dtype=np.int64),
            target=out,
            r=int(slide_radius),
        )

        if (not feasible) and warn_on_infeasible:
            v = np.asarray(data, dtype=np.int64)
            y = np.asarray(out, dtype=np.int64)
            n = v.size
            V = np.cumsum(v)
            Y = np.cumsum(y)

            # どの時刻で破っているかを特定
            viol_low = np.where(Y < L)[0]
            viol_up = np.where(Y > U)[0]

            parts = []
            parts.append(
                f"Target histogram may be infeasible under ±{int(slide_radius)} minute sliding constraints "
                f"(prefix necessary/sufficient check failed)."
            )

            if viol_low.size > 0:
                t0 = int(viol_low[0])
                parts.append(
                    f"First lower-bound violation at t={t0}: prefix={int(Y[t0])}, "
                    f"lower={int(L[t0])} (needs +{int(L[t0]-Y[t0])})."
                )
            if viol_up.size > 0:
                t1 = int(viol_up[0])
                parts.append(
                    f"First upper-bound violation at t={t1}: prefix={int(Y[t1])}, "
                    f"upper={int(U[t1])} (needs -{int(Y[t1]-U[t1])})."
                )

            # 追加のヒント：start_valueが強すぎる典型パターン
            if start_value is not None:
                sv = int(start_value)
                # t=0の上限: V[r]
                rr = int(slide_radius)
                upper0 = int(V[min(rr, n - 1)]) if n > 0 else 0
                if sv > upper0:
                    parts.append(
                        f"Hint: start_value={sv} exceeds feasible immediate supply V[{rr}]={upper0}. "
                        f"Even the first minute may not be achievable within ±{rr}."
                    )

            warnings.warn(" ".join(parts), RuntimeWarning)

        return out

    lkr_gradual_transition_step4_result = kernel_regression_gradual_transition_step4(
        test_data,
        window=window_size,
        start_value=start_value,
        transition_length=5,
        extrap_max=5,
        slope_clip_factor=2.0,
        slope_window=5,
        slide_radius=5,
        warn_on_infeasible=True
    )
    print(f"\nLKR with Gradual Transition Step4 result:")
    print(f"Original sum: {np.sum(test_data)}, Smoothed sum: {np.sum(lkr_gradual_transition_step4_result)}")
    fig_lkr_gradual_transition_step4 = plot_individual_comparison(
        test_data,
        lkr_gradual_transition_step4_result,
        'Localized Kernel Regression with Gradual Transition (Step4)',
        'LKR Gradual Transition Step4'
    )
    plt.show()
    return


if __name__ == "__main__":
    app.run()

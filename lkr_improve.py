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
    return Optional, np, plt, test_data


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

        # カーネル関数の定義
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
    def lkr_sum_preserve(data: np.ndarray, window: int = 10, kernel: str = 'triangular') -> np.ndarray:
        """
        局所的カーネル回帰（総計保存・整数出力版）
        - 指定ウィンドウ内でカーネル重み付け平均
        - 左端固定、総計厳密保存
        - 整数出力
        """
        n = len(data)

        if n <= 1:
            return data.copy().astype(np.int64)

        # カーネル関数の定義
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

        # 浮動小数点でスムージング
        smoothed = np.zeros(n, dtype=np.float64)
        smoothed[0] = float(data[0])  # 左端固定

        bandwidth = float(window)

        for i in range(1, n):
            # ウィンドウ範囲（左端を除外）
            start = max(1, i - window)
            end = min(n, i + window + 1)

            weights = []
            values = []

            # 各点の重みを計算
            for idx in range(start, end):
                distance = abs(idx - i)
                weight = get_kernel_weight(float(distance), bandwidth, kernel)
                if weight > 0:
                    weights.append(weight)
                    values.append(float(data[idx]))

            # 重み付き平均
            if len(weights) > 0 and sum(weights) > 0:
                smoothed[i] = sum(w * v for w, v in zip(weights, values)) / sum(weights)
            else:
                smoothed[i] = float(data[i])

        # 総計保存のための調整（浮動小数点）
        original_sum = int(np.sum(data))
        smoothed_sum = np.sum(smoothed)

        adjusted = smoothed.copy()

        # 左端以外を比例スケール
        if abs(smoothed_sum - smoothed[0]) > 1e-10:
            target_sum_without_first = original_sum - smoothed[0]
            current_sum_without_first = smoothed_sum - smoothed[0]

            if current_sum_without_first > 0:
                scale_factor = target_sum_without_first / current_sum_without_first
                adjusted[1:] = smoothed[1:] * scale_factor

        # 整数化（四捨五入）
        result = np.round(adjusted).astype(np.int64)
        result[0] = int(data[0])  # 左端は元の値を厳密に保持

        # 丸め誤差による総計のずれを修正
        current_total = np.sum(result)
        diff = original_sum - current_total

        if diff != 0:
            # 左端以外で調整
            # 差分を最も値が大きい要素から順に配分
            indices = np.argsort(result[1:])[::-1] + 1  # 降順、左端除外

            # 差分を1ずつ配分
            for i in range(abs(diff)):
                idx = indices[i % len(indices)]
                if diff > 0:
                    result[idx] += 1
                else:
                    result[idx] -= 1
                    # 負にならないようにチェック
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
        """
        Greville補正付き局所カーネル回帰（ミラーリング版）

        右端付近のデータを鏡映して仮想点を生成
        """
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
    window_size_slider = mo.ui.slider(1, 50, value=10, label='Window Size')
    window_size_slider
    return (window_size_slider,)


@app.cell(hide_code=True)
def _(
    Optional,
    np,
    plot_individual_comparison,
    plt,
    start_value_slider,
    test_data,
    window_size_slider,
):
    def kernel_regression_greville_flexible_start(
        data: np.ndarray, 
        window: int = 10, 
        kernel: str = 'triangular',
        extrapolation_points: Optional[int] = None,
        start_value: Optional[int] = None,
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
    
        target_sum_without_first = original_sum - int(left_value)
        current_sum_without_first = np.sum(smoothed[1:])
    
        if abs(current_sum_without_first) > 1e-10:
            scale = target_sum_without_first / current_sum_without_first
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


    start_value = start_value_slider.value
    window_size = window_size_slider.value

    lkr_flexible_start_result = kernel_regression_greville_flexible_start(
        test_data, window=window_size, start_value=start_value
    )
    print(f"\nLKR with Flexible Start Value result:")
    print(f"Original sum: {np.sum(test_data)}, Smoothed sum: {np.sum(lkr_flexible_start_result)}")
    fig_lkr_flexible_start = plot_individual_comparison(
        test_data,
        lkr_flexible_start_result,
        'Localized Kernel Regression with Flexible Start Value',
        'LKR with Flexible Start'
    )
    plt.show()
    return


if __name__ == "__main__":
    app.run()

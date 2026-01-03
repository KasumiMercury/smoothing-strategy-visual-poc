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


@app.cell
def _(Optional, Tuple, np):
    import warnings
    from typing import Dict, Any

    def _prefix_bounds_from_hist(x: np.ndarray, r: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        スライドで実現可能な累積和の上下限
          L[t] = sum_{k<=t-r} x[k]
          U[t] = sum_{k<=t+r} x[k]
        """
        x = np.asarray(x, dtype=np.int64)
        n = x.size
        V = np.cumsum(x)

        def V_at(idx: int) -> int:
            if idx < 0:
                return 0
            if idx >= n - 1:
                return int(V[-1])
            return int(V[idx])

        L = np.empty(n, dtype=np.float64)
        U = np.empty(n, dtype=np.float64)
        for t in range(n):
            L[t] = float(V_at(t - r))
            U[t] = float(V_at(t + r))
        return L, U


    def _affine_project_sum_and_anchor(y: np.ndarray, total: float, left: float) -> np.ndarray:
        """
        y[0]=left, sum(y)=total を満たす最近点射影
        """
        y = np.asarray(y, dtype=np.float64).copy()
        n = y.size
        y[0] = left
        if n == 1:
            y[0] = total
            return y

        # 残りの合計を合わせる
        rest_target = total - left
        rest_sum = float(np.sum(y[1:]))
        delta = (rest_target - rest_sum) / float(n - 1)
        y[1:] += delta
        return y


    def _bounded_isotonic_regression(z: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """
        PAVA
          minimize ||y - z||^2
          yは減少しない
        """
        z = np.asarray(z, dtype=np.float64)
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        n = z.size
        if lo.size != n or hi.size != n:
            raise ValueError("lo/hi must have same length as z")

        values = np.clip(z, lo, hi).copy()
        w = np.ones(n, dtype=np.float64)

        starts = list(range(n))
        ends = list(range(n))
        vals = values.tolist()
        ws = w.tolist()

        i = 0
        while i < len(vals) - 1:
            if vals[i] <= vals[i + 1] + 1e-15:
                i += 1
                continue

            s = starts[i]
            e = ends[i + 1]
            wsum = ws[i] + ws[i + 1]
            vavg = (vals[i] * ws[i] + vals[i + 1] * ws[i + 1]) / wsum

            # 区間内lo/hi
            block_lo = float(np.max(lo[s:e + 1]))
            block_hi = float(np.min(hi[s:e + 1]))
            if block_lo > block_hi:
                mid = 0.5 * (block_lo + block_hi)
                vavg = mid
            else:
                vavg = float(np.clip(vavg, block_lo, block_hi))

            starts[i] = s
            ends[i] = e
            vals[i] = vavg
            ws[i] = wsum

            del starts[i + 1], ends[i + 1], vals[i + 1], ws[i + 1]

            if i > 0:
                i -= 1

        y = np.empty(n, dtype=np.float64)
        for s, e, v in zip(starts, ends, vals):
            y[s:e + 1] = v

        y = np.clip(y, lo, hi)

        for k in range(1, n):
            if y[k] < y[k - 1]:
                y[k] = y[k - 1]
        return y


    def _project_prefix_band(y: np.ndarray, L: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        累積和制限への射影
        """
        y = np.asarray(y, dtype=np.float64)
        Y = np.cumsum(y)
        Yp = _bounded_isotonic_regression(Y, L, U)
        yp = np.empty_like(y)
        yp[0] = Yp[0]
        yp[1:] = np.diff(Yp)
        return yp


    def _check_feasible_prefix(values: np.ndarray, target: np.ndarray, r: int) -> bool:
        v = np.asarray(values, dtype=np.int64)
        y = np.asarray(target, dtype=np.int64)
        if v.size != y.size:
            return False
        L, U = _prefix_bounds_from_hist(v, r)
        V = np.cumsum(v)
        Y = np.cumsum(y)
        return (int(v.sum()) == int(y.sum())) and np.all(Y >= L - 1e-9) and np.all(Y <= U + 1e-9)


    def _D2_matrix(n: int) -> np.ndarray:
        """
        2階差分行列 D2
        """
        if n < 3:
            return np.zeros((0, n), dtype=np.float64)
        D = np.zeros((n - 2, n), dtype=np.float64)
        for t in range(n - 2):
            D[t, t] = 1.0
            D[t, t + 1] = -2.0
            D[t, t + 2] = 1.0
        return D


    def _local_cost_rounding(
        left: int,
        rest: np.ndarray,
        total: int,
        smooth_weight: float = 0.25,
    ) -> np.ndarray:
        """
        整数化
        貪欲ヒープで丸め込み差異を吸収
        """
        import heapq

        rest = np.asarray(rest, dtype=np.float64)
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
        rest_full[1:] = rest

        lam = float(smooth_weight)
        ver = np.zeros(n, dtype=np.int64)

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
            return smooth_local_cost(i, new_val) - smooth_local_cost(i, old)

        def mc_add(i: int) -> float:
            return fit_delta(i, int(out[i]) + 1) + lam * smooth_delta(i, int(out[i]) + 1)

        def mc_sub(i: int) -> float:
            return fit_delta(i, int(out[i]) - 1) + lam * smooth_delta(i, int(out[i]) - 1)

        def push(heap, i: int, mode: str):
            if i <= 0:
                return
            if mode == "sub" and out[i] <= 0:
                return
            ver[i] += 1
            mc = mc_add(i) if mode == "add" else mc_sub(i)
            heapq.heappush(heap, (mc, i, ver[i]))

        def pop_valid(heap, mode: str):
            while heap:
                mc, i, vnow = heapq.heappop(heap)
                if vnow != ver[i]:
                    continue
                if mode == "sub" and out[i] <= 0:
                    continue
                return mc, i
            return None

        mode = "add" if diff > 0 else "sub"
        heap = []
        for i in range(1, n):
            push(heap, i, mode)

        steps = diff if diff > 0 else -diff
        for _ in range(steps):
            picked = pop_valid(heap, mode)
            if picked is None:
                raise RuntimeError("rounding stuck (unexpected)")
            _, i = picked
            if mode == "add":
                out[i] += 1
            else:
                out[i] -= 1

            for j in (i - 1, i, i + 1):
                if 1 <= j < n:
                    push(heap, j, mode)

        if int(out.sum()) != int(total):
            raise RuntimeError("rounding sum mismatch (unexpected)")
        return out

    def smooth_throttle_optimized_preview(
        data: np.ndarray,
        start_value: Optional[int] = None,
        r: int = 5,
        lam_smooth: float = 10.0,          # 2階差分
        lam_fit: float = 1.0,              # 追従
        max_iter: int = 2000,
        step: Optional[float] = None,
        proj_iters: int = 8,               # Dykstra反復
        clamp_start_value: bool = True,
        return_integer: bool = True,
        rounding_smooth_weight: float = 0.25,
        warn_on_infeasible: bool = True,
        seed: int = 0,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)

        x = np.asarray(data)
        if x.ndim != 1:
            raise ValueError("data must be 1-D")
        if x.size == 0:
            return {"y_float": np.array([], dtype=np.float64), "y_int": np.array([], dtype=np.int64)}

        if not np.issubdtype(x.dtype, np.integer):
            if np.any(np.floor(x) != x):
                raise ValueError("data must contain integers only")
        if np.any(x < 0):
            raise ValueError("data must be non-negative")

        x = x.astype(np.int64)
        n = x.size
        total = float(np.sum(x))

        left = int(x[0]) if start_value is None else int(start_value)

        # prefix bounds
        L, U = _prefix_bounds_from_hist(x, int(r))

        # anchor の実現可能域
        lo0 = int(np.ceil(L[0] - 1e-9))
        hi0 = int(np.floor(U[0] + 1e-9))

        if clamp_start_value:
            if left < lo0 or left > hi0:
                if warn_on_infeasible:
                    warnings.warn(
                        f"start_value={left} is outside feasible prefix band at t=0 "
                        f"([{lo0}, {hi0}]) for ±{r}. Clipping to feasible range.",
                        RuntimeWarning
                    )
                left = int(np.clip(left, lo0, hi0))
        else:
            if left < lo0 or left > hi0:
                raise ValueError(f"start_value={left} infeasible for ±{r}: must be in [{lo0},{hi0}]")

        D2 = _D2_matrix(n)
        # ヘッセ行列の最大固有値（Lipschitz定数）
        H = lam_fit * np.eye(n, dtype=np.float64)
        if D2.shape[0] > 0:
            H += lam_smooth * (D2.T @ D2)

        if step is None:
            # eigvalsh=対称行列
            Lmax = float(np.max(np.linalg.eigvalsh(H)))
            step = 1.0 / max(Lmax, 1e-9)

        y = x.astype(np.float64).copy()
        y = _affine_project_sum_and_anchor(y, total=total, left=float(left))
        y = np.maximum(y, 0.0)
        y = _project_prefix_band(y, L, U)
        y = _affine_project_sum_and_anchor(y, total=total, left=float(left))

        # ダイクストラ射影
        def proj_C(v: np.ndarray) -> np.ndarray:
            z = v.copy()
            pA = np.zeros_like(z)
            pB = np.zeros_like(z)
            pC = np.zeros_like(z)

            for _ in range(int(proj_iters)):
                a = _affine_project_sum_and_anchor(z + pA, total=total, left=float(left))
                pA = (z + pA) - a
                z = a

                b = np.maximum(z + pB, 0.0)
                pB = (z + pB) - b
                z = b

                c = _project_prefix_band(z + pC, L, U)
                pC = (z + pC) - c
                z = c

            z = _affine_project_sum_and_anchor(z, total=total, left=float(left))
            z = np.maximum(z, 0.0)
            return z

        last_obj = None
        for it in range(int(max_iter)):
            grad = lam_fit * (y - x.astype(np.float64))
            if D2.shape[0] > 0:
                grad += lam_smooth * (D2.T @ (D2 @ y))

            y_new = proj_C(y - step * grad)

            fit = 0.5 * lam_fit * float(np.sum((y_new - x) ** 2))
            smt = 0.0
            if D2.shape[0] > 0:
                d2 = D2 @ y_new
                smt = 0.5 * lam_smooth * float(np.sum(d2 ** 2))
            obj = fit + smt

            if last_obj is not None and abs(last_obj - obj) / (abs(last_obj) + 1e-9) < 1e-9:
                y = y_new
                last_obj = obj
                break

            y = y_new
            last_obj = obj

        y[0] = float(left)
        y = _affine_project_sum_and_anchor(y, total=total, left=float(left))
        y = np.maximum(y, 0.0)

        result: Dict[str, Any] = {
            "y_float": y.copy(),
            "objective": float(last_obj) if last_obj is not None else None,
            "iterations": it + 1,
            "step": float(step),
            "params": {
                "r": int(r),
                "lam_fit": float(lam_fit),
                "lam_smooth": float(lam_smooth),
                "proj_iters": int(proj_iters),
                "left": int(left),
            },
        }

        if return_integer:
            y_int = _local_cost_rounding(
                left=int(left),
                rest=y[1:],
                total=int(np.sum(x)),
                smooth_weight=float(rounding_smooth_weight),
            )
            feasible = _check_feasible_prefix(x, y_int, int(r))
            if (not feasible) and warn_on_infeasible:
                warnings.warn(
                    f"Integer-rounded target may be infeasible under ±{r} minute sliding (prefix check failed). "
                    f"Consider increasing lam_fit (closer to data), reducing lam_smooth (less aggressive smoothing), "
                    f"or allowing post-rounding repair.",
                    RuntimeWarning
                )
            result["y_int"] = y_int
            result["feasible_int"] = bool(feasible)

        return result

    def print_smoothing_summary(data: np.ndarray, out: Dict[str, Any]) -> None:
        x = np.asarray(data, dtype=np.int64)
        y = out["y_float"]
        print(f"n={x.size}, sum(x)={int(x.sum())}")
        print(f"objective={out.get('objective')}, iterations={out.get('iterations')}, step={out.get('step')}")
        print(f"y_float: sum={float(np.sum(y)):.6f}, y[0]={y[0]:.6f}, min={float(np.min(y)):.6f}")
        if "y_int" in out:
            yi = out["y_int"]
            print(f"y_int:   sum={int(np.sum(yi))}, y[0]={int(yi[0])}, min={int(np.min(yi))}, feasible_int={out.get('feasible_int')}")
    return print_smoothing_summary, smooth_throttle_optimized_preview


@app.cell(hide_code=True)
def _(mo):
    lam_smooth_slider = mo.ui.slider(0.0, 50.0, 0.5, label="Smoothing Weight (lam_smooth)", value=10.0)
    lam_smooth_slider
    return (lam_smooth_slider,)


@app.cell(hide_code=True)
def _(mo):
    lam_fit_slider = mo.ui.slider(0.1, 10.0, 0.1, label="Fit Weight (lam_fit)", value=1.0)
    lam_fit_slider
    return (lam_fit_slider,)


@app.cell(hide_code=True)
def _(mo, test_data):
    start_value_slider = mo.ui.slider(0, 60, 1, label="Start Value", value=int(test_data[0]))
    start_value_slider
    return (start_value_slider,)


@app.cell(hide_code=True)
def _(lam_fit_slider, lam_smooth_slider, start_value_slider):
    lam_smooth_value = lam_smooth_slider.value
    lam_fit_value = lam_fit_slider.value
    start_value_value = start_value_slider.value
    return lam_fit_value, lam_smooth_value, start_value_value


@app.cell(hide_code=True)
def _(
    lam_fit_value,
    lam_smooth_value,
    plot_individual_comparison,
    plt,
    print_smoothing_summary,
    smooth_throttle_optimized_preview,
    start_value_value,
    test_data,
):
    result = smooth_throttle_optimized_preview(
        data=test_data,
        start_value=start_value_value,
        r=5,
        lam_smooth=lam_smooth_value,
        lam_fit=lam_fit_value,
        max_iter=2000,
        step=None,
        proj_iters=8,
        clamp_start_value=True,
        return_integer=True,
        rounding_smooth_weight=0.25,
        warn_on_infeasible=True,
        seed=0,
    )
    print_smoothing_summary(test_data, result)
    fig = plot_individual_comparison(
        original=test_data,
        smoothed=result["y_int"],
        title="Smoothing Throttle Optimized Preview",
        method_name="Optimized Smoother"
    )
    plt.show()
    return


if __name__ == "__main__":
    app.run()

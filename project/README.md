# RV Prediction Pipeline (`src`)

这个目录包含一个可扩展的时序 RV（二分类）预测流水线：

- 数据：`yfinance` 拉取，失败自动 fallback 到 synthetic 数据
- 特征：内置 `core` + 可插拔 feature selection
- 模型：可插拔 model registry + Optuna 调参
- 训练：支持 CLI 与 Python 调用

---

## 1) 目录结构与职责

- 顶层（`project/`）
  - `train.py`: 训练入口 CLI
  - `eval.py`: 评估入口 CLI（薄入口，委托 `src/eval/reporting.py` + `src/eval/plots.py`）
  - `main.py`: 训练 → 评估一键入口（顶部 `CONFIG` dict）
  - `prompts/eval_llm_prompts.md`: LLM 生成/审核 Prompt 模板
  - `cache/`: 训练与评估产物缓存目录
  - `src/`: 主要业务代码

- `src/data/`
  - `load_data.py`: 获取真实数据 + fallback 逻辑
  - `synthetic.py`: 生成假数据（便于离线测试）
  - `spliter.py`: `TimeSeriesSplitWithPurge`
- `src/features/`
  - `core.py`: 构建基础特征与标签（`build_dataset`）
  - `rv.py`: 历史/前瞻 RV 计算
  - `steps.py`: 特征步骤注册与组合
  - `selection.py`: 特征选择（`tree_topk` / `lasso` / `ttest`）
- `src/models/`
  - `objectives/`: 各模型的 Optuna objective
  - `registry.py`: 模型注册、实例化
  - `tuner.py`: 单模型/多模型调参
  - `fit.py`: 最优模型全量拟合与持久化
  - `evaluator.py`: OOF 评估
- `src/metrics/`
  - `probabilistic.py`: 概率预测指标
  - `dm_test.py`: DM 检验
  - `baseline.py`: baseline 与 per-observation loss
- `src/eval/`（评估流程实现）
  - `data.py`: 读取 run artifacts（`y.csv` / `oof_predictions.csv` / `pipeline.json`）
  - `regime.py`: 宏观指标（VIX MA30）聚类划分 + 分组混淆矩阵
  - `plots.py`: 评估图（CM / timeline / cluster scatter / comparison）
  - `llm_briefing.py`: SHAP 降维、双 Agent 生成/审核、fallback
  - `reporting.py`: `run_dm_evaluation()` 与终端输出
- `src/evaluation/`: 对外稳定导出层（避免 `eval` 包名冲突）
- `src/utils/`
  - `cache.py`: cache 根目录与 run 目录管理、模型 bundle 读写
  - `artifacts.py`: 训练产物落盘
- `src/pipline.py`: 主流程 `RVPipeline`
- `src/plots.py`: 训练后图表输出
- `src/trains/`: 早期实验目录（当前主流程不依赖，保留用于对照）

---

## 2) 如何使用（CLI）

在 `project` 目录下运行（推荐使用 `uv`）：

```bash
cd project
uv sync
uv run python train.py --synthetic --n-trials 5
```

常用参数：

- `--symbol`, `--start`, `--end`: 标的与时间区间
- `--models`: 选择模型（`logistic xgb lgbm`）
- `--n-trials`: 每个模型的 Optuna trial 数
- `--synthetic`: 强制使用假数据
- `--feature-selection`: `tree_topk` / `lasso` / `ttest` / `none`
- `--top-k-features`: 特征选择保留的数量
- `--horizon`: 预测窗口（默认 5）

示例（真实数据优先，失败自动 fallback）：

```bash
uv run python train.py \
  --symbol SPY \
  --start 2020-01-01 \
  --end 2024-12-31 \
  --models logistic xgb lgbm \
  --n-trials 30 \
  --feature-selection tree_topk \
  --top-k-features 4
```

---

## 2.1) CLI 参数总览（完整）

下面的参数列表与当前代码实现对齐（`train.py` / `eval.py`）。

### `train.py` 参数

| 参数 | 类型 | 默认值 | 用途 |
|---|---|---|---|
| `--symbol` | `str` | `"SPY"` | 训练标的代码（传给数据加载层） |
| `--start` | `str` | `"2018-01-01"` | 训练起始日期 |
| `--end` | `str` | `"2024-12-31"` | 训练结束日期 |
| `--models` | `list[str]` | `["logistic", "xgb", "lgbm"]`（以注册表实时为准） | 参与调参/比较的模型集合 |
| `--n-trials` | `int` | `30` | 每个模型的 Optuna trial 数量 |
| `--synthetic` | `flag` | `False` | 启用后强制使用 synthetic 数据（不拉 yfinance） |
| `--horizon` | `int` | `5` | 前瞻预测窗口（用于标签构造） |
| `--label-method` | `str` | `"median"` | 标签构造规则，候选：`median` / `expanding` |
| `--feature-selection` | `str \| None` | `None` | 特征选择方法；候选：`tree_topk` / `lasso` / `ttest` / `none`（`none` 会被转换为 `None`） |
| `--top-k-features` | `int` | `5` | 特征选择保留数量（对 top-k 类方法生效） |
| `--n-splits` | `int` | `5` | 时序 CV 折数 |
| `--train-window` | `int` | `252` | 每折训练窗口长度（交易日） |
| `--test-size` | `int` | `21` | 每折验证窗口长度 |
| `--gap-size` | `int` | `5` | 训练与验证之间的 gap（防泄漏） |
| `--output-dir` | `str \| None` | `None` | run 目录父路径；为空时使用 `cache/runs` |

> 说明：`--models` 与 `--feature-selection` 的候选项来自注册表，后续新增模型/选择器后，CLI 会自动扩展可选值。

### `eval.py` 参数

| 参数 | 类型 | 默认值 | 用途 |
|---|---|---|---|
| `--run-dir` | `str` | 无（必填） | 指向单次训练产物目录（通常位于 `cache/runs/<run-id>`），用于 DM 评估、regime CM 与 LLM briefing |
| `--baselines` | `str` | 空（全部运行） | 逗号或空格分隔，如 `prior,rv_persist,garch`。合法值：`prior`, `rv_persist`, `garch` |
| `--shap-max-samples` | `int` | `100` | LinearExplainer 背景样本上限；传 `<=0` 表示用全量样本（消除 SHAP 警告） |
| `--regime-method` | `str` | `"kmeans"` | 宏观指标聚类方法。候选：`quantile` / `gmm` / `kmeans` / `agglomerative` / `birch` / `all`（一次跑全部并额外出 comparison 图） |
| `--n-regimes` | `int` | `2` | 聚类簇数 / 分位箱数（必须 `>= 2`）。簇按 **宏观指标均值从高到低** 重标为 `Cluster_1` … `Cluster_n` |
| `--save-dir` | `str` | 空 → `{run_dir}/plots/` | 评估 PNG 输出目录；文件名始终带 method 与 n，如 `confusion_matrices_kmeans_n2.png` |

### `main.py`

默认无 CLI 参数。训练与评估配置都在顶部 `CONFIG` dict 中调整：

```bash
uv run python main.py
```

会先跑完整训练，再用产出的 `run_dir` 自动调用 `evaluate()`（DM + regime CM + 默认落盘到 `{run_dir}/plots/`）。

相关 `CONFIG["eval"]` 字段：

| 键 | 默认值 | 用途 |
|---|---|---|
| `baselines` | `None`（全部） | 如 `["prior", "rv_persist"]` |
| `shap_max_samples` | `-1`（全量） | SHAP 背景样本上限；`<=0` → 全量 |
| `regime_method` | `"kmeans"` | 同 `eval.py --regime-method` |
| `n_regimes` | `2` | 同 `eval.py --n-regimes` |
| `save_dir` | `None` | `None` → `{run_dir}/plots/` |

单独评估示例：

```bash
uv run python eval.py --run-dir cache/runs/<your-run-dir> --regime-method kmeans --n-regimes 2
uv run python eval.py --run-dir cache/runs/<your-run-dir> --regime-method all --n-regimes 3
uv run python eval.py --run-dir cache/runs/<your-run-dir> --baselines prior,garch --shap-max-samples 0
```

## 3) 如何通过代码调用并训练（Python API）

```python
from project.src.pipline import RVPipeline

pipeline = RVPipeline(
    symbol="SPY",
    start_date="2018-01-01",
    end_date="2024-12-31",
    models=["logistic", "xgb", "lgbm"],
    n_trials=20,
    use_synthetic=False,
    feature_selection="lasso",
    top_k_features=4,
)

result = pipeline.run()
print(result.best_model_name, result.best_auc)
print(result.selected_features)
print(result.plot_dir)
```

如果你只想分步调试：

```python
pipeline.load_data()
pipeline.build_features()
pipeline.run_feature_selection()   # 可选
pipeline.create_cv()
pipeline.tune_model("logistic")
```

---

## 4) 如何使用 Eval（以及如何改各个环节）

### CLI 用法

```bash
cd project
uv run python eval.py --run-dir cache/runs/<your-run-dir>
uv run python eval.py --run-dir cache/runs/<your-run-dir> --regime-method gmm --n-regimes 3
uv run python eval.py --run-dir cache/runs/<your-run-dir> --regime-method all --n-regimes 2
```

输出包括：

- DM 检验（Brier / Log-loss，按所选 baseline）
- Regime 分簇混淆矩阵（`Overall` + `Cluster_1`…`Cluster_n`，按簇内宏观指标均值从高到低）
- 评估图（默认写入 `{run_dir}/plots/`）
- `llm_briefing`（极简 SHAP + 双 Agent 审核结果）
- JSON：`{run_dir}/config/dm_test_report.json`

### Regime 聚类（宏观指标 = VIX）

所有方法都在同一维宏观序列上分簇：优先 `^VIX` 的 30 日滚动均值；拉不到则用 OOF `Close` 的年化滚动波动再 MA30 作为 fallback。

| `--regime-method` | 行为 |
|---|---|
| `quantile` | 等频分位切成 `n` 箱，再按均值降序重标号 |
| `gmm` | `GaussianMixture(n_components=n)`，按 `means_` 降序重标号 |
| `kmeans`（默认） | `KMeans(n_clusters=n)`，按簇均值降序重标号 |
| `agglomerative` | `AgglomerativeClustering(n_clusters=n)`，按均值降序重标号 |
| `birch` | `Birch(n_clusters=n)`，按均值降序重标号 |
| `all` | 上述五种各跑一遍；各自出 CM/timeline/scatter，并额外写 `cluster_comparison_n{N}.png` |

重标规则（所有方法统一）：算每簇在宏观序列上的均值 → 从高到低记为 `Cluster_1` … `Cluster_n` → 对每簇分别算 confusion matrix（再加 Overall）。

### 评估图文件名

| 产物 | 文件名 |
|---|---|
| 分簇混淆矩阵 | `confusion_matrices_{method}_n{N}.png` |
| 时间轴着色 | `macro_regime_timeline_{method}_n{N}.png` |
| 经典 cluster scatter | `cluster_scatter_{method}_n{N}.png` |
| 多方法对照（仅 `all`） | `cluster_comparison_n{N}.png` |

图标题同样带方法名与 `n`，例如 `Confusion Matrices — KMeans (n=2)`。

### Python API 用法

```python
from pathlib import Path
from project.src.evaluation.reporting import run_dm_evaluation, print_report
from project.eval import evaluate

# 仅算报告（不含默认绘图封装）
report = run_dm_evaluation(
    Path("cache/runs/<your-run-dir>"),
    save_report=True,
    regime_method="kmeans",
    n_regimes=2,
)
print_report(report)

# 推荐入口：DM + 打印 + 默认写到 {run_dir}/plots/
evaluate(
    "cache/runs/<your-run-dir>",
    regime_method="all",
    n_regimes=3,
)

# 自定义 sklearn 聚类器（任意 fit/predict 或 fit_predict）
from sklearn.cluster import SpectralClustering
evaluate(
    "cache/runs/<your-run-dir>",
    n_regimes=2,
    clusterer=SpectralClustering(n_clusters=2, affinity="nearest_neighbors"),
)
```

`regime_method="all"` 时，报告里有 `regime_splits[<method>]`；`regime_split` 仍指向主方法（优先 `kmeans`）以保持兼容。

### 如何改 Eval 的各个环节（按职责改）

- **读取 run 数据**：改 `src/eval/data.py` 的 `load_run_data()`
- **Regime 划分逻辑**：改 `src/eval/regime.py`
  - `fetch_vix_ma_series()` / `build_macro_indicator()`: 宏观指标数据源与 fallback
  - `assign_regime_clusters()`: 分簇算法 registry + 均值降序重标号
  - `build_macro_regime_labels()`: 串联 indicator + clustering
  - `group_confusion_matrices()`: 分组 CM
- **评估绘图**：改 `src/eval/plots.py` 的 `save_eval_plots()` 等
- **LLM 与 SHAP**：改 `src/eval/llm_briefing.py`
  - `_extract_lean_shap_context()`: top-k 特征压缩规则
  - `_generate_and_review_brief()`: 双 Agent 串联逻辑
  - `_build_mock_brief()`: API 失败时的降级文本
  - Prompt 模板文件：`prompts/eval_llm_prompts.md`
- **总报告结构**：改 `src/eval/reporting.py` 的 `run_dm_evaluation()`
- **CLI 入口**：`eval.py`（尽量保持薄入口）

环境变量（`.env`）：

- `LLM_API_KEY`
- `LLM_MODEL`
- `CACHE_PATH`（可选，覆盖默认 `./cache`）

---

## 5) 如何加更多 Feature

特征构建已解耦为「逐步添加」模式：

- `features/steps.py`：每个 `add_`* 函数只负责新增一列
- `features/core.py`：`FeatureStep` 注册表 + `apply_features()` 串联步骤

### 当前默认 Feature 列表（已与代码同步）

`features/core.py` 的 `DEFAULT_FEATURE_STEPS` 当前默认包含以下 49 个特征（按执行顺序）：

`ret_1d`, `hl_range`, `rv_hist_5d`, `rv_hist_20d`, `gk_rv`, `rs_rv`, `yz_rv`, `volume_zscore`, `parkinson_rv`, `ewma_rv`, `downside_rv`, `vol_of_vol`, `rv_accel`, `ret_skew_20d`, `ret_kurt_20d`, `max_drawdown_20d`, `ret_autocorr_1_20d`, `abs_ret_lag1`, `rv_ratio_5_20`, `vix_level`, `vix_change_1d`, `vix_term_structure`, `vix_rv_spread`, `vvix_level`, `skew_index_level`, `yield_change_1d`, `dxy_change_1d`, `gold_ret_1d`, `day_of_week`, `is_month_end`, `is_quarter_end`, `dist_from_ma50`, `bb_width`, `overnight_ret`, `intraday_ret`, `overnight_vol_5d`, `garch_vol_1d`, `gjr_garch_vol_1d`, `egarch_vol_1d`, `garch_vol_ratio_5_20`, `garch_rv_spread`, `har_rv_forecast_1d`, `kalman_price_medium`, `kalman_ret_1d_medium`, `kalman_trend_medium`, `kalman_resid_medium`, `kalman_resid_zscore_medium`, `kalman_state_var_medium`, `close_to_kalman_medium`

其中：

- 仅依赖标的自身 OHLCV 的特征可直接从主行情构建（如 `gk_rv` / `yz_rv` / `downside_rv` / GARCH / HAR-RV / Kalman）。
- 宏观/跨资产特征依赖 `load_data.py` 自动附加的市场上下文列：`vix`, `vix3m`, `vvix`, `skew`, `tnx`, `dxy`, `gold`。
- 数据层会尝试拉取 `^VIX`, `^VIX3M`, `^VVIX`, `^SKEW`, `^TNX`, `DX-Y.NYB`, `GC=F`；若某个 ticker 缺失，会自动跳过，不阻塞训练流程。
- GARCH / HAR-RV / Kalman 特征依赖 `arch` / `scikit-learn` / `pykalman`；并统一 `shift(1)`，按开盘前可用信息对齐（不使用当日收盘价）。
- Kalman 在 `log(Close)` 上做 local-linear-trend filtering（`process_var=1e-5`, `obs_var=1e-3`），再按特征需要回映到价格语义；非正价按缺失处理。
- HAR-RV 用单日 Garman-Klass 年化波动构造 \(RV^{(d/w/m)}\)（1/5/22），滚动窗内 `StandardScaler` + Ridge（`alpha=1.0`, `fit_window=750`, `min_obs=252`）做一步预测，负值 `clip` 到 0 后 `shift(1)`。
- DM 评估除 prior / RV persistence baseline 外，还会比较 `garch_vol_1d` 的 Logistic 校准 baseline（`garch_vol_1d_logistic_calibration`）。

**尺度/变换标注**（只标该 step 自身对输入做的变换；下游派生不重复继承）：

| 标注 | 含义 | 特征 |
|---|---|---|
| **log** | 对价格/收益显式取 `log`，或在 log 空间建模/输出残差·状态 | `garch_vol_1d`, `gjr_garch_vol_1d`, `egarch_vol_1d`, `kalman_price_medium`, `kalman_trend_medium`, `kalman_resid_medium`, `kalman_resid_zscore_medium`, `kalman_state_var_medium` |
| **比值** | `P₁/P₂`、`pct_change`、`(P-A)/A` 一类相对价格/水平比（非 log） | `ret_1d`, `hl_range`, `max_drawdown_20d`, `dxy_change_1d`, `gold_ret_1d`, `dist_from_ma50`, `bb_width`, `kalman_ret_1d_medium`, `close_to_kalman_medium` |
| **log+比值** | 典型 `log(P₁/P₂)`：比值后再取对数（或等价的对数收益） | `parkinson_rv`, `gk_rv`, `rs_rv`, `yz_rv`, `overnight_ret`, `intraday_ret`, `overnight_vol_5d`, `har_rv_forecast_1d` |
| **波动比值** | 波动特征之间相除（不是价格比） | `rv_ratio_5_20`, `garch_vol_ratio_5_20` |

其余未列出的默认特征为水平值、差分、滚动统计或 z-score 等，**不做**价格 log / `P₁/P₂` 比值变换。

### Feature 含义速查表

| Feature | 主要作用（干什么） | 变换 | 计算来源（怎么来的） |
|---|---|---|---|
| `ret_1d` | 日收益率，最基础价格变化信号 | 比值 | `Close.pct_change()` |
| `hl_range` | 日内振幅，衡量当天价格波动区间 | 比值 | `(High - Low) / Close` |
| `rv_hist_5d` | 5 日历史已实现波动率（短期） | — | `ret_1d` 的 5 日滚动标准差年化 |
| `rv_hist_20d` | 20 日历史已实现波动率（中期） | — | `ret_1d` 的 20 日滚动标准差年化 |
| `gk_rv` | Garman-Klass 波动率，利用 OHLC 降噪 | log+比值 | `log(H/L)`, `log(C/O)` 组合后滚动均值年化 |
| `rs_rv` | Rogers-Satchell 波动率，对趋势更稳健 | log+比值 | `log(H/O), log(L/O), log(C/O)` 组合后滚动均值年化 |
| `yz_rv` | Yang-Zhang 波动率，考虑隔夜跳空 | log+比值 | 隔夜/开收/RS 的对数收益分量组合后年化 |
| `volume_zscore` | 成交量异常程度（放量/缩量） | — | `Volume` 相对 20 日均值和标准差的 z-score |
| `parkinson_rv` | Parkinson 波动率，高低价区间估计波动 | log+比值 | `log(High/Low)` 的滚动二阶矩年化 |
| `ewma_rv` | 更重视近期冲击的波动率估计 | — | `ret_1d^2` 做 EWMA 后开方年化 |
| `downside_rv` | 下行半方差，强调负收益风险 | — | 仅对负 `ret_1d` 平方做滚动均值后开方年化 |
| `vol_of_vol` | 波动率的波动，识别状态稳定性 | — | `rv_hist_5d` 的 20 日滚动标准差 |
| `rv_accel` | 波动率加速度，识别波动状态切换 | — | `rv_hist_5d - rv_hist_5d.shift(5)` |
| `ret_skew_20d` | 收益偏度，识别左尾风险倾向 | — | `ret_1d` 的 20 日滚动偏度 |
| `ret_kurt_20d` | 收益峰度，识别厚尾/极端波动风险 | — | `ret_1d` 的 20 日滚动峰度 |
| `max_drawdown_20d` | 近期最深回撤，衡量尾部压力 | 比值 | `Close / rolling_max - 1`，再取滚动最小值 |
| `ret_autocorr_1_20d` | 收益一阶自相关，反映趋势/均值回复状态 | — | `ret_1d` 在 20 日窗口内计算 lag=1 自相关 |
| `abs_ret_lag1` | 昨日绝对收益，短期冲击记忆 | — | `abs(ret_1d).shift(1)` |
| `rv_ratio_5_20` | 短中期波动率相对强弱 | 波动比值 | `rv_hist_5d / rv_hist_20d` |
| `vix_level` | 隐含波动率水平 | — | `^VIX` 收盘值 |
| `vix_change_1d` | 隐含波动率日变动 | — | `vix.diff().shift(1)` |
| `vix_term_structure` | VIX 期限结构，识别恐慌/backwardation | — | `vix3m - vix` |
| `vix_rv_spread` | 隐含-已实现波动率价差，均值回复信号 | — | `vix / 100 - rv_hist_20d` |
| `vvix_level` | “波动率的波动率”水平 | — | `^VVIX` 收盘值 |
| `skew_index_level` | 尾部风险溢价强弱 | — | `^SKEW` 收盘值 |
| `yield_change_1d` | 利率冲击代理 | — | `tnx.diff().shift(1)` |
| `dxy_change_1d` | 美元风险偏好代理 | 比值 | `dxy.pct_change().shift(1)` |
| `gold_ret_1d` | 避险资产变动代理 | 比值 | `gold.pct_change().shift(1)` |
| `day_of_week` | 周内日历效应 | — | 交易日 `dayofweek` |
| `is_month_end` | 月末效应标记 | — | 日期是否月末（0/1） |
| `is_quarter_end` | 季末效应标记 | — | 日期是否季末（0/1） |
| `dist_from_ma50` | 趋势偏离程度 | 比值 | `(Close - MA50) / MA50` |
| `bb_width` | 布林带宽度，波动状态非线性表达 | 比值 | `4 * rolling_std(Close,20) / rolling_mean(Close,20)` |
| `overnight_ret` | 隔夜收益，捕捉跳空信息 | log+比值 | `log(Open / Close.shift(1))` |
| `intraday_ret` | 日内收益，分离日内波动驱动 | log+比值 | `log(Close / Open)` |
| `overnight_vol_5d` | 隔夜波动率（短期） | log+比值 | `overnight_ret` 的 5 日滚动标准差年化 |
| `garch_vol_1d` | GARCH(1,1)-t 一步波动预测 | log | 对 `log(Close).diff()` 滚动拟合后年化，再 `shift(1)` |
| `gjr_garch_vol_1d` | GJR-GARCH 一步波动，捕捉杠杆效应 | log | 同上，GARCH + `o=1`，年化后 `shift(1)` |
| `egarch_vol_1d` | EGARCH 一步波动，非对称波动 | log | 同上，EGARCH(1,1)-t，年化后 `shift(1)` |
| `garch_vol_ratio_5_20` | GARCH 预测相对历史 RV 强弱 | 波动比值 | `garch_vol_1d / rv_hist_20d` |
| `garch_rv_spread` | GARCH 预测与历史实现波动价差 | — | `garch_vol_1d - rv_hist_20d` |
| `har_rv_forecast_1d` | HAR-RV 一步波动预测 | log+比值 | 日度 GK（`log(H/L)`, `log(C/O)`）的 d/w/m，Scaled-Ridge 后 `clip≥0` 再 `shift(1)` |
| `kalman_price_medium` | Kalman 去噪价格水平（medium） | log | 对 `log(Close)` 做 filtering，再 `exp` 回映，`shift(1)` |
| `kalman_ret_1d_medium` | 去噪价格日收益 | 比值 | 回映价格的 `pct_change().shift(1)` |
| `kalman_trend_medium` | 去噪状态趋势分量（log 漂移） | log | log 空间速度项，`shift(1)` |
| `kalman_resid_medium` | 对数价格相对去噪 log 水平的残差 | log | `(log(Close) - filtered_log).shift(1)` |
| `kalman_resid_zscore_medium` | 对数残差标准化 | log | log 残差 20 日 z-score 后 `shift(1)` |
| `kalman_state_var_medium` | 状态不确定性（log 水平） | log | Kalman log 水平状态方差，`shift(1)` |
| `close_to_kalman_medium` | 现价相对去噪价偏离率 | 比值 | `((Close - kalman_price) / kalman_price).shift(1)` |

### 方式 A：新增一个 feature step（推荐）

1. 在 `features/steps.py` 添加函数：

```python
def add_close_to_ma10(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ma_10 = out["Close"].rolling(10).mean()
    out["close_to_ma10"] = out["Close"] / ma_10 - 1
    return out
```

1. 在 `features/core.py` 的 `DEFAULT_FEATURE_STEPS` 注册：

```python
FeatureStep("close_to_ma10", add_close_to_ma10),
```

或运行时注册：

```python
from project.src.features.core import FeatureStep, register_feature
from project.src.features.steps import add_close_to_ma10

register_feature(FeatureStep("close_to_ma10", add_close_to_ma10))
```

确保：

- 每个 step 只加一列，且函数返回新 DataFrame
- 特征只使用历史信息（避免未来泄漏）
- 依赖其他列的 step 要排在后面（如 `rv_hist_*` 依赖 `ret_1d`）

### 方式 B：自定义 feature 列表

```python
from project.src.features.core import build_dataset, get_feature_steps

steps = get_feature_steps(["ret_1d", "hl_range", "rv_hist_5d"])
X, y, meta = build_dataset(df, feature_steps=steps)
```

### 方式 C：新增特征选择方法（可扩展）

在 `features/selection.py` 中：

1. 继承 `BaseFeatureSelector`
2. 实现 `fit(self, X, y)`，并设置：
  - `self.scores_`
  - `self.selected_features_`
3. 使用 `register_selector("your_method", YourSelector)` 注册

然后 CLI 即可用 `--feature-selection your_method`（如果你同步把参数 choices 加入入口）。

---

## 6) 如何加更多模型

当前模型是通过 `models/registry.py` + `models/objectives/*` 解耦管理的。

新增模型步骤：

1. 新建 objective 文件
  例如：`models/objectives/catboost.py`，实现：
  - 函数签名与现有 objective 保持一致  
  - 返回 CV 平均 AUC（float）
2. 在 `models/registry.py` 注册 objective
  - 把模型名加入 `MODEL_REGISTRY`
3. 在 `models/registry.py` 的 `build_model()` 中增加实例化逻辑
  - 用于 OOF 评估阶段创建训练模型
4. （可选）更新 `train.py` 参数 choices
  - 这样 CLI `--models` 会显示你的新模型名

最小注册示例：

```python
from project.src.models.registry import register_model

def objective_my_model(trial, X, y, cv):
    ...
    return mean_auc

register_model("my_model", objective_my_model)
```

---

## 7) 训练部分：如何改 Optuna 目标

当前默认目标：

- 在 `src/models/tuner.py` 里 `optuna.create_study(direction="maximize")`
- 各 objective（`src/models/objectives/*.py`）返回一个要被“最大化”的分数（目前是 AUC）

### 方案 A：继续 `maximize`，但替换优化指标

适合指标“越大越好”（例如 F1、收益、Calmar 等）。

步骤：

1. 修改对应 objective（如 `src/models/objectives/xgb.py`）里每折计算的 metric
2. 返回 CV 平均 metric（float）
3. 保持 `direction="maximize"` 不变

### 方案 B：改成最小化目标（如 log loss / brier）

步骤：

1. 在 `src/models/tuner.py` 把
  - `direction="maximize"` 改为 `direction="minimize"`
2. 在 objective 返回“越小越好”的值（如 mean log-loss）

注意：当前 `ModelResult.best_auc` 字段名是历史命名；如果你切换成非 AUC 目标，建议同步重命名为通用名（如 `best_score`），避免语义混淆。

### 多目标优化（可选）

如需同时优化多个指标，可改用 `optuna.create_study(directions=[...])` + Pareto 策略；这会牵涉 `ModelResult`、排序逻辑和终端输出，需要大幅度重构。

---

## 8) Cache 机制说明

每次运行会在 `cache/runs/` 下自动创建一个目录，命名格式：

```text
{best_model}-{feature_selection}-{timestamp}
```

示例：`lgbm-tree_topk-20260623_080630`（无特征选择时为 `lgbm-none-...`）

目录结构：

```text
cache/
  data/                         # 共享 yfinance 数据缓存
  runs/
    lgbm-tree_topk-20260623_080630/
      config/
        pipeline.json           # 全部 pipeline 参数
        cv.json                 # CV 切分配置
        best_params.json        # 最优模型超参数
        model_comparison.json   # 各模型 AUC 与参数
        feature_selection.json  # 特征选择结果
      data/
        raw.csv                 # 原始 OHLCV
        X.csv                   # 训练特征
        y.csv                   # 标签
        meta.csv                # 画图用元数据
        oof_predictions.csv     # OOF 预测概率
      model/
        best_model.joblib       # 最优模型权重 + scaler + feature_names
      plots/
        *.png                       # 训练图 + eval 图（CM / timeline / cluster_*）
```

训练图由 `src/plots.py` 写入；eval 后还会追加例如：

- `confusion_matrices_kmeans_n2.png`
- `macro_regime_timeline_kmeans_n2.png`
- `cluster_scatter_kmeans_n2.png`
- （`--regime-method all` 时）`cluster_comparison_n2.png`

训练完成后终端会打印 `result.run_dir` 和 `result.plot_dir`。

加载已保存模型：

```python
from project.src.utils.cache import RunCache

bundle = RunCache.load_model_bundle("cache/runs/lgbm-tree_topk-20260623_080630")
model = bundle["model"]
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]
```

可用 `--output-dir /path/to/parent` 指定 run 目录的父路径（默认 `cache/runs`）。

### Cache 的几个关键点

- `cache/data/`：共享数据缓存（跨 run 复用）
- `cache/runs/<run-id>/`：一次训练对应一份完整快照（可复现）
- `run_dir` 是训练和 eval 串联的核心输入（`eval.py --run-dir ...`）
- 删除某个 `run` 只会影响该次结果，不影响其他 run
- 清理策略建议：
  - 保留最近 N 次 run
  - 定期删除过大的 `plots/` 和历史 run
  - 重要 run 先备份 `config/ + model/ + data/oof_predictions.csv`

---

## 9) 常见问题

- `yfinance` 无法访问：使用 `--synthetic`，或依赖自动 fallback
- 训练很慢：降低 `--n-trials`，先只跑一个模型（例如 `--models logistic`）
- 结果不稳定：固定时间区间、固定特征选择、提高样本量


# ML Experiment Workbench

An interactive Streamlit application that wraps the RV (Realized Volatility) prediction pipeline into a multi-page workbench for FINS5557 Applied AI in Finance.

## Quick Start

```bash
cd project
uv run streamlit run app.py
```

The app opens at `http://localhost:8501` by default.

## Prerequisites

- Python ≥ 3.13
- [uv](https://docs.astral.sh/uv/) for dependency management
- A `.env` file with Gemini API credentials (optional, for LLM features):

```
LLM_API_KEY=your-google-api-key
LLM_MODEL=gemini-3.5-flash
```

## Pages

### Page 1 — Data & Feature Engineering

1. Enter a ticker symbol (e.g. `SPY`), date range, or toggle synthetic data
2. Click **Load Data** — head/tail preview appears
3. Select which features to compute from the 48 available
4. Click **Build Features** to generate the enriched DataFrame
5. Optionally describe a new feature in natural language and click **Generate Feature with Gemini**
6. Configure the label horizon (1–20 days) and method, then click **Build Dataset (X, y)**

### Page 2 — Train/Test Split

1. Adjust the train ratio slider (default 0.8)
2. Click **Split Data** — time-ordered split with stats
3. Configure cross-validation parameters (`n_splits`, `train_window_size`, `test_size`, `gap_size`)
4. Click **Create CV Object** — split visualization appears

### Page 3 — Model Training

1. Select one or more models (logistic, XGBoost, LightGBM)
2. Set the number of Optuna trials (default 10 for speed)
3. Click **Train Models** — progress shown in real time
4. Results table displays best AUC and hyperparameters per model

### Page 4 — Evaluation & Insight

1. Click **Compute OOF Predictions** — Brier score and log-loss metrics appear
2. View ROC curve and predictions overlay charts
3. Click **Compute Feature Importance** for a ranked bar chart
4. Click **Generate LLM Report** for an AI-written diagnostic summary (falls back to rule-based if API key is missing)

## Project Structure

```
project/
├── app.py                          # Streamlit entry point
├── src/
│   ├── streamlit_app/              # Streamlit adapter layer (new)
│   │   ├── session_manager.py      # Session state keys & helpers
│   │   ├── plots.py                # Figure-returning plot wrappers
│   │   ├── llm_helper.py           # Gemini feature gen & briefing
│   │   └── pages/
│   │       ├── data_feature.py     # Page 1
│   │       ├── train_test.py       # Page 2
│   │       ├── model_train.py      # Page 3
│   │       └── evaluation.py       # Page 4
│   ├── data/                       # Data loading (yfinance, synthetic)
│   ├── features/                   # Feature engineering (48 steps)
│   ├── models/                     # Model registry, Optuna tuner, evaluator
│   ├── eval/                       # LLM briefing, regime analysis
│   ├── metrics/                    # Brier, log-loss, Diebold-Mariano
│   └── plots.py                    # Original disk-saving plot functions
├── pyproject.toml
└── .env                            # API keys (not committed)
```

## Docker deploy

Build and run from this `project/` directory. Teaching-pack Parquets
(`cache/precomputed/*.parquet`, `grid_results.parquet`, `manifest.json`) and
**grid experiment runs** under `cache/precomputed/runs/` (excluding `custom_*`)
are baked into the image. SQLite (`data/workspaces.db`) is ephemeral inside
the container (no volume).

```bash
cd project
docker build -t rv-lab .
docker run --rm -p 8501:8501 \
  -e ALLOW_ANON=0 \
  -e CLASS_INVITE_CODE=fins5557 \
  -e LLM_API_KEY=your_key \
  -e LLM_MODEL=gemini-2.0-flash \
  rv-lab
```

Open http://localhost:8501. Auth is required by default (`ALLOW_ANON=0`);
override with `-e ALLOW_ANON=1` for anonymous local demos.

## Key Design Decisions

- **No modifications to existing modules** — all Streamlit logic lives in `src/streamlit_app/`
- **Session state** connects pages; changing upstream data automatically clears downstream results
- **`@st.cache_data`** on data loading and dataset building avoids redundant network calls
- **LLM feature generation** uses `exec()` with restricted builtins (`pd`/`np` only)
- **Optuna** defaults to 10 trials for interactive speed; increase as needed

## Batch Pipeline (Original CLI)

The original non-interactive pipeline is still available:

```bash
cd project
uv run python -m project.main
```

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
  - `eval.py`: 评估入口 CLI（薄入口，委托 `src/evaluation/reporting.py`）
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
  - `regime.py`: VIX regime 划分与分组混淆矩阵
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
| `--run-dir` | `str` | 无（必填） | 指向单次训练产物目录（通常位于 `cache/runs/<run-id>`），用于 DM 评估与 LLM briefing |

### `main.py`

`main.py` 当前没有 CLI 参数，仅打印一行示例文本。

---

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
```

输出包括：

- DM 检验（Brier / Log-loss）
- Regime 分组混淆矩阵
- `llm_briefing`（极简 SHAP + 双 Agent 审核结果）

### Python API 用法

```python
from pathlib import Path
from project.src.evaluation.reporting import run_dm_evaluation, print_report

report = run_dm_evaluation(Path("cache/runs/<your-run-dir>"), save_report=True)
print_report(report)
```

### 如何改 Eval 的各个环节（按职责改）

- **读取 run 数据**：改 `src/eval/data.py` 的 `load_run_data()`
- **Regime 划分逻辑**：改 `src/eval/regime.py`
  - `fetch_vix_ma_series()`: 宏观指标数据源
  - `build_macro_regime_labels()`: 阈值规则（默认中位数）
  - `group_confusion_matrices()`: 分组统计细节
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
        *.png
```

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


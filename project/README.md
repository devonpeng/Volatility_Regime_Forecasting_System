# RV Prediction Pipeline (`src`)

这个目录包含一个可扩展的时序 RV（二分类）预测流水线：
- 数据：`yfinance` 拉取，失败自动 fallback 到 synthetic 数据
- 特征：内置 `core` + 可插拔 feature selection
- 模型：可插拔 model registry + Optuna 调参
- 训练：支持 CLI 与 Python 调用

---

## 1) 目录结构与职责

- `data/`
  - `load_data.py`: 获取真实数据 + fallback 逻辑
  - `synthetic.py`: 生成假数据（便于离线测试）
  - `spliter.py`: `TimeSeriesSplitWithPurge`
- `features/`
  - `core.py`: 构建基础特征与标签（`build_dataset`）
  - `rv.py`: 历史/前瞻 RV 计算
  - `selection.py`: 特征选择（`tree_topk` / `lasso` / `ttest`）
- `models/`
  - `objectives/`: 各模型的 Optuna objective
  - `registry.py`: 模型注册、实例化
  - `tuner.py`: 单模型/多模型调参
  - `evaluator.py`: OOF 评估
- `pipline.py`: 主流程 `RVPipeline`
- `plots.py`: 训练后图表输出

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
  --start 2018-01-01 \
  --end 2024-12-31 \
  --models logistic xgb lgbm \
  --n-trials 30 \
  --feature-selection tree_topk \
  --top-k-features 4
```

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

## 4) 如何加更多 Feature

特征构建已解耦为「逐步添加」模式：

- `features/steps.py`：每个 `add_*` 函数只负责新增一列
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

2. 在 `features/core.py` 的 `DEFAULT_FEATURE_STEPS` 注册：

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

## 5) 如何加更多模型

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

## 6) 训练产物与 Cache 结构

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

---

## 7) 常见问题

- `yfinance` 无法访问：使用 `--synthetic`，或依赖自动 fallback
- 训练很慢：降低 `--n-trials`，先只跑一个模型（例如 `--models logistic`）
- 结果不稳定：固定时间区间、固定特征选择、提高样本量

# multi_rules：多意图拆分规则后处理实验（不训练模型）

与 `app/` 完全解耦，只读引擎首轮计划，不改线上链路。

```bash
python3 experiments/multi_rules/run_raw.py                     # 一次 LLM/条，缓存原始计划 raw_plans.json
python3 experiments/multi_rules/apply.py --no-rules --out experiments/multi_rules/base.csv
python3 tools/eval_3domain.py experiments/multi_rules/base.csv    # 基线
python3 experiments/multi_rules/apply.py                          # 套 rules.postprocess
python3 tools/eval_3domain.py experiments/multi_rules/result.csv  # 加规则
```

3domain_revised（263 条）：工具链 58.2% → 98.5%，整条全对 50.2% → 93.9%。
注意：规则是看着这套数据的错误模式写的，没有留出集，数值偏乐观；接入 app 前需在别的多意图集上复测。

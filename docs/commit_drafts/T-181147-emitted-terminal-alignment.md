# 建议 Commit

```text
feat(training): 补齐实际输出终点验收并增加可选终点保持

保留专家长度ADE/FDE，新增原生末点、联合目标支撑、后缀点数及全量逐样本诊断。
新增默认禁用的终点保持损失和一致teacher补齐，保持旧checkpoint默认兼容并拒绝跨策略恢复。
准备同LT30起点的LT31配对验证；正式学习收益及完整闭环尚待验收。
```

## 本轮代码与配置范围

- model/terminal_hold.py、model/__init__.py；model/network.py仅澄清原终点损失注释。
- training/trainer.py。
- metrics/emitted_terminal.py、metrics/prediction_analysis.py。
- scripts/analyze_predictions.py。
- tests/test_terminal_hold.py、tests/test_emitted_terminal.py、tests/test_prediction_analysis.py。
- configs/training/net-v1-terminal-hold-control.yaml、net-v1-terminal-hold-hold.yaml。
- spec/emitted_terminal_alignment.md及相关README/ARCHITECTURE/LEARNING/training_guide、LT31计划和日志。

## 实际验证

68项相关单测通过；真实同权重8训练+4验证/臂各1次新Adam更新，7.956秒预检通过；2918训练专家末点全部联合达标；归档LT30各300条输出的新公共指标复算一致；167代码/11输入/2配置SHA、4段PowerShell解析、2段嵌入Python编译、两个合成阶段筛选用例与Spec结构/差异检查通过。

预检仅验证实现，不代表终点保持有效。正式LT31未启动，无闭环回合或部署；未git add/commit/push。仓库有大量前轮改动，人工提交时按本轮范围审查，不一并收入无关修改。

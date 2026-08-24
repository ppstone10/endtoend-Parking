# 第三方声明

本文件记录项目实现所参考的第三方算法、来源、作者和许可证信息，便于审计、
分发与后续合规检查。项目自写说明使用中文；第三方专有名称、版权声明、
许可证标识以及未来可能收录的许可证原文保持原文，避免改变其法律含义。

## Reeds–Shepp 算法参考

`planner/reeds_shepp.py` 实现了 James A. Reeds III 与 Lawrence A. Shepp 在
*Optimal paths for a car that goes both forwards and backwards*
（Pacific Journal of Mathematics，1990）中发表的公式。实现过程中交叉核对了：

- Open Motion Planning Library (OMPL), `ReedsSheppStateSpace.cpp`, Copyright (c) 2010
  Rice University, BSD-3-Clause License；
- PythonRobotics, `reeds_shepp_path_planning.py`, Copyright (c) 2016 Atsushi Sakai and
  contributors, MIT License。

上述项目的许可证原文与当前源码以其上游仓库为准。本项目没有引入二者作为
运行时依赖；若未来直接纳入其源码或实质性代码片段，应同时在分发物中保留
对应版权声明和完整许可证文本。

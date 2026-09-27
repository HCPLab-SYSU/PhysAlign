# PhysGraph Pass 1–5 标注流程

整条流程把“题图和题干中直接可观察的事实”与“求解产生的推导”分开。Pass 1–4
构建答案盲化的 Observed-PhysGraph（G_obs）；Pass 5 只在人工批准后，把数据集原有
标准解答切分并对齐到 G_obs。

## Pass 1：视觉图元提取

识别图片中可定位的文字、点、线、曲线、箭头、区域、坐标轴与元件符号，记录
0–1000 归一化几何和可见文本。不创建物理实体，也不解释图元的物理含义。

## Pass 2：视觉—物理语义绑定

把 Pass 1 图元绑定到题图明确表达的物理实体，例如物块、导线、力箭头或电阻。
图中文字、视觉形状和物理实体保持为不同节点，通过 `labels`、`represents` 等绑定。

## Pass 3：题干 Grounding

从题干和问题中提取精确文本 mention、明确给定的物理量和约束，并与已有实体
绑定。选项不是事实来源；不得从正确选项、常识或答案补全条件。

## Pass 4：Observed-PhysGraph 组装

合并前三阶段，加入仅由 IMAGE 或 TEXT 直接支持的关系，校验所有 ID、provenance
与 evidence。不得输出未画出的力、由运动状态推断的量、辅助构造或任何解题结论。

Pass 4 是可导出的 G_obs。框架要求人工审核批准后才将其视为最终标注。

## Pass 5：标准解答步骤对齐

Pass 5 可以看到数据集原有的标准解答，但不能重新求解。它按原文顺序切分原子
步骤，并为每一步记录直接使用的 G_obs 节点、物理量、关系和约束 ID。Pass 5 不会
反向修改 Pass 1–4。

## 统一约束

- IMAGE 事实必须指向视觉 evidence；TEXT 事实必须保留精确 quote。
- 不确定内容进入 `ambiguities`，不得猜测或伪精确。
- ID 在阶段间稳定，后续阶段不得静默重编号前序实体。
- 所有阶段只接受 JSON Schema 允许的字段和枚举。
- 修改上游标注后，应重新校验所有依赖的下游阶段。

权威机器约束位于 `scripts/physgraph_annotation_lib.py` 与工作区 `schemas/`；公开模型
提示词位于 `physgraph_annotation/prompts/`。

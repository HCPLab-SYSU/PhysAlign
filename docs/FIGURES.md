# PhysAlign 结果可视化

入口为仓库根目录的 [`plot_results.py`](../plot_results.py)，实现位于独立包
[`physalign_viz/`](../physalign_viz/)。输入是**同一冻结 study 的已评分模型运行**，输出包括
18 类图、16 类表、离线 HTML 图册、完整精度数据和来源哈希。缺少必要观测的图会注明原因并跳过。
绘图不调用模型，可以在评测完成后用普通 CPU 机器执行。

已有冻结计划包含 `physalign/*.py` 的代码哈希。可视化放在同级独立包中，避免因增加绘图功能而使既有
study 失效；它直接复用已有评分、聚合和 bootstrap 函数。更改核心评分代码仍须遵守原来的冻结规则。

## 安装与生成

从仓库根目录运行。可视化依赖与 GPU 推理依赖分开安装：

```bash
python -m pip install -r requirements-viz.txt
```

三个模型完成 `score-study` 后：

```bash
python plot_results.py --study studies/final-v1 --run runs/final-v1/qwen35-9b --run runs/final-v1/qwen35-27b --run runs/final-v1/internvl35-8b --output figures/final-v1
```

输出目录须是新目录，且位于 study/run 目录之外。每次可使用 `figures/final-v2` 等新名字保留前一版。
无需重新推理，也无需先运行 `panel-report`。图表的模型顺序与 `--run` 顺序一致。
默认显示名称为 `Qwen3.5-9B`、`Qwen3.5-27B`、`InternVL3.5-8B`；日志中的原始模型 ID 完整保留。
可用 `--labels labels.json` 传入“原始模型 ID → 展示名称”的 JSON 映射。

只导出表格与数据时不需要导入 Matplotlib：

```bash
python plot_results.py --study studies/final-v1 --run runs/final-v1/qwen35-9b --run runs/final-v1/qwen35-27b --run runs/final-v1/internvl35-8b --output figures/tables-v1 --tables-only
```

更多参数：

| 参数 | 行为 |
|---|---|
| `--formats pdf svg png` | 默认同时输出三种格式；可只选部分格式 |
| `--dpi 450` | 默认 PNG 分辨率；可设为 600；矢量 PDF/SVG 不依赖 PNG 分辨率 |
| `--width 5.5` | 默认物理宽度，单位英寸；不要导出后再盲目缩小到不可读 |
| `--no-diagnostic-ci` | 不额外计算探索性分层、Gold 转移和解题分箱 CI；保留原报告 CI，模型间配对 CI 仍按冻结配置计算 |
| `--allow-incomplete` | 允许读取尚有 pending 请求、但已按相同状态评分的运行；不会把未服务项改为零或删除 |
| `--allow-non-scientific` | 仅用于 smoke/replay 等检查；每张图和每种表格式都保留非正式结果标识 |

首次运行会对每个模型输出 `audited_requests` 进度，最后输出图册路径、成功图数、缺少条件的图数和表数。
额外 bootstrap 的耗时取决于母题数、模型数、分层数与冻结重采样次数。


## 全部图与表

各图的英文完整说明保存在 `captions.md` 和图册中，说明统计集合、单位、对照参考与解释边界。

| 图 ID | 形式 | 内容 |
|---|---|---|
| 01_main_results | 多模型点图 + CI | CAcc、BAcc、JAcc、BAcc\|C、原题 SolveAcc |
| 02_reading_binding_quadrants | 水平堆叠条 | L 上四象限；C1B0 使用橙色斜线强调 |
| 03_gold_reading_control | Raw/Gold 哑铃图 + 差值图 | 同一配对 B 的变化与 Gold − Raw CI |
| 04_solving_association | 两组均值 + 差值图 | 式 (8) 中解对/解错母题的 b_i |
| 05_type_heatmap | 三面板热力图 | 类型 × 模型的识别、绑定、条件绑定错误 |
| 06_candidate_count | 跨模型折线图 | 候选数量 K 与分层 BAcc |
| 07_panel_count | 跨模型折线图 | 冻结面板数量与分层 BAcc；需要元数据 |
| 08_distance_bands | 有序分档点图 | 原图像素距离档位与 BAcc；不把不同宽度档位伪装成连续距离 |
| 09_interface_controls | 横向差值图 | 无图保留几何、无图无几何、重绘参考相对匹配主实验 |
| 10_permutation_stability | 两面板散点图 | Raw/Gold 下规范预测一致性与绑定准确率变化 |
| 11_solving_by_binding | 跨模型分箱折线图 | 母题绑定分数与该分箱中的 SolveAcc；探索性 |
| 12_human_reference | 同子集配对点图 | 模型与同接口人工 BAcc；需要推理前封存的人工证据 |
| 13_gold_transitions | 四类配对转移堆叠条 | Gold 下绑定错→对、对→错、双对、双错 |
| 14_dataset_profile | 数据组成条形图 | 类型 B/L 数量、候选数量分布和母题/簇总量 |
| 15_service_and_invalidity | 两面板点图 | 服务完成率与 Raw 已服务输出中的 JSON 对象无效率 |
| 16_token_usage | 输入/输出 token 箱线图 | 实际记录的 Raw token 用量；不是配置预算或推理成本排名 |
| 17_shortcut_baselines | 两面板基线对照 | 全 B 随机概率、最近区域合格子集；分开标明集合 |
| 18_model_contrasts | 有正负号的差值矩阵 | 行模型 − 列模型的 BAcc，配对 CI 在表 16 |

| 表 ID | 内容 |
|---|---|
| 01_main | 主结果：CAcc/BAcc/JAcc/BindErr\|C/SolveAcc 与可用 CI |
| 02_gold | 配对 Raw、Gold、差值、探针数、母题数 |
| 03_quadrants | 四象限的精确显示值 |
| 04_solving | 可靠答案母题数、解对/解错组大小、b_i 均值与差值 |
| 05_controls | 每个控制条件的参考值、控制值、差值与规范一致性 |
| 06_by_type | 各类型分数和 B/L 支持量 |
| 07_strata | 候选数/面板数/距离分层值、支持量与实际权重 |
| 08_binding_bins | 每个绑定分箱的母题数、来源簇数、SolveAcc |
| 09_gold_transitions | 四类绑定转移的份额与可用 CI |
| 10_human | Raw/Gold 同子集模型、人工及差距 |
| 11_service | 计划、完成、缺失、pending、Raw 对象无效数量 |
| 12_fields | 各必需输出字段的无效数量和相应分母 |
| 13_baselines | 随机/最近区域对照及最近区域的全池加权覆盖率 |
| 14_dataset | 每类 B/L 探针、母题和 Gold 探针数量 |
| 15_nonempty_packets | 预定非空信息包子集覆盖率与 Raw/Gold 结果 |
| 16_model_contrasts | 模型间配对差值与共享来源簇 CI |

每张表提供 CSV、Markdown、LaTeX；前四张核心表另外输出 PDF、SVG、PNG 排版预览。
长表保留所有行，用 `longtable` 跨页并重复表头；HTML 长表只展示前 100 行，下载文件保留全部行。

## 数学与来源约束

- **计算顺序不变**：逐探针 C、独立 B、J=C×B → 类型内母题均值 → 贡献母题等权 → 冻结类型权重 → 最后计算 JAcc/CAcc。
  `BindErr|C` 的 CI 由 `[l,u]` 转为 `[1-u,1-l]`，不能保留原端点顺序。
- **四象限共用 L**：q11=JAcc，q10=CAcc−JAcc，q01=BAcc_L−JAcc，q00=1−CAcc−BAcc_L+JAcc。
  图中 C1B0 不是 `1−BAcc`，也不是 `1−JAcc/CAcc`。
- **配对范围**：Gold 与配对 Raw 共享探针、母题与类型权重。`wrong_to_correct` 等转移先逐探针计算，再聚合；
  错→对减对→错等于配对 Gold−Raw。这里是局部正确读取对照，不能叫作选项上下文干预。
- **母题求解**：式 (8) 的 b_i 是母题所有 B 探针的普通均值，不重新按类型等权；
  SolveAcc 的分母是事前有可靠答案的母题。母题级关联不把一题的多个探针当独立题目。
- **缺失和无效不同**：未服务/pending 保留预定分母，点估计缺失并保留界限；模型已返回的无效对象/字段按原评分计错。
  图中缺值不画为 0；表中 `--` 表示未定义。完整精度界限在 `figure_data.json` 中。
- **共享抽样**：直接调用已有来源簇 bootstrap，保留重复抽中母题的重数。Raw 全量、Gold 配对、可靠答案母题分别使用对应抽样范围。
  模型差值在每次相同抽样上重算两个模型后相减，不从各自区间端点相减。出现无定义重采样时保留 withheld CI。
  区间的置信水平、种子、重复次数均读取冻结 study，不在图上偷偷改为另一种误差条。
- **分层为探索性诊断**：候选数、面板数、距离档位成员来自冻结元数据；子集内重新计算母题/类型权重，与既有报告的分层规则相同。
  不跨档平滑或拟合趋势。分箱边界 `[0,.2,.4,.6,.8,1]` 固定在绘图版本；左闭右开、最后包含 1。
  不同模型的同一母题可能落入不同箱，表 08 明示支持量。任一合格母题 b_i 或原题正确性缺失时，该模型整条分箱曲线 withheld。
  这些诊断不是对事前主假设的追加确认，也没有对多重比较作显著性校正。
- **控制参考正确**：置换使用相同渲染器的 identity reference，而非图像呈现不同的主 Raw。
  最近区域只在批准子集与模型比较；随机基线先算每项正确概率，再按原权重聚合，不计算 `1/平均 K`。
- **输入审计**：读取首次响应日志，核对计划、请求、来源图片、运行身份和报告，再从响应重新评分并核对主指标、配对结果、控制、原题自动判分、CI 点值及人工封存。
  不重写评测结果。人工原题判分沿用用户已评分报告中的人工决定，保留其 `manual_grades_hash` 与报告哈希；绘图不是新的人工判分步骤。
  多模型必须同一计划、不同 model ID，并使用相同解码/thinking/token 上限配置。模型各自原生预处理仍完整保留在数据中，不能据此宣称计算量完全相等。

数据文件中的分数是 0–1 比例，图表展示为百分数；差值单位是百分点（pp）。CSV 为**与排版表一致的显示值**，
可能包含 CI 字符串；如需数值复算，应读取 `figure_data.json` 的未舍入值。

## 投稿排版与复现

默认物理宽度为 **5.5 英寸**，可通过 `--width` 按目标版式调整。

采用白底、蓝/橙/青的模型配色，辅以圆/方/三角形标记；主要百分比轴保留 0–100 的共同范围，差值图显示零线。
不使用 3D、雷达图、双 y 轴或把无序指标连接成趋势线。PDF 内嵌 TrueType 字体，SVG 用字形路径保证跨机器显示，
没有依赖操作系统中文字体。字体和导出行为参见
[Matplotlib 配置文档](https://matplotlib.org/stable/users/explain/configuration.html)
与 [字体文档](https://matplotlib.org/stable/users/explain/text/fonts.html)。

在论文导言区准备：

```latex
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{longtable} % 附录长表；不要嵌套在 table 浮动体中
```

正文示例：

```latex
\input{figures/final-v1/tables/01_main.tex}

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/final-v1/figures/02_reading_binding_quadrants.pdf}
  \caption{Use the exported caption and adapt the conclusion to the measured results.}
  \label{fig:reading-binding}
\end{figure}
```

表标题位于上方，使用 booktabs 横线，无竖线；CI 放在点值下方。表列宽按 `\linewidth` 分配。
`figure_manifest.json` 记录绘图代码哈希、软件版本、尺寸、来源计划与输出文件哈希；
`figure_data.json` 保留每个模型报告哈希、推理配置、诊断定义与完整值。HTML 图册完全离线，不依赖外部字体或脚本。

测试命令：

```bash
python -m unittest discover -s tests -v
```

新增测试覆盖非均衡母题手算反例、Gold 子集分母、四象限、式 (8)、共享簇重数、分箱边界、
缺失服务、过期报告、CI 端点变换、人工封存匹配、合成标识、来源不变与真实图像导出尺寸。
图像渲染测试需要 `requirements-viz.txt`；未安装时该项明确跳过，其他表格与数学测试仍可运行。

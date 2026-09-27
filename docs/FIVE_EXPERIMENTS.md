# 前五个实验的完整运行流程

对应 Proposal 第 5–6 节、附录 D–F 及中文 idea 的主评测、对照和统计设计。
以下命令在项目根目录执行。

## 实验与代码

| 实验 | 执行条件 | `study_report.json` 结果 |
|---|---|---|
| 读取—绑定主评测 | `main.raw`，原公开输入和全部图片 | `experiments.main.raw_all`：CAcc/BAcc/JAcc、条件错误率、四象限、类型明细与 CI |
| 正确局部识别对照 | `main.gold`，仅增加批准的局部转录 | `experiments.main.paired`：同一集合 Raw/Gold、BAcc_gold、配对差值 |
| 原题正常求解 | 每个母题一次 `solve.solve`，原题及原图，独立上下文 | `original_solving`：SolveAcc、式 (8) 关联、原题答对但局部绑定出错的母题 |
| 捷径与接口对照 | 随机期望、最近区域、无图、仅改编号、仅重排、同时变化 | `uniform_random_full/by_type`、`nearest_region`、`control_comparisons` 与各条件指标 |
| 数据及人类同接口审计 | 真实浏览器作答、人工审核、争议裁决、运行前封存 | `human_same_interface`、`human_model_same_subset`、独立人工报告与 `seal.json` |

`study.py` 冻结输入；`hf_adapter.py` 推理；`controls.py` 构造对照；`solving.py` 判原题；
`human.py` 提供人工界面；`study_reporting.py` 评分；`panel.py` 比较模型。
所有统计复用已有 `scoring.py / metrics.py / bootstrap.py`。

## 样例验证：不加载模型

准备对照视图需要 Pillow，核心评分仍只用标准库。

```bash
python -m pip install Pillow==11.3.0
python evaluate.py draft-study --dataset examples/synthetic --output plans/starter-study-spec.json
python evaluate.py prepare-study --dataset examples/synthetic --spec plans/starter-study-spec.json --split framework_development --output studies/starter-five-smoke
python evaluate.py run-study --study studies/starter-five-smoke --output runs/starter-five-smoke --dry-run
python evaluate.py run-study --study studies/starter-five-smoke --output runs/starter-five-smoke --adapter smoke
python evaluate.py score-study --study studies/starter-five-smoke --run runs/starter-five-smoke
```

默认生成 **54 个请求**：6 Raw、3 Gold、6 无图保留几何、9 重绘参考、3 组各 9 个置换请求、3 原题。
随机和最近区域离线计算。`smoke` 返回空 JSON，报告标记 `scientific_run:false`，不代表模型结果。
随附合成样例只供开发，不会成为未见 test。

已有目录不会被首次准备/运行覆盖。复现使用新目录，续跑使用 `--resume`。
代码哈希在准备时固定；修改评测模块后须建立新 study。

## 完整数据和原题答案

支持现有 starter 格式及 [`EVALUATION.md`](EVALUATION.md) 定义的 `physalign_eval_bundle_v1`。
先生成并人工审核 spec：

```bash
python evaluate.py draft-study --dataset /data/physalign-final --output plans/final-study-spec.json
```

`original_problems[母题ID]` 有两部分：

- `original`：未经改写的原题、原选项、原图 ID 顺序、来源及 `unchanged_original_verified`。
  starter 从 source manifest 提取。原生包可增加清单覆盖的 `public/problems.json` 数组，
  每项为 `problem_id/user/asset_ids`。无材料保持 `null`，不能拼接局部探针或 Gold 来代替原题。
- `answer_key`：独立可靠答案或人工评分参考，草稿统一为 `null`。必须在看到模型输出前确定，
  只进入 private 文件，不进入模型请求。

例如，**仅在真实答案确为 B 时**填写：

```json
{"kind":"choice","accepted":["B"],"reliable":true,"source_reference":"实际答案来源及版本"}
```

| 判分 `kind` | 字段和规则 |
|---|---|
| `choice` / `exact` | `accepted` 字符串数组；去首尾空白后精确匹配，区分大小写 |
| `choice_set` | `accepted` 为正确选项集合；模型 `answer` 必须是字符串数组 |
| `quantity` | `expected/unit_scales/absolute_tolerance`；精确数值与单位规则，不猜容差 |
| `human` | `reference_answer` 保存标准答案及评分要点；实际人工判 0/1 |

原题输出要求一个 JSON 对象，最终答案在 `answer`，可含 `explanation`。
自动判分不从自由文本猜选项、不修复 JSON。复杂推导、多问或不适合精确匹配的题用 `human`。
没有可靠答案的母题不进入预定 SolveAcc 分母，并单独列出；有答案而未服务/未判分的母题保留为缺失，
给出覆盖率与上下界，不能计错或偷偷移除。

`controls[探针ID]` 字段：

- `nearest`：只用于数量/标签归属题，声明 `field/anchor_id/eligibility_basis`，其他为 `null`。
  所有候选须在锚点所在原图有可比区域，不能根据模型错误选子集。
- `no_geometry_compatible:true` 须有 `no_geometry_basis`，说明去掉视觉几何后接口仍明确。默认关闭。
- `panel_count`：人工注释的原图面板数；未知为 `null`，不能用定位图片数量代替。

spec 顶层 `permutations_per_mode/permutation_seed` 冻结置换；默认每种变换 1 次，可实验前增加。
`human_sample_mothers` 指定人工母题；`null` 按 `human_seed` 固定抽最多 20 道母题。
更大或分层的人工样本应事前列出母题，不按模型错误筛选。

```bash
python evaluate.py prepare-study --dataset /data/physalign-final --spec plans/final-study-spec.json --split test --output studies/final-v1 --bootstrap-resamples 2000 --seed 2027
```

`test` 要求上游正式发布资格、母题/重复簇不跨 split、排除开发保留题，以及**每个 B 探针都有批准的 Gold**。
无额外转录的探针也须显式导出经审核的空包；脚本不会补造未导出的 Gold。
额外保留簇可传 `--reservation-file reservations.json`，格式为 `{"problem_ids":[...],"cluster_ids":[...]}`。
跨数据包近重复关系须由上游全局去重提供。

## 服务器依赖和模型准备

使用 Python 3.10+，建议独立环境。先装与服务器驱动相容的 PyTorch/torchvision CUDA 组合。
驱动支持 CUDA 12.4 wheel 时，可使用官方列出的组合：

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-eval.txt
python -m pip freeze > environment-lock.txt
```

配对依据：[PyTorch 官方安装表](https://pytorch.org/get-started/previous-versions/)。
实际版本须适合服务器驱动。本地未执行 GPU 推理，仍需在服务器开发数据上验证环境与显存。
评测依赖固定 Transformers 5.17.0、Accelerate 1.15.0、Pillow 11.3.0。
默认 BF16、SDPA，不要求 FlashAttention 或量化组件。

| 模型 ID | 配置 | GPU |
|---|---|---|
| `Qwen/Qwen3.5-9B` | `configs/server/qwen35-9b.json` | 0,1 |
| `Qwen/Qwen3.5-27B` | `configs/server/qwen35-27b.json` | 2,3,4,5 |
| `OpenGVLab/InternVL3_5-8B-HF` | `configs/server/internvl35-8b.json` | 6,7 |

之后在服务器准备官方 HF 完整本地目录，包括权重、配置、processor、tokenizer 和 chat template。
**代码不会自动下载模型**。以下只对已存在文件计算哈希，清单存放在模型目录之外：

```bash
python evaluate.py freeze-model --model-path /models/Qwen3.5-9B --model-id Qwen/Qwen3.5-9B --output models/qwen35-9b.snapshot.json
python evaluate.py freeze-model --model-path /models/Qwen3.5-27B --model-id Qwen/Qwen3.5-27B --output models/qwen35-27b.snapshot.json
python evaluate.py freeze-model --model-path /models/InternVL3_5-8B-HF --model-id OpenGVLab/InternVL3_5-8B-HF --output models/internvl35-8b.snapshot.json
```

首次/续跑核对完整快照清单和哈希。默认每卡 18 GiB 权重预算、输入上限 32768 token、
输出上限 2048 token、贪心解码、单请求 batch；这个分配为激活/KV 留空间，但不保证任意多图题都能容纳。

正式生成之前，可只运行处理器检查所有输入：

```bash
python evaluate.py inspect-model-inputs --study studies/final-v1 --adapter-config configs/server/qwen35-9b.json --output plans/preflight-qwen9.json
python evaluate.py inspect-model-inputs --study studies/final-v1 --adapter-config configs/server/qwen35-27b.json --output plans/preflight-qwen27.json
python evaluate.py inspect-model-inputs --study studies/final-v1 --adapter-config configs/server/internvl35-8b.json --output plans/preflight-internvl8.json
```

它在 CPU 上使用**同一编码函数**解码全部图片并应用实际 chat template/processor，保存 token 数、
图像张量形状、输入预算及“输入 + 预留输出”的原生上下文检查；不加载模型、不生成答案、不验证 GPU 峰值显存。
随后在开发数据实际推理，检查显存和处理后小标签可见性，再固定配置用于正式 test。

`processor_kwargs:{}` 使用各自原生默认值；实际处理配置、处理器/分词器版本写入 run manifest。
开发阶段可配置模型原生支持的图像预算参数，但三个模型不保证接受同一参数或有同样预处理。
改变清晰度、图像预算、精度、GPU 分配或输出预算须在正式实验前完成，不能某题失败后自动降采样、删图或重答。

Qwen 显式设置 `enable_thinking=False`；InternVL 保持原有系统提示，不加入其特殊 thinking 提示。
三模型配置均记录 `thinking:false`，未额外运行 thinking 消融。
已对照 [Qwen3.5 模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)、
[InternVL3.5 HF 模型卡](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF)、
[Transformers InternVL 接口](https://huggingface.co/docs/transformers/model_doc/internvl)，
并检查 Transformers 5.17.0 的自动模型映射、图像消息编码和像素张量路径。

## 人工同接口作答、审核和裁决

人工流程不需要 GPU。每个实际参与者使用不同匿名编号，两组交叉得到同一批探针的 Raw/Gold。
单人不会看到同一探针的另一条件；无 Gold 的探针两组均答 Raw。参与者不能跨组重复作答或兼任审核者。

```bash
python evaluate.py human-create --study studies/final-v1 --sessions human_sessions/final-v1 --participant reader01 --cohort 0
python evaluate.py human-create --study studies/final-v1 --sessions human_sessions/final-v1 --participant reader02 --cohort 1
python evaluate.py human-serve --study studies/final-v1 --session human_sessions/final-v1/answer-reader01 --port 8765
```

打开打印的带 token 地址。只监听 `127.0.0.1`；远程可用
`ssh -L 8765:127.0.0.1:8765 user@server` 转发，再本地打开该地址。
其他参与者各开自己的会话，多人并发用不同端口。中断后重新打开原会话即可。
界面展示原样 system/user、全部图片及 asset ID、局部包和相同 JSON 输出合同。
点击图片可查看原尺寸；提交保存图片自然/显示尺寸。首次提交不可修改，不反馈正确答案。
盲测路径不读取 `private.json`，HTTP 不提供文件下载路由。

审核者独立运行：

```bash
python evaluate.py human-create --study studies/final-v1 --sessions human_sessions/final-v1 --participant audit01 --mode audit
python evaluate.py human-serve --study studies/final-v1 --session human_sessions/final-v1/audit-audit01 --port 8766
python evaluate.py human-report --study studies/final-v1 --sessions human_sessions/final-v1 --output plans/human-final-report.json
```

审核预定人工母题的主条件、重绘参考及适用的无几何条件，检查源版本、锚点、候选覆盖、歧义、
Gold 泄漏、最小标签可见性、坐标和遮挡。备注记录实际尺寸/可见性问题和处理方式。
重绘坐标变换为 `x'=x,y'=y+32`；原图不缩放，标题 32 px、边框 2 px，字体/Pillow 版本固定。
检查失败不能提交 accept。出现歧义或审核分歧时，创建裁决会话查看此前记录，逐题填写理由：

```bash
python evaluate.py human-create --study studies/final-v1 --sessions human_sessions/final-v1 --participant adjudicator01 --mode adjudicate
python evaluate.py human-serve --study studies/final-v1 --session human_sessions/final-v1/adjudicate-adjudicator01 --port 8767
```

确需改题或改图应修正上游并建立新 study，不能改旧答案日志。
人工评分先在探针内平均参与者，再按母题/类型聚合，输出同探针的 Raw/Gold 差值。
一致率只用真实同条件重复回答；没有重复参与者时为 `null`。

```bash
python evaluate.py seal-study --study studies/final-v1 --sessions human_sessions/final-v1
```

最终 test 运行前须有实际人工覆盖、预定审核全部通过，且每个歧义报告都有对应裁决。
`seal.json` 保存证据和哈希；model run 引用运行前的封存版本，不能倒填事后审核。
模型与人类差距只在**同一人工母题/探针子集**比较，不拿全量模型分数减小样本人类分数。
仓库没有预填真人结果。

## 8 张 RTX3090 启动与恢复

```bash
python scripts/run_panel.py --study studies/final-v1 --output runs/final-v1 --dry-run
python scripts/run_panel.py --study studies/final-v1 --output runs/final-v1
python scripts/run_panel.py --study studies/final-v1 --output runs/final-v1 --resume
```

三个进程分别可见 2/4/2 张卡，模型组内使用 Accelerate 分片，不用 `torchrun` 创建八份副本。
日志在 `runs/final-v1/launcher_logs/`。拒绝 CPU/disk offload 和不匹配的可见 GPU 数。
单模型 Linux 启动例：

```bash
CUDA_VISIBLE_DEVICES=0,1 python evaluate.py run-study --study studies/final-v1 --output runs/final-v1/qwen35-9b --adapter hf --adapter-config configs/server/qwen35-9b.json
```

改变 GPU 分组时同步修改启动器和配置的 `gpu_count`，并在新运行前确定。
各探针/条件/原题重新建立上下文。保存首次实际输出、生成 token ID、带特殊 token 的解码、usage、
finish reason、revision 和配置。评分使用去特殊 token 后的整个新生成后缀，不剪 JSON、不修复或重答错题。
OOM/上下文超限记基础设施缺失，不自动变更输入或设置。

## 离线评分和跨模型比较

```bash
python evaluate.py score-study --study studies/final-v1 --run runs/final-v1/qwen35-9b
python evaluate.py score-study --study studies/final-v1 --run runs/final-v1/qwen35-27b
python evaluate.py score-study --study studies/final-v1 --run runs/final-v1/internvl35-8b
python evaluate.py panel-report --study studies/final-v1 --run runs/final-v1/qwen35-9b --run runs/final-v1/qwen35-27b --run runs/final-v1/internvl35-8b --output runs/final-v1/panel-report.json
```

若原题使用 `human` 判分，先导出实际作答，真人填写 `correct/grader_id/human_confirmed/rationale` 再评分：

```bash
python evaluate.py grading-queue --study studies/final-v1 --run runs/final-v1/qwen35-9b --output plans/solve-grades-qwen9.json
python evaluate.py score-study --study studies/final-v1 --run runs/final-v1/qwen35-9b --grades plans/solve-grades-qwen9.json
```

每个模型用自己的评分文件，逐项校验 plan/run/request、响应及评分参考哈希。
`panel-report` 汇总主指标和配对 Gold 的模型间差异；原题、人工差距及全部控制详情在各自 `study_report.json`。

论文图表可在评分完成后用 CPU 生成，独立于模型推理：

```bash
python -m pip install -r requirements-viz.txt
python plot_results.py --study studies/final-v1 --run runs/final-v1/qwen35-9b --run runs/final-v1/qwen35-27b --run runs/final-v1/internvl35-8b --output figures/final-v1
```

18 类图、16 类表的设计、导出格式与解释边界见 [`FIGURES.md`](FIGURES.md)。可视化位于独立包，
新增绘图代码不更改已冻结的评测核心哈希。

| 文件 | 内容 |
|---|---|
| `studies/.../plan.json` | 公开请求、固定集合/权重/置换/分层、哈希 |
| `studies/.../private.json` | 评分合同、原题答案、最近区域预测；不向模型传递 |
| `studies/.../images/` | 未修改的原图/导出视图副本及单独重绘的控制图 |
| `runs/.../manifest.json` | 数据/代码/模型/处理器版本、推理配置、封存版本 |
| `requests/attempts/results/` | 输入快照、持久化调用日志、首次响应或缺失 |
| `predictions.jsonl` | 按计划顺序导出的原始预测 |
| `study_scores.jsonl` | 逐项 C/B/J、字段有效性、规范物理预测 |
| `study_report.json` | 单模型五实验指标、支持量、缺失、CI、诊断 |

## 数学与对照的固定解释

1. 逐项 C、独立 B、J=C×B → 母题内同类型平均 → 贡献母题等权 → 固定类型权重 → 条件比例。
   B/L 分别冻结权重，默认集合内类型等权；错误四象限仅在 L 分解。
2. Raw/Gold 差值必须同探针同权重；starter 的 3 条配对 Gold 不能减全量 6 条 Raw。
   未服务项保留分母、点估计置空并给上下界。
3. 式 (8) 的 `b_i` 为母题全部 B 探针的**普通均值**；bootstrap 只抽预定可靠答案的 `(A_i,b_i)` 记录。
   正确/错误组为空则关联差未定义；关联不解释为因果效应。
4. 默认 2000 次母题/来源簇 bootstrap，同条件和模型共享抽样次数，保留重复母题重数。
   全量、配对子集各用自己的抽样范围；缺固定类型或条件分母为零时不重配权重，记录无定义次数并保留空 CI。
5. 随机基线先算各项概率再聚合，不能算 `1/平均候选数`。集合/关系未声明分布时不套单目标公式，
   完整混合池随机值明确未定义，不删掉这些探针凑基线。
6. 最近区域是原图像素尺度的矩形边界欧氏距离，重叠/相接为零；先换算横纵尺度再合并，
   精确有理数比较平方距离、按冻结原始候选 index 破平局。同子集比较模型和启发式并报告全池覆盖。
7. `no_image_geometry` 只去像素，保留公开文本和坐标，明确称“无图、保留几何”。
   `no_image_no_geometry` 另去视觉坐标，不补实体描述，保留原文定位，仅用事前批准的兼容子集。
8. 置换的烙印编号须同步修改，所以从原图重绘；**所有置换与 `permutation_reference` 用同一渲染器**。
   主实验保留原导出图；重绘参考相对主实验的变化单独报告。置换差值相对重绘参考计算，
   候选编号、列表、文字引用、图片编号和私有映射同步更新，读取锚点不变。
   当前 Gold 仅接受 `{anchor_id,text}` 的原样转录，不把物理文字当候选编号替换；
   其他候选引用/描述字段须有上游明确的新合同，当前严格拒绝。
9. 比较规范物理预测与准确率变化；两个无效输出不算稳定物理预测，一致错误仍计错。
   候选数、面板数、距离分层事前确定；距离分档 0、(0,10]、(10,50]、(50,∞) 原图像素。

## 验证范围

```bash
python -m unittest discover -s tests -v
```

覆盖精确数学反例、合成样例传图、独立原题、置换规范身份、缺失分母、抽样范围、人工会话/封存、
评分证据哈希、三模型 SDK 测试替身、处理器预检查、超长输入/OOM/无效输出。
测试替身不代表模型或真人实验。本次未下载模型权重，未执行三个模型的 GPU 推理；
正式结果须在服务器运行，并提供独立可靠原题答案和实际人工测量。

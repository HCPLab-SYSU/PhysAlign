# PhysAlign 评测脚本

本框架实现 Raw / Gold 局部探针评测流程，直接使用已经导出的提示词、
候选、锚点和图片，并调用现有 `scoring.py`、`metrics.py`、`bootstrap.py`。

要求 Python 3.10+，核心框架仅依赖标准库。真实模型的 SDK、权重和推理由后续适配器提供。

## 运行

在项目根目录执行，输出目录位于数据包之外：

```powershell
python evaluate.py prepare --dataset examples/synthetic --split framework_development --plan plans/starter.json
python evaluate.py run --plan plans/starter.json --output runs/starter-smoke --adapter smoke
python evaluate.py score --run runs/starter-smoke
```

标准 Python 环境也支持 `python -m physalign`，其参数与 `evaluate.py` 相同。

`prepare` 在调用模型之前完成校验并写出计划。默认冻结 2,000 次 bootstrap、种子 2027、
95% 置信水平，以及最多两次基础设施重试。流程调试可显式使用 `--bootstrap-resamples 0`。
`--conditions raw` 只调度 Raw；默认 `--conditions raw gold` 调度所有已有的匹配 Gold。
`--problem-id` 可重复指定，选择单位是完整母题，其类型权重会在运行前重新冻结。
计划文件不可覆盖；改变数据或设置时应准备新计划并使用新的运行目录。

续跑命令：

```powershell
python evaluate.py run --plan plans/starter.json --output runs/starter-smoke --adapter smoke --resume
```

`smoke` 读取实际图片字节，返回固定的 `{}`。报告中的 `scientific_run=false` 明确表示这只是
流程检查。`replay` 可回放已有公开请求的响应，同样标为非科学实验。真实模型运行应指定
`--adapter your_module:create_adapter --adapter-config configs/model.json`。

## 数据流和文档约束

| 阶段 | 输入与职责 | 约束 |
|---|---|---|
| `dataset.py` | 校验公开记录、图片及私有评分定义 | 保留原文、候选顺序、锚点、图片顺序；不自动改写或生成探针 |
| `planning.py` | 冻结母题、类型、评测集合、配对、权重、哈希 | 只按模型运行前的元数据选择，核对最终审核记录 |
| `runner.py` | 公开文本和图像字节传给适配器，保存响应 | 每条探针和每个条件使用独立请求，运行阶段不读私有评分文件 |
| `reporting.py` | 离线加载私有答案，逐项评分、汇总 | 核对计划、输入快照、首次响应和源文件；不请求模型解释或修复输出 |

脚本不要求模型生成 PhysGraph、完整物理解答或推理链。

Raw / Gold 必须属于同一逻辑探针，并保持同一系统提示词、局部问题、输出字段、候选与
图片顺序。公开用户提示词只能在 `[Local reading information]` 部分不同。
信息包只接受锚定转录项 `{"anchor_id":"R1","text":"..."}`，禁止夹带 `owner` 等绑定字段。
样例的非空信息包还必须等于私有键指定的转录，并对应已经通过的最终信息包审核。
Gold 的复制读取不会产生 `C` 或 `J` 分数。

代码能验证字段、身份、哈希、最终审核证据及已批准信息包的一致性。
“文字内容是否间接泄露绑定”“读取是否确有必要”“候选是否物理合理”等语义资格仍须上游审核，
不能通过增加一个审核标志来自动获得。本框架不会把模型审核解释成人工审核。

## 统计口径

样例包有 6 条 Raw，只有 `sample_04`、`sample_05`、`sample_06` 导出了 Gold：

| 报告部分 | 固定集合 | 含义 |
|---|---|---|
| `raw_all` | 6 条 Raw | 全部绑定探针 B，以及其中具有显式读取目标的 L |
| `paired` | 上述 3 条探针的 Raw 和 Gold | 同一子集上的配对比较；不能解释为全部 6 条的 Gold 效果 |
| `nonempty_packets` | 对应评测集合中实际具有非空包的探针 | 使用该子集独立冻结的类型权重与母题支持 |

不存在的 Gold 不会被补成 Raw，不会计作服务故障。存在但为空的信息包仍可形成真实配对，
但不会进入非空包子集。已计划的请求若发生基础设施缺失，则保留在原集合和分母中。

计算顺序始终是：逐探针计算 `C`、独立的 `B` 和 `J=C*B`；同类型内先求各母题均值，
再对有贡献的母题等权平均，随后按固定类型权重汇总；最后计算 `JAcc/CAcc`。
全 B 与 L 的权重分别冻结。样例的任务类型取 `task_id`，因此 `T03-image` 与 `T03-text`
属于同一 T03 类型；接口字段另外保留，不能偷偷改变类型权重。

默认各集合内部的类型等权。自定义权重通过 `prepare --weights weights.json` 提前提供：

```json
{
  "full": {"binding": {"T02": 0.25, "T03": 0.75}},
  "paired": {"binding": {"T03": 1.0}}
}
```

每个作用域可分别指定 `binding`、`joint`、`packet`、`binding_only`。提供的权重必须恰好覆盖
该集合中的类型且和为 1；不会自动归一化。未提供的集合按该集合元数据确定等权。

bootstrap 的采样单位是母题或预先声明的重复来源簇；同一抽样中 Raw / Gold 共享母题重复次数，
所有均值和条件比例重新计算。全量与配对子集各有自己的采样范围。
某次抽样丢失必需类型，或条件分母为零，会记录未定义次数并保留空区间，不能删除这些抽样
或按剩余类型重新分配权重。JSON `null` 表示未定义，不代表准确率为零。

服务缺失导致相应点估计为 `null`，`summaries` 同时给出固定分母下的上下界和加权服务覆盖率。
均匀候选基线先计算每条单目标探针的正确别名概率，再执行同样的母题/类型聚合。
集合或关系任务没有默认的随机分布；混合评测池会显式标记此基线未定义，不擅自删掉这些任务。

## 模型适配接口

实现一个可导入模块，提供 `create_adapter(config)`，返回具有 `info` 和 `generate(request)` 的对象。
核心类型位于 `physalign.adapters`。`info` 示例：

```python
from physalign.adapters import AdapterInfo, ModelRequest, ModelResponse, InfrastructureError

info = AdapterInfo(
    name="your-adapter",
    model_id="your-model",
    revision="exact-model-revision",
    implementation_version="1",
    settings_json='{"temperature":0,"max_output_tokens":128,"seed":2027}',
    preprocessing_json='{"resize":false,"image_detail":"original"}',
)
```

`revision` 无法获得时可以是 `None`，不能捏造固定版本。记录模型实际接受的解码参数、预算、
图像预处理策略，以及适配器实际生效的默认值。认证信息从环境等渠道读取，不写入元数据。

`generate(request: ModelRequest) -> ModelResponse` 收到的只有：

- `request_id`：此次计划请求的唯一标识，可用于供应商幂等键。
- `system`、`user`：公开导出中的原始消息。
- `images`：按公开顺序排列的不可变元组；每项包含 `asset_id`、MIME、尺寸、SHA-256 和真实 `bytes`。
- `settings_json`：本次运行冻结的模型设置。

`request.image_data_urls()` 可生成同顺序的图片 data URL。适配器必须将它们作为图像内容发送，
并保持 `asset_id` 与图像的对应，不能把磁盘路径或 base64 当普通问题文本。原始图和导出的
定位视图都要传入，不能凭图片数量或模型输出自动删图、裁剪或降采样；必要预处理应事先声明。

每次 `generate` 都应构造全新上下文，只有当前请求的公开消息与图像，禁止前一探针消息、
Raw 回答、`previous_response_id` 或人工修复提示。可以复用已加载的权重或网络客户端。
框架不向适配器传 `Probe`、答案、候选的私有身份映射或审核文件。Python 适配器并非操作系统
安全沙箱，接入的适配器必须遵守这一边界；当前测试验证了内置执行路径不打开私有文件。

将模型**原始最终输出字符串**放入 `ModelResponse.text`，保持原样，不提取/修复 JSON，
不拿推理内容替换最终答案。可记录 `finish_reason`、`usage_json`、供应商返回的模型版本和请求 ID。
报告会列出实际返回的所有版本，并标记运行中出现多个版本的情况；这不能被当成固定版本的比较。

仅在确定没有收到可供评测的响应时抛出 `InfrastructureError("stable_error_code")`。
不支持当前图像规模等不可重试故障可指定 `retryable=False`。错误码不应含认证信息。
拒答、空的实际输出、格式错误、答错、预算截断都返回 `ModelResponse`，不得触发正确性重试。
部分输出一旦已收到就应保留，不能因为传输随后断开而丢弃它去获取第二份答案。
适配器应关闭会重新执行生成的隐藏 SDK 重试；服务重试总数由本框架统一管理。

`ReplayAdapter` 的文件格式是 `{公开请求内容指纹: 响应对象}`。指纹计算为
`fingerprint(request.public_snapshot() 去掉 request_id)`，响应对象字段与
`ModelResponse.record()` 一致；可通过 `--adapter replay --adapter-config replay-config.json`
调用，配置内容为 `{"file":"path/to/replay.json"}`。

## 首次输出、故障与恢复

每次调用前持久保存公开输入快照和开始记录；调用完成后先持久保存原始输出，再生成终态记录。
重试使用完全相同的请求和设置，只允许“无响应的可重试基础设施错误”，最多初次加两次重试。

| 恢复时的日志状态 | 行为 |
|---|---|
| 已有完整终态 | 校验一致性并复用，不再调用模型 |
| 已有完成的响应，但尚未写终态 | 从已保存的首次响应恢复终态 |
| 已保存无响应基础设施错误 | 按原计划剩余预算继续重试 |
| 有开始记录但没有完成记录 | 标记 `interrupted_request_unknown_response`，不重新调用 |
| 尚未开始 | 正常发起首次请求 |

不能确认崩溃时是否收到过回答的请求保守保留为缺失。程序错误会停止运行，不会被包装成
答错或自动重试。`--resume` 核对数据、模型配置、适配器来源和代码版本，拒绝混用不同实现。
同一个运行目录有操作系统文件锁，不能并发写入或边推理边评分。哈希用于发现不一致，
不构成对恶意重写全部日志的密码学认证。

## 输出文件

```text
plans/starter.json             运行前冻结的计划、数据与评分代码哈希、集合和权重
runs/starter-smoke/
  manifest.json               计划、模型配置、适配器及代码来源、运行 ID
  requests/<id>.json          公开消息和图像身份/字节数快照
  attempts/<id>.<n>.*.json    每次尝试的开始/完成记录及原始响应
  results/<id>.json           从尝试记录推导的最终首次输出或基础设施缺失
  predictions.jsonl          按冻结计划顺序导出的原始预测
  status.json                有终态的输出、基础设施缺失和尚未形成终态的数量
  scores.jsonl               逐项 C/B/J、接口、解析结果和字段有效性、集合 F1
  report.json                全量 Raw、配对报告、类型明细、支持量、区间、基线和版本信息
```

`score` 不访问模型。它会核对 `predictions.jsonl` 与独立结果/尝试记录的一致性，不能只改预测
导出文件来更换答案。默认要求运行有所有终态；`--allow-incomplete` 允许诊断未完成运行，
所有未完成项保持为缺失并单独列出，不会缩小评测集合。报告可重复生成，也可用 `--output`
写到另一个 JSON 路径；不会覆盖无关文件。数据包搬迁时，`run` 和 `score` 可用 `--dataset`
指定哈希一致的新位置。

原始响应始终保存在预测及尝试日志里。`scores.jsonl` 的 `parsed_output` 是方便检查的解析副本；
若无关字段的极大数值等无法重新序列化，它会为空并由 `parsed_output_status` 明确说明，
不改变请求字段的评分。续跑后需要重新执行 `score`，以更新此前生成的诊断报告。

## 已有样例与后续完整数据

现有样例采用 `physalign_eval_starter_samples_v1` 和 `physalign_public_qa_v1`，直接支持，
无需修改或迁移。私有键、简化答案、母题归属和最终 `result/audit/blind` 记录会交叉核验。
较早导出快照中的 `pending_review` 不会覆盖最终审核结论。样例仅供框架开发，
内置包由程序从零生成，只有 3 个合成母题，不能作为正式 test。审核字段是测试桩，未执行真人或模型审核。

完整数据也可导出为本框架明确规定的 `physalign_eval_bundle_v1`。这是新增的输入契约，
不声称当前未展示的数据已经采用此格式。公开部分沿用样例的字段、八个消息分节及附件格式，
不自动猜测未声明的上游格式。图片目前支持 PNG/JPEG；定位支持样例的 `bbox_1000`，以及
明确的文本定位 `{"kind":"text","section":"statement","start":2,"end":6,"text":"2 kg"}`。
文本 section 可为 `statement/question/options`；偏移是对应原样分节字符串的 Python Unicode
码点半开区间，不是字节偏移。图片尺寸校验基于编码头及冻结文件哈希，完整解码由模型适配器负责。

原生包必须包含 `manifest.json`、`FILE_MANIFEST.json`、`public/qa_raw.json`、图片，以及
`private/probes.json`。可选 `public/qa_gold.json`。`FILE_MANIFEST.json` 是相对 POSIX 路径到
SHA-256 的平面映射，覆盖所有声明文件且不包含自身；不得使用绝对路径、上级目录或链接逃逸。
`manifest.json` 至少声明 `schema_version` 和 `image_paths_relative_to:"public"`。

`private/probes.json` 是数组，每条私有记录示例：

```json
{
  "instance_id": "p1",
  "logical_probe_id": "logical-p1",
  "problem_id": "mother-A",
  "probe_type": "T03",
  "split": "development",
  "review_status": "model_approved",
  "human_reviewed": false,
  "cluster_id": "source-cluster-A",
  "binding": {
    "kind": "single",
    "domains": [{"field":"owner","candidates":{"E1":"object-a","E2":"object-b"}}],
    "gold": ["object-a"]
  },
  "readings": [{"field":"read","anchor_id":"R1","expected":"2 kg",
                "rule":{"kind":"ocr","normalizer_id":"ocr_label_v2_1"}}],
  "permitted_gold_packet": [{"anchor_id":"R1","text":"2 kg"}]
}
```

`probe_type` 必须与公开 `task_id` 相同。`domains` 使用**有序数组**：单目标/集合只有一个
字段；关系必须依次为主体、关系、客体三个字段，`gold` 同序，不能按 JSON 键的字母顺序推断。
集合使用 `kind:"set"`，公开输出字段必须是数组。关系只有经任务批准的对称关系才能设置
`symmetric:true`。等价目标通过事先批准的别名到规范身份映射表达，不能运行后放宽答案。

`readings:[]` 表示绑定独占任务，不进入 L。数量读取必须显式使用
`{"kind":"quantity","unit_scales":{"kg":"1","g":"0.001"},"absolute_tolerance":"0"}`；
默认 OCR 不会进行物理单位换算。多个读取目标各自有字段和锚点，必须全部正确才有 `C=1`。
`permitted_gold_packet` 必须是上游审核后冻结的完整信息包；`null` 表示没有导出 Gold，
`[]` 表示导出了空包。这一许可字段不是自动的语义泄漏检测器。

同一母题只能属于一个 split 和一个来源簇，同一重复簇也不能跨 split。
`split=test` 还要求包明确声明 `formal_release_eligibility_checked:true`，并排除保留母题和簇。
框架自动读取数据包清单中的 `DEVELOPMENT_RESERVATION.json`，不内置任何真实母题 ID。
历史开发、提示调试和训练中已暴露的母题应由数据维护者提供；
`--reservation-file` 可追加 `{"problem_ids":[...],"cluster_ids":[...]}`；如果已暴露母题不在新包中，
应使用上游全局去重映射提供其 `cluster_ids`，脚本无法凭空识别未声明的跨包近重复。

## 验证与范围

```powershell
python -m unittest discover -s tests -v
```

测试覆盖既有指标手算与有理数对照、完整评测的非均衡母题聚合、合成样例图片逐字节传递、
Raw / Gold 配对、私有文件读取隔离、无效输出与基础设施缺失、首次结果恢复、哈希与拆分校验，
以及集合、关系和数量契约。对合成样例的测试还会核对运行前后源文件哈希不变。

基础主评测之外，已增加前五个实验的模型接口、可配置 GPU 分组启动、原题自动/人工判分、
控制条件、人工同接口作答和审核。新入口与完整步骤见
[`FIVE_EXPERIMENTS.md`](FIVE_EXPERIMENTS.md)；本页的 `prepare/run/score` 基础命令仍可用。
五实验报告的论文图表生成入口为 `plot_results.py`，设计和命令见 [`FIGURES.md`](FIGURES.md)。
观察事实辅助求解和选项上下文干预仍不属于这次五个实验的执行范围。
`option_diagnostic` 数学接口保留；没有据此自动构造第七个实验。

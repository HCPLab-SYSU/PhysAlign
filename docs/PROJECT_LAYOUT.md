# 框架目录与数据边界

| 位置 | 用途 |
|---|---|
| `physalign/`、`evaluate.py` | 输入校验、计划冻结、模型请求、评分、统计、控制实验和人工评测 |
| `PhysAlign_Benchmark_Kit/` | 标注转换、审核和 benchmark 数据导出；附独立合成示例 |
| `PhysAlign_data_build/` | 原始题目适配、答案盲化、Pass 1–5 标注与本地审核；附独立合成图片 |
| `server_eval/` | 可选模型/API 适配器、批量运行、补充评测和结果比较 |
| `physalign_viz/`、`plot_results.py` | 可选图表、表格及离线图册 |
| `configs/` | 模型设置、GPU 调度及结果比较示例 |
| `examples/synthetic/` | 程序生成的三母题、六探针、九请求开发包 |
| `scripts/` | 合成样例生成器与可配置模型启动器 |
| `tests/`、`server_eval/tests/` | 框架与适配器回归测试 |
| `docs/` | 数据合同和各模块的操作说明 |

项目不携带真实数据集、个人实验结果、模型权重、论文草稿、访问凭据或浏览器资料。
`examples/synthetic/private/` 是合成评分答案和审核测试桩；其中 `private_key.json` 指探针答案映射，
并非身份凭据。合成样例的审核字段不代表真人或模型已完成审核，也不代表正式 benchmark。

核心流程可用标准库运行：

```bash
python evaluate.py prepare --dataset examples/synthetic --split framework_development \
  --bootstrap-resamples 0 --plan plans/example.json
python evaluate.py run --plan plans/example.json --output runs/example-smoke --adapter smoke
python evaluate.py score --run runs/example-smoke
```

`smoke` 返回固定输出，报告明确标记 `scientific_run=false`。详见 [EVALUATION.md](EVALUATION.md)。
需要重建示例时运行 `python scripts/build_synthetic_example.py --output <新目录>`；生成器拒绝覆盖已有目录。

本地数据与输出建议放在 `.gitignore` 已排除的 `datasets/`、`models/`、`plans/`、`studies/`、
`runs/`、`logs/`、`figures/` 等目录中。凭据通过环境变量或本地 `.env.local` 提供；
`.env.example` 只保留空值模板。API 配置中的 `api.example.invalid` 必须改成使用者自己的服务地址。

未见测试的开发保留列表由数据维护者提供。框架读取数据包清单中的
`DEVELOPMENT_RESERVATION.json` 和显式 `--reservation-file`；不内置真实母题、来源簇或旧实验身份。

完整 CPU 测试依赖与入口：

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
python -m unittest discover -s server_eval/tests -v
python -m unittest discover -s PhysAlign_Benchmark_Kit/physalign_converter/tests -v
python -m unittest discover -s PhysAlign_Benchmark_Kit/tests -v
python PhysAlign_Benchmark_Kit/tools/share_kit.py verify
```

各测试套件单独运行，避免同名测试模块冲突。模型相关测试使用替身；真实模型推理的环境与
预检见 [SERVER_SETUP.md](SERVER_SETUP.md)。

数据构造入口独立安装、独立测试：

```bash
python -m pip install -e "./PhysAlign_data_build[dev]"
python -m pytest PhysAlign_data_build/tests -q
node PhysAlign_data_build/tests/test_physgraph_geometry_focus.js
python PhysAlign_data_build/scripts/audit_release.py
```

运行示例时先进入 `PhysAlign_data_build/`，按该模块已有文档执行 `doctor`、`prepare`、
`serve`。四个阶段的结果批准后，Benchmark Kit 的 `convert.py inventory --workspace ...`
可直接接收该标注工作区；相对媒体路径按工作区位置解析。Pipeline 的 `export` 输出
是 Observed-PhysGraph JSONL，不等同于已完成 QA 审核的 benchmark 发布包。
整个链路仍需经过 Kit 的构造、人工审核和冻结，再交给 `evaluate.py`。

根目录 `.github/workflows/ci.yml` 覆盖整个项目；模块内部的 CI 用于单独分发 Pipeline。

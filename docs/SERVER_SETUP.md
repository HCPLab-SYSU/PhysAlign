# 本地模型与 GPU 运行

核心评测只需 Python 3.10+。本地模型推理另外需要与硬件相容的 CUDA PyTorch、torchvision 和
`requirements-eval.txt` 中的依赖；绘图使用 `requirements-viz.txt`。模型权重由使用者自行准备，
不随框架分发。配置文件是适配器的参考设置，实际设备的显存和上下文容量需要预检。

## 模型快照与配置

从项目根目录运行，为完整的本地权重目录建立文件哈希清单，例如：

```bash
python evaluate.py freeze-model \
  --model-path /path/to/Qwen3.5-9B \
  --model-id Qwen/Qwen3.5-9B \
  --output models/qwen35-9b.snapshot.json
```

`configs/server/` 保存各适配器的配置示例。检查 `snapshot_manifest`、精度、显存放置预算、
处理器参数和生成设置，再用于自己的环境。快照含机器路径及权重哈希，应保留在本地；
默认 `.gitignore` 已排除 `models/`。冻结后更改权重或配置，需重新准备预检及运行。

## 用合成数据检查 study 流程

```bash
python evaluate.py draft-study --dataset examples/synthetic --output plans/example-spec.json
python evaluate.py prepare-study --dataset examples/synthetic \
  --spec plans/example-spec.json --split framework_development \
  --bootstrap-resamples 0 --output studies/example

python evaluate.py inspect-model-inputs --study studies/example \
  --adapter-config configs/server/qwen35-9b.json --output plans/example-preflight.json
```

处理器预检不加载模型权重，但会实际解码所有图片并检查输入预算。此处数据及审核记录都是
合成测试材料，结果只用于检查接口。正式数据需要自己的审核、保留清单和输入预检。

## GPU 分配

`scripts/run_panel.py` 从 `configs/panel.example.json` 读取模型与 GPU 分配。该文件演示
三模型、八张卡的分组；使用时按自己的硬件修改或另建配置，删除不需要的 job。
各 job 的 `config` 相对项目根目录解析，GPU 编号不可重复或重叠。

```bash
python scripts/run_panel.py --study studies/example --output runs/example \
  --panel-config configs/panel.example.json --dry-run
python scripts/run_panel.py --study studies/example --output runs/example \
  --panel-config configs/panel.example.json
```

也可直接运行单个适配器：

```bash
CUDA_VISIBLE_DEVICES=0,1 python evaluate.py run-study \
  --study studies/example --output runs/example-single \
  --adapter hf --adapter-config configs/server/qwen35-9b.json
python evaluate.py score-study --study studies/example --run runs/example-single
```

运行目录记录首次响应、模型设置和代码哈希。确认旧进程退出后才使用 `--resume`；
续跑会验证配置与输入一致性。对 `test` split 的 study，启动前还会检查实际人工测量与审核封存，
流程见 [FIVE_EXPERIMENTS.md](FIVE_EXPERIMENTS.md)。

## 其他入口

- [NEW_OPEN_MODELS.md](NEW_OPEN_MODELS.md)：额外本地模型适配器与处理器差异。
- [API_MODELS.md](API_MODELS.md)：OpenAI 兼容 HTTP 服务配置。
- `server_eval/two_experiments.py --help`：将审核后的 benchmark release 转成评测包并调度模型。
- `server_eval/run_*.sh`：Linux Bash 包装器，默认使用 PATH 中的 Python；可用 `PHYSALIGN_PY` 指定解释器。

自动化测试使用模型 SDK 替身，不下载权重，不代表真实 GPU 推理已验证。

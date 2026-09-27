# 四个新增开源模型：服务器评测说明

本入口把以下检查点加入现有 PhysAlign Raw/Gold 计划，不新建数据划分，也不改变请求顺序。

| CLI 名称 | 精确 Hugging Face ID | RTX 3090 | 加载方式 |
|---|---|---:|---|
| `gemma4-26b-a4b` | `google/gemma-4-26B-A4B-it` | 4 卡 | 原生 `AutoModelForMultimodalLM` |
| `glm46v-flash` | `zai-org/GLM-4.6V-Flash` | 2 卡 | 原生 `AutoModelForMultimodalLM` |
| `molmo2-8b` | `allenai/Molmo2-8B` | 2 卡 | 本地冻结的 checkpoint custom code |
| `kimi-vl-a3b` | `moonshotai/Kimi-VL-A3B-Instruct` | 2 卡 | 本地冻结的 checkpoint custom code |

## 冻结协议

与原评测相同的部分：同一份 `plans/evaluation.json`；原始 prompt、图像字节、asset ID 和附件顺序；BF16、每卡 18 GiB 权重放置上限；32K 输入、2048 输出；greedy decoding、`seed=2027`；每个样本独立无历史；保存完整生成后缀，不抽取答案或修复 JSON；复用同一 runner、计分器和 bootstrap plan。

Gemma 4、GLM-4.6V 与 Molmo2 冻结为 SDPA。Kimi-VL 当前官方 custom code 不接受 SDPA，且 `eager` 对长多图序列需要二次方注意力内存，因此冻结为其原生支持的 `flash_attention_2`；该 checkpoint 必需差异及 `flash-attn` 版本会写入配置和 run manifest，不能在正式 run 中静默更改。

四个模型都记录 `thinking=false`。Gemma 4 与 GLM-4.6V 通过 chat template 的 `enable_thinking=false` 实现；Molmo2 与 Kimi-VL-A3B-Instruct 没有对应模板开关，manifest 会如实记录。

不同视觉编码器不能共用同一个 processor 参数，因此使用预先冻结的相近多图预算：

- Gemma 4：每图最多 280 个 soft tokens；
- GLM-4.6V：每图 64--256 个 merge 后视觉 tokens；
- Molmo2：`max_crops=1`，即一个局部 crop 加 checkpoint 强制的全局 crop；
- Kimi-VL：每图最多 1024 个 merge 前 patch，再做原生 2x2 merge；padding 后的精确长度由预检记录。

这些设置只限制模型原生 resize/crop，不删除附件或修改原图文件。CPU 预检会真实解码每个请求的全部图片，核对图片数量、processor 张量与输入长度。任一请求超过 32K 或模型原生上下文时，GPU 推理不会开始。

Molmo2 的官方 chat template 不支持 `system` role，并强制 user/assistant 交替。因此适配器将原 system 字符串、两个换行和原 user 字符串合并成一个 user turn，不改写两段原文；这个差异记录为 `system_transport`。其官方模板还会把图像占位符放在文本前面，asset ID 文本和图像列表顺序仍保留。

## 1. 环境和本地检查点

先安装与现有环境匹配的 CUDA PyTorch/torchvision，再安装：

```bash
python -m pip install -r requirements-eval.txt
python -m pip install flash-attn==2.8.3.post1 --no-build-isolation
```

安装 FlashAttention-2 需要服务器存在与 PyTorch CUDA 版本兼容的 CUDA toolkit/NVCC；RTX 3090 属于其支持的 Ampere GPU。不要把 Kimi 临时改成 `eager` 后混入正式结果。

上传后先做入口与 CPU 合约检查：

```bash
bash -n server_eval/run_new_open_models.sh
python -m unittest server_eval.tests.test_new_open_models
```

包装器默认查找当前 PATH 中的 `python3` 或 `python`；也可显式执行 `export PHYSALIGN_PY="$(command -v python)"`。

四个模型须提前下载为完整本地目录，包含权重、config、tokenizer、processor 和 chat template；Molmo2/Kimi 还须包含仓库 Python 文件。运行阶段强制 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，不会临时联网或漂移 revision。Gemma 若要求许可，请先在 Hugging Face 账户接受许可。

## 2. 在服务器冻结快照

```bash
bash server_eval/run_new_open_models.sh gemma4-26b-a4b freeze \
  --model-path /models/gemma-4-26B-A4B-it
bash server_eval/run_new_open_models.sh glm46v-flash freeze \
  --model-path /models/GLM-4.6V-Flash
bash server_eval/run_new_open_models.sh molmo2-8b freeze \
  --model-path /models/Molmo2-8B
bash server_eval/run_new_open_models.sh kimi-vl-a3b freeze \
  --model-path /models/Kimi-VL-A3B-Instruct
```

分别生成 `models/gemma4-26b-a4b.snapshot.json`、`glm46v-flash.snapshot.json`、`molmo2-8b.snapshot.json` 和 `kimi-vl-a3b.snapshot.json`。manifest 会哈希全部 checkpoint 文件；不要在 freeze 后更新或删除仓库代码。

## 3. 全量预检

```bash
bash server_eval/run_new_open_models.sh panel preflight
```

四个摘要必须都是 `over_budget: 0`。预检证明 processor/上下文预算正确，但不能代替真实 RTX 3090 峰值显存测试。若数据包搬过位置但内容未变：

```bash
export PHYSALIGN_NEW_DATASET=/path/to/evaluation-bundle
bash server_eval/run_new_open_models.sh panel preflight
```

## 4. 8 卡调度

默认分配：GPU 0--3 跑 Gemma；GPU 4--5 先跑 GLM、完成后跑 Kimi；GPU 6--7 跑 Molmo2。

```bash
mkdir -p logs/new-open-models-v1
nohup bash server_eval/run_new_open_models.sh panel run \
  > logs/new-open-models-v1/launcher.log 2>&1 </dev/null &
```

`panel all` 会依次执行预检、推理和完整性计分。重启后继续运行相同命令即可；runner 只恢复缺失请求，不覆盖第一次成功响应。

如需改物理卡号：

```bash
export PHYSALIGN_GEMMA4_GPUS=0,1,2,3
export PHYSALIGN_GLM46V_GPUS=4,5
export PHYSALIGN_KIMIVL_GPUS=4,5
export PHYSALIGN_MOLMO2_GPUS=6,7
```

## 5. 状态、计分和结果目录

```bash
bash server_eval/run_new_open_models.sh panel status
bash server_eval/run_new_open_models.sh panel score
```

默认结果目录为：

```text
runs/local-models/gemma4-26b-a4b-nonthinking/
runs/local-models/glm46v-flash-nonthinking/
runs/local-models/molmo2-8b-nonthinking/
runs/local-models/kimi-vl-a3b-nonthinking/
```

单模型调试可运行 `bash server_eval/run_new_open_models.sh glm46v-flash preflight`，再依次执行 `run` 和 `score`。

Molmo2/Kimi 的 `trust_remote_code=true` 仅允许从已冻结且逐文件哈希的本地目录加载。快照、配置、预检报告、适配器源码或核心评测源码变化后，旧预检/旧 run 不能继续混用。

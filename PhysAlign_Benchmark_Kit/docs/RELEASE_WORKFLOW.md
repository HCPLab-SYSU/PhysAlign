# 审核结果接入与正式数据导出（工具包 v1.1）

本轮新增的是**数据侧的审核与发布层**，不是模型评测器。旧 `native-draft` 仍生成不可变草稿；新入口是 `tools/native_release.py`。不要使用底层旧 `freeze` 命令处理 native 数据。

当前可以发布 T02-single、T03-image、T03-text。发布后，外部评测项目可读取公共 QA 和真实图片，独立读取私有映射进行判分；这里没有 runner/scorer，也没有声称论文的控制实验、人类上限或研究结论已完成。

## 1. 两种准入方式，不能混淆

| 方式 | 必要证据 | 未抽样数据 | 适用场景 |
|---|---|---|---|
| 逐项审核发布 | 同一母题的全部拟导出探针均通过；负责人确认 | 不导出 | 已逐题审核的小集合；开发集；不做总体错误率声明 |
| 批次审计发布 | 完整开发闭环、预先批准政策、固定随机样本、独立二审、风险检查、通过统计门槛、负责人确认 | 可在批次依据下导出，但标为 `human_item_review=not_sampled` | 需要批次审计的测试候选数据 |

旧开发页默认的 25 道母题始终是 development，不会因“通过”自动变成 test。批次主样本不从这 25 道中抽，开发样本不用于估计发布池缺陷率。

**不需要重做旧的通过记录。** 新入口支持原 `physalign_decisions_v1` 导出文件。旧“退回”记录没有结构化错误分类，接入后是 unresolved，需要在新页面明确是孤立严重源问题、系统性错误、信息/版本问题，还是仍未决；不能默认为安全的孤立错误。

## 2. 路径、版本与数据安全

以下单行命令从工具包根目录执行，适合 Bash；其中 Python 命令也可在 Windows CMD/PowerShell 单行执行。先激活安装过 requirements.txt 的 Python 环境。

```bash
cd "/path/to/PhysAlign_Benchmark_Kit"
python -B tools/share_kit.py verify
python -B -X utf8 tools/native_release.py --help
```

组员把第一行改成自己的工具包路径。其他示例相对于工具包根目录；按实际位置替换。

全部真实输出必须放在工具包外，输出目录必须不存在。Windows 推荐短路径，如 `D:/pa/review1`、`D:/pa/release1`，避免 WinError 206。发布会复制冻结来源用于追溯，需预留额外磁盘空间。

新增发布流程遇到 Windows 临时读取占用时会有限重试目录 rename；绝不会删除目标目录或降级为覆盖写入。持续权限错误请解决文件权限/占用后使用新输出目录；`.staging-*` 是未发布产物，不是正式集。

旧转换器 43 个文件保持原字节，原草稿/旧审核目标仍可复验。新发布代码有自己的版本指纹；修改发布代码后需新建对应审核项目/批次，不能手改哈希让旧记录通过。审核 JSON 是本地可信记录，不是数字签名；负责人必须核实审核人和证据真实性。

## 3. 第一步：创建独立的发布审核项目

对当前这份真实草稿：

```bash
python -u -B -X utf8 tools/native_release.py review-init --draft "../data/physalign_native_draft_v1" --output "../data/pa_review_v1"
```

它先完整复验旧草稿，再复制为新的只读来源快照，生成 `../data/pa_review_v1/review.html`。原 `data/physalign_native_draft_v1/` 不修改；正在打开的旧页面不受影响。

新页面包含全部生成探针，支持开发/测试候选、接口、母题和状态筛选。图片和来源都来自新项目的相对路径；可以把整个 `pa_review_v1` 文件夹单独交给获授权的审核员，但不要只发 HTML。

打开新页面后：

1. 导入旧开发页导出的审核 JSON，保留已有通过记录和首次独立判断。
2. 填真实审核人。选择一母题，完成它的全部探针。
3. 根据 Raw 公开图文盲答，再展开金标准、Gold 实际文本及原来源。
4. 检查六项。空 packet/文本分支只在确实不适用时注明并确认，不编造读数。
5. 通过或退回；退回必须选择问题分类并写原因。
6. 导出到项目目录外，例如 `../data/pa_reviews/dev_decisions.json`（请提前用文件管理器创建 `pa_reviews` 文件夹）。

新页面有未提交内容时会提醒，但仍没有后台自动保存。点击通过/退回只进入本页内存，关闭/刷新前务必导出。导入时不会静默覆盖已有内存记录；要更换文件，先导出、重开页面再导入。

首次独立判断一旦提交即在页面中只读保留；后续复核意见写在 note 中，不把首次盲答改成金标准。导入旧记录同样保留其原盲答。

**此页面只审核，不改框、不改 owner、不改原 Pass。** 系统性问题必须修正规则/可信来源后重新生成受影响版本；不能在草稿中手改答案。

## 4. 接入审核结果，查看完成和遗漏

假设实际导出的文件已放到以下路径：

```bash
python -u -B -X utf8 tools/native_release.py review-import --project "../data/pa_review_v1" --decisions "../data/pa_reviews/dev_decisions.json" --output "../data/pa_dev_receipt_v1"
```

输出 `receipt.json` 包含原始导出文件、规范化记录、完整通过的母题、未决/退回母题及 pending 数量。它本身还不是发布批准。

多人审不同题，可以重复 `--decisions` 参数；同一题不同的已提交记录会拒绝合并，不实行“最后文件覆盖”。同一审核人的多个下载版本请只提交最新一份，不把 `(1)`、`(2)` 全部混进去。需要仲裁的记录应保留原意见，形成明确的新审核文件，不用覆盖掩盖分歧。

## 5. 路径 A：逐项审核通过的母题直接准备发布

### 5.1 选择发布划分

导出已完成的开发母题：

```bash
python -u -B -X utf8 tools/native_release.py release-plan --project "../data/pa_review_v1" --reviews "../data/pa_dev_receipt_v1" --split development --output "../data/pa_dev_request_v1"
```

如果你另外审核了测试候选母题，导入那份最终审核结果，使用 `--split test`：

```bash
python -u -B -X utf8 tools/native_release.py review-import --project "../data/pa_review_v1" --decisions "../data/pa_reviews/test_item_decisions.json" --output "../data/pa_test_receipt_v1"
python -u -B -X utf8 tools/native_release.py release-plan --project "../data/pa_review_v1" --reviews "../data/pa_test_receipt_v1" --split test --output "../data/pa_test_request_v1"
```

只有完整通过的母题会被选择；未审齐/退回的母题整体不导出。已知系统性、信息/版本或 unresolved 问题会阻止发布申请，不能通过只选其他题绕过规则级缺陷。

输出 `request.json` 固定入选/排除成员、划分、证据和目标根。数量较少是如实反映支持范围，不宣称它继承了整个候选池的错误率保证。

### 5.2 负责人确认，不自动盖章

复制 `pa_test_request_v1/approval.TEMPLATE.json` 到项目和 request 目录外，例如 `pa_reviews/test_release_approved.json`。保留 `schema_version` 与 `target_root`，由负责人实际检查后填写：

```json
{
  "schema_version": "native_release_signoff_v1",
  "target_root": "保留工具生成的原值，不是填写这段文字",
  "status": "approved",
  "reviewer": "真实负责人",
  "evidence_ref": "可追溯的审核/讨论记录编号或路径",
  "confirmations": {
    "source_and_rules_checked": true,
    "known_systemic_and_unresolved_issues_zero": true,
    "near_duplicates_and_external_splits_checked": true,
    "not_selected_by_model_performance": true,
    "review_evidence_authentic": true,
    "data_use_authorized": true
  }
}
```

这六项是真实责任确认，不是让你为了跑通机械改成 true。尤其“近重复/外部划分”需要和组员协调；工具只保证当前快照内的精确去重与开发保留，未实现跨团队近重复检测或自动合并。不要根据被测模型表现决定哪些题留下。

### 5.3 正式导出与复验

```bash
python -u -B -X utf8 tools/native_release.py export --project "../data/pa_review_v1" --request "../data/pa_test_request_v1" --approval "../data/pa_reviews/test_release_approved.json" --output "../data/pa_release_test_v1"
python -u -B -X utf8 tools/native_release.py validate --path "../data/pa_release_test_v1"
```

开发集同理，使用开发 request 和对应批准文件。必须明确标为 development，不能把上述开发命令的产物改名成测试集。

## 6. 路径 B：低人工成本的批次审计与发布

### 6.1 批准政策并建立唯一 campaign

先完成开发母题全部探针的检查、规则修复和 `review-import`；然后：

```bash
python -u -B -X utf8 tools/native_release.py campaign-init --output "../data/pa_campaign_v1"
```

将 `policy.TEMPLATE.json` 复制到外部 `pa_reviews/policy_approved.json`，按第 5.2 节填写真实负责人/证据/确认。此处 target_root 绑定的是 campaign 与固定审计政策，不是某个发布申请。

实现采用以下固定政策：主随机样本 `min(100,N)` 母题，独立随机二审 `min(25,n)` 母题；单侧 97.5% 超几何上界；5% 母题级门槛；最多两轮。附加风险预算具体固定为最多 **20 道母题**，覆盖缺失的来源、接口/选择器/packet 规则、语言、表示、修复和多图标签。20 是本实现需负责人预先确认的工程预算，不是会议标准；覆盖不足就不允许批次继承。

每个真实数据扩充批次登记一个 campaign。**不得反复 campaign-init 来更换种子、清空失败史或获得第三次尝试。** 本地文件不是中心化审计服务，团队必须保留登记和失败记录。

### 6.2 冻结候选总体并抽样

```bash
python -u -B -X utf8 tools/native_release.py audit-plan --project "../data/pa_review_v1" --campaign "../data/pa_campaign_v1" --policy-approval "../data/pa_reviews/policy_approved.json" --development-review "../data/pa_dev_receipt_v1"
```

候选总体为去重后有探针且不属于开发集的全部母题，不能传任选的好题列表。输出固定的 `pa_campaign_v1/round1/`：

- `audit_plan.json`：完整候选、主随机样本、附加风险样本、基础二审样本、每母题的 blind-first 探针、政策和源版本。
- `primary.html`：首审。
- `secondary.html`：独立二审，不含首审意见。
- `adjudication.html`：必要时第三方仲裁。

页面相对引用 `pa_review_v1/source/`，请保持 campaign 与审核项目的相对位置；给审核员一起传这两个数据目录。两者都是私有审核材料，不在代码分享包内。

### 6.3 首审与二审

首审检查主样本及附加风险样本中每母题的全部探针。每母题必须先检查预先指定的 blind-first 探针；页面顺序已安排，不能先审其他条再制造盲答记录。

二审先检查 `audit_plan.json` 的 secondary_mothers 中全部探针，再补首审问题母题。页面标出基础必审母题；其他母题也可选择。每母题同一角色由一名审核员负责；同一探针二审人与首审人必须不同。

导出为外部 `pa_reviews/audit1_primary.json` 和 `audit1_secondary.json`。原开发审核文件不能冒充主随机审计，也不能直接导入二审角色。

### 6.4 检查是否齐全、是否需要升级

```bash
python -u -B -X utf8 tools/native_release.py audit-assess --project "../data/pa_review_v1" --audit "../data/pa_campaign_v1/round1" --primary "../data/pa_reviews/audit1_primary.json" --secondary "../data/pa_reviews/audit1_secondary.json" --output "../data/pa_audit1_check_v1"
```

`accepted: false` 时不能发布；命令成功写出报告不等于审计被接受。看 `pending_primary_probe_ids`、`pending_secondary_probe_ids`、`unresolved_probe_ids` 和 `blockers`。补审后重新导出完整的最新文件，以新 output 再 assess，仍属于同一次抽样，不重新 audit-plan。

二审发现首审漏掉的严重问题时，本实现保守升级为**全部主样本和风险样本**二审；未补齐不通过。分歧/未决问题交给不同于前两人的仲裁者，只审核需仲裁条目并导出；再加：

```bash
python -u -B -X utf8 tools/native_release.py audit-assess --project "../data/pa_review_v1" --audit "../data/pa_campaign_v1/round1" --primary "../data/pa_reviews/audit1_primary.json" --secondary "../data/pa_reviews/audit1_secondary.json" --adjudication "../data/pa_reviews/audit1_adjudication.json" --output "../data/pa_audit1_check_v2"
```

已记录的严重发现保守保留在原随机母题发现数中，不因隔离或仲裁而改报零发现；只有确认的孤立严重缺陷母题才从剩余集上界中扣除。附加风险母题不加入主随机分母。系统性或严重信息缺陷会阻止整个规则版本发布，不允许只删除抽中的坏题。

区间对“该审计流程能识别的缺陷”负责，不保证无漏检。全部已知严重缺陷必须隔离，5% 不是允许保留已知错题。

### 6.5 审计接受后生成发布申请

```bash
python -u -B -X utf8 tools/native_release.py release-plan --project "../data/pa_review_v1" --assessment "../data/pa_audit1_check_v2" --split test --output "../data/pa_batch_request_v1"
```

若实际通过的是 `check_v1`，替换为那份。接下来仍需复制 request 中的批准模板、由负责人确认，再按第 5.3 节 export/validate。政策批准、抽样完成、发布批准是不同记录，不能相互替代。

### 6.6 第一轮失败与最多两轮停止

缺审核/缺二审不是重新抽样的理由，先续审原面板。第一轮完整评估失败，或已发现系统性缺陷并需修复时，保留 assessment；修复规则/来源后创建新草稿及审核项目，再用**同一个 campaign**申请第二轮：

```bash
python -u -B -X utf8 tools/native_release.py audit-plan --project "../data/pa_review_v2" --campaign "../data/pa_campaign_v1" --policy-approval "../data/pa_reviews/policy_approved.json" --development-review "../data/pa_dev_receipt_v2" --previous-assessment "../data/pa_audit1_check_v2"
```

新的源草稿可以由同一转换器重新生成；若修改了发布层自身的代码，旧 campaign 会因实现指纹变化而拒绝，需团队保留既有失败预算并设计明确迁移，不能用新 campaign 重置预算。当前没有自动迁移工具。

第二轮必须是新的候选内容版本，不能对同一份未修改数据换种子重抽。第一轮标记的问题母题必须从新候选中隔离，或确实改变其已审核内容；系统性/信息问题会把检查范围扩大到同一选择器/packet 规则的全部旧母题，包括未抽中的题。仅改输出目录名无效。内容变化本身不是“已修好”的证明，第二轮审计和负责人确认仍不可省略。

第二轮仍失败：不允许第三轮。可以暂停，或仅保留真正逐项确认的样本，不作总体批准声明。对无系统性/未决故障的审计结果，可显式选择已逐项完整通过的小集合，复用实际审核证据：

```bash
python -u -B -X utf8 tools/native_release.py release-plan --project "../data/pa_review_v2" --assessment "../data/pa_audit2_check_v1" --audited-subset --split test --output "../data/pa_audited_subset_request_v1"
```

这条命令不会继承批准给未抽样题，也不会把该缩水集合描述成原总体随机审计已通过。仍需负责人 signoff 后 export。

## 7. 正式输出怎么交给外部评测项目

```text
pa_release_test_v1/
  release.json                         # 版本、划分、B/L/P 数量和范围
  FILE_MANIFEST.json                   # 全目录逐文件校验
  README.md
  public/
    qa_raw.json                        # B：全部获准绑定探针
    qa_gold.json                       # P：仅获准非空 packet；不是每条 Raw 都配一条 Gold
    images/<sha256>.<ext>               # 实际原图/定位图；按字节去重
  private/
    mappings.json                      # 保持 v2.1 私有映射、read 真值、R/E/H 与来源
    membership.json                    # 母题/探针/划分/真实审核依据及诊断
    pools.json                         # B/L/P 固定 probe_ids 与 mother_ids
    frozen/<instance_id>.json          # 验证后的内容、授权链与私有 key
    signoff.json
    request/                           # 固定申请、真实审核证据及前轮失败链
    project/source/                    # 原候选快照，含排除记录；不改原数据
    contract_schemas/                  # v2.1 私有映射及包络 Schema
```

公共记录统一字段：`schema_version, instance_id, logical_probe_id, task_id, interface, language, split, condition, input`。`input.messages` 为 `{system, user}`；`input.attachments` 给出 asset_id、path、sha256、width、height。

**图片 path 以 `public/` 为根目录。** 例如 `images/abc.png` 对应 `pa_release_test_v1/public/images/abc.png`，没有指向原作者 C 盘路径。

外部 runner 每次只发送该记录的 `input.messages` 和对应真实图片字节，独立上下文处理每个探针/条件。不要发送整个记录集、`private/`、完整 preview/review HTML、金标准、source paths 或审核备注。私有数据只留给服务器端评分/追溯。

T03-text 不进入图像读取 L，也没有伪造的 Gold/packet；Raw 和 Gold 的配对只在相同 P 成员上进行。这里冻结的是数据成员，不实现模型失败/拒答/缺失的统计处理，外部评分器仍需正确处理计划分母和母题聚类。

复验会重新检查来源、几何、原文定位、原生依赖闭合、审批证据、实际图像字节、完整公私映射及公共字段白名单；即使有人重新计算目录文件清单，也不能靠修改导出 QA 的答案、路径或成员蒙混过关。

## 8. 分享给组员与以后扩充

分享更新后的 `PhysAlign_Benchmark_Kit/`，不用分享本机旧 ZIP。本轮不生成压缩包。新增数据先按 README 转换成新 native 草稿，再走本教程。

不要把真实 project/campaign/release 或下载的 review JSON 放入工具包。每批保存工具版本、源版本、审核文件及 campaign 失败史。跨批次/团队合并要复核源身份、近重复及开发/测试边界；此工具不自动替代这一研究决策。

当前真实数据没有被本轮实现自动批准或导出。测试中出现的 SYNTHETIC 审核/批准仅是合成 fixture；生产 export 会拒绝内置合成来源。`--synthetic-example` 只用于合成测试，并强制标记为非 Benchmark 数据，不用于真实发布。

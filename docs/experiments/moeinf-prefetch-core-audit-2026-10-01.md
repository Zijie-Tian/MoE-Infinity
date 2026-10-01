# MoE-Infinity 预取核心改动复审（2026-10-01）

> **审计后的处理：** 用户随后要求撤销 DFlash 相关修改。共享 executor/prefetcher 及配套测试已恢复为 `f7f56df`，本报告分析对象仍是回退前的 `286970…` 快照，不应把其行号或新增回归直接套到回退后的文件。C++/LFU 与对应测试保留；回退证据位于 `.omx/research/moeinf-dflash-scope-revert-2026-10-01/`。

## 审查结论

**不建议把当前修改视为已证明通用正确、保持原系统行为等价的补丁。** 用户对修改核心机制的担忧合理。确有合理的阶段接线、重复 API 调用消除和安全改进，但 `correct_prefetch` 的补漏行为被整体移除，不只是去重；当前又有已复现的失败隔离回归和预测归属问题。

本轮只读审查，未修改源码、测试、native binary、实验结果或提交。新增本文与 `.omx/research/moeinf-prefetch-code-audit/` 审计资料。结果表继续保留为“修改后本地版本的限定实验观测”，不能升级为上游实现/论文原样复现或全路径代码正确性证明。

## 范围与版本

- 工作区：HEAD `f7f56df0cff8b5b609316dcd96be66443c07bab7`，源码/测试 diff SHA256 `2869700f303aba0d3c4ecbbfe85223e83bc6b0eabd060d516713964833660f06`。
- 修改前对照：`f7` 加保存的 pre-prefetch patch `7c6284…`。该 patch 没有 `expert_executor.py` 或 `expert_prefetcher.py` 的修改，所以这两个文件的对照就是 f7 源码。
- 主范围：Python 普通预测/纠错/dispatch 生命周期；相关 native 传输、状态与统计边界；相关测试和实验覆盖。
- 独立 Python、native、契约反方三路只读审核。当前安装角色没有正式 `code-reviewer` / `architect` 路由，本结论是基于上述证据的 scoped 审查，不冒称正式 merge-ready 批准。

## 1. 必须纠正之前的表述：没有删除 prefill/decode，但确实改了核心预取

### 阶段标记

测试 wrapper 的 `dispatch_with_phase` 根据本实验固定 batch=1 的 token 行数选择 PREFILL 或 DECODE。这在 frozen benchmark 中，不是删除 prefill/decode，也不自动修复仓库全部 HF/serving 的阶段接线。普通 phase policy 遇到 MIXED 会返回，因此这个 wrapper 使本实验的预取真正触发。

此规则适用于当前 512-token prefill / 单 token decode，不可不加验证用于多请求混合批次或 chunked prefill。

### 重复 predictor 调用

旧 executor 在 overlap 分支先调用一次默认 MIXED，再调用一次实际 phase。改为一次实际 phase 调用有依据：

- phase policy 启用时，旧 MIXED 调用常是 no-op，真正调用只有后一次。
- phase policy 未启用时，两次可能提交同一集合。

所以删除多余 API 调用是合理方向，但不能声称旧路径一定发生两次物理 H2D——native 还可能做同节点去重。

### `correct_prefetch`：实际是“有补漏 → 无补漏”

旧 f7 的函数计算 `missed = actual_expert_ids - predicted`，然后调用两次 `prefetch_experts_list`：一次带 phase，一次使用默认 MIXED。旧 executor 给它的 target 是 **L+1**，给它的 actual IDs 却来自当前 **L**。

当前代码同时做了两件事：

1. 将预测与真正对应的当前目标层关联，用 L 的实际路由评估 L−1 的预测。
2. 删除了所有 `missed` 补发，把公共 `correct_prefetch` 变成只返回 TP/FP/FN 的函数。

第 1 项有逻辑依据，第 2 项不是行为等价去重。旧行为可被视为“把当前层 actual union 作为下一层辅助候选”的 hedge，不能仅因名称或错层核算就断言该候选扩展完全不应存在。

具体例子：下一层 logits 预测为 `{1,2}`，当前层实际用了 `{1,2,3}`。旧路径还会向下一层补发 `3`（两次 API 调用）；当前路径不补发 `3`。**预取集合、缓存占用、带宽和命中时机会因此变化。** 最小去重与取消整个补漏机制必须分别说明。

原旧测试要求 correction 发出 miss；新测试被改成不发，只证明符合新设计，不证明新设计保持旧契约。

## 2. 确定问题与证据

| 严重度 | 问题 | 证据与影响 |
|---|---|---|
| HIGH | 提前执行 correction 破坏失败隔离 | `expert_executor.py:505–547` 已提交真实 expert 工作后，无 try 调用 correction；pending 直到 `572–582` 才建立。旧 failure-safe correction 位于 wait 之后的 try 内。纠错异常可直接中断 dispatch，破坏“预取失败不改变等待结果”的契约。 |
| HIGH：契约变化 | 整体移除公开 correction 补发 | `expert_prefetcher.py:552–580` 当前没有任何 miss 发请求；f7:501–504 有两次。目标层核算修正不自动证明可以删除未来候选 hedge。 |
| MEDIUM | 错 phase/target 可借用无归属的旧预测 | `_consume_prediction` (`expert_prefetcher.py:534–550`) 对缺失 key 借用 `_last_speculative_prediction`。CPU 探针：只记录 `(DECODE,3,g17,{2})`，纠正 `(PREFILL,1)` 却返回 generation0、predicted/TP `{2}`，真正 key 仍在。不能声称所有 TP/FP/FN 都严格归属于正确层/阶段。 |
| MEDIUM | Python 样本 archive 常开、无上限 | `expert_prefetcher.py:713–748` 不论 overlap policy 是否启用都 drain，并 `trace.extend(samples)`；普通应用没有 benchmark 的 take 消费者时，可长期线性增长。可选统计异常也可能从 executor finally 逃出，替代本已成功的推理返回。 |
| MEDIUM | ordinary / overlap 身份与失败取消未闭合 | 两个 generation counter 从 1 开始；显式 expert-set path 记录 generation 却返回 None。混用/切换路径可能错配或不能加入失败取消列表。本实验 controller=off 不覆盖这些情况。 |
| MEDIUM：新原生风险 | 未同步的到达时设备读 | `expert_dispatcher.cpp:621–624` 在取得 claim 前读取非原子 `Node::device`，prefetch 会在 `model_topology.cpp:225` 写同一对象。存在 C++ data-race 风险，主要涉及新增 provenance 采样；本轮未证明其实际导致漏 expert 或表格某个数字错误。 |

### Root 直接复现的失败

未修改的测试：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD" \
  /home/zijie/anaconda3/envs/moeinf/bin/python -m pytest -q \
  tests/python/dflash/test_capability_orthogonality.py::test_prefetch_exceptions_cannot_change_waited_expert_output
```

结果：**1 failed，exit=1**，在 `expert_executor.py:546` 抛出 `RuntimeError("correct boom")`，没有返回测试期望的 `legacy-output`。

完整 CPU 选定组为 65 passed / 3 failed。仅上述 failure 已直接归因于本次提前 correction；另两个失败没有在本轮做完整旧版本复现，不将它们擅自算成新增或存量。日志见下方索引。

## 3. native 问题必须区分新旧，不能归因过度

对当前选定普通、managed、routed、单 GPU 正常路径，实际路由的 expert 由 demand dispatch 提交；删除额外 corrected-miss prefetch **不意味着真实 expert 被完全漏算**。有 pending-key 单 owner、claim 持有至 commit/release、事务清理及已驻留 no-op 等安全改进。

但这些不是无条件保证。复审同时发现/确认的边界：

- release 附近单次 demand CAS 失败后再读 flag，可能进入 fatal 的窗口在 **7c 已存在**，不是本轮新增。
- wait-for-prefetch 后忽略重获 CAS 结果也是存量。成功 commit 后的 exact-active no-op 和 pending-dispatch 防淘汰限制了当前普通路径的竞争，故不能声称它就是本次两次停滞的根因。只有失败/无 active entry等额外前提下更广泛路径的可达性仍待验证。
- advanced `schedule_prefetch_tensors` 的 running/terminal accounting、layer=-1 取消、managed resize、detach teardown 等是存量 broader-path 问题；本实验关闭 controller/resize，不能把它们算成本次新增，也不能被主表覆盖。
- 本轮新 ordinary cancellation 的 keep tensor-ID 与 node-ID 转换不一致，在失败/取消路径有风险，成功主矩阵没有验证这种调用。

36 个 C++ 测试主要验证 manager/helper，不是完整 TaskPool/dispatcher 的并发故障测试，也不是 TSAN 或所有 CUDA 交错证明。

## 4. 以前“测试通过”实际证明了什么？

- 20 个成功配置和 4 个明确 OOM，及 5,008 个核验，证明已保存结果的完整性、限定运行及账目/时间线一致性；**5,008 是检查条目，不是 5,008 种独立系统条件或故障注入。**
- 628 项短测覆盖四模型、4 GiB LRU、batch1、64输入/8输出的匹配 OFF/ON。不能外推所有容量、所有策略、更长服务、observe/enforce、DFlash 异常回退或原始上游等价。
- 主表使用修改后 local native2f；旧 OFF 完整表使用 native772。不是原代码与修改后代码的全量严格 A/B，更不是完整 HF reference 数值一致性证明。
- 删除补漏候选的行为变化会影响 cache hit，不能仅因为没有改 gate 或淘汰分数就认为“只开预取、其它系统逻辑没变”。
- 原来两次 native wait_expert 停滞仍只保留证据并重跑；没有声明根因修复。本轮审查也没有证明某个具体竞争就是其根因。

## 5. 后续应怎样处理（本轮没有执行）

1. 将“重复调用消除”和“取消公开补漏语义”拆开；明确复现原机制还是评估新调度变体。若目标是保留原机制，需保留一个明确的补发路径或兼容接口，而不是把整个 API 静默改成纯统计。
2. 先恢复可选 correction/telemetry 的错误隔离与 pending/cleanup 所有权，再讨论合入。
3. 修正精确 phase/target 的 missing/empty 区分，限制 trace 为 opt-in/有界，统一或命名空间化 generation；分别覆盖普通与 advanced 路径。
4. 同一模型/输入/cache/precision、同一 native 构建做独立开关差分：原纠错、只去重、只目标层核算修正、取消补漏，分别看发请求集合和 cache hit。当前表不可代替这种因果消融。
5. 需同步处理新增 native provenance 读写保护，并以并发/故障测试补证；本轮不自行修代码、revert、commit 或启动 GPU 重测。

## 审计证据与归档边界

- [Python 代码与 CPU probes 审计](../../.omx/research/moeinf-prefetch-code-audit/README.md)
- [完整 CPU 选定测试日志](../../.omx/research/moeinf-prefetch-code-audit/cpu-selected-tests.log)
- [Root 直接复现的失败](../../.omx/research/moeinf-prefetch-code-audit/root-failure-isolation-test.log)
- [独立契约反方审计](../../.omx/research/moeinf-prefetch-code-audit/contract-adversarial-review-2026-10-01.md)
- [窄化、区分新旧的 native 审计](../../.omx/research/moeinf-prefetch-code-audit/fresh-readonly-native-audit.md)
- [已归档实验结果与上一版修改说明](./moeinf-prefetch-cache-hit-2026-09-30.md)

上一版说明中的“删除重复预取/纠错时再发当前 miss”需要以本文的更精确分析为准：旧 executor 传的是 L+1，不应把旧全部补发描述成 L demand 的完全重复。原结果及旧说明保留为审计历史，未改写实验值。

**结论：核心代码确实改过；部分方向合理，但公开机制变化没有等价证明，且已有已复现回归。当前不应直接认定补丁通用正确或正式合入。**

# MoE-Infinity 专家缓存实验：预取 ON 全表与代码改动说明

> **2026-10-01 后续回退说明：** 按用户要求，三个 DFlash 测试、共享 `expert_executor.py` / `expert_prefetcher.py` 及两个配套 unit-test 期待已恢复为 `f7f56df` 版本。真正 LFU、C++ 缓存实现和 C++ 单元测试保留。本文数字属于回退前 `286970…` 修改版本的历史实验，**不再声称对应当前工作树，也不可直接用当前共享 Python 接口运行旧 frozen 预取观测 harness**。原数据与修改前补丁均保留，未改写成绩。

- 实验日期：2026-09-30。
- 保存日期：2026-09-30。
- 适用版本：本机修改后的 MoE-Infinity；不是未经修改的上游发行版。
- Git 基线：`f7f56df`；基线之后的 LFU、预取修复及测试修改均未提交。
- 本文先保存最终结果，再说明哪些改动影响功能、哪些只是观测与校验。

## 结论与统计口径

四模型 × 4/8/16 GiB × 本地 C++ 管理器中的 managed LRU / resident LFU（真正 LFU 是此前按用户要求新增的实现），共 24 个配置：**20 个完整成功、4 个明确 CUDA OOM**。独立验证 **5008 项 PASS、0 FAIL、0 INCOMPLETE**；失败不记为 0%。

以下主表为 **decode**，汇总两次测量请求，排除一次 warmup。每请求输入 512 token、生成 64 token，batch=1、BF16，单个物理 GPU；仅使用 GPU 1/2/3/4。缓存容量为共享 residency manager 的逻辑预算，包含被管理的非路由参数，**不是 expert-only 或全卡显存硬上限**。

- Hit = native 到达/就绪分类中的 `prefetch_ready_first_use + reuse_hit`。
- Miss = `prefetch_late_first_use + demand_miss`；预取已启动但需要时未就绪仍算 miss。
- 查找单位为一次层调用中一个 unique active `(layer, expert)`；decode batch=1 每层等于该层 routed top-k。
- Python `is_tensor_on_device` 仅为时间线交叉检查，不作为异步预取的主命中计数。
- 相对于本次预取 OFF 的本地基线，保留 LRU/LFU 淘汰排序、模型 gate 实现、BF16 和请求预算；但预取接线、目标层纠错及并发生命周期确有功能性修复，不能理解为只加日志。所有 baseline commit 之后的代码修改仍未提交。

## 4 GiB

| 模型 | LRU Hit | LRU Miss | LFU Hit | LFU Miss |
|---|---:|---:|---:|---:|
| DeepSeek-V2-Lite | 30.12% | 69.88% | 14.25% | 85.75% |
| Mixtral-8x7B-v0.1 | 0.76% | 99.24% | 0.76% | 99.24% |
| Qwen1.5-MoE-A2.7B | 28.21% | 71.79% | 31.45% | 68.55% |
| Qwen2-57B-A14B | 0.03% | 99.97% | 0.03% | 99.97% |

## 8 GiB

| 模型 | LRU Hit | LRU Miss | LFU Hit | LFU Miss |
|---|---:|---:|---:|---:|
| DeepSeek-V2-Lite | 50.24% | 49.76% | 40.73% | 59.27% |
| Mixtral-8x7B-v0.1 | 0.71% | 99.29% | 0.76% | 99.24% |
| Qwen1.5-MoE-A2.7B | 45.64% | 54.36% | 63.46% | 36.54% |
| Qwen2-57B-A14B | N/A（OOM） | N/A（OOM） | N/A（OOM） | N/A（OOM） |

## 16 GiB

| 模型 | LRU Hit | LRU Miss | LFU Hit | LFU Miss |
|---|---:|---:|---:|---:|
| DeepSeek-V2-Lite | 78.14% | 21.86% | 73.46% | 26.54% |
| Mixtral-8x7B-v0.1 | 0.73% | 99.27% | 0.78% | 99.22% |
| Qwen1.5-MoE-A2.7B | 79.46% | 20.54% | 92.01% | 7.99% |
| Qwen2-57B-A14B | N/A（OOM） | N/A（OOM） | N/A（OOM） | N/A（OOM） |

## 如何理解结果

**Mixtral 仍很低。** 所有成功配置的 decode `reuse_hit` 都为 0，主表约 0.7–0.8% 来自少量提前就绪的预取。当前预取未解决跨层循环淘汰：预取可以提前搬运，并不自动增加缓存可保留的工作集。

**Qwen2-57B** 在 4 GiB 的两个策略均完成测量，但命中仅 8/28,224 ≈ 0.0283%；8/16 GiB 的两策略均 hash-proven OOM，0 个完整请求。这不意味着没有生成任何 token，也不意味着模型不支持。

**不能从 cache hit 直接推出 TPS。** Late-prefetch 即便算 miss，仍可能提前做了一部分搬运；反之错误/未使用的预取会占带宽和缓存。此次只作缓存观测，不给出吞吐提升或精准 PCIe 流量节省的结论。

当前使用现有普通预测路径：将**当前层** router logits 按 token 取均值、top-k 专家编号预测为**下一层**候选；不是下一层真实 gate、未来 oracle，也不冒称论文 EAMC。实际预测与下一层路由逐条关联，详见 `scientific-interpretation-notes.md`。

## 就绪、迟到、复用分解（decode）

| 容量 GiB | 模型 | 策略 | 总查找 | 复用命中 | 就绪预取首次用 | 迟到预取首次用 | 到达缺失 |
|---:|---|---|---:|---:|---:|---:|---:|
| 4 | DeepSeek-V2-Lite | LRU | 19656 | 5913 | 7 | 225 | 13511 |
| 4 | DeepSeek-V2-Lite | LFU | 19656 | 2760 | 40 | 219 | 16637 |
| 4 | Mixtral-8x7B-v0.1 | LRU | 8064 | 0 | 61 | 828 | 7175 |
| 4 | Mixtral-8x7B-v0.1 | LFU | 8064 | 0 | 61 | 837 | 7166 |
| 4 | Qwen1.5-MoE-A2.7B | LRU | 12096 | 3389 | 23 | 147 | 8537 |
| 4 | Qwen1.5-MoE-A2.7B | LFU | 12096 | 3794 | 10 | 107 | 8185 |
| 4 | Qwen2-57B-A14B | LRU | 28224 | 0 | 8 | 411 | 27805 |
| 4 | Qwen2-57B-A14B | LFU | 28224 | 0 | 8 | 401 | 27815 |
| 8 | DeepSeek-V2-Lite | LRU | 19656 | 9861 | 15 | 178 | 9602 |
| 8 | DeepSeek-V2-Lite | LFU | 19656 | 7982 | 24 | 221 | 11429 |
| 8 | Mixtral-8x7B-v0.1 | LRU | 8064 | 0 | 57 | 836 | 7171 |
| 8 | Mixtral-8x7B-v0.1 | LFU | 8064 | 0 | 61 | 834 | 7169 |
| 8 | Qwen1.5-MoE-A2.7B | LRU | 12096 | 5507 | 14 | 101 | 6474 |
| 8 | Qwen1.5-MoE-A2.7B | LFU | 12096 | 7664 | 12 | 72 | 4348 |
| 16 | DeepSeek-V2-Lite | LRU | 19656 | 15335 | 24 | 131 | 4166 |
| 16 | DeepSeek-V2-Lite | LFU | 19656 | 14384 | 56 | 100 | 5116 |
| 16 | Mixtral-8x7B-v0.1 | LRU | 8064 | 0 | 59 | 827 | 7178 |
| 16 | Mixtral-8x7B-v0.1 | LFU | 8064 | 0 | 63 | 822 | 7179 |
| 16 | Qwen1.5-MoE-A2.7B | LRU | 12096 | 9595 | 17 | 61 | 2423 |
| 16 | Qwen1.5-MoE-A2.7B | LFU | 12096 | 11124 | 6 | 13 | 953 |

## 可比性限制

- 新版本四模型 OFF/ON 64/8 短测均通过：8/8、4/4 配对，628 检查，输出及有序路由相同。
- 完整矩阵每个成功配置内部三个请求的输出/有序路由均一致；但 **DeepSeek 16 GiB LRU ↔ LFU** 的 decode 路由不一致。DeepSeek LFU：4 ↔ 8 GiB 可比；凡涉及 16 GiB（4 ↔ 16、8 ↔ 16）均存在路由差异。上述差值不能严格归因于策略或容量，不作因果排名。
- 其余成功容量内 LRU/LFU 配对通过 prompt、输出、有序路由、分母比对；结论仍仅限单个固定 prompt 的这个实验，不能泛化为整个数据集。
- 旧 OFF 完整数据来自 native772，新 ON 来自 native2f，是历史参照而非完整同 binary A/B。旧 phase 编码 MIXED=2，新明确绑定 PREFILL/DECODE；原始 hash 不能不加区分地比较。

## 故障、重跑与审计历史

- DeepSeek 16-LFU 首次因 legacy H2D 转换计数遗漏一次 managed GPUFetch fallback，被旧校验拒绝。旧结果没有补成 PASS；只在新 frozen artifact 中把 legacy 视作 native 记录搬运账目的子集，重新跑完整 512/64/1+2。
- DeepSeek 8-LFU 首次虽输出相同，但重复有序路由不同，独立验证拒绝；另跑完整请求，最终接受 attempt-2。
- Qwen1.5 8-LFU 和 Mixtral 4-LFU 首次在原生 `wait_expert` 停滞，保存 SIGUSR1 栈与终止证据；重跑成功不代表首次成功，也不声称根因已修复。
- Qwen1.5 4-GiB 两首次调度因 root 手动重跑与队列调度租约相撞，在加载模型前被拒绝，无数据；随后在 GPU2 串行完整重跑。
- 所有实际产生的旧失败 raw/log/terminal 文件及 scheduler 外层日志均保留，`prior_attempts` 记录其路径；不得拿部分成功请求替代完整结果。Qwen1.5 4-GiB 两次加载前的租约拒绝未创建 raw、主 log 或 terminal，只由现存 scheduler 外层日志及 journal 的 returncode=75 证明，不存在的文件不是丢失的测量记录。
- 六个明确 override 使用 harnessA512，其余使用856；AST/runner证明仅校验 legacy 子集和诊断变化，所有协议、观察器、主指标、路由、native 均不变。混合版本只按 cell hash 严格批准。
- H2D 指标是 **native recorded 参数搬运账目**，不是独立 PCIe 带宽实测。legacy 子集与 native 账目不必相等；不据此宣称精确物理流量节省。

## 文件与复现

- 最终逐配置 raw/GPU/store/版本/历史：`prefetch-manifest.executed.json`
- 验证结果：`prefetch-validation.json`；汇总：`prefetch-summary.json`、`prefetch-summary.csv`
- 原始结果、日志、终止 sidecar：`results/`、`logs/`
- 校验变体证明：`harness-equivalence-evidence.json`；历史 OFF hash：`provenance/off-baseline-pins.json`
- native源码变更说明：`prefetch-native-review-notes.md`；预取机制解读：`scientific-interpretation-notes.md`
- 复现说明：`reproduce.md`


## 代码修改究竟是为了什么？

**确实修改过代码，也并非全部只是加日志。** 目的分为：模型适配、实现此前明确要求的真正 LFU、使已有普通专家预取按正确阶段/目标层运行、保护参数搬运与执行的并发安全，以及把命中和搬运账目统计准确。没有用新的缓存优化算法或更有利的实验设置替代原来的任务来“让结果好看”。

### 按开发阶段区分

| 阶段 | 改动 | 目的与性质 |
|---|---|---|
| `96695c9 → f7f56df`：已经提交的测试基线 | Qwen2-MoE 适配；Mixtral runtime/checkpoint 名称匹配；DeepSeek 路径大小写与层编号；position IDs 设备对齐；CUDA 构建兼容 | 让四个既定模型能在当前依赖和设备上正确初始化、映射与执行；是模型/运行适配，不是新增缓存策略。 |
| 同一已提交基线 | 路由 expert 与 generic 节点驻留键分离；多 victim 淘汰与 reservation 计账；`wait_expert` 期间释放 GIL | 修复假命中、空间不足和等待期间的并发问题；会影响真实执行，并不是纯日志。 |
| `f7f56df → pre-prefetch 7c6284…`：此前已完成、仍未提交 | 真正 resident-cache LFU 以及 native 配置/读回 | 按用户要求替代旧 LFU-like：在当前驻留周期累计需求访问，命中和 miss 都计一次；预取不增频；真正淘汰清零；最少频率优先，平局按旧需求序列及 logical key。**这是新增策略实现**，已存在于本次预取测试之前。 |
| 同一 LFU 阶段 | 忙/执行中/pending-dispatch 节点的淘汰资格检查；原子 eviction claim；事务 RAII 清理 | 防止还在使用的参数被搬回 CPU，或 admission/淘汰失败遗留状态。影响候选资格/生命周期，但不增加另一个“聪明”淘汰评分。 |
| `7c6284… → final 286970…`：本次预取工作 | 显式 PREFILL/DECODE；正确关联目标层；删除重复预取/纠错时再发当前 miss | 让现有预测预取真正运行，并修复错层与重复发请求；属于**功能性 bugfix**。 |
| 本次预取工作 | 传输唯一所有权；已驻留 no-op；持有状态直到 commit/release；失败回滚 | 防止 demand/prefetch 重复或冲突地搬同一个 expert，防止参数还没可用就被当作可消费；属于**并发安全修复**。 |
| 本次预取工作 | native ready/late/reuse 分类、任务 issued/terminal 账目、取消任务扣账、事件与字节计数 | 使统计能完整收敛、能区分“参数显示 CUDA”和“使用时已经 ready”；主要是**观测与记账修复**，不是提高 hit 的算法。 |
| 六个明确 fresh 重跑使用的 A512 artifact | legacy H2D 等于 native 的错误校验改为子集诊断 | legacy 只统计部分 ApplyPrecisionTarget 搬运，native 还记录 managed GPUFetch fallback，二者不应强求相等。只改变校验/诊断，主命中公式、协议、路由、生成及 native 均未变；旧失败没有被补写成成功。 |

### 三项功能性修复的直白解释

1. **让预取开关真正生效。** 直接 HF `generate` 原先可能处于默认 `MIXED` 阶段，而该普通预测路径会直接返回。测试 wrapper 把多 token dispatch 标为 prefill、单 token 标为 decode；两阶段 top-k 使用模型实际 routed top-k。这里不是更换预测器，而是接通已有路径。
2. **修正“给哪一层预取”。** L−1 的预测应在真实走到 L 时与 L 的实际 expert 集合比较，而不是把当前层实际路由拿去纠正下一层。纠错现在只统计 TP/FP/FN，不拿真实路由做未来 oracle，也不把当前 miss 再重复发送为预取。
3. **保证搬运/执行不打架。** 预取先拿到节点所有权，再申请缓存、H2D、提交和释放；同一个参数已有驻留时直接结束为 no-op，失败时回收状态。这会改变旧 bug 路径中的调度及淘汰时机，因此不能声称它对原先行为“完全没有影响”。

> 本轮保留既有 LRU/LFU 排序规则，不代表整个旧实现与修复后的实现行为完全等价。真实预取执行与安全资格检查的修复可能影响观测到的命中率，结果应归属于本文记录的**修改后本地版本**。

### 没有采用哪些“让它通过”的手段

- 没有新增或启用 ARC、论文 EAMC、历史路由优先级策略、按层固定配额或人为 pin 一组 expert 来改善主表。
- 没有提前读取下一层真实 gate，或用测量后的真实路由反过来当未来预测。普通预测仍是**当前层 logits 均值 top-k 编号 → 下一层候选**，其预测质量本身并不保证好。
- 没有主动修改模型真实 gate 数学/选 expert 实现，或用预测集合替代 `router_mask`。
- 没有量化、切换模型、减少请求长度、缩小失败配置的 cache、改为 CPU expert 执行，或把同一个模型分到多卡以避开 OOM。
- 没有把“预取正在做/迟到”计作 hit；没有把 OOM、停滞或部分请求的结果计作 0% 或完整成功。
- 没有丢弃旧失败来覆盖历史；六个结果来自不同新进程的完整 `512/64/1+2` 重跑。
- 没有声称已修复两次 `wait_expert` 停滞的根因。此次只保留证据并做有界 fresh 重试。

### 验证能证明什么，不能证明什么

- 36 个真实候选链接 C++ 测试、native 小 GPU 例子、四模型短程 OFF/ON 和最终 5,008 项结果校验支持实现与账目可靠；不是把阈值调低来掩盖少拷贝、漏任务或错误命中。
- A512 的校验变化有逐函数 AST、runner diff 和 hash 证据，允许的两套 artifact 只按具体 cell 固定批准；实际 native 是同一个 `2f832c4…`。
- **不是完整模型与原版 HF 的数值一致性证明。** 未主动修改 gate，也不等于每次异步执行的路由逐项相同：DeepSeek 16 GiB 的路由差异已观察到，仍必须披露。
- 旧 OFF 完整表来自较早本地 native772；不能把新 ON 与旧 OFF 的差值直接写成“仅打开一个开关导致的纯收益”，更不能当作未经修改上游或论文完整复现成绩。

### 审核代码的主要入口

以下行号对应本次保存时工作区；它们是定位入口，之后编辑可能移动行号。

| 目的 | 文件 / 入口 |
|---|---|
| LFU 排序与安全候选 | [`core/prefetch/expert_residency.cpp`](../../core/prefetch/expert_residency.cpp)，`SelectVictim`（约 181–225 行） |
| LFU 每次需求增频及 ready/late/reuse 分类 | 同文件 `RecordAccess`（约 797–868 行） |
| LRU/LFU 真正配置及回读 | [`moe_infinity/runtime/model_offload.py`](../../moe_infinity/runtime/model_offload.py)，约 743–790 行 |
| 预取传输所有权/失败回滚/no-op | [`core/prefetch/task_scheduler.cpp`](../../core/prefetch/task_scheduler.cpp)，`RunTrackedRoutedPrefetch`（约 31–136 行） |
| 取消任务 terminal 扣账及计数聚合 | 同文件 `FetchExec`、`StartExec`、`TrackPrefetchTerminal`、`GetPrefetchPipelineStats` |
| 当前层→下一层普通预测、目标层纠错 | [`moe_infinity/memory/expert_prefetcher.py`](../../moe_infinity/memory/expert_prefetcher.py)，约 436–580 行 |
| 实际 expert dispatch 与纠错顺序 | [`moe_infinity/distributed/expert_executor.py`](../../moe_infinity/distributed/expert_executor.py)，约 480–568 行 |
| HF 阶段绑定与完整实验协议 | [`benchmark_prefetch.frozen-856e4314.py`](../../.omx/research/moeinf-prefetch-matrix/benchmark_prefetch.frozen-856e4314.py)，`dispatch_with_phase` / `runtime_config` |
| legacy 子集校验及 scoped 变体证明 | [`benchmark_prefetch.frozen-a512d061.py`](../../.omx/research/moeinf-prefetch-matrix/benchmark_prefetch.frozen-a512d061.py)、[`harness-equivalence-evidence.json`](../../.omx/research/moeinf-prefetch-matrix/harness-equivalence-evidence.json) |

## 最终版本与原始证据索引

- Git HEAD：`f7f56df0cff8b5b609316dcd96be66443c07bab7`。
- 实验源代码 patch SHA256：`2869700f303aba0d3c4ecbbfe85223e83bc6b0eabd060d516713964833660f06`（基线之后的 source/test diff）。
- native `_store` SHA256：`2f832c4b51ecf2ad46cd2eef8b0f1323ba714c85e068e2eae4a1a34507419e57`。
- 逐格模型路径、实际 GPU/store、版本及旧失败：[最终 executed manifest](../../.omx/research/moeinf-prefetch-matrix/prefetch-manifest.executed.json)。
- [独立验证 5,008 项](../../.omx/research/moeinf-prefetch-matrix/prefetch-validation.json)、[精确汇总 JSON](../../.omx/research/moeinf-prefetch-matrix/prefetch-summary.json)、[CSV](../../.omx/research/moeinf-prefetch-matrix/prefetch-summary.csv)。
- [完整复现说明](../../.omx/research/moeinf-prefetch-matrix/reproduce.md)、[实现/数据独立审计](../../.omx/research/moeinf-prefetch-matrix/final-audit.md)。

**本次保存文档没有重新运行实验，也没有再改源代码、stage 或 commit。** 结果表是已验证结果的归档，不是重新生成或调高的成绩。

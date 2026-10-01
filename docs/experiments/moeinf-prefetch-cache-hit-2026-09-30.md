# MoE-Infinity 普通 MoE 推理：专家预取与缓存命中率测试报告

- 实验日期：2026-09-30。
- 报告修订日期：2026-10-01。
- 研究对象：普通自回归 MoE 推理中的专家参数预取，以及 LRU/LFU 在不同缓存容量下的 decode 命中率。
- 实验规模：4 个模型 × 3 档容量 × 2 种策略，共 24 个配置；专家预取均开启。

> **版本说明：** 本文保留历史实验结果，测量对象是 `f7f56df` 基线加源代码补丁 `286970…` 的本地版本，不是未经修改的上游版本。后续部分 Python 预取改动已撤回，保留的 LFU、原生缓存实现及测试已提交到 `620b343`。因此，本文数字**不代表当前提交重新运行的结果**，旧实验脚本也不能直接套用到当前 Python 接口。此次仅重写报告，没有重新测量或更改原始数据。

## 实验目的与模型

本实验回答两个问题：

1. 开启专家参数预取后，四个模型在 decode 阶段的缓存命中率、未命中率分别是多少？
2. 在相同模型、请求和容量条件下，LRU 与 LFU 的表现如何；容量由 4 GiB 增加到 8/16 GiB 后，结果怎样变化？

| 模型 | 本地模型路径 |
|---|---|
| DeepSeek-V2-Lite | `/home/zijie/models/DeepSeek-V2-Lite` |
| Mixtral-8x7B-v0.1 | `/home/zijie/models/Mixtral-8x7B-v0.1` |
| Qwen1.5-MoE-A2.7B | `/home/zijie/models/Qwen1.5-MoE-A2.7B` |
| Qwen2-57B-A14B | `/home/zijie/models/Qwen2-57B-A14B` |

## 测试设置

| 项目 | 设置 |
|---|---|
| 执行方式 | 单 GPU 运行一个任务，在物理 GPU 1、2、3、4 上并行调度 |
| 生成入口 | Hugging Face `generate()` 路径，`use_native_engine=False` |
| 生成设置 | `do_sample=False`、`num_beams=1` |
| 参数精度 | BF16，未量化 |
| expert 计算位置 | GPU；按需或通过预取将参数从 CPU 内存搬至 GPU |
| batch size | 1 |
| 每请求长度 | 输入 512 token，生成 64 token |
| 请求次数 | 1 次 warmup，随后 2 次正式测量 |
| 主表统计范围 | 两次正式测量的 decode；排除 warmup 和 prefill |
| 缓存策略 | managed LRU、resident LFU |
| 缓存容量 | 4、8、16 GiB，按字节设置并读回校验 |
| 专家预取 | 开启，prefill/decode 的预取 top-k 均取模型 routed top-k |
| 其他设置 | 不启用 attention/KV offload 或自适应专家精度 |

**容量口径：** 4/8/16 GiB 是共享驻留管理器（residency manager）的逻辑预算，包含它所管理的非路由参数，**不是纯 expert 缓存容量，也不是整张 GPU 的总显存硬上限**。底层设备池另外按构造比例 0.75 设置。因此，设置了某个管理器预算，并不意味着整个模型一定能在该 GPU 上运行。

`use_native_engine=False` 只说明没有使用原生的生成循环；专家调度、缓存管理和参数搬运仍使用 C++ 实现，并不表示 expert 改为 CPU 计算。

## 本次使用的缓存与预取机制

### LRU 与 LFU

- **LRU：** 在允许淘汰的驻留对象中，优先淘汰最久未访问的对象。
- **LFU：** 在当前驻留周期内累计需求访问次数，优先淘汰次数最少的对象；频率相同时按较旧的需求访问序列、再按逻辑键确定顺序。需求命中和 miss 均计一次，预取不增加频率，真正淘汰后重新载入会重新计数。
- 两者使用相同的管理器容量与准入路径；正在使用、被保护或尚未完成加载的对象还受生命周期约束。

这里测试的是已实现的 resident LFU，不是此前的旧 LFU-like 策略。

### 专家参数预取

实验使用当前层路由信息预测下一层候选专家：

```text
当前层 router logits
  → 按 token 求均值
  → 选取 top-k 专家编号
  → 提前搬运下一层相应 expert 参数
  → 到达下一层后，仍按该层真实 gate 决定计算哪些 expert
```

它预测的是**需要搬运的参数**。没有提前读取下一层真实 gate，也没有用预测集合替代实际路由；不能将其称为论文 EAMC 的完整实现。

对应原始配置：

```python
prefetch = True
speculative_prefetch = True
speculative_prefetch_overlap = True
overlap_prefetch_policy = "off"
```

这里的 `speculative_prefetch` 是仓库中“预测式专家参数预取”的接口名称；`overlap_prefetch_policy="off"` 表示没有启用另一套重叠预算控制器，**不表示关闭专家预取**。

实验脚本根据层调用的 token 行数显式绑定 PREFILL/DECODE 阶段。实验快照还包含预取目标层关联、纠错行为、并发加载生命周期及计数方面的本地改动，**不只是增加日志，也不是未经修改的上游行为**。

## 命中率如何计算

统计单位为一次层调用中一个去重的活跃 `(layer, expert)` 查找。对于本实验 batch=1 的 decode，每层的查找数等于该层 routed top-k。

| 原生分类 | 含义 | 主指标归类 |
|---|---|---|
| `reuse_hit` | 复用已驻留的专家参数 | Hit |
| `prefetch_ready_first_use` | 预取参数第一次被需要时已经就绪 | Hit |
| `prefetch_late_first_use` | 预取已发起，但第一次被需要时尚未就绪 | Miss |
| `demand_miss` | 到达时缺失，需要需求加载 | Miss |

```text
Hit = reuse_hit + prefetch_ready_first_use
Miss = prefetch_late_first_use + demand_miss
总查找 = Hit + Miss

Hit rate = Hit / 总查找
Miss rate = Miss / 总查找
```

先汇总两次正式测量的计数，再计算比例；以下百分比保留两位小数。参数显示在 CUDA 上，不代表异步搬运已经完成，因此 Python `is_tensor_on_device` 只用于交叉检查，不作为主命中计数。

**最终状态：20 个配置完成全部请求，4 个配置因 CUDA OOM 未完成。** OOM 记为 N/A，而不是 0% 命中率。

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

## 就绪、迟到、复用分解（decode）

下表为两次正式测量的累计计数，可直接复算上述百分比。

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

## 结果解释

### Mixtral：预取没有解决跨层循环淘汰

六个成功配置的 decode `reuse_hit` 都为 0，约 0.7–0.8% 的命中来自少量提前就绪的预取。与此前的跨层循环淘汰分析一致，这些配置没有形成可观测到的跨次需求复用。

预取改变的是参数搬运的时机，不会自动扩大可保留的工作集，也不会自动给各层分配受保护的缓存空间。因此，开启预取并不保证能够解决缓存循环淘汰。

### Qwen1.5：该请求下增加容量有明显收益

在这次固定请求中，LRU 命中率从 28.21% 增至 79.46%，LFU 从 31.45% 增至 92.01%。成功配置的同容量策略配对通过了输入、输出、有序路由和分母比对；这不是对所有输入分布的普遍结论。

### DeepSeek：记录结果，但不能忽略路由差异

表中可见较大容量下的命中率较高，但 16 GiB 的 LRU 与 LFU 存在 decode 路由差异；LFU 的 4/8 GiB 与 16 GiB 之间也存在路由差异。

因此，相关差值不能全部归因于缓存策略或容量，不据此对这些配置作严格的因果排名。

### Qwen2-57B：低命中与 OOM 必须分开

4 GiB 的两种策略都完成测量，但均仅命中 8/28,224 次，约 0.0283%。8/16 GiB 的四个配置由失败日志与终止记录确认为 OOM，没有完整完成请求；这不等于模型不支持，也不意味着运行中从未生成 token。

### 命中率不等于吞吐率

迟到的预取虽然计入 Miss，仍可能已经完成一部分搬运；错误或未使用的预取则可能额外占用带宽和缓存。本报告不据此推导 token/s 提升或精确的 PCIe 流量节省。H2D 数字来自原生代码的参数搬运账目，不是独立硬件带宽测量。

## 验证与可比性边界

- **5,008 项归档结果检查通过，0 FAIL、0 INCOMPLETE。** 这是配置、日志、计数和结果一致性等检查项，不是 5,008 次独立实验，也不等于 24 个配置全部推理成功。
- 同一实验版本下，四模型进行过输入 64 / 输出 8 的短程 OFF/ON 检查：8 个任务、4 对配置、628 项检查通过，配对输出及有序路由相同。该短测不能替代完整容量/策略矩阵的 OFF/ON 对照。
- 完整矩阵每个成功配置内部，三个请求的输出及有序路由一致。DeepSeek 16 GiB 的跨配置例外如上；其余成功的同容量 LRU/LFU 配对通过相应一致性检查。
- 历史 OFF 完整表使用较早的 native772，本文 ON 表使用 native2f；两者不是完整同二进制的 A/B 实验，不能把差值全部解释为“只打开预取开关”的收益。
- 全部结果只覆盖单个固定 prompt、batch=1 和指定长度，不能泛化为整个数据集或在线多请求服务。
- 单元测试、短测及账目检查不能证明所有并发情形都安全，也不是与原版 Hugging Face 模型的完整数值一致性证明。

## 失败与重跑记录

历史失败未被成功重跑覆盖，原始数据、日志和终止记录按实际产生情况保留。

| 配置或事件 | 处理与限制 |
|---|---|
| DeepSeek 16 GiB LFU | 首次因旧 H2D 校验遗漏 managed GPUFetch fallback 被拒绝；将旧计数视为原生搬运账目的子集后，使用新脚本完整重跑，没有把旧结果直接改为 PASS。 |
| DeepSeek 8 GiB LFU | 首次重复请求的有序路由不同，被拒绝；另跑完整请求，最终采用 attempt-2。 |
| Qwen1.5 8 GiB LFU、Mixtral 4 GiB LFU | 首次在原生 `wait_expert` 停滞，保存栈和终止证据；重跑成功不代表停滞根因已经修复。 |
| Qwen1.5 4 GiB 的两种策略 | 首次调度因任务租约冲突在加载前被拒绝；随后在 GPU2 串行完整重跑。加载前拒绝没有生成主 raw/log/terminal，只由调度日志和 returncode=75 证明。 |
| Qwen2-57B 8/16 GiB 的两种策略 | 保留 OOM 分类，不用部分请求补成完整结果。 |

六个指定重跑采用 A512 冻结脚本，其余采用 856 版本。两者使用同一个 native 二进制；已归档的 AST/启动脚本比对将差异限定为旧 H2D 子集校验与诊断变化，主命中公式、请求协议和生成路径不变。每个配置采用哪个脚本，由最终清单记录。

## TODO：低命中率问题与更大显存 GPU 验证

以下是后续研究与实验清单，**尚未执行，也不表示已经实现新的缓存策略**。更换设备后应建立新的结果目录和版本记录，不覆盖本文历史数据。

### 待验证的问题

**核心假设：参数会重复使用，但在跨层遍历后再次被需要之前，已经被缓存淘汰。** prefill 结束后缓存偏向后几层，随后从前几层开始 decode；之后每个 token 又依次遍历各层，可能持续出现循环淘汰。增加 warmup 不一定能消除这种容量与复用间隔不匹配。

按本地 BF16 权重尺寸，仅计算 routed expert 参数：

| 模型 | MoE 层数 × routed top-k | 单个 expert | 一个 decode token 涉及的 routed 参数工作集 | 乐观容量对照 |
|---|---:|---:|---:|---|
| Mixtral-8x7B-v0.1 | 32 × 2 | 336 MiB | 21 GiB，64 个层—expert 对象 | 4/8/16 GiB 最多放 12/24/48 个 |
| Qwen2-57B-A14B | 28 × 8 | 52.5 MiB | 11.484375 GiB，224 个对象 | 4 GiB 最多放 78 个 |
| Qwen1.5-MoE-A2.7B | 24 × 4 | 16.5 MiB | 1.546875 GiB，96 个对象 | 4 GiB 最多放 248 个 |
| DeepSeek-V2-Lite | 26 × 6 | 16.5 MiB | 2.513671875 GiB，156 个对象 | 4 GiB 最多放 248 个 |

尺寸依据见[本地权重预检记录](../../.omx/research/moeinf-four-models/results/preflight-patched.json)。不同层的同编号 expert 是不同参数对象；“工作集”不是实测每 token 的 H2D 流量。这些换算不包含其他参数、KV cache、激活、临时工作区和分配器开销，实际共享预算还会被非路由对象占用。

对于 Mixtral，同一 expert 在相邻 decode token 间复用，结构上至少跨过其他 31 层的 62 个不同 expert；16 GiB 理想情况下只能容纳 48 个。Qwen2-57B 对应至少跨过 216 个，4 GiB 最多容纳 78 个。这支持普通 LRU 的循环淘汰解释，但不是任意缓存策略都必然零命中的证明，也不是已经逐事件测得的精确复用距离。

- [ ] **直接验证循环淘汰。** 记录按 token/层排列的实际 expert 请求、命中类型、准入、淘汰对象及原因、每层驻留量，重建复用距离和相应参数字节量。观察是否确实在再次使用前被后续层挤出，而不是仅凭模型总大小推断。旧 OFF 的快照与结构分析见[零命中审查](../../.omx/research/moeinf-capacity-sweep/mixtral-zero-audit.json)，不能把它当成本次 ON 的逐事件轨迹。
- [ ] **验证 resident LFU 的频率积累限制。** 分别统计需求访问频率分布、驻留时长、淘汰后重新计数的次数，以及同频平局的淘汰决定。旧 OFF 的测量结束快照中，Mixtral 驻留对象频率均为 1，支持“第二次访问前已淘汰”的解释；需要在新版本、新容量下重新验证，不能预设所有 LFU 都无效。
- [ ] **分开分析预测失误与预取迟到。** 同时记录下一层预测集合与实际集合的交集、预取发起/完成及需求到达时序、ready/late/reuse/demand-miss、未使用预取被淘汰的数量及搬运账目。判断低命中来自预测不准、提前量不足还是已预取参数再次被挤出，不把“发起预取”当作“命中”。
- [ ] **区分两个千问模型。** Qwen2-57B 的 4 GiB 是极低命中，而 8/16 GiB 是 OOM；Qwen1.5 在较大容量已有明显命中，不能合并成“千问整体零命中”。对 OOM 单独记录失败阶段和显存占用，不填入 0% 命中率。

### 更大显存 GPU 上的实验计划

- [ ] **先建立可复现的新测试基线。** 当前 Python 接口与历史冻结脚本不完全兼容；启动前核对或适配观测接口，验证策略读回、阶段划分和命中计数，再冻结源码、native 二进制与测试脚本。不能只更换 GPU 就将历史脚本视为可直接运行，也不能把新结果标成历史版本的复测。
- [ ] **先确认设备和显存预算，再确定容量矩阵。** 记录 GPU 型号/显存、CPU—GPU 连接、驱动、软件及 native 版本；检查管理器预算、构造设备池、非路由参数、KV/激活/工作区和峰值显存。21 GiB、11.48 GiB 是单 token 的 routed 参数工作集估算，**不是总显存最低要求，也不保证超过后就高命中**；不同 token 的路由可能变化，长期工作集可能更大。
- [ ] **优先扫描 Mixtral、Qwen2-57B 的容量转折区间。** 在显存允许的前提下，Mixtral 可考虑 4/8/16/20/24/32 GiB；Qwen2-57B 可考虑 4/8/12/16/24 GiB。它们是候选管理器预算，最终档位需根据真实显存余量确定。先复测原有可运行档位，再观察跨越上述工作集附近时 `reuse_hit` 是否上升；不要直接假定某档足够。
- [ ] **建立同设备、同版本的 LRU/LFU × 预取 OFF/ON 对照。** 在新 GPU 上重新跑小容量基线，固定模型、精度、请求、代码、二进制及统计口径；不能将新 GPU 的 ON 与旧 GPU 的历史 OFF 差值解释为单独的预取收益。每张 GPU 同时只运行一个任务，使用哪些卡及最终测试范围在实际启动前确认。
- [ ] **先沿用当前协议，再扩展负载。** 第一阶段保持 batch=1、512 输入/64 输出、1 warmup + 2 次测量；优先解决两大模型的问题，再以 Qwen1.5 和 DeepSeek 作为对照。第二阶段增加 prompt 类型、重复次数及 decode 长度，区分冷启动与稳态复用，仍将 prefill/decode 分开报告。
- [ ] **校验路由与结果可比性。** 保存输入/输出和有序路由指纹；跨策略或容量出现路由差异时，单独披露，不把命中率差值全部归因于缓存。必要时用同一冻结路由轨迹做离线策略对照，并明确它不替代真实模型运行。
- [ ] **输出容量曲线和逐层分解。** 主要报告 decode Hit/Miss、四类到达计数、每层驻留量、需求及预取搬运账目、未使用预取淘汰量和 OOM/停滞状态。保存全部失败及重跑记录，不能只展示最终成功配置；增加容量后是否仍有循环淘汰，应由这些记录共同判断。

### 容量仍不足时的备选研究方向

- [ ] 在完成上述基线验证后，单独设计按层配额、保护部分层/高复用 expert、流式处理其他 expert 或准入旁路的对照，检查是否能避免全局缓存被逐层扫描冲刷。
- [ ] 如研究跨淘汰保留频率历史或带衰减的频率策略，应明确将其标为新的策略变体，不与当前 resident LFU 混称；新策略与基线保持相同预算，并同时评估路由正确性和预取浪费。

研究动机可用一个理想例子说明：循环访问 A→B→C→D，而缓存只有三个槽，LRU 在稳态可能每次都 miss；若保护 A、B，让 C、D 共用第三槽，稳态可有 50% 命中。**这是解释机制的示例，不是四个模型的已测优化收益。**

## 实验版本与原始证据

### 版本标识

| 项目 | 标识 |
|---|---|
| 实验时 Git 基线 | `f7f56df0cff8b5b609316dcd96be66443c07bab7` |
| 实验源码补丁 SHA256 | `2869700f303aba0d3c4ecbbfe85223e83bc6b0eabd060d516713964833660f06` |
| 原生 `_store` SHA256 | `2f832c4b51ecf2ad46cd2eef8b0f1323ba714c85e068e2eae4a1a34507419e57` |
| 主要冻结脚本 | `benchmark_prefetch.frozen-856e4314.py` |
| 六个指定重跑脚本 | `benchmark_prefetch.frozen-a512d061.py` |

这些标识描述**当时实际测量的版本**，不是用当前提交号替换历史版本。复现这些历史数字需要相应源码快照、native 二进制、冻结脚本及模型输入；用当前代码重新测试应另外记录版本和结果。

### 本机归档索引

以下证据位于本机 `.omx/research/moeinf-prefetch-matrix/`。该目录不随 Git 提交分发，单独克隆仓库不会自动获得这些原始文件。

- [最终逐配置清单](../../.omx/research/moeinf-prefetch-matrix/prefetch-manifest.executed.json)：模型路径、GPU、容量、策略、脚本版本及历史尝试。
- [结果检查](../../.omx/research/moeinf-prefetch-matrix/prefetch-validation.json)、[汇总 JSON](../../.omx/research/moeinf-prefetch-matrix/prefetch-summary.json)、[汇总 CSV](../../.omx/research/moeinf-prefetch-matrix/prefetch-summary.csv)。
- [原始结果目录](../../.omx/research/moeinf-prefetch-matrix/results/)、[日志目录](../../.omx/research/moeinf-prefetch-matrix/logs/)。
- [主要冻结脚本](../../.omx/research/moeinf-prefetch-matrix/benchmark_prefetch.frozen-856e4314.py)、[重跑冻结脚本](../../.omx/research/moeinf-prefetch-matrix/benchmark_prefetch.frozen-a512d061.py)、[脚本差异证据](../../.omx/research/moeinf-prefetch-matrix/harness-equivalence-evidence.json)。
- [普通专家预取机制说明](../../.omx/research/moeinf-prefetch-matrix/scientific-interpretation-notes.md)、[历史复现说明](../../.omx/research/moeinf-prefetch-matrix/reproduce.md)、[历史 OFF 版本记录](../../.omx/research/moeinf-prefetch-matrix/provenance/off-baseline-pins.json)。

**本次报告修订不改变三张主表、20 行计数分解、失败分类或原始实验文件；没有重新运行模型，也没有修改实现代码。**

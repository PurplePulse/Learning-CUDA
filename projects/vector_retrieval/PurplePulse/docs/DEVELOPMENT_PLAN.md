# 开发计划

## 第一阶段：正确性地基

- [x] 定义带版本号的二进制文件格式
- [x] 实现数据生成、读取和检查
- [x] 实现 CPU 精确检索参考答案
- [x] 实现容易理解的 CUDA 距离计算 baseline
- [x] 建立自动正确性与边界测试
- [x] 在中等规模上复验 L2、内积和 Cosine 的 CPU/GPU 一致性

## 第二阶段：精确检索性能

- [x] 将 Top-K 从 CPU 移到 GPU
- [x] 实现分块与两阶段 Top-K
- [x] 将局部 Top-K 从有序插入改为无序候选 + 最差项跟踪
- [x] 为 K=1/10 增加编译期模板特化
- [x] 实现 FP16 输入、GPU 16 位存储和 FP32 累加
- [x] 扫描 batch=1/4/8/16/32/64/128 的吞吐与延迟
- [x] 融合 Exact 距离计算与局部 Top-K，删除完整距离矩阵写回
- [x] 用 nsys 记录 Top-K 优化前后的 kernel 时间线
- [x] 完成 ncu 可用性诊断；确认赛事平台宿主权限限制并在报告中记录，不虚构计数器数据

## 第三阶段：IVF-Flat

- [x] 用确定性采样 k-means 训练聚类中心并构建倒排表
- [x] 使用连续桶布局保存和加载版本化索引
- [x] 实现 CPU 固定 nprobe 正确性基线
- [x] 对比 exact search 的 recall@K
- [x] 实现第一版 GPU 常驻索引、候选扫描和 Top-K
- [x] 将中心选择迁到 GPU，并消除候选位置 H2D
- [x] 使用每个 probe 一个 block 提高单 query 候选扫描并行度
- [x] 对相似度度量使用球面 k-means，修复大 nlist 下的桶倾斜
- [x] 使用 block 归约并行选择中心，替代单线程有序插入
- [x] 完成 100k × 128、K=10 的首轮 nlist × nprobe 扫描
- [x] 使用 warp 协作改善桶内向量读取与距离计算，并保留 scalar A/B 基线
- [x] 百万规模扫描 nlist=256/512/1024 与对应 nprobe
- [x] 在百万规模扫描 IVF batch=1/8/16/32/64/128
- [x] 对 K=1/10 完成首轮质量与吞吐对比
- [x] CLI 报告 exact/IVF 显式 GPU 缓冲区，并提供自动 CSV sweep 脚本
- [x] 使用 K=10/50/100 模板化 warp heap 扩展 IVF 局部 Top-K
- [x] 使用 probe 有序列表 k-way merge 高效归并 K=50/100
- [x] 百万规模完成 K50/K100 的 batch 与 nprobe 首轮扫描
- [x] 将 metric 编译期特化，并删除每候选无效的 warp shuffle 归约
- [x] 将小于等于 8 KiB 的 query 缓存到 block 共享内存
- [x] 为大 K 使用 compact 2-warp block，保留 warp2/4/8 A/B 路径
- [x] 优化后重新扫描 K10/K50/K100 的 batch 与 nprobe

## 第四阶段：Agent 场景创新

- [x] 根据归一化中心分数质量为每个 query 自适应选择 nprobe
- [x] 实现 masked/grouped 两种变长 GPU 调度并完成 A/B
- [x] 与相同平均 probe 的固定 nprobe 完成百万级公平实验
- [x] 根据负结果保留实验接口但不替换固定策略默认值
- [x] 在 GPU 候选扫描中融合元数据过滤和记忆评分
- [x] 与分离重排序方案进行公平实验

## 第五阶段：正式实验与报告

- [x] 百万向量、128 维、1000 queries 固定数据集
- [x] 记录 QPS、run/batch P50/P99、显存和 recall@K
- [x] 完成 FAISS CPU/GPU Flat/IVF-Flat 与自研引擎首轮公平对照
- [x] 保存 FAISS 与自研 nprobe 扫描原始 CSV
- [x] 输出 recall-latency、recall-QPS 和 batch-QPS 曲线图
- [x] 记录赛事平台无法开放 ncu 计数器的原因、影响范围和替代证据
- [x] 完成最终总结报告与复现说明

最新优化后，K10 在 nprobe=160/224 分别约为 FAISS GPU IVF 的
1.47×/1.64×；K50 的差距缩小到约 1.31×/1.42×，K100 缩小到约
1.47×/1.50×。首轮 score-mass 自适应 nprobe 在均匀合成数据上未超过固定
策略，已作为有证据的负结果保留。GPU 元数据融合在 100k 正式 A/B 中比
Top-100 后独立重排快约 2.23×，且避免了先截断导致的记忆候选损失。正式
曲线和总结报告已经生成。Exact fused 在百万规模 K10/K50/K100 上相对矩阵
路径分别加速约 3.13×/7.16×/10.59×，显式 GPU 缓冲区减少约 33%。赛事
平台不开放 ncu 所需宿主权限，诊断证据和替代验证方式已写入最终报告。

## 第六阶段：Agent 应用方向（后续开发）

> 本节以下内容属于 Agent 应用方向，不改变项目主体。PurplePulse 仍是一套
> 可独立构建、运行和评测的 C++/CUDA GPU 向量检索引擎；Agent 是用于验证
> 元数据融合检索价值的真实工作负载，Pi 只是首个轻量客户端。

### 6.1 范围与边界

- PurplePulse 核心继续负责 Exact、IVF-Flat、GPU Top-K、常驻索引、批量
  查询、元数据前置过滤以及语义/时间/重要性融合评分。
- Agent 侧只负责提取记忆、生成 query、调用检索接口和组装上下文，不在
  PurplePulse 内嵌 LLM，也不让 Agent 框架接管索引与排序逻辑。
- 第一版采用 session 结束后的批量摄取和下一 session 的在线检索；暂不实现
  实时增量索引、分布式服务、多 Agent 编排或 Web UI。
- 接口保持框架无关，Pi adapter 放在 `clients/pi/`，后续可以增加其他 Agent
  客户端而无需修改 CUDA 核心。

### 6.2 最小可交付链路

- [ ] 定义 Agent Memory JSONL 格式，包含 content、memory_type、project_id、
  session_id、timestamp、importance 和 source_type。
- [ ] 实现 session ingestion，将事实、经历、成功步骤和失败教训映射到文本、
  embedding 与现有 PurplePulse metadata 文件。
- [ ] 增加 embedding provider 抽象；演示实现可替换，不把具体模型写死在
  检索引擎中。
- [ ] 保存 `vector_id -> content/provenance` 映射，使 GPU Top-K ID 能恢复成
  可注入 Agent 上下文的记忆文本。
- [ ] 提供最小常驻查询接口，复用已加载的 IVF 索引和 GPU 工作区，避免每个
  Agent tool call 都重新初始化 GPU。
- [ ] 实现框架无关的 `memory_search` 与 `memory_record` 客户端协议。
- [ ] 实现 Pi adapter：`memory_search` 工具、session 结束记忆记录和带来源的
  context builder。
- [ ] 建立双 session CUDA 开发演示：后一 session 在空上下文下召回前一
  session 的成功方案或失败教训。

### 6.3 评测与验收

- [ ] 对比 No Memory、纯语义检索和 PurplePulse fused memory 三组方案。
- [ ] 记录记忆 Recall@K/MRR、错误重复率、任务成功率和注入 token 数。
- [ ] 记录检索 QPS、P50/P99、GPU kernel 时间、显存以及错误 project/session
  的过滤泄漏率。
- [ ] 验证 Agent adapter 关闭后，原有 CTest、百万规模 benchmark 和 CLI
  行为完全不变。
- [ ] 在 README 中先展示 GPU 引擎架构与性能，再将 Agent Memory 作为应用
  案例；Agent 内容不覆盖或替代 Exact/IVF/FAISS 主线。

### 6.4 本阶段明确不做

- MiniDeepSeek 或其他小模型训练。
- MoE、GRPO、DPO 或 learned retrieval policy。
- Base/Delta 在线索引、后台 compaction 和分布式多租户调度。
- 完整 RAG、复杂网页、多个 Agent 框架同时适配。

完成上述最小链路后再根据时间评估在线增量索引；任何扩展不得降低
PurplePulse 核心测试、性能复现和代码可解释性。

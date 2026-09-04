# PurplePulse GPU 向量检索引擎

这是 2026 夏季训练营 CUDA 方向“GPU 向量检索引擎”项目的独立实现。

项目从容易理解、结果正确的精确检索 baseline 逐步演进到 GPU Top-K、
IVF-Flat、多轮性能优化、实验性自适应 nprobe，以及面向 Agent 长期记忆的
GPU 元数据过滤和融合评分。

## 当前功能

- FP32/FP16 向量库与查询文件读写
- FP16 在 GPU 上保持 16 位存储，距离计算使用 FP32 累加
- L2、内积和余弦相似度
- CPU 精确检索参考实现
- CUDA 精确距离计算 baseline
- 四种 GPU Top-K：simple、block、two-stage，以及不生成完整距离矩阵的
  fused 路径
- 精确检索与 IVF 均支持 K = 1、10、50、100（也支持其他 K≤100）
- 批量查询
- GPU 数据库和工作区常驻显存，区分一次性初始化与稳定查询性能
- CPU/GPU 结果自动对比测试
- IVF-Flat CPU baseline：采样 k-means、连续倒排桶、索引保存/加载、固定
  `nprobe` 查询和 recall@K
- L2 使用普通 k-means；内积和余弦使用归一化中心的球面 k-means，避免
  相似度聚类因中心模长造成严重桶倾斜
- GPU IVF：中心、桶 offsets 和索引向量常驻显存，GPU 完成中心选择、
  每桶 block 并行扫描、局部 Top-K 与最终归并；支持 scalar、2/4/8-warp
  和按 K 自动选择并发度的 compact 扫描路径
- 实验性 `score_mass` 自适应 nprobe：依据每个 query 的中心分数分布选择
  probe 前缀，支持实际 probe 明细导出和 masked/grouped 调度 A/B
- Agent 记忆检索：时间戳、重要性、会话 ID、来源类型，支持 fused GPU
  过滤/评分与独立 GPU rerank A/B

IVF-Flat 的索引、CPU 正确性基线、全 GPU 查询路径、K=1/10/50/100 以及
100k/百万规模参数扫描、桶扫描优化、FAISS CPU/GPU 对照和首轮自适应
nprobe 实验和 GPU 元数据融合实验已经完成，请以
`docs/DEVELOPMENT_PLAN.md` 为准。

## 正式结果与报告

完整方法、实验口径、负结果和复现命令见
[`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md)；本轮正式环境、训练差异、
数据/索引校验和、完整命令和延迟定义见
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)。图表由原始 CSV 直接生成：

![Recall–QPS](results/figures/recall-qps.svg)

![Batch–QPS](results/figures/batch-qps.svg)

重新生成全部图表：

```bash
python3 scripts/plot_results.py \
  --results-dir results/formal_100k15 \
  --batch-results-dir results/million \
  --output-dir results/figures
```

## 编译

```bash
export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build
ctest --test-dir build --output-on-failure
```

`native` 会在配置时检测当前 GPU，因此切换 GPU 型号时不需要修改
README。如果要生成能在多种 GPU 上运行的二进制，可显式传入用分号分隔的
多个架构编号。切换设备后建议删除旧 `build/` 或使用新的构建目录，
避免复用 CMake 缓存中的旧架构。

## 生成测试数据

```bash
python3 scripts/generate_data.py \
  --database data/database.bin \
  --queries data/queries.bin \
  --num-vectors 10000 \
  --num-queries 20 \
  --dim 128 \
  --dtype fp32 \
  --metric l2 \
  --seed 2026
```

## 执行检索

```bash
./build/vector_search \
  --database data/database.bin \
  --queries data/queries.bin \
  --params configs/exact_l2.conf \
  --backend gpu \
  --output results/exact_l2.txt
```

结果文件中每一行依次为 `query_id vector_id score`。

### 融合 Exact 路径

设置 `distance_mode=warp`、`topk_mode=fused` 后，每个 warp 在计算距离时
直接维护局部 Top-K，每个 query/chunk 只写回 K 个候选，再由一个小型 kernel
完成最终归并。该路径不再物化 `batch_size × num_vectors` 的完整距离矩阵。
K=10/50/100 的参考配置分别为
`configs/exact_million_k{10,50,100}_fused_batch64.conf`。

百万条 FP32、128 维、1000 queries、Cosine、batch=64、预热 1 次并重复 5 次
的当前测试结果如下；旧 matrix 路径保留用于 A/B：

| K | matrix QPS | fused QPS | 加速 | matrix / fused 缓冲区 |
|---:|---:|---:|---:|---:|
| 10 | 1775.10 | 5563.72 | 3.13× | 734.34 / 488.44 MiB |
| 50 | 715.87 | 5123.32 | 7.16× | 732.49 / 488.94 MiB |
| 100 | 443.78 | 4700.80 | 10.59× | 732.53 / 489.56 MiB |

所有成对结果的 Top-K ID 集合一致，记录的最大分数误差为 0。原始数据见
`results/exact_fused/million_compare.csv`，可重新运行：

```bash
python3 scripts/benchmark_exact_fused.py \
  --search-binary ./build/vector_search \
  --database data/exact_million/cosine_db.bin \
  --queries data/exact_million/cosine_q.bin \
  --baseline-configs configs/exact_l2_batch64.conf,configs/exact_million_k50_batch64.conf,configs/exact_million_k100_batch64.conf \
  --fused-configs configs/exact_million_k10_fused_batch64.conf,configs/exact_million_k50_fused_batch64.conf,configs/exact_million_k100_fused_batch64.conf \
  --output-dir results/exact_fused/million_outputs \
  --csv results/exact_fused/million_compare.csv \
  --warmup 1 --repeat 5
```

## IVF-Flat baseline

先构建可持久化索引：

```bash
./build/ivf_build \
  --database data/database.bin \
  --output data/database.ivf \
  --nlist 1024 \
  --iterations 15 \
  --training-samples 100000
```

再运行 CPU IVF 正确性/质量基线；`--nprobe` 可覆盖配置文件，便于扫描：

```bash
./build/ivf_search \
  --index data/database.ivf \
  --queries data/queries.bin \
  --params configs/ivf_flat.conf \
  --nprobe 16 \
  --backend gpu \
  --warmup 1 \
  --repeat 5 \
  --output results/ivf_flat.txt

python3 scripts/evaluate_recall.py \
  --exact results/exact_l2.txt \
  --approximate results/ivf_flat.txt
```

输出同时包含 recall、最低单 query recall，以及按名次计算的平均/最大绝对
分数误差。

`configs/ivf_flat_100k_{balanced,high_recall,exact_check}.conf` 是
100k × 128、K=10 实验的三档参考配置。`nlist` 必须与索引构建参数一致；
迁移到百万级或不同数据分布时，应重新扫描 `nlist/nprobe`，不要直接把这组
参数当成通用最优值。`ivf_flat_100k_scalar_baseline.conf` 保留旧的
thread-per-vector 桶扫描，用于和默认 warp 协作版本做 A/B 对照。
`nprobe` 可取 `1..nlist`；使用 `nprobe=nlist` 可作为结果完整性校验，但
近似检索应根据目标 recall 和延迟选取更小的值。

百万规模 K=10 初测提供四档配置：

- `ivf_flat_million_fast.conf`：偏吞吐。
- `ivf_flat_million_balanced.conf`：约 0.87 recall 的平衡档。
- `ivf_flat_million_high_recall.conf`：约 0.97 recall。
- `ivf_flat_million_near_exact.conf`：约 0.99 recall。

大 K 使用独立模板实例和 warp 共享内存堆；`warp_compact` 在 K≤10 时使用
8 warps、K>10 时使用 2 warps，以减少每个 block 的共享内存并提高 probe
并发。推荐从 `ivf_flat_million_k50.conf` 或
`ivf_flat_million_k100.conf` 开始；实测默认 batch 分别为 32 和 64，并提供
batch=8/32/64/128 以及 warp2/4/8 的 A/B 配置。百万级 K10 的 nlist=256
三档配置默认 batch=64。切换数据、维度或 GPU 后仍应重新扫参。

这些 recall 只对应固定随机种子生成的测试数据。可用脚本在新数据或新 GPU
上重新生成 CSV，而不是沿用旧机器的结论：

```bash
python3 scripts/benchmark_ivf_sweep.py \
  --search-binary ./build/ivf_search \
  --index data/million/inner_fp32_nlist256_spherical_100k15.ivf \
  --queries data/million/inner_fp32_q.bin \
  --params configs/ivf_flat_million_batch64.conf \
  --exact results/million/inner_exact_gpu_k10.txt \
  --output-dir results/million/sweep_n256 \
  --csv results/million/sweep_n256.csv \
  --nlist 256 \
  --nprobes 64,128,160,192,224,256 \
  --training-samples 100000 \
  --training-iterations 15
```

## FAISS 公平对照

`scripts/benchmark_faiss.py` 直接读取相同的 PurplePulse 数据文件，支持
FAISS CPU/GPU、Flat/IVF-Flat、多个 K/nprobe，并把索引构建/加载、GPU
转移、稳定查询 QPS、run/batch P50/P99 及其样本数、recall、平均/最大绝对
分数误差和 FAISS GPU 内存分类写入 CSV。
运行它的 Python 环境需要 NumPy 和与当前 CUDA 环境兼容的 FAISS 包。

```bash
python3 scripts/benchmark_faiss.py \
  --database data/million/inner_fp32_db.bin \
  --queries data/million/inner_fp32_q.bin \
  --reference-results results/million/inner_exact_gpu_k100.txt \
  --index-dir data/million/faiss \
  --output-dir results/million/faiss_gpu \
  --csv results/million/faiss_gpu.csv \
  --backends gpu \
  --index-types flat,ivf_flat \
  --top-ks 10,50,100 \
  --nlist 256 \
  --nprobes 128,160,192,224 \
  --batch-size 64 \
  --train-samples 100000 \
  --iterations 15 \
  --seed 2026 \
  --warmup 1 \
  --repeat 5
```

脚本默认将 FAISS CPU 线程数固定为 1；可用 `--cpu-threads` 显式改变。
FAISS Flat 会先与 PurplePulse exact 结果交叉验证，候选 ID 不一致时终止。
长矩阵每完成一项都会立即更新 CSV，中断时不会丢失已经完成的行。

## 实验性自适应 nprobe

固定策略仍是默认。设置 `nprobe_policy=score_mass` 后，`nprobe` 表示允许的
最大值；策略将排序后的中心分数转换为相对概率质量，并在
`adaptive_nprobe_min..nprobe` 内按 `adaptive_nprobe_step` 选择每个 query
的实际 probe 数。`adaptive_temperature` 使用当前 query 的中心分数跨度归一化，
因此不依赖绝对分数尺度。

```bash
python3 scripts/benchmark_adaptive_nprobe.py \
  --search-binary ./build/ivf_search \
  --index data/million/inner_fp32_nlist256_spherical.ivf \
  --queries data/million/inner_fp32_q.bin \
  --params configs/ivf_flat_million_adaptive_k10.conf \
  --exact results/million/inner_exact_gpu_k10.txt \
  --output-dir results/million/adaptive_k10 \
  --csv results/million/adaptive_k10.csv \
  --max-nprobe 224 --min-nprobe 64 --step 16 \
  --temperature 0.2 --target-masses 0.80,0.90,0.93,0.97,0.99
```

百万级均匀合成数据上的首轮实验没有超过相同平均 probe 的固定策略：平均
151.632 probes 的代表点在 K10/K50/K100 上分别慢约 1.0%/2.6%/3.6%，
recall 也略低。因此配置和脚本用于研究与复现，尚不作为推荐生产默认值。
切换到更异质的真实查询分布后应重新校准并重新判断是否保留。

## Agent 记忆检索

元数据文件按原始向量 ID 保存 `timestamp`、`importance`、`session_id` 和
`source_type`。`fused` 在桶扫描读取向量前执行过滤，并在局部 Top-K 前融合
语义分数、重要性和指数时间衰减；`rerank` 先取语义 Top-(K×factor)，再由
独立 GPU kernel 过滤和重排，作为 A/B baseline。L2 仍保持分数越低越好，
内积和 Cosine 保持越高越好。

```bash
python3 scripts/generate_memory_metadata.py \
  --output data/memory.bin --num-vectors 100000 --seed 2026

./build/ivf_search \
  --index data/database.ivf --queries data/queries.bin \
  --params configs/ivf_flat_memory_fused.conf \
  --memory-metadata data/memory.bin --memory-mode fused \
  --filter-session-id any --filter-source-type 1 \
  --backend gpu --warmup 2 --repeat 5 \
  --output results/memory_fused.txt
```

使用 `scripts/benchmark_memory_modes.py` 可在完全相同的过滤、评分和批量参数
下运行 fused/rerank 并保存 CSV。`any` 表示不限制对应的离散元数据字段；
若过滤后不足 K 个候选，输出会诚实返回少于 K 条，而不会写入伪造 ID。

索引使用版本化二进制格式，保存 FP32 聚类中心、桶 offsets、原始向量 ID
以及按桶连续排列的 FP32/FP16 原始向量。GPU backend 会将这些索引数据
常驻显存；每批只上传 queries，中心选择、桶扫描和 Top-K 均在 GPU 完成。

GPU backend 会先创建常驻检索引擎，数据库只上传一次，然后让
`--warmup` 和 `--repeat` 的每轮查询复用同一份 GPU 数据和工作区。
程序分别输出：

- GPU 一次性初始化和数据库 H2D 时间。
- 估算冷启动端到端时间。
- 数据库常驻后的查询时间和 QPS。
- 引擎显式分配的 GPU 缓冲区总量。
- query H2D、距离 kernel、Top-K kernel 和结果 D2H 分段时间。

`configs/exact_l2_batch*.conf` 保留生成完整距离矩阵的历史 batch 扫描配置。
新实验推荐以 fused batch=64 配置为起点；融合路径会根据 batch size 调整每个
query 的 chunk 数，使总并发块数维持在合理范围，同时避免产生过多局部候选。
切换数据规模、维度或 GPU 后仍应重新扫描 batch size。

## 项目原则

1. 正确性优先，每个 GPU 版本都和 CPU 参考答案比较。
2. 先实现简单版本，再逐步优化；不直接堆叠难以解释的 CUDA 技巧。
3. 每次优化都保存测试环境、参数和优化前后的性能数据。
4. Agent 是应用场景，CUDA 检索引擎始终是项目核心。

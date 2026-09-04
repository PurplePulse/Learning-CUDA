# PurplePulse 正式实验复现记录（2026-09-04）

本文固定本轮正式结果的数据、训练、延迟和环境口径。公开结果文件不包含原始
数据与索引；二者通过生成命令和 SHA-256 校验和审计。

## 1. 环境

- GPU：NVIDIA GeForce RTX 4090 D，24,564 MiB，compute capability 8.9。
- Driver：570.124.06。
- OS：Ubuntu 24.04.1 LTS。
- CUDA Toolkit / NVCC：12.8 / 12.8.61。
- CMake / Ninja / GCC：3.31.4 / 1.11.1 / 13.3.0。
- FAISS 专用 venv：FAISS 1.14.1 GPU，NumPy 2.2.6，Python 3.12.3；CPU
  对照固定 1 线程。系统 Python 不含 FAISS，不用于 FAISS 正式实验。
- 构建类型：Release，`CMAKE_CUDA_ARCHITECTURES=native`。

GPU 机上的部署目录不是 Git checkout。正式实验前，核心源码和本地基准提交
`fea41f6` 的 SHA-256 逐文件一致；随后新增的评测字段不改变检索 kernel。
完整环境输出、测试与 profiler 摘要保存在 `../evidence/2026-09-04/`。

## 2. 数据与索引

```bash
python3 scripts/generate_data.py \
  --database data/million/inner_fp32_db.bin \
  --queries data/million/inner_fp32_q.bin \
  --num-vectors 1000000 \
  --num-queries 1000 \
  --dim 128 \
  --dtype fp32 \
  --metric inner_product \
  --seed 2026
```

| 文件 | SHA-256 |
|---|---|
| `inner_fp32_db.bin` | `d4dde1418ea7626e5226e594e592f9ae5ffbe4c412c326395d96388f95df9800` |
| `inner_fp32_q.bin` | `a58ebe43fce865b71a2a651791286fa0bd1567348e07d6c0601b69beaebceb45` |
| `inner_fp32_nlist256_spherical_100k15.ivf` | `f5bd87fcc72d37164fa931436e5bfd293dfb7f953d09cb28169a77c7cdadc415` |
| FAISS Flat index | `760e3a94f24c5becfa21e89a57f67395013d5ad5375c20eb3f5d5cfb4c3720a1` |
| FAISS IVF-Flat index（100k/15/seed 2026） | `a66a63bb28ed9e70ae83c042f57a8ca79de3ee01a6414377452a7610738431b1` |

PurplePulse 正式索引的构建命令为：

```bash
./build/ivf_build \
  --database data/million/inner_fp32_db.bin \
  --output data/million/inner_fp32_nlist256_spherical_100k15.ivf \
  --nlist 256 \
  --iterations 15 \
  --training-samples 100000
```

构建耗时 67,678 ms，桶大小 min/avg/max 为
3734/3906.25/4091，无空桶，索引大小 520,133,160 bytes。

两种实现都使用 100,000 个训练向量和 15 次迭代，但不强制共享中心：

- PurplePulse 从全库等间距、确定性抽取训练样本，初始中心也等间距选择；
  当前实现没有单独的训练随机种子。
- FAISS 用 NumPy `default_rng(2026)` 无放回抽样，并把聚类 seed 设为 2026。

所以这是相同训练预算和检索参数下的端到端系统比较，不是相同桶扫描比较。

## 3. 正式查询命令

Exact 的 K=10/50/100 均使用 batch=64、预热 5 次、正式重复 100 次：

```bash
for k in 10 50 100; do
  ./build/vector_search \
    --database data/million/inner_fp32_db.bin \
    --queries data/million/inner_fp32_q.bin \
    --params configs/exact_million_k${k}_fused_batch64.conf \
    --backend gpu \
    --output results/formal_100k15/exact_k${k}.txt \
    --warmup 5 --repeat 100
done
```

IVF 对 nprobe=128/160/192/224 扫描。K=10 的示例为：

```bash
python3 scripts/benchmark_ivf_sweep.py \
  --search-binary ./build/ivf_search \
  --index data/million/inner_fp32_nlist256_spherical_100k15.ivf \
  --queries data/million/inner_fp32_q.bin \
  --params configs/ivf_flat_million_batch64.conf \
  --exact results/formal_100k15/exact_k10.txt \
  --output-dir results/formal_100k15/ivf_k10 \
  --csv results/formal_100k15/ivf_k10.csv \
  --nlist 256 --nprobes 128,160,192,224 \
  --training-samples 100000 --training-iterations 15 \
  --warmup 5 --repeat 100
```

K=50/100 分别替换为对应的 batch64 配置和 Exact 文件。FAISS 使用同一数据、
K、batch、nlist、nprobe、训练预算、预热和重复次数；完整命令保存在
`../evidence/2026-09-04/commands.md`。

## 4. 延迟与质量定义

- `run P50/P99`：一次处理全部 1000 queries 的端到端稳定查询时间。
- `batch P50/P99`：每个 batch 的 GPU 时间，从 query H2D 开始，到 Top-K
  ID/score D2H 完成为止；包含 H2D、所有检索 kernel 和 D2H。
- 稳定查询不包含数据库/索引的一次性加载、一次性 H2D 和引擎初始化；这些
  时间另列，不能混入常驻 QPS。
- GPU 正式项每个 run 有 16 个 batch；100 次重复共产生 1600 个 batch 延迟
  样本。单线程 CPU Flat 重复 20 次，共 320 个 batch 延迟样本。
- `recall@K` 是返回 ID 集合与 Exact Top-K 的交集比例。
- `mean/max absolute score error` 按名次比较近似 Top-K 与 Exact Top-K 的
  分数绝对差；它衡量漏召回造成的分数质量损失，不等同于同 ID 数值误差。
  Exact kernel A/B 的分数误差则按相同候选 ID 比较。

## 5. 自动验证

```bash
ctest --test-dir build --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 99 ./build/unit_tests
```

本轮 CTest 7/7 通过，`compute-sanitizer` 报告 0 errors。Nsight Systems 的
Exact K10 代表性采样显示 fused 距离+局部 Top-K kernel 占 GPU kernel 时间
约 99.9%；文本摘要已入库，原始 `.nsys-rep` 因体积和机器相关性不提交。

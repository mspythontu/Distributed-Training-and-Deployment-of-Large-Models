# Megatron 配置目录说明

本目录存放 Megatron-LM 三条链路（预训练 / SFT / GRPO）的参数配置，
由 `scripts/megatron/*.sh` 以 `source` 方式加载。

| 文件 | 用途 | 对应脚本 |
|---|---|---|
| `pretrain_qwen2.5-3b.env` | 预训练 / 继续预训练参数（含 Qwen2.5-3B 架构） | `scripts/megatron/pretrain_qwen.sh` |
| `sft_qwen2.5-3b.env` | SFT 监督微调参数（低学习率、短调度） | `scripts/megatron/sft_qwen.sh` |

自定义训练时用 `--conf` 指定自己的 env，或直接复制一份修改。

---

## 并行度组合速查（`world_size == TP x PP x CP x DP`）

单节点内优先用 TP（NVLink 带宽高），跨机优先扩 DP / PP（走网络，通信量小）。

| GPU 数 | 推荐 TP | 推荐 PP | CP | DP | 适用场景 |
|---|---|---|---|---|---|
| 1 | 1 | 1 | 1 | 1 | 调试（需关闭 TP 相关优化） |
| 2 | 1 | 1 | 1 | 2 | 小模型调试，纯数据并行 |
| 4 | 2 | 1 | 1 | 2 | 单节点 4 卡，TP 覆盖 NVLink 组 |
| 8 | 2 | 1 | 1 | 4 | **单节点 8 卡最常用**（TP 不超 NVLink 域） |
| 8 | 2 | 2 | 1 | 2 | 显存吃紧时加大 PP 换显存 |
| 16 | 2 | 2 | 1 | 4 | 两节点，PP 跨机 + DP 并行 |
| 32 | 2 | 4 | 1 | 4 | 大模型 / 长序列 |

### 取值约束（务必满足，否则启动即报错）

1. **`TP x PP x CP` 必须整除 `world_size`**，余下的商即 `DP`
2. **`PP > 1` 时**，`NUM_LAYERS` 必须能被 `PP` 整除（36 层可选 PP = 1/2/3/4/6/9/12/18/36）
3. **`TP` 建议 <= 单节点 GPU 数**：跨节点的 TP 会把 attention 通信打到网络上，性能骤降
4. **`NUM_ATTN_HEADS` 必须能被 `TP` 整除**：Qwen2.5-3B 为 16 头，TP 可选 1/2/4/8/16
5. **`NUM_QUERY_GROUPS`(GQA)=2 也必须能被 `TP` 整除**：因此 **TP 最大只能取 2**（除非关闭 GQA）
6. **必须满足 `GBS % (MBS x DP) == 0`**，否则梯度累积步数非法

> 注意 **Qwen2.5-3B 特别注意**：GQA 分组数为 2，意味着 TP 最大值为 2。
> 想用更大 TP 需改用无 GQA 的模型，或接受 `tp=2` 后依靠 PP/DP 扩展并行度。

### 调参建议

- **先求能跑起来**：`TP=1, PP=1`，只调 `GBS`/`MBS` 直到显存不 OOM
- **再提吞吐**：逐步加 TP（<= NVLink 域），同步开启 `SEQ_PARALLEL=1`
- **仍放不下**：加 PP（拖慢但省显存）或启用 offload
- **长序列**：加大 `CP`，或配合 `--seq-length` 调整
- **显存 vs 吞吐**：`TP` 提单步速度，`PP` 降单卡显存，`DP` 提样本吞吐

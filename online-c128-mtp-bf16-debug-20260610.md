# online-c128-mtp-bf16-debug-20260610

记录时间：2026-06-10

## 当前代码状态

- 本地分支：`deepseek_v4_dev_zjy_online`
- 当前 HEAD：`1f5d72739 deepseek v4: support online c128 bf16 state`
- 当前业务未提交文件：
  - `python/sglang/jit_kernel/csrc/deepseek_v4/online_c128_mtp.cuh`
  - `python/sglang/jit_kernel/dsv4/online_c128_mtp.py`
  - `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
  - `python/sglang/srt/model_executor/pool_configurator.py`
- 当前未跟踪文件/目录：
  - `.agents/`
  - `codex_tmp/`
  - `deepseek_v4_pr_description.md`
  - `deepseek_v4_pr_description_zh.md`

## 本轮目标

继续定位 `SGLANG_OPT_USE_ONLINE_COMPRESS=1` + `SGLANG_EXPERIMENTAL_ONLINE_C128_MTP=1`
+ `SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16` 在并行3容器上的服务问题，并判断问题来自：

- 上一个 commit；
- 当前未合入改动；
- 远端镜像/基线代码；
- 服务启动参数组合。

## 远端环境

- 机器：并行3
- IP：`115.191.2.23`
- SSH：
  ```bash
  ssh -i /Users/bytedance/Downloads/key/perfTest.pem root@115.191.2.23 -J jumpecs-lf.byted.org
  ```
- 容器：`deepseek-v4-pro-dev-zjy-online`
- 镜像：
  ```text
  iaas-gpu-cn-beijing.cr.volces.com/serving/sglang:v0.5.12.post1.dev.dsv4.online.zjy
  ```
- 当前已确认：并行3上没有残留 `sglang serve` / `sglang::` 进程，仅保留容器本身。

## 关键结论

当前 hang 不是当前未提交的 BF16/online C128 MTP 改动直接引起的。

复现到的核心触发条件：

```text
--dp-size 8
--enable-dp-attention
--speculative-algo EAGLE
默认 overlap schedule / Spec v2
```

即使不设置：

```bash
SGLANG_OPT_USE_ONLINE_COMPRESS=1
SGLANG_EXPERIMENTAL_ONLINE_C128_MTP=1
SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16
```

plain `DP attention + EAGLE` 也会在同镜像基线容器中复现 request timeout。

卡住位置：

```text
scheduler_components/batch_result_processor.py
process_batch_result_decode / process_batch_result_idle
result.copy_done.synchronize()
```

说明更像是 `DP attention + EAGLE + overlap/spec v2` 的同步路径问题，而不是 online C128 MTP BF16 state kernel 自身问题。

## 已验证 workaround

在服务命令中添加：

```bash
--disable-overlap-schedule
```

效果：

- plain `DP attention + EAGLE` 正常返回；
- `online C128 MTP + BF16 state + DP attention + EAGLE` 正常返回；
- 日志显示切到 Spec v1：
  ```text
  Spec v1 is used for eagle/eagle3/standalone speculative decoding because overlap schedule is disabled.
  ```

## 重要测试结果

### 干净 git worktree 测试

远端 clean worktree `7a681d9b5d` / `f055dd6ce0` 使用 `PYTHONPATH=$SRC_ROOT/python` 跑 plain
`DP attention + EAGLE` 可以正常返回。说明不是简单的远端 git HEAD commit 本身导致。

### 同镜像基线容器测试

使用同镜像新建临时容器，不叠加当前 BF16/online 改动，plain `DP attention + EAGLE` 仍超时。

日志目录：

```text
/data00/eval_results/codex_online_baseline_20260610_154802_plain
```

结果：

```text
READY=1
CURL_EXIT=124
LATENCY=120.003
REQUEST_OK=0
```

py-spy 栈显示卡在：

```text
cuEventSynchronize
torch/cuda/streams.py:synchronize
process_batch_result_decode / process_batch_result_idle
```

### plain + disable overlap

日志目录：

```text
/data00/eval_results/plain_eagle_dp_disable_overlap_20260610_075621
```

结果：

```text
READY=1
CURL_EXIT=0
LATENCY=2.673
REQUEST_OK=1
```

### online C128 MTP + BF16 + disable overlap

日志目录：

```text
/data00/eval_results/online_mtp_bf16_disable_overlap_20260610_080113
```

结果：

```text
READY=1
CURL_EXIT=0
LATENCY=0.891
REQUEST_OK=1
```

关键配置确认：

```text
DSV4 compressed attention: experimental online c128 + MTP enabled
c128_state_pool_size=5510
```

## 对当前未提交改动的判断

当前未提交改动中：

- `online_c128_mtp.cuh` / `online_c128_mtp.py` 只在 `SGLANG_OPT_USE_ONLINE_COMPRESS=1`
  且 `SGLANG_EXPERIMENTAL_ONLINE_C128_MTP=1` 时进入；
- `model_runner_kv_cache_mixin.py` / `pool_configurator.py` 当前只去掉
  `online C128 MTP + bf16 state` 的 guard；
- plain 失败场景没有启用 online C128 MTP，也没有启用 bf16 state。

因此当前 request hang 不应归因于这 4 个未提交业务改动。

## 后续建议

1. 当前功能验证可以先使用 `--disable-overlap-schedule` 作为服务 workaround。
2. 若要修复根因，应继续查镜像基线里的 `DP attention + EAGLE + overlap/spec v2` 路径。
3. 重点代码范围：
   - `python/sglang/srt/managers/scheduler.py`
   - `python/sglang/srt/managers/scheduler_components/batch_result_processor.py`
   - `python/sglang/srt/managers/overlap_utils.py`
   - `python/sglang/srt/speculative/eagle_worker_v2.py`
   - `python/sglang/srt/layers/attention/deepseek_v4_backend.py`
4. 直接 overlay clean worktree 的 `speculative/`、`managers/`、`deepseek_v4_backend.py`
   会出现 API 不兼容，说明镜像里这些改动是成组引入的，不能用单目录替换做可靠二分。

## 本轮新增的本地诊断脚本

均在 `codex_tmp/` 下：

- `run_online_mtp_matrix.sh`
- `run_service_minimal_matrix.sh`
- `run_eagle_variant_matrix.sh`
- `run_eagle_dp_isolation_matrix.sh`
- `start_online_mtp_bf16_tp8_service.sh`
- `run_specv2_dp_eagle_check.sh`
- `run_commit_eagle_dp_check.sh`
- `run_commit_eagle_dp_no_precompile_check.sh`
- `run_current_eagle_dp_no_precompile_matrix.sh`
- `run_plain_eagle_dp_no_precompile.sh`
- `run_overlay_plain_eagle_dp_check.sh`

这些脚本都是临时诊断脚本，不是业务代码。

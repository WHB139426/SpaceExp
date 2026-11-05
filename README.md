# SpaceExp

从现有点云场景启动一个可探索的第一视角环境。普通 Python 运行 agent，后台常驻一个 Blender 进程，按工具请求渲染。

## 运行

```bash
pip install -r requirements.txt
python main.py
```

在 `main.py` 顶部选择 `MODEL`，并通过 `MODEL_GENERATION_CONFIGS[MODEL]` 设置该模型的生成与 thinking 参数。`MODEL_SERVER_CONFIG` 配置服务地址和超时，`AGENT_CONFIG` 配置工具循环上限和历史图片预算，`SCENE_CONFIG` 集中配置场景目录、Blender 路径、渲染参数、步长和眼高。模型名通过 `agent/protocols/__init__.py` 中的字典选择协议。

场景目录需要 `reconstruction.ply` 和 `cameras.json`；相机 JSON 使用已有的内外参及 `W/H`，无需原视频或 `frames/`，不加载重建模型。初始化沿用 SpatialHarness 的地面估计、首帧水平朝向和眼高逻辑。

## 模型工具

只有 `camera_observe()` 和 `camera_action(key, magnitude=None)`，每次成功调用都附上一张当前第一视角 PNG。

- `move_forward`：沿当前视角的水平朝向前进，保持高度。
- `turn_left` / `turn_right`：向相机自己的左侧 / 右侧转动，保持位置和俯仰。
- `magnitude`：前进距离使用 PLY 坐标单位，转角使用度数；省略时使用顶部默认值。米制校准后的点云距离近似为米。

环境没有碰撞检测。观察与动作都由模型按需调用，不提供读图、地图、规划、状态、reset、undo 或其他动作工具。

## 会话输出

每次启动在 `outputs/chat_<时间戳>/` 保存 `conversation.json`、观察 PNG、对应相机元数据 JSON 和 Blender 的 `render.log`。相机元数据用于本地检查，不作为数值状态附加给模型。

`/new` 创建新的会话目录，清空聊天并恢复初始相机位姿，继续使用同一个已加载场景；旧目录保持原样。`/exit` 退出并关闭 Blender。

普通 Python 依赖只有 OpenAI SDK；渲染端使用 Blender 自带的 `bpy` 和 NumPy。未使用 PyTorch、Open3D、VGGT 或 DA3。

## vLLM 启动命令

## Qwen3.8-Flash-Next-FP8 (vLLM 0.31.0)

Uses the native 262,144-token context window without RoPE scaling overrides.

```bash
CUDA_VISIBLE_DEVICES=2,3 \
VLLM_ENGINE_READY_TIMEOUT_S=3600 \
VLLM_PLE_CPU_OFFLOAD=1 \
vllm serve /data4/haibo/weights/Qwen3.8-Flash-Next-FP8 \
  --served-model-name qwen3.8-flash-next \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --no-enable-flashinfer-autotune \
  --allowed-local-media-path /data4/haibo \
  --limit-mm-per-prompt '{"image":100,"video":100}'
```

## Qwen3.8-27B-FP8 (vLLM 0.31.0)

Requires `transformers>=5.8.0`. Uses the native 262,144-token context window
without RoPE scaling overrides.

```bash
CUDA_VISIBLE_DEVICES=2 \
VLLM_ENGINE_READY_TIMEOUT_S=3600 \
vllm serve /data4/haibo/weights/Qwen3.8-27B-FP8 \
  --served-model-name qwen3.8-27b \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --no-enable-flashinfer-autotune \
  --allowed-local-media-path /data4/haibo \
  --limit-mm-per-prompt '{"image":100,"video":100}'
```

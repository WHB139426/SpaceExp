# SpaceExp

从现有点云场景启动一个可探索的第一视角环境。普通 Python 运行 agent，后台常驻一个 Blender 进程，按工具请求渲染。

## 运行

在 `main_agent.py` 顶部选择 `MODEL`，并通过 `MODEL_GENERATION_CONFIGS[MODEL]` 设置该模型的生成与 thinking 参数。`MODEL_SERVER_CONFIG` 配置服务地址和超时，`AGENT_CONFIG` 配置工具循环上限和历史图片预算，`SCENE_CONFIG` 集中配置场景目录、Blender 路径、渲染参数和步长。模型名通过 `agent/protocols/__init__.py` 中的字典选择协议。

手动探索使用 `python main_env.py`，无需模型服务。场景和渲染参数在该文件顶部的 `SCENE_CONFIG` 中独立配置。启动后自动保存初始观察，随后每行输入一个动作，例如 `move_forward 0.3`、`turn_left 30`、`turn_up 15`；省略幅度则使用配置的默认步长。`observe` 再次观察，`help` 显示命令，`/new` 在新目录重置环境，`/exit` 退出。输出保存在 `outputs/env_<时间戳>/`；终端显示当前第一视角图片和最新俯视轨迹图的路径，打开保存的 PNG 即可查看。

场景目录需要 `reconstruction.ply` 和 `cameras.json`；无需原视频或 `frames/`，不加载重建模型。初始位置、完整姿态（包括俯仰和侧倾）及内参使用第一帧的记录；仅将 OpenCV 相机坐标转换为 Blender 相机坐标，不调平、不修改眼高。渲染继续采用 `resolution` 指定的方形画面，保留原水平视场并扩展竖直视场。

## 模型工具

只有 `camera_observe()` 和 `camera_action(key, magnitude=None)`，每次成功调用都附上一张当前第一视角 PNG。

- `move_forward`：沿独立保存的水平前进方向移动，保持相对场景向上轴的高度及镜头姿态。初始前进方向来自第一帧视线在地面上的投影。
- `turn_left` / `turn_right`：绕场景向上轴旋转，同时更新水平前进方向和镜头朝向，保持位置和俯仰。
- `turn_up` / `turn_down`：只调整镜头俯仰，保持位置和水平前进方向；俯仰限制在 -90° 至 90°。
- `magnitude`：前进距离使用 PLY 坐标单位，转角使用度数；省略时使用顶部默认值。米制校准后的点云距离近似为米。

低头和抬头不会改变前进时的高度；镜头垂直朝上或朝下时仍保留已有水平前进方向。若第一帧本身垂直朝上或朝下，则根据其相机右轴确定初始水平前进方向。初始镜头位姿不自动调平。环境默认启用场景外边界空气墙，不检测家具和内部障碍物。观察与动作都由模型按需调用，不提供读图、地图、规划、状态、reset 或 undo 工具。

## 会话输出

Agent 每次启动在 `outputs/chat_<时间戳>/` 保存 `conversation.json`、观察 PNG、对应相机元数据 JSON 和 Blender 的 `render.log`。动作图片及配套 JSON、轨迹图的文件名包含动作和幅度，例如 `0002_move_forward_0.3.png`、`0002_move_forward_0.3.json`、`0002_move_forward_0.3_top_down.png`；被边界阻止的动作追加 `_blocked`，例如 `0003_move_forward_100_blocked.png`。普通观察仍为 `0001_observe.png`。另外覆盖更新一张 `top_down.png` 方便查看最新状态。两个入口使用同一套渲染与记录逻辑。

`trajectory.json` 累计记录每次观察的位置、镜头视线 `forward`、水平前进方向 `heading`、相机上方向、动作和图像路径。`magnitude` 保存补全默认值后的动作幅度，也是文件名中的幅度；`applied_magnitude` 保存实际执行幅度，俯仰被 ±90° 上限截断时可能更小。被边界阻止时记录 `blocked=true`、英文 `reason` 和 `applied_magnitude=0`，位置和姿态保持原样，并保存一张当前观察；这些动作状态也通过工具结果返回给模型。普通观察的幅度为 `null`、`blocked=false`、`reason=null`。前进单位为米，转动单位为度。原地转头、受阻动作和重复观察都会保留记录。图中青色箭头表示下一次前进方向，颜色条使用观察序号。模型始终只接收第一视角图片，不接收本地位姿和俯视图。

`SCENE_CONFIG` 的 `trajectory_resolution`、`trajectory_samples` 和 `trajectory_max_height` 分别控制俯视图背景的分辨率、采样数和地面以上的可见高度，用于裁掉天花板；`trajectory_dpi` 和 `trajectory_arrow_length` 控制图表输出 DPI 和朝向箭头长度。`up` 统一定义水平移动、左右转动和本地地图的向上方向；默认从场景估计，也可以手动覆盖，不改变初始相机位姿。每张相机元数据 JSON 中的 `top_down_frame` 保存投影坐标轴、原点、中心和范围。隐藏文件 `.top_down_background.png` 缓存背景，地图范围未变时不会重复渲染背景。

自动估计使用全部相机上方向的中位数作为竖直参考，并优先在点云高度较低的 20% 采样点中寻找地面平面；候选平面及最终拟合结果与参考方向的夹角均不超过 30°，同时检查相机和场景点是否位于其上方。若没有受足够点支持的平面，则保留多帧估计的竖直方向、近似地面高度，并在图中标注 `approximate floor`；相机方向本身缺乏稳定共识时直接报错，需手动配置 `up`。

`boundary_enforced=True` 默认启用移动限制。每次前进会逐格检查从当前位置到终点的完整线段，包含跨格拐角和沿网格线移动的情况；任一部分离开内缩区域就拒绝整步，不会跨越外部凹口后跳回场景内，也不会自动缩短动作或移动起点。转向和观察始终可用。若起点本身在内缩范围外，前进也会受阻，需检查边界参数或初始视点。

`boundary_preview=True` 控制本地俯视图是否显示边界，关闭显示不会关闭移动限制。黄色线表示场景外轮廓，绿色线表示按 `camera_radius` 向内收缩后的允许移动范围。边界使用完整点云中距地面 -0.1 米至 `trajectory_max_height` 的点进行投影，填满所有封闭的内部空洞，保留与外部连通的凹口和 L 形结构，并选取与初始位置连通的区域；若起点不在轮廓内，则选取最近区域并标注起点异常。家具和内部障碍物不会因此变成碰撞体。

两个入口顶部的 `boundary_cell_size=0.05` 控制网格精度，`boundary_close_radius=0.15` 控制小缺口修补半径，`camera_radius=0.15` 控制内缩距离。设 `boundary_enforced=False` 可恢复仅预览。每个会话的 `boundary.npz` 保存外轮廓掩码、内缩掩码、坐标基、参数、起点标记和 `boundary_enforced` 状态，并用 `outer_only=True` 标明只使用外边界；加载场景时计算一次，`/new` 复用并保存到新目录。

`/new` 创建新的会话目录，清空聊天并恢复初始相机位姿，继续使用同一个已加载场景；旧目录保持原样。`/exit` 退出并关闭 Blender。

普通 Python 运行 agent 和 NumPy / Matplotlib 轨迹绘图；渲染端使用 Blender 自带的 `bpy` 和 NumPy。探索过程不调用 PyTorch、Open3D、VGGT 或 DA3。

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

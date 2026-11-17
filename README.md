# interbeat —— 跨应用互动节拍

多个独立音序器应用经 UDP 共享速度、节拍相位和启停状态，每个应用用本地
时钟映射到共享节拍网格并渲染真实音频。节拍映射经过去抖动、限速的 PI 回路，
网络抖动不会抖动到可听节拍。

## 图形入口（互动节拍界面）

**互动节拍的图形入口是生产面板**，多端制作面板在同一页面里完成：

```bash
python3 scripts/run_panel.py            # 默认 http://127.0.0.1:8410/
python3 scripts/run_panel.py --port 8410
```

浏览器打开后即可：

- 播放 / 停止（在共享网格的量化边界生效）
- 拖动速度滑杆改变 BPM；在“主时钟”下拉框切换主时钟
- 在每端的步进编辑器里改节拍、改乐句长度
- 注入真实 UDP 丢包 / 抖动 / 重排，宕掉或复活主时钟
- 启用“监听声音”听到内核实际渲染的混音
- 在“运行诊断”卡片查看运行状态，以及**跨端节奏**与**监听声音**两类异常标志、
  异常所在阶段和告警列表
- “导出证据 JSON”得到可脱离本进程、离线重放的证据文档

面板页面由内核状态经 SSE 驱动，画面上的播放头、命中闪烁、偏差表都来自内核，
不是本地动画。

## 命令行（仅辅助图形入口）

命令行只做互动节拍面板的辅助工作：构造受控输入、离线诊断证据文档。
它不替代图形入口，也不读取实时会话。

```bash
# 构造缓慢漂移输入（跟随端时钟 -200 ppm）并诊断
python3 scripts/beattrace.py construct --kind drift --drift-ppm -200 --out out/drift.json
python3 scripts/beattrace.py diagnose out/drift.json

# 构造节拍相位突跳输入并诊断
python3 scripts/beattrace.py construct --kind jump --beats 0.5 --out out/jump.json
python3 scripts/beattrace.py diagnose out/jump.json

# 构造渐变速度输入（单向分段变速）并诊断
python3 scripts/beattrace.py construct --kind ramp --tempo 120 --tempo-end 168 --out out/ramp.json
python3 scripts/beattrace.py diagnose out/ramp.json

# 离线重放面板导出的证据（无需原进程）
python3 scripts/beattrace.py diagnose out/panel-evidence.json --json

# 打印当前阈值与“原因 → 阶段”映射
python3 scripts/beattrace.py maps
```

`construct` 与 `diagnose` 都只读写独立证据文档。相同 `--seed` 的复跑产生
逐字节一致的证据与诊断，互动节拍的复跑证据因此可对齐比较。

## 分层执行证据

每次播放决策的证据链，逐阶段记录，记录过程不改变任何播放决策：

| 阶段 | 记录内容 |
| --- | --- |
| clock | 节点时钟刻度（节点时间、可用时的参考时间） |
| filter | PI 估计：相位误差、调整量、锁定状态、链路延迟 |
| map | 节拍映射硬跳变（首锁 / 采纳校验 / 重锁 / 显式相位跳变） |
| tempo / master / play | 速度、主时钟、启停变化 |
| hit | 实际触发时刻：应触发时刻、缓冲区区间、放置偏移、映射误差 |
| output | 实测音频起音 vs 期望时刻（输出延迟） |

单个触发时刻的偏差可加性分解为：

```
输出偏差 = 映射误差 + 调度迟到 + 音频输出延迟
```

- **映射误差**：本地节拍映射相对主时钟网格的相位差；
- **调度迟到**：缓冲区已晚于应触发时刻，放置被夹到缓冲区首帧；
- **音频输出延迟**：实测起音相对放置时刻的延迟。

故障原因与所在阶段在 `interbeat/diag_maps/thresholds.json` 显式给出。
相位漂移会区分**渐变速度/时钟漂移**（相位误差单向累积，或单向分段变速，
定位 `clock/filter`）与**节拍相位突跳**（单次阶跃，定位 `map`）。

### 缺失与开销

- 没测到的量记为“未测量”，并给出 `unmeasured` 提示；**不会用 0 或默认值顶替**。
- 映射文件缺失、为空或不完整是直接报错，**不回退到内置默认值**。
- 每个阶段是固定容量的滚动缓冲：容量、采样、淘汰计数都写进证据快照。
  滚出窗口的部分标注为不可重放，属于信息级提示，不伪造完整性。
- 面板的诊断按 0.5 s 的较慢节奏运行，快照有上限，不进入每个音频 tick 的路径。

## 测试

```bash
PYTHONPATH=.pylocal python3 -m pytest tests -q
```

诊断相关用例在 `tests/test_diagnostics.py`：漂移与突跳的原因区分、三类误差
分离测量、离线重放一致性、缺失量不记零、空/坏映射直接报错、记录不改变播放
决策、有界淘汰可见。

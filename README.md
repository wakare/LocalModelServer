# 独立本机模型服务

阅读助手和其他项目共用 `http://127.0.0.1:8080/v1`。独立服务负责模型身份、排队、
按需加载、故障恢复与空闲卸载。它不导入 Qt、Sudachi、Anki 或阅读助手的数据库。
推理后端使用独立目录中的 llama.cpp CUDA 运行时与 GGUF，或 Strata 引擎（`backend: "strata"`）。

## 源码仓库

本目录既是安装目录，也是服务的源码仓库（`git@github.com:wakare/LocalModelServer.git`）。
源码在 `src/local_model_service`，测试在 `tests`，打包配置在 `packaging`，`config/service.example.json`
是当前部署配置的示例。模型、运行时、打包后的 `app`、凭证、日志和实际使用的 `service.json` 都由 `.gitignore`
排除。阅读助手（LanguageLearnerForPC）以可编辑方式安装本仓库：`pip install -e D:\Work\LocalModelService`。

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,build]"
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check src tests
.venv\Scripts\python -m PyInstaller packaging\local_model_service.spec --noconfirm --distpath <临时目录>
```

部署新版本：确认服务空闲后执行 `app\LocalModelService.exe stop`，把旧 `app` 改名保留，复制新打包的
`LocalModelService` 目录为 `app`，再执行 `start`。回滚就是换回旧目录。

### Strata 后端

`service.json` 的 `model.backend` 设为 `"strata"`，`model.strata_config` 指向 Strata 安装器写出的
`strata-<model>.json`（当前为 `D:\Work\Strata\strata-iq3_s.json`，权重在 `D:\Work\Strata-data`）。服务用 Strata 自带的
`.venv` 启动 `serve/server.py`，凭证经 `STRATA_API_KEY` 环境变量传入，等 `/health` 报告 `loaded` 后才放行请求。
模型 id 取自 Strata 配置的 `model_name`（`qwen3.8-flash-next-iq3_s`）。Strata 配置里的 `reasoning_effort` 设为 `none`，
与原 Qwen 默认不思考一致；请求可自行设置 `reasoning_effort`。Strata 冷启动约 45 秒，常驻时占约 52 GB 内存与约 23.7 GB 显存，
因此空闲卸载为 900 秒、单次推理超时 600 秒。dashboard 的「选择模型」可在 Strata 与 Qwen 预设之间切换。

## 本机独立安装目录

当前完整安装位于 `D:\Work\LocalModelService`：`app` 保存服务程序，`models` 保存模型权重，
`runtimes` 保存 llama.cpp 运行时，根目录保存 `service.json`、私有控制凭证和日志。
阅读助手仍保留自己的设置和数据库，通过原来的本机接口调用服务；Pi 的接口地址也不变。
原模型目录中的 NLLB 也随权重迁移，阅读助手的翻译设置使用新的绝对路径。
本机当前默认模型为 Strata 上的 Qwen3.8-Flash-Next IQ3_S（128k 上下文，2026-10-11 起；之前为
Qwen3.8-27B Q4_K_M + DFlash2，16384 上下文，仍可在 dashboard 切回）。
迁移校验记录位于安装目录的 `migration` 子目录。
旧模型与运行时已分别改名为原数据目录下的 `models.migrated-20261011`、
`runtimes.migrated-20261011`，旧程序目录为 `%USERPROFILE%\.local-model-service.migrated-20261011`。
这些是可恢复备份，服务不再使用它们。执行工具的自动审批拒绝了永久删除，因此 C 盘暂未释放这部分空间。

## 日常操作与 dashboard

安装后的入口位于 `D:\Work\LocalModelService\app\LocalModelService.exe`。
双击该程序，或双击桌面的「本机模型服务」快捷方式，会先确保服务开启，再用默认浏览器
打开带操作权限的 dashboard。关闭浏览器不会停止服务，登录自启也不会自动打开浏览器。

控制台地址：`http://127.0.0.1:8080/dashboard`。直接访问可以查看状态；要使用操作按钮，
请通过上面的程序或桌面入口打开。控制台展示实际模型身份、加载状态、活动和等待请求、
空闲卸载倒计时、网关/模型进程、恢复退避及错误原因。队列曲线记录当前页面打开后的样本。
可见时每 3 秒刷新，页面隐藏或关闭时停止查询，状态查询不加载模型，也不延长空闲计时。

- **开启**：双击桌面入口。模型仍按需加载，开启服务不会占用模型显存。
- **关闭服务**：dashboard 的「关闭服务」同时关闭网关与模型后端；网页随后显示离线。
  再次双击桌面入口即可重新开启。手动关闭不会取消下一次用户登录自启。
- **重启服务**：dashboard 的「重启服务」先停止旧实例，再启动新实例；页面会自动恢复。
- **释放模型资源**：只卸载当前模型，保留本机接口。下一次计算请求会重新加载模型。
- **选择模型**：从「选择模型」下拉框选择，然后点「使用此模型」。包含预设和资源目录
  `models` 下的其他主模型 GGUF；草稿文件不会作为主模型单独列出。缺少文件的预设不可选，
  草稿加速不可用且允许兼容回退时会显示提示。
  切换保存到 `service.json`，释放旧后端并重新生成实际模型身份，网关继续运行。
  新模型保持待命，首次计算请求才加载；查看和选择模型本身不占模型显存。
- 网页操作在存在活动或排队请求时拒绝关闭、重启、卸载和切换，等待请求完成后再操作。

PowerShell 中的独立安装操作，无需源码仓库或虚拟环境：

```powershell
$serverExe = 'D:\Work\LocalModelService\app\LocalModelService.exe'
function Invoke-ModelService([string]$action) {
    $operation = Start-Process -FilePath $serverExe -ArgumentList $action `
        -WindowStyle Hidden -PassThru
    $operation.WaitForExit()
    if ($operation.ExitCode -ne 0) { throw "操作失败，请检查 launcher.log" }
}
Invoke-ModelService start
Invoke-ModelService stop
Invoke-ModelService restart
Invoke-ModelService dashboard
Invoke-RestMethod http://127.0.0.1:8080/status |
    Select-Object state,active,queued,service_pid,pid,effective_model,fallback,reason
```

这些是分别执行的操作示例。CLI 的 `stop` / `restart` 会取消尚未完成的请求，适用于明确
需要停止服务的场景；CLI 会等待旧端口完成关闭，再允许启动新实例，重复 `stop` 安全返回。
Windows 打包程序不显示控制台输出，失败时查看 `launcher.log` / `service.log`；
上面的 `WaitForExit()` 只等待操作程序退出。不要使用 `Start-Process -Wait` 启动这些管理命令，
因为 PowerShell 会连同新创建的常驻子进程一起等待，命令将持续停留。

开发环境管理当前独立安装时，显式指定同一份配置：

```powershell
.venv\Scripts\python -m local_model_service --config D:\Work\LocalModelService\service.json start
.venv\Scripts\python -m local_model_service --config D:\Work\LocalModelService\service.json stop
.venv\Scripts\python -m local_model_service --config D:\Work\LocalModelService\service.json restart
.venv\Scripts\python -m local_model_service --config D:\Work\LocalModelService\service.json dashboard
```

dashboard 操作使用配置目录中私有的 `dashboard.key` 凭证。启动入口把凭证放在 URL fragment
中，页面取出后立即清除地址栏片段并仅保留于当前标签页会话；凭证不会通过 HTTP URL、日志
或公开页面发送。跨站请求、过期实例操作及嵌入 iframe 均被拒绝。使用 HTTPS 外部网页访问
本机控制接口不受支持；第三方项目的计算请求仍走原来的 OpenAI 兼容接口。

## 初始化与运行

开发环境：

```powershell
.venv\Scripts\python -m local_model_service init --connect-assistant
.venv\Scripts\python -m local_model_service start
.venv\Scripts\python -m local_model_service status
```

默认从 `%LOCALAPPDATA%\KindleAnkiBridge\settings.json` 导入一次模型运行参数，
将程序与 GGUF 路径解析为绝对路径，写入独立配置：

`%USERPROFILE%\.local-model-service\service.json`

这是未指定配置路径的开发环境初次初始化位置。当前本机部署使用上面的独立目录。
打包程序优先查找 EXE 同目录或其父目录的 `service.json`，因此双击迁移后的程序即可操作
新安装。`--config` 可显式选择配置；`LOCAL_MODEL_SERVICE_DIR` 可覆盖自动查找结果。
部署后的配置和可执行文件均可独立于 Codex 与源码仓库运行；控制台显示实际服务目录和启动命令。

`--connect-assistant` 会备份原设置，将助手切换到服务端点并关闭旧的进程托管选项。
运行中的旧版助手应先退出，随后启动本次构建的新版助手。
此后的服务配置与助手配置相互独立；助手设置不会重启或修改共享模型。
新版助手每次开始本地任务时读取并跟随服务当前选择，同时固定该任务的模型身份和缓存。
切换后助手无需手动修改旧模型名称；其他项目应重新读取 `/v1/models` 获取当前模型 id。

可以指定配置位置：

```powershell
.venv\Scripts\python -m local_model_service --config D:\Models\service.json init `
  --assistant-settings "$env:LOCALAPPDATA\KindleAnkiBridge\settings.json" `
  --assets-dir "$env:LOCALAPPDATA\KindleAnkiBridge"
.venv\Scripts\python -m local_model_service --config D:\Models\service.json serve
```

Windows 当前用户登录自启由 `install-startup` 注册到 HKCU Run，不增加定时任务。
服务运行独立于阅读助手；关闭助手不会关闭它。

```powershell
.venv\Scripts\python -m local_model_service install-startup
.venv\Scripts\python -m local_model_service uninstall-startup
.venv\Scripts\python -m local_model_service stop
```

这是当前用户的后台服务模式，注销后会退出。若要在无人登录时运行，可由操作系统服务宿主
运行 `serve`，固定配置、资源路径和运行账户；该模式不由上述登录自启命令安装。

独立 Windows 可执行文件可这样构建：

```powershell
.venv\Scripts\python -m PyInstaller packaging\local_model_service.spec --noconfirm
```

`dist\LocalModelService` 包含独立运行所需的 Python 运行时，不含 Qt、词典或模型权重。
`LocalModelService.exe --config <绝对配置路径> serve` 可在后台运行；该文件为无控制台程序，
状态通过 HTTP 或上面的 Python CLI 查询。开发环境的登录自启使用 pythonw；部署时可以
由打包后的程序执行同样的 `install-startup` 子命令，将自启入口固定到安装位置。

## 其他项目接入

无需安装阅读助手的依赖。任何支持 OpenAI chat completions 协议的客户端均可调用：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
model = client.models.list().data[0].id
result = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "将日语句子翻译为中文：覚悟はしていた。"}],
    max_tokens=128,
)
print(result.choices[0].message.content)
```

支持 `stream=True` 的 SSE 响应，也支持 `response_format` 等 llama.cpp 的 chat 参数。
当前只暴露 chat completions 推理接口，不提供模型下载、执行命令或任意文件访问接口。
服务只绑定 `127.0.0.1`，拒绝浏览器 Origin 和非本机 Host；网页项目应由其本机后端调用。
可通过 `api_key_env` 配置可选的 API key 环境变量；配置该变量但值为空时服务拒绝启动。
私有 llama 后端使用随机端口和独立随机凭证，不作为其他项目的公共入口。

## 身份和缓存

`GET /v1/models` 返回模型列表及 `local_model_service` 元数据；`GET /status` 和
`GET /health` 返回服务状态。健康端点的 HTTP 200 表示网关可用，模型可以处于 `sleeping`。
以上查询都不加载模型，也不重置空闲计时。

身份指纹包含实际模型别名、主/草稿 GGUF 的绝对路径、大小和修改时间、运行时文件及 DLL
身份、实际运行参数。它用于运行配置和缓存隔离，不是对全部 GGUF 内容进行每次 SHA256
扫描。加载完成后还会检查私有后端 `/v1/models` 的实际别名；仅端口可用或健康检查通过
不足以放行推理。

客户端可以将元数据中的 `identity` 放进请求头 `X-Expected-Model-Identity`。
服务在真正执行前检查，身份变化返回 HTTP 409 / `MODEL_IDENTITY_CHANGED`，此请求未执行
推理。成功响应带 `X-Local-Model-Identity` 和 `X-Local-Model-Effective`。
阅读助手会在后台工作线程中先查询身份，把身份纳入词汇、语法和原句翻译缓存；遇到加载
回退时重新查询并最多重提一次，按照最终身份写入缓存。不同工作线程使用独立配置快照。

DFlash2 缺失或加载失败时可回退普通模型，`fallback`、`reason`、`effective_model` 和
身份指纹会同步变化。回退选择在本服务实例中保留，避免每次空闲唤醒重复尝试失败配置。
调整配置后应停止并重新启动服务。

## 队列、恢复和资源

- 只有一个推理工作线程和一个后端槽位；最多等待 32 个请求，队列满返回 HTTP 429。
- `X-Request-Priority` 可设为 -10 至 10。助手快速翻译使用 10；等待时间会增加请求权重，
  防止后台请求一直被新请求挤占。运行中的请求不会被抢占。
- 请求总时限默认 300 秒，包含排队和加载；推理读写超时默认 90 秒。过期和断开连接的
  请求会取消，不再留在队列中执行；正在进行的上游连接也会关闭。
- 后端加载失败最多尝试两次并退避；DFlash2 到普通模型额外只允许一次回退。运行中故障
  会结束当前请求，由后续请求重新加载，不会悄悄重复已经提交的推理请求。
- 服务启动时不加载、不预热模型。最后一个请求完成且队列清空后，默认 60 秒关闭后端，
  释放其显存和模型内存。健康查询、状态查询及助手启动不会延长模型常驻时间。
- 模型卸载后工作线程等待事件，没有定期 GPU 查询、健康轮询或后台预热；仅保留轻量
  HTTP 网关和线程。Windows Job Object 保证服务主进程异常退出时清理其模型子进程。
- 日志位于服务配置目录：`service.log` 轮转上限 1 MiB × 4；`backend.log` 在加载前按
  10 MiB 阈值轮转，保留三份历史。服务日志不记录请求正文或凭证。

模型切换请先停止服务，修改 `service.json` 中的 `model`，并同步客户端模型别名。
空闲时间可以调整 `idle_seconds`，必须大于零；越短越快释放资源，下一次调用的冷启动
次数也越多。常驻网关本身会保留少量内存，模型不会在空闲卸载后继续占用显存。

停止与回滚：先执行 `stop`、`uninstall-startup`，再恢复初始化时生成的助手设置备份。
当前源码中的助手只使用服务模式；恢复旧的应用托管行为还需要切回旧版助手。
模型评测会使用独立后端，正式评测前应停止共享服务，避免其收到其他项目请求而加载模型。

# FaceSwap Studio · AI 换脸雏形

一个**开箱即跑**的 AI 换脸原型：浏览器上传两张图，点一下就出结果。
模型权重已经随项目内置，不需要 GPU，也不需要注册任何云服务。

```
            ┌─ YuNet 检测 5 关键点
源脸照片 ────┤  ArcFace 对齐 → w600k_r50 提身份向量 → 经 emap 投影
            └────────────────────────────────┐
                                             ▼
目标照片 ── YuNet 检测 ── 对齐到 128×128 ──→ inswapper_128 生成新脸
                                             │
        CodeFormer 人脸修复（128→512 重建细节）
                                             │
             颜色微调 → 细节回填（碎发/纹理）→ 逆仿射 → 椭圆羽化融合 → 结果
```

核心是 **inswapper_128 + ArcFace** 这条主流链路（FaceFusion / Roop / Deep-Live-Cam 同款）。
和「把源脸像素贴过去」的几何法有本质区别：inswapper 是在特征空间里**重画**一张脸 ——
姿态、表情、光照属于目标，五官身份属于源脸。

用 `tools/compare.py` 可以随时出对比图看效果（不需要起服务）：

```bash
python tools/compare.py --source pic/1.png --target pic/2.png --swap-both
```

---

## 快速开始

### 本机已经配好的环境（推荐，直接跑）

这台机器上已经建好了一个带 OpenCV 的独立环境，**不需要再装任何东西**：

```bash
"C:/Users/86156/.workbuddy/binaries/python/envs/fs39/Scripts/python.exe" backend/app.py
```

然后浏览器打开 <http://127.0.0.1:8000>。或者直接双击 `start.bat`。

### 在别的机器上从零开始

```bash
# 1. 装依赖
pip install -r requirements.txt

# 2. 启动
python backend/app.py

# 3. 浏览器打开
# http://127.0.0.1:8000
```

### 环境踩坑记录

- **OpenCV 必须用 4.x**。OpenCV 5.0 已经彻底移除了 `CascadeClassifier`
  （整个 objdetect 模块在 Python 绑定里都没了），装上 5.x 经典引擎会不可用。
  `requirements.txt` 里已经卡了 `opencv-python-headless>=4.8,<5`。
  引擎做了懒加载，所以即使环境不对，服务照样能启动，只是 `/api/engines`
  会把这个引擎标成不可用并给出可读原因，不会直接崩。
- `opencv-python` 的 wheel 有 50MB 左右，网络差的环境建议用国内镜像：
  `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "opencv-python-headless>=4.8,<5"`
- 如果只装 `opencv-python-headless`，功能完全够用（本项目不做窗口显示）。
- **`typing_extensions` 版本冲突**：如果用 `python -m venv --system-site-packages`
  继承了系统环境的旧版 `typing_extensions`，FastAPI / pydantic 会报
  `cannot import name 'Sentinel'`。解决办法是强制装进虚拟环境：
  ```bash
  pip install --ignore-installed "typing_extensions>=4.14.1"
  ```

---

## 两种引擎

项目内置两个引擎，服务端启动时自动选择「当前可用的最强引擎」。

| 引擎 | 依赖 | 效果 | 速度 | 说明 |
|---|---|---|---|---|
| `deep` | onnxruntime + 模型（已内置） | ★★★★★ | ~0.9s CPU / 张 | **默认引擎**。inswapper_128 + ArcFace 身份向量，与 FaceFusion/Roop 同款主流方案 |
| `classic` | 仅 opencv-python | ★★★☆☆ | ~0.3s / 张 | YuNet 关键点几何对齐 + 泊松融合。兜底方案 |
| `insightface` | insightface 包 + 模型 | ★★★★★ | ~3s CPU | 官方包版实现（需编译，一般用 deep 即可） |

### deep 引擎的模型（已随项目内置）

`models/` 下三个文件，缺失时引擎自动显示不可用并给出原因：

| 文件 | 大小 | 作用 | 下载源 |
|---|---|---|---|
| `inswapper_128.onnx` | 554MB | 人脸生成网络 | ModelScope: `chwshuang/inswapper_128.onnx` |
| `w600k_r50.onnx` | 174MB | ArcFace 提取源脸身份向量 | ModelScope: `cunkai/ComfyUI_Notebook`（buffalo_l/w600k_r50.onnx） |
| `codeformer.onnx` | 360MB | 人脸修复，把 128 的脸重建到 512（可选，缺失自动跳过） | hf-mirror: `facefusion/models-3.0.0` |
| `yunet_2023mar.onnx` | 232KB | 人脸检测 + 5 关键点 | GitHub opencv_zoo |

命令行重新下载：

```bash
curl -L -o models/inswapper_128.onnx "https://www.modelscope.cn/api/v1/models/chwshuang/inswapper_128.onnx/repo?FilePath=inswapper_128.onnx"
curl -L -o models/w600k_r50.onnx "https://modelscope.cn/api/v1/models/cunkai/ComfyUI_Notebook/repo?Revision=master&FilePath=buffalo_l/w600k_r50.onnx"
curl -L -o models/yunet_2023mar.onnx "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
```

### classic 引擎的关键点检测（YuNet，可选但强烈推荐）

`classic` 引擎检测人脸时优先使用 OpenCV 官方的 YuNet 轻量模型（约 230KB，
提供双眼/鼻尖/双嘴角 5 个精准关键点），没有模型时自动退回 Haar 级联。
**有 YuNet 和没有 YuNet 的贴合精度差距非常大**（Haar 的嘴部检测在大图上经常偏移，
导致整张贴歪、出光晕）。模型已随项目放在 `models/yunet_2023mar.onnx`，
如果丢了，重新下载：

```bash
curl -L -o models/yunet_2023mar.onnx \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
```

### 备选的 insightface 官方包版（一般不需要）

`insightface` 引擎走的是官方 `insightface.model_zoo` 的 FaceAnalysis + INSwapper
组合（代码见 `backend/engines/insightface_engine.py`，没有魔改），好处是省事，
代价是要装 `insightface` 包（需要编译，比较费劲），而且 CPU 上比 `deep` 慢 3 倍左右。

```bash
pip install insightface onnxruntime            # GPU 换成 onnxruntime-gpu
set INSWAPPER_MODEL=models\inswapper_128.onnx  # Windows
python backend/app.py
```

装好之后可以在接口里显式指定 `engine=insightface` 用它对比效果。注意
`auto_engine()` 的优先级是 **deep → insightface → classic**，因为 `deep`
内置了模型、永远可用，所以默认走的还是 `deep`。

### deep 引擎的四个关键细节（都是踩过坑才写对的）

这部分是整个项目最值钱的地方 —— 同样是「inswapper + ArcFace」，下面四步
做错任何一步，出来的人脸都会一眼假。

**1. 身份向量必须先过 `emap` 投影，否则换出来的是「路人脸」**

inswapper 模型的 `source` 输入**不是** ArcFace 的原始 embedding，而是要乘上
模型图里藏着的最后一个 initializer（一个 512×512 的方阵，社区习惯叫 `emap`）：

```python
latent = normed_embedding @ emap        # 先投影到 inswapper 自己的身份空间
latent /= np.linalg.norm(latent)        # 再归一化
```

漏掉这一步，模型读到的是一个「无效身份」，会输出一张跟谁都不像的通用脸。
实测自换脸（源=目标）的身份余弦相似度：**漏掉 emap 是 0.05，补上是 0.85**。
`emap` 首次使用时自动从 onnx 图里抽出来缓存成 `models/emap.npy`。

**2. inswapper 的输入输出格式和常见直觉不一样**

- 输入：`x/255` 的 RGB（`swapRB=True`，不做 ±1 归一化）
- 输出：`pred * 255`（不是 `pred * 127.5 + 127.5`）

这两条反了就会得到一张灰蒙蒙的糊图。`deep_engine._swap_face` 里有注释标着。

**3. 蒙版椭圆的位置要按对齐空间实测标定**

ArcFace 对齐之后人脸在画布里的位置是**固定**的，所以蒙版可以用固定椭圆，
但参数必须以对齐空间为基准量出来。曾经写成「五官质心与嘴心的中点」，
圆心落到 `y≈0.75*size`，椭圆整体下坠 0.15×画布 —— 额头整块露在蒙版外，
融合接缝正好横在发际线上，非常显眼。现在的参数：

```
眼睛 y≈0.46 · 鼻尖 y≈0.64 · 嘴巴 y≈0.82 · 发际 y≈0.25 · 下巴 y≈0.93
→ 人脸外接椭圆：圆心 (0.50, 0.585)，半轴 (0.305, 0.385)
```

**4. 颜色迁移要「轻」，细节回填要「只压暗」**

- inswapper 的输出**本来就已经继承了目标图的色调**（实测 L/a/b 均值误差 <2），
  所以 LAB 迁移只该轻轻推一把。早期用的是「直线拼接」的权重公式，明明只想微调，
  却把 a 通道从 134 压到 129，整张脸发白。现在改成严格插值
  `out = (1-w)*src + w*matched`，并且 a/b 通道不再压缩方差（保留血色），
  过饱和的唇色交给 `tanh` 软限幅单独处理。
- **细节回填**：把目标原图的高频明暗调制 `k = ref / gaussian(ref)` 乘回生成脸，
  用来恢复被 inswapper 抹掉的**碎发、睫毛、皮肤纹理**（inswapper 只输出 128px
  的光板脸，搭在脸上的发丝会被整片抹掉，这是最明显的破绽）。
  但 `k` 的上界必须卡在 **1.0**（只压暗、不提亮）：允许提亮会把目标脸上的
  高光/腮红再叠一层，实测脸颊 L 从 164 被抬到 172（目标自己才 167），脸就"发白"。
  另外眼睛和嘴唇要单独**降权到 25%** —— 这两块位置会被对齐到同一处、但形状
  因人而异，硬叠会把目标的牙缝、睫毛糊到生成脸上，变成一团灰黑。

**5. 修复网络接进来之后，仿射矩阵必须跟着分辨率缩放**

inswapper 只输出 128px，直接贴回大图是"塑料脸"。本项目接了 **CodeFormer**
（FaceFusion 官方 ONNX 转换版，360MB）：把 128 的脸双三次放大到 512 重建细节，
再贴回去。接它的时候踩了一个隐蔽的坑 —— 对齐矩阵 `matrix` 是「原图 → 128 对齐
空间」的，换到 512 空间后**必须整体乘 4（线性部分和平移量一起）**，否则脸会以
1/4 尺寸被甩到图角落上，且引擎把形状错误静默吞掉，表面看起来"像没换"。
排查手段：看 `cos(结果, 目标)`，如果接近 1.0，说明目标原脸原封不动地留在那了。

```python
work_matrix = matrix * (work_size / _SWAP_SIZE)   # 512/128 = 4.0
```

CodeFormer 接口实测（和常见资料略有出入，以实测为准）：
- 输入 `input` = (1,3,512,512) RGB，归一化到 **[-1,1]**（`(x/255-0.5)/0.5`）
- 输入 `weight` = **double 标量**（0-d ndarray，`np.array(0.7)`，传裸的
  `np.float64` 会报 `'numpy.float64' object is not iterable`）
- 输出 `output` = (1,3,512,512)，区间 **[-1,1]**（min≈-1 / max≈1，用 `y*0.5+0.5` 解码）
- 图里还有 `logits`/`features` 两个输出，`sess.run(["output"], feed)` 只算需要的那个

`weight` 是保真度权重：0=画质猛但可能"换头"，1=最忠实于输入。换脸场景
**0.6~0.8** 比较稳（默认 0.7）。CPU 单次推理约 1.5s，开了它整条链路 ~0.8s → ~5s。

模型下载（缺失时引擎自动降级为不修复，不会报错）：

```bash
curl -L -o models/codeformer.onnx \
  "https://hf-mirror.com/facefusion/models-3.0.0/resolve/main/codeformer.onnx"
```

### 效果验收指标

`tools/compare.py` 会直接打印这三个数字，调参时照着看就行：

| 指标 | 健康值 | 含义 |
|---|---|---|
| `cos(结果, 源)` | > 0.60 | 身份确实换过来了（源/目标互不相同的两人约 0.12） |
| `cos(结果, 目标)` | 明显小于上一项 | 没有只是把目标原脸留在那 |
| `脸_LAB偏差` / `脖_LAB偏差` | 各通道 \|Δ\| < 6 | 脸没有发白/发灰，和脖子没有断层 |

---

## 项目结构

```
faceswap-studio/
├─ backend/
│  ├─ app.py                      FastAPI 服务：上传、检测、换脸、历史
│  └─ engines/
│     ├─ base.py                  引擎抽象层（SwapEngine / SwapOptions / FaceBox）
│     ├─ classic.py               经典几何法引擎（零模型）
│     ├─ insightface_engine.py    深度学习引擎（可选）
│     └─ __init__.py              引擎注册表 + 自动降级逻辑
├─ frontend/
│  ├─ index.html                  三栏工作台：素材 / 参数 / 结果
│  ├─ styles.css                  浅色主题样式
│  └─ app.js                      上传、调参、前后对比滑块、历史
├─ tools/
│  ├─ compare.py                  出「源|目标|结果」三联图 + 身份/颜色指标
│  └─ selftest.py                 12 项管线自检，不需要真实人脸照片
├─ uploads/  outputs/             运行时产生的图片
├─ docs/灵感与产品方向.md          玩法、商业化、合规、技术选型
├─ start.bat                      Windows 一键启动
└─ requirements.txt
```

### 跑自检

```bash
python tools/selftest.py
```

它会用程序画两张「假脸」当素材，检查：输出尺寸、脸区域是否真的被改动、
背景是否被误伤、色彩匹配是否生效、强度=0 是否等于原图、蒙版参数是否响应、
退化输入是否抛可读错误、结果能否写盘 —— 共 12 项，全部不依赖网络和真实照片。

---

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查 |
| GET | `/api/engines` | 列出引擎及可用状态 |
| GET | `/api/history` | 最近的结果记录 |
| POST | `/api/detect` | 表单字段 `image`，返回人脸框 + 眼睛/嘴关键点 |
| POST | `/api/swap` | 字段 `source` / `target` / `options`(JSON) / `engine` |
| POST | `/api/swap/batch` | 多张 `sources` + 一张 `target`，批量出图 |

`options` 可调字段（都带默认值）：

```json
{
  "target_face_index": 0,     // 目标图里第几张脸（按面积从大到小）
  "source_face_index": 0,
  "feather": 45,              // 蒙版羽化 0-100
  "mask_scale": 1.28,         // 蒙版范围 0.6-2.0
  "color_match": true,        // LAB 肤色迁移，开不开差别巨大
  "face_restore": true,       // CodeFormer 重建脸部细节（需要 codeformer.onnx）
  "restore_weight": 0.7,      // 修复保真度 0-1：0=画质猛 1=最像原脸
  "blend_mode": "seamless",   // seamless | feather
  "strength": 100,            // 结果混入比例 0-100
  "enhance": false            // 锐化
}
```

命令行调一下就能验证接口：

```bash
curl -X POST http://127.0.0.1:8000/api/swap ^
  -F "source=@me.jpg" -F "target=@group.jpg" ^
  -F "options={\"blend_mode\":\"seamless\"}"
```

---

## 当前版本的已知边界

这个雏形故意做得「轻」，所以有几处明显的天花板，别当成产品直接用：

1. **CodeFormer 的"发明力"有限但存在**。它能把睫毛、皮肤纹理、牙齿重建得
   非常真实，但保真度权重调太低（<0.4）会慢慢长出一张"别人的脸"，把换上去的
   身份稀释掉。默认 0.7 是实测平衡点。
2. **表情差异大时会露馅**。目标大笑露齿、源脸闭嘴时，模型重建的牙齿容易糊
   （CodeFormer 会修一部分，但不是总能修好）。身份向量只带五官身份不带表情，
   所以表情整体跟着**目标**走 —— 这也是为什么 deep 引擎不会出现经典引擎那种
   「目标笑了换上去的脸不笑」的问题。
3. **大角度侧脸会崩**（超过约 ±60° 基本没救，这是 inswapper 本身的训练范围）。
   检测端倒是稳的，YuNet 到 30°~45° 都还能出关键点。
4. **无视频支持**。视频需要逐帧换脸 + 时序平滑（避免抖动闪烁）+ 音画同步，
   工作量是图片版的 3~5 倍。
5. **无水印、无授权校验、无内容审核**。这三件事在真实产品里是必须的，见
   `docs/灵感与产品方向.md` 的合规章节。
6. **历史存在内存里**，重启即丢。要持久化请接 SQLite / 云数据库。
7. **多脸场景只能按面积索引选脸**，没有做人脸跟踪 / 多脸同时替换的 UI。

---

## 合规提醒

换脸技术本身中立，但**未经他人同意把他人的脸换到任何内容上，在绝大多数
司法辖区都是违法的**。做产品前请务必：拿到肖像权授权、打不可去除的水印、
对公开传播的图片做来源标记（C2PA / 元数据）、对明显违规内容做事前拦截。
本雏形仅供学习、研究和经本人同意的娱乐用途。

# 실습 환경 세팅 (setting.md)

> **AI 에이전트(Claude Code)에게:** 이 문서는 수강생이 "이 setting.md 파일을 읽고 세팅을 완료해 줘"라고 요청했을 때 따라야 할 절차입니다. 아래 단계를 **위에서부터 순서대로** 모두 수행하고, 각 단계의 **검증 항목**을 직접 실행해서 확인하세요. 검증이 실패하면 원인을 고쳐서 다시 시도하고, 끝내 해결하지 못하면 어떤 단계에서 무엇이 막혔는지 그대로 보고하세요. 마지막에는 "최종 보고" 형식으로 결과를 정리하세요.

## 0. 실습 개요와 전제

- 데이터: KITTI 3D object detection (이미 다운로드된 `archive` 폴더). 이 문서가 있는 폴더가 **archive 루트**입니다. 이하 `<ARCHIVE>`로 표기합니다.
- 실습 목표: RGBD 카메라 입력을 가정하여, 단안 depth estimator(Depth Anything V2 Metric-Outdoor)의 추정 depth를 D 채널로 붙인 4채널(RGBD) 입력을 만들고, RGB만 사용하는 object detector를 학습하여 **차량 좌표계 기준 특정 영역 안의 물체를 검출**하는 시스템을 만든다.
- 이 문서가 다루는 범위: (1) 파이썬 환경, (2) 보조 스크립트 설치, (3) **주행(drive) 단위 7:1:2 데이터 분할**, (4) 라벨 시각화와 depth 추정 동작 확인, (5) **7,481 프레임 전체의 depth map 생성**. 모든 수강생이 **완전히 같은 분할**과 **같은 depth map**을 쓰도록 분할과 depth estimator는 이 문서에 고정되어 있습니다.
- **Depth estimator는 반드시 Depth Anything V2 – Metric Outdoor (Large)를 사용합니다.** Hugging Face 모델 ID는 `depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf`입니다. 다른 모델(Indoor 버전, Small/Base 크기, relative depth 버전, UniDepth, ZoeDepth 등)로 바꾸거나, 추론 설정(해상도, 정밀도, 후처리)을 임의로 바꾸지 마세요. 그래야 모든 수강생의 D 채널이 같아져서 성능을 비교할 수 있습니다. 이 모델을 쓸 수 없는 문제가 생기면 대체하지 말고 수강생에게 보고하세요.
- 라벨은 `training/label_2`의 7,481 프레임만 사용합니다. `testing/`은 라벨이 없으므로 이 실습의 train/val/test 분할에 쓰지 않습니다.

### 지켜야 할 안전 규칙
- `<ARCHIVE>`의 **원본 데이터(`training/`, `testing/`, `gt_database/`, `devkit_object/`, `*.pkl`)는 절대 수정, 삭제, 이동하지 않습니다.** 읽기 전용으로만 사용합니다.
- 생성물은 아래 폴더에만 만듭니다: `<ARCHIVE>/tools/`(스크립트), `<ARCHIVE>/splits/`(분할 목록), `<ARCHIVE>/outputs/`(결과물).
- 다운로드된 데이터 폴더 안에서 파이썬을 실행하지 마세요(현재 디렉터리의 모듈이 import될 수 있음). 스크립트는 항상 **절대 경로**로, 작업 디렉터리를 `<ARCHIVE>` 밖(예: 사용자 홈이나 임시 폴더)으로 둔 채 실행하고, 데이터 경로는 인자로 넘깁니다.
- 외부 다운로드는 `pip` 패키지와 Hugging Face 모델 가중치(`depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf`, 약 1.3 GB)뿐입니다. 그 외의 것은 받지 않습니다.

## 1. 폴더 확인

`<ARCHIVE>`를 정하고 다음을 확인하세요.

- `training/image_2`, `training/velodyne`, `training/calib`, `training/label_2`, `devkit_object/mapping/train_mapping.txt`가 있어야 합니다.
- `training/label_2`의 파일은 7,481개, `training/image_2`의 파일은 7,481개여야 합니다.

하나라도 다르면 수강생에게 폴더가 올바른지 확인해 달라고 알리고 중단합니다.

## 2. 파이썬 환경

**요구:** Python 3.10 이상(실습 기준 3.14에서 검증됨). 먼저 `python --version`으로 확인합니다.

필요한 패키지: `numpy`, `pillow`, `matplotlib`, `packaging`, `transformers`, `torch`.

```
python -m pip install numpy pillow matplotlib packaging transformers
```

**주의할 점(실제로 겪은 문제):**
- `matplotlib`이 `ModuleNotFoundError: No module named 'packaging'`로 실패할 수 있습니다. 위 명령에 `packaging`을 포함한 이유입니다.
- `pip install`이 사용자 site-packages(`AppData\Roaming\Python`)에 설치되면, `python -I`(isolated 모드)는 그 경로를 무시합니다. 검증과 실행에는 `-I` 없이 `python`을 쓰되, 작업 디렉터리를 `<ARCHIVE>` 밖으로 둡니다.
- **PyTorch는 반드시 CUDA(GPU) 빌드여야 합니다.** 6단계의 전체 depth 생성은 GPU에서 약 40분(RTX 5080 기준 프레임당 0.34초)이지만 CPU에서는 약 10시간이라 사실상 불가능합니다. 기본 `pip install torch`는 CPU 빌드(`+cpu`)가 설치될 수 있으므로 **그대로 두지 마세요.**
  1. `nvidia-smi`로 NVIDIA GPU와 드라이버가 보이는지 확인하고 첫 줄의 `CUDA UMD Version`(또는 `CUDA Version`)을 읽습니다.
  2. 그 버전에 맞는 CUDA 빌드를 설치합니다. 이미 CPU 빌드가 깔려 있으면 같은 버전 번호라서 pip가 "이미 설치됨"으로 건너뛰므로 **`--force-reinstall --no-deps`**를 붙입니다.
     - CUDA 13.x 드라이버: `python -m pip install torch --index-url https://download.pytorch.org/whl/cu130 --force-reinstall --no-deps`
     - CUDA 12.8~12.9 드라이버: `python -m pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps` (이 인덱스는 최신 torch가 2.11까지일 수 있음)
     - **RTX 50 시리즈(Blackwell)는 CUDA 12.8 이상 빌드가 필수**입니다. 그보다 오래된 빌드는 GPU를 인식해도 커널이 없어 실패합니다.
     - 인덱스에 따라 이름이 다를 수 있으니 안 되면 `https://pytorch.org/get-started/locally/`의 명령을 따르세요. 실행에 필요한 다른 패키지는 위 `pip install` 명령으로 이미 설치돼 있습니다.
  3. 확인: `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"`에서 `True`와 GPU 이름이 나와야 합니다.
  - **GPU가 없거나 위 방법으로도 `False`이면 6단계를 시작하지 말고** 수강생에게 상황을 알리세요. 이 경우 5단계의 `000010` 한 장 확인은 CPU로 해도 되지만, 전체 depth 생성은 GPU가 있는 환경에서 해야 합니다. 어떤 빌드를 설치했는지 최종 보고에 적습니다.
- Windows에서 Hugging Face가 "To support symlinks on Windows..." 경고를 내는 것은 정상이며 무시해도 됩니다.

**검증:** 다음이 오류 없이 출력되어야 합니다.
```
python -c "import numpy, PIL, matplotlib, packaging, transformers, torch; print('ok', torch.__version__, torch.cuda.is_available())"
```

## 3. 보조 스크립트 설치 (`<ARCHIVE>/tools/`)

`<ARCHIVE>/tools/` 폴더를 만들고, 아래 네 파일을 **내용을 그대로** 저장하세요. 이미 같은 이름의 파일이 있으면 내용이 아래와 같은지 비교하고, 다르면 수강생에게 알린 뒤 덮어쓸지 확인합니다.

### 3-1. `tools/make_split.py` — 주행 단위 7:1:2 분할

주행 → 분할 배정이 코드에 **고정**되어 있어서 난수에 의존하지 않고, 누구가 실행해도 같은 결과가 나옵니다.

```python
"""Drive-level 7:1:2 split of KITTI training frames (instructor-fixed assignment).

Usage: python make_split.py <archive_root>
Writes <archive_root>/splits/{train,val,test}.txt (frame ids, 6 digits, one per line).
Frames of one raw drive always go to the same split (no leakage between neighbouring frames).
The drive->split assignment is hard-coded below (found with seed 960), so it does not depend on any RNG.
"""
import sys, os

DRIVES = {
  "train": [
    "2011_09_26_drive_0001_sync",
    "2011_09_26_drive_0009_sync",
    "2011_09_26_drive_0013_sync",
    "2011_09_26_drive_0015_sync",
    "2011_09_26_drive_0017_sync",
    "2011_09_26_drive_0018_sync",
    "2011_09_26_drive_0019_sync",
    "2011_09_26_drive_0022_sync",
    "2011_09_26_drive_0023_sync",
    "2011_09_26_drive_0027_sync",
    "2011_09_26_drive_0032_sync",
    "2011_09_26_drive_0035_sync",
    "2011_09_26_drive_0036_sync",
    "2011_09_26_drive_0046_sync",
    "2011_09_26_drive_0052_sync",
    "2011_09_26_drive_0056_sync",
    "2011_09_26_drive_0060_sync",
    "2011_09_26_drive_0061_sync",
    "2011_09_26_drive_0064_sync",
    "2011_09_26_drive_0079_sync",
    "2011_09_26_drive_0084_sync",
    "2011_09_26_drive_0086_sync",
    "2011_09_26_drive_0087_sync",
    "2011_09_26_drive_0093_sync",
    "2011_09_26_drive_0095_sync",
    "2011_09_26_drive_0101_sync",
    "2011_09_26_drive_0106_sync",
    "2011_09_26_drive_0113_sync",
    "2011_09_26_drive_0117_sync",
    "2011_09_28_drive_0001_sync",
    "2011_09_28_drive_0002_sync",
    "2011_09_28_drive_0016_sync",
    "2011_09_28_drive_0021_sync",
    "2011_09_28_drive_0035_sync",
    "2011_09_28_drive_0038_sync",
    "2011_09_28_drive_0043_sync",
    "2011_09_28_drive_0047_sync",
    "2011_09_28_drive_0053_sync",
    "2011_09_28_drive_0054_sync",
    "2011_09_28_drive_0065_sync",
    "2011_09_28_drive_0068_sync",
    "2011_09_28_drive_0071_sync",
    "2011_09_28_drive_0075_sync",
    "2011_09_28_drive_0077_sync",
    "2011_09_28_drive_0078_sync",
    "2011_09_28_drive_0086_sync",
    "2011_09_28_drive_0089_sync",
    "2011_09_28_drive_0090_sync",
    "2011_09_28_drive_0094_sync",
    "2011_09_28_drive_0095_sync",
    "2011_09_28_drive_0096_sync",
    "2011_09_28_drive_0098_sync",
    "2011_09_28_drive_0100_sync",
    "2011_09_28_drive_0102_sync",
    "2011_09_28_drive_0103_sync",
    "2011_09_28_drive_0104_sync",
    "2011_09_28_drive_0106_sync",
    "2011_09_28_drive_0108_sync",
    "2011_09_28_drive_0113_sync",
    "2011_09_28_drive_0117_sync",
    "2011_09_28_drive_0119_sync",
    "2011_09_28_drive_0121_sync",
    "2011_09_28_drive_0122_sync",
    "2011_09_28_drive_0125_sync",
    "2011_09_28_drive_0126_sync",
    "2011_09_28_drive_0132_sync",
    "2011_09_28_drive_0134_sync",
    "2011_09_28_drive_0135_sync",
    "2011_09_28_drive_0136_sync",
    "2011_09_28_drive_0138_sync",
    "2011_09_28_drive_0141_sync",
    "2011_09_28_drive_0143_sync",
    "2011_09_28_drive_0145_sync",
    "2011_09_28_drive_0146_sync",
    "2011_09_28_drive_0149_sync",
    "2011_09_28_drive_0153_sync",
    "2011_09_28_drive_0154_sync",
    "2011_09_28_drive_0155_sync",
    "2011_09_28_drive_0156_sync",
    "2011_09_28_drive_0160_sync",
    "2011_09_28_drive_0161_sync",
    "2011_09_28_drive_0162_sync",
    "2011_09_28_drive_0165_sync",
    "2011_09_28_drive_0166_sync",
    "2011_09_28_drive_0167_sync",
    "2011_09_28_drive_0174_sync",
    "2011_09_28_drive_0177_sync",
    "2011_09_28_drive_0179_sync",
    "2011_09_28_drive_0184_sync",
    "2011_09_28_drive_0185_sync",
    "2011_09_28_drive_0186_sync",
    "2011_09_28_drive_0187_sync",
    "2011_09_28_drive_0191_sync",
    "2011_09_28_drive_0195_sync",
    "2011_09_28_drive_0199_sync",
    "2011_09_28_drive_0201_sync",
    "2011_09_28_drive_0204_sync",
    "2011_09_28_drive_0205_sync",
    "2011_09_28_drive_0208_sync",
    "2011_09_28_drive_0209_sync",
    "2011_09_28_drive_0216_sync",
    "2011_09_28_drive_0220_sync",
    "2011_09_29_drive_0026_sync",
    "2011_09_29_drive_0071_sync"
  ],
  "val": [
    "2011_09_26_drive_0005_sync",
    "2011_09_26_drive_0011_sync",
    "2011_09_26_drive_0057_sync",
    "2011_09_26_drive_0091_sync",
    "2011_09_26_drive_0096_sync",
    "2011_09_26_drive_0104_sync",
    "2011_09_28_drive_0045_sync",
    "2011_09_28_drive_0087_sync",
    "2011_09_28_drive_0168_sync",
    "2011_09_28_drive_0183_sync",
    "2011_09_28_drive_0198_sync"
  ],
  "test": [
    "2011_09_26_drive_0002_sync",
    "2011_09_26_drive_0014_sync",
    "2011_09_26_drive_0020_sync",
    "2011_09_26_drive_0028_sync",
    "2011_09_26_drive_0029_sync",
    "2011_09_26_drive_0039_sync",
    "2011_09_26_drive_0048_sync",
    "2011_09_26_drive_0051_sync",
    "2011_09_26_drive_0059_sync",
    "2011_09_26_drive_0070_sync",
    "2011_09_28_drive_0034_sync",
    "2011_09_28_drive_0037_sync",
    "2011_09_28_drive_0039_sync",
    "2011_09_28_drive_0057_sync",
    "2011_09_28_drive_0066_sync",
    "2011_09_28_drive_0070_sync",
    "2011_09_28_drive_0080_sync",
    "2011_09_28_drive_0082_sync",
    "2011_09_28_drive_0110_sync",
    "2011_09_28_drive_0128_sync",
    "2011_09_28_drive_0171_sync",
    "2011_09_28_drive_0192_sync",
    "2011_09_28_drive_0214_sync",
    "2011_09_28_drive_0222_sync",
    "2011_09_29_drive_0004_sync",
    "2011_10_03_drive_0047_sync"
  ]
}

root = sys.argv[1]
drive_of = [l.split()[1] for l in open(os.path.join(root, "devkit_object", "mapping", "train_mapping.txt")) if l.strip()]
assert len(drive_of) == 7481, len(drive_of)
split_of = {d: s for s, ds in DRIVES.items() for d in ds}
assert len(split_of) == sum(len(v) for v in DRIVES.values()) == len(set(drive_of)), "drive list mismatch"

out = os.path.join(root, "splits")
os.makedirs(out, exist_ok=True)
for s in DRIVES:
    ids = [i for i, d in enumerate(drive_of) if split_of[d] == s]
    with open(os.path.join(out, s + ".txt"), "w", newline="\n") as f:
        f.write("".join("%06d\n" % i for i in ids))
    print(s, len(ids), "frames,", len(DRIVES[s]), "drives")
```

### 3-2. `tools/viz_kitti.py` — 한 샘플의 모든 라벨 시각화

2D 박스, 투영된 3D 박스, LiDAR BEV를 한 장에 그립니다. 사용법: `python viz_kitti.py <ARCHIVE> <프레임번호 6자리> <출력.png>`

```python
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image

root, idx, out = sys.argv[1], sys.argv[2], sys.argv[3]
split = "training"
COLORS = {"Car": "#00e5ff", "Van": "#ffd600", "Truck": "#ff9100", "Pedestrian": "#ff1744",
          "Person_sitting": "#f50057", "Cyclist": "#76ff03", "Tram": "#d500f9",
          "Misc": "#b0bec5", "DontCare": "#9e9e9e"}

img = np.array(Image.open(f"{root}/{split}/image_2/{idx}.png").convert("RGB"))
pts = np.fromfile(f"{root}/{split}/velodyne/{idx}.bin", dtype=np.float32).reshape(-1, 4)

calib = {}
for line in open(f"{root}/{split}/calib/{idx}.txt"):
    if ":" in line:
        k, v = line.split(":", 1)
        calib[k] = np.array(v.split(), dtype=np.float64)
P2 = calib["P2"].reshape(3, 4)
R0 = np.eye(4); R0[:3, :3] = calib["R0_rect"].reshape(3, 3)
Tr = np.eye(4); Tr[:3, :] = calib["Tr_velo_to_cam"].reshape(3, 4)

objs = []
for line in open(f"{root}/{split}/label_2/{idx}.txt"):
    t = line.split()
    if not t:
        continue
    objs.append(dict(type=t[0], trunc=float(t[1]), occ=int(t[2]), alpha=float(t[3]),
                     bbox=list(map(float, t[4:8])), hwl=list(map(float, t[8:11])),
                     loc=list(map(float, t[11:14])), ry=float(t[14])))


def corners_cam(o):
    h, w, l = o["hwl"]
    x = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y = [0, 0, 0, 0, -h, -h, -h, -h]
    z = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    c, s = np.cos(o["ry"]), np.sin(o["ry"])
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return R @ np.array([x, y, z]) + np.array(o["loc"]).reshape(3, 1)


def project(c3):
    p = P2 @ np.vstack([c3, np.ones((1, 8))])
    return (p[:2] / p[2]).T


EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

# LiDAR -> rect camera coordinates (for BEV we use camera x/z, same frame as labels)
xyz1 = np.hstack([pts[:, :3], np.ones((len(pts), 1))])
cam = (R0 @ Tr @ xyz1.T).T[:, :3]
mask = cam[:, 2] > 0
cam, inten = cam[mask], pts[mask, 3]

fig = plt.figure(figsize=(18, 12), constrained_layout=True)
gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], height_ratios=[1, 1])
ax_img = fig.add_subplot(gs[0, 0]); ax_3d = fig.add_subplot(gs[1, 0]); ax_bev = fig.add_subplot(gs[:, 1])

# 1) 2D boxes
ax_img.imshow(img); ax_img.set_title(f"{idx}: 2D boxes (all labels, incl. DontCare)"); ax_img.axis("off")
for i, o in enumerate(objs):
    col = COLORS.get(o["type"], "white")
    l, t, r, b = o["bbox"]
    ax_img.add_patch(plt.Rectangle((l, t), r-l, b-t, fill=False, ec=col, lw=1.5,
                                   ls="--" if o["type"] == "DontCare" else "-"))
    ax_img.text(l, t-2, f"{i}:{o['type']}", color="black", fontsize=7,
                bbox=dict(fc=col, ec="none", pad=0.8, alpha=0.85))

# 2) 3D boxes projected
ax_3d.imshow(img); ax_3d.set_title("3D boxes projected (P2) + [trunc / occ / dist z]"); ax_3d.axis("off")
for i, o in enumerate(objs):
    if o["type"] == "DontCare":
        continue
    col = COLORS.get(o["type"], "white")
    c3 = corners_cam(o)
    if (c3[2] <= 0.1).any():
        continue
    uv = project(c3)
    for a, b in EDGES:
        ax_3d.plot(*zip(uv[a], uv[b]), color=col, lw=1.4)
    ax_3d.add_patch(Polygon(uv[[0, 1, 5, 4]], closed=True, fc=col, alpha=0.25, ec="none"))  # front face
    ax_3d.text(uv[4:, 0].min(), uv[4:, 1].min()-3, f"{i}:{o['type']} t{o['trunc']:.2f} o{o['occ']} z{o['loc'][2]:.1f}",
               color="black", fontsize=7, bbox=dict(fc=col, ec="none", pad=0.8, alpha=0.85))
ax_3d.set_xlim(0, img.shape[1]); ax_3d.set_ylim(img.shape[0], 0)

# 3) BEV
ax_bev.scatter(cam[:, 0], cam[:, 2], s=0.3, c=inten, cmap="gray", vmin=0, vmax=1)
ax_bev.set_facecolor("#101418")
for i, o in enumerate(objs):
    if o["type"] == "DontCare":
        continue
    col = COLORS.get(o["type"], "white")
    c3 = corners_cam(o)
    poly = c3[[0, 2], :][:, [0, 1, 2, 3]].T
    ax_bev.add_patch(Polygon(poly, closed=True, fill=False, ec=col, lw=1.5))
    # heading arrow from center along ry
    cx, cz = o["loc"][0], o["loc"][2]
    ax_bev.annotate("", xy=(cx + np.cos(o["ry"]) * o["hwl"][2]/2, cz - np.sin(o["ry"]) * o["hwl"][2]/2),
                    xytext=(cx, cz), arrowprops=dict(arrowstyle="->", color=col, lw=1.2))
    ax_bev.text(cx, cz, str(i), color=col, fontsize=8, ha="center", va="bottom")
ax_bev.set_xlim(-30, 30); ax_bev.set_ylim(0, 70); ax_bev.set_aspect("equal")
ax_bev.set_xlabel("x (m, right)"); ax_bev.set_ylabel("z (m, forward)")
ax_bev.set_title("BEV (LiDAR points, camera frame) + heading")

# legend + table
present = sorted({o["type"] for o in objs})
handles = [plt.Line2D([0], [0], color=COLORS.get(t, "w"), lw=3, label=f"{t} ({sum(o['type']==t for o in objs)})") for t in present]
fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=10)
fig.suptitle(f"KITTI training/{idx} - {len(objs)} labels, {len(pts)} LiDAR points", fontsize=14)
fig.savefig(out, dpi=110)

print(f"{'#':>2} {'type':<11}{'trunc':>6}{'occ':>4}{'alpha':>7}  {'bbox(l,t,r,b)':<26}{'hwl':<20}{'xyz':<22}{'ry':>6}")
for i, o in enumerate(objs):
    print(f"{i:>2} {o['type']:<11}{o['trunc']:>6.2f}{o['occ']:>4}{o['alpha']:>7.2f}  "
          f"{str([round(v) for v in o['bbox']]):<26}{str(o['hwl']):<20}{str(o['loc']):<22}{o['ry']:>6.2f}")
```

### 3-3. `tools/depth_da2.py` — Depth Anything V2 Metric으로 depth 추정

사용법: `python depth_da2.py <ARCHIVE> <프레임번호 6자리> <출력 폴더>`. 출력 폴더에 `<id>_depth.npy`(float32, H×W, 단위 m)와 RGB/depth/LiDAR 비교 그림 `<id>_rgb_depth.png`를 저장하고, LiDAR 대비 AbsRel, RMSE, δ<1.25를 출력합니다.

```python
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

root, idx, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
split = "training"

img = Image.open(f"{root}/{split}/image_2/{idx}.png").convert("RGB")
W, H = img.size

proc = AutoImageProcessor.from_pretrained(MODEL)
model = AutoModelForDepthEstimation.from_pretrained(MODEL).eval()
dev = "cuda" if torch.cuda.is_available() else "cpu"
model.to(dev)

with torch.no_grad():
    inp = proc(images=img, return_tensors="pt").to(dev)
    pred = model(**inp).predicted_depth  # (1,h,w), metres
    depth = torch.nn.functional.interpolate(pred[None], size=(H, W), mode="bicubic",
                                            align_corners=False)[0, 0].cpu().numpy()
np.save(f"{outdir}/{idx}_depth.npy", depth.astype(np.float32))

# LiDAR sparse depth (for sanity check)
calib = {}
for line in open(f"{root}/{split}/calib/{idx}.txt"):
    if ":" in line:
        k, v = line.split(":", 1)
        calib[k] = np.array(v.split(), dtype=np.float64)
P2 = calib["P2"].reshape(3, 4)
R0 = np.eye(4); R0[:3, :3] = calib["R0_rect"].reshape(3, 3)
Tr = np.eye(4); Tr[:3, :] = calib["Tr_velo_to_cam"].reshape(3, 4)
pts = np.fromfile(f"{root}/{split}/velodyne/{idx}.bin", dtype=np.float32).reshape(-1, 4)
cam = (R0 @ Tr @ np.hstack([pts[:, :3], np.ones((len(pts), 1))]).T)
cam = cam[:, cam[2] > 0.1]
uvw = P2 @ cam
u, v, z = uvw[0] / uvw[2], uvw[1] / uvw[2], cam[2]
ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
u, v, z = u[ok].astype(int), v[ok].astype(int), z[ok]
gt = np.zeros((H, W), np.float32); gt[v, u] = z
m = gt > 0
pd_ = depth[m]; gd = gt[m]
absrel = np.mean(np.abs(pd_ - gd) / gd)
rmse = np.sqrt(np.mean((pd_ - gd) ** 2))
d1 = np.mean(np.maximum(pd_ / gd, gd / pd_) < 1.25)
ratio = np.median(pd_ / gd)
print(f"device={dev}  depth range pred: {depth.min():.2f}-{depth.max():.2f} m")
print(f"LiDAR-covered pixels: {m.sum()} ({100*m.mean():.2f}% of image)")
print(f"AbsRel={absrel:.4f}  RMSE={rmse:.3f} m  delta<1.25={d1:.4f}  median(pred/gt)={ratio:.3f}")

vmax = 60
fig, ax = plt.subplots(3, 1, figsize=(14, 13), constrained_layout=True)
ax[0].imshow(img); ax[0].set_title(f"RGB  (training/{idx})"); ax[0].axis("off")
im = ax[1].imshow(depth, cmap="turbo_r", vmin=0, vmax=vmax)
ax[1].set_title("Depth Anything V2 Metric-Outdoor-Large (predicted, m)"); ax[1].axis("off")
fig.colorbar(im, ax=ax[1], fraction=0.025, pad=0.01, label="depth (m)")
ax[2].imshow(img)
ys, xs = np.nonzero(m)
sc = ax[2].scatter(xs, ys, c=gt[m], s=3, cmap="turbo_r", vmin=0, vmax=vmax)
ax[2].set_title(f"LiDAR sparse depth overlay (reference) - AbsRel {absrel:.3f}, RMSE {rmse:.2f} m, d1 {d1:.3f}")
ax[2].axis("off")
fig.colorbar(sc, ax=ax[2], fraction=0.025, pad=0.01, label="depth (m)")
fig.savefig(f"{outdir}/{idx}_rgb_depth.png", dpi=90)
```

### 3-4. `tools/depth_all.py` — 전체 프레임 depth 생성 (6단계에서 사용)

Depth Anything V2 – Metric Outdoor (Large)로 프레임들의 depth를 추정해 KITTI 형식 16비트 PNG(`값/256 = m`)로 저장합니다. 모델 ID와 추론 설정은 모든 수강생이 같아야 하므로 **수정하지 마세요.**

```python
"""Estimate metric depth (Depth Anything V2 Metric-Outdoor-Large) for KITTI training frames.

Usage:
  python depth_all.py <archive_root> [--ids splits/train.txt] [--limit N] [--out DIR] [--allow-cpu]
Requires a CUDA GPU by default (exits with an error otherwise).

Output: <out>/<id>.png  (default out = <archive_root>/outputs/depth)
  16-bit PNG, KITTI depth format: depth_m = pixel_value / 256.0  (0 = invalid). Size = original image size.
Frames whose output already exists are skipped, so an interrupted run can simply be restarted.
"""
import os, sys, time, argparse
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--ids", default=None, help="text file with one 6-digit frame id per line (default: all 7481)")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--allow-cpu", action="store_true", help="run on CPU if no CUDA GPU (very slow)")
a = ap.parse_args()

root = a.root
out = a.out or os.path.join(root, "outputs", "depth")
os.makedirs(out, exist_ok=True)
if a.ids:
    ids = [l.strip() for l in open(a.ids) if l.strip()]
else:
    ids = ["%06d" % i for i in range(7481)]
if a.limit:
    ids = ids[:a.limit]
todo = [i for i in ids if not os.path.exists(os.path.join(out, i + ".png"))]
print(f"{len(ids)} frames requested, {len(ids) - len(todo)} already done, {len(todo)} to do -> {out}", flush=True)

if not torch.cuda.is_available() and not a.allow_cpu:
    sys.exit("CUDA GPU not available (torch %s). Install a CUDA build of PyTorch (see setting.md, step 2), "
             "or pass --allow-cpu to run on CPU (~5 s/frame, ~10 h for all frames)." % torch.__version__)
dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = AutoImageProcessor.from_pretrained(MODEL)
model = AutoModelForDepthEstimation.from_pretrained(MODEL).eval().to(dev)
print("device:", dev, flush=True)

t0 = time.time()
for n, i in enumerate(todo, 1):
    img = Image.open(os.path.join(root, "training", "image_2", i + ".png")).convert("RGB")
    W, H = img.size
    with torch.no_grad():
        inp = proc(images=img, return_tensors="pt").to(dev)
        pred = model(**inp).predicted_depth
        d = torch.nn.functional.interpolate(pred[None], size=(H, W), mode="bicubic",
                                            align_corners=False)[0, 0].cpu().numpy()
    q = np.clip(np.round(d * 256.0), 1, 65535).astype(np.uint16)
    tmp = os.path.join(out, i + ".tmp.png")
    Image.fromarray(q).save(tmp)
    os.replace(tmp, os.path.join(out, i + ".png"))  # atomic: never leaves a half-written file
    if n % 50 == 0 or n == len(todo):
        el = time.time() - t0
        print(f"{n}/{len(todo)}  {el/n:.2f}s/frame  ETA {el/n*(len(todo)-n)/60:.1f} min", flush=True)
print("done", flush=True)
```

## 4. 데이터 분할 생성

작업 디렉터리를 `<ARCHIVE>` 밖으로 두고 실행합니다.
```
python <ARCHIVE>/tools/make_split.py <ARCHIVE>
```
`<ARCHIVE>/splits/train.txt`, `val.txt`, `test.txt`가 생성됩니다(프레임 번호 6자리, 한 줄에 하나, 오름차순). 번호 `000123`은 `training/image_2/000123.png`, `training/label_2/000123.txt` 등에 대응합니다.

**검증(모두 일치해야 함):**

| 분할 | 주행 수 | 프레임 수 | 비율 |
|---|---|---|---|
| train | 104 | 5,239 | 70.0% |
| val | 11 | 749 | 10.0% |
| test | 26 | 1,493 | 20.0% |

- 세 파일의 합은 7,481이고 서로 겹치는 프레임이 없어야 합니다.
- MD5가 아래와 같아야 합니다(줄바꿈은 `\n` 기준. 값이 다르면 분할이 어긋난 것이므로 반드시 원인을 찾으세요).

```
a3c3849342f0861e700433d255f2a2c3  splits/train.txt
4e35d081b80e0fd1121cdbe58cc85301  splits/val.txt
c84c85c120e23b2c6a898e92e7e3af88  splits/test.txt
```
  (확인 예: `python -c "import hashlib,sys;print(hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest())" <파일>`)

### 분할 방식 (참고)
- `devkit_object/mapping/train_mapping.txt`의 각 줄은 프레임 번호 순서대로의 raw 주행 이름을 담고 있으며, 7,481 프레임은 **141개의 주행**에서 나왔습니다. 같은 주행의 인접 프레임은 거의 같은 장면이라서, 프레임을 무작위로 섞어 나누면 train과 test에 비슷한 장면이 섞여 성능이 부풀려집니다. 그래서 **같은 주행은 항상 같은 분할**에 넣었습니다.
- 7:1:2 프레임 비율과 클래스(Car, Pedestrian, Cyclist) 분포가 최대한 비슷하도록 시드를 탐색해서(시드 960) 배정을 정했고, 그 결과를 `make_split.py`에 고정했습니다. 클래스 비율은 train/val/test 순으로 대략 Car 69.6/10.4/19.9%, Pedestrian 66.3/12.9/20.8%, Cyclist 67.7/11.5/20.8%입니다(주행 단위라서 7:1:2에서 약간 벗어납니다).

## 5. 동작 확인 (샘플 `000010`)

작업 디렉터리를 `<ARCHIVE>` 밖으로 두고 `<ARCHIVE>/outputs/check/`를 만들어 실행합니다.

```
python <ARCHIVE>/tools/viz_kitti.py <ARCHIVE> 000010 <ARCHIVE>/outputs/check/000010_labels.png
python <ARCHIVE>/tools/depth_da2.py <ARCHIVE> 000010 <ARCHIVE>/outputs/check
```

**검증(기대값):**
- `viz_kitti.py`: 표에 라벨 13줄(Car 8, Pedestrian 1, DontCare 4)이 출력되고 이미지가 저장됩니다.
- `depth_da2.py`: 첫 실행에서 모델 가중치(약 1.3 GB)를 받습니다. 출력은 대략 다음과 같아야 합니다(GPU/CPU 차이로 마지막 자릿수는 조금 다를 수 있습니다).
  - `LiDAR-covered pixels: 16429 (3.53% of image)`
  - `AbsRel≈0.108, RMSE≈5.46 m, delta<1.25≈0.92, median(pred/gt)≈1.03`
  - 예측 depth 범위 약 4.6~78 m
  - `<ARCHIVE>/outputs/check/000010_depth.npy`(375×1242, float32)와 `000010_rgb_depth.png` 생성
- 생성된 두 PNG를 직접 열어서 3D 박스가 차량에 맞고 depth 맵이 RGB의 구조와 일치하는지 눈으로 확인하세요.

AbsRel이 0.2보다 크게 다르거나 median(pred/gt)이 0.8~1.25를 벗어나면 모델 로딩이나 이미지 경로에 문제가 있는 것이므로 원인을 찾으세요.

## 6. 전체 프레임 depth map 생성

**모델:** Depth Anything V2 – Metric Outdoor (Large), `depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf`. `tools/depth_all.py`가 이 모델을 고정해서 쓰므로 스크립트의 모델 ID, fp32 정밀도, 전처리, 후처리를 **바꾸지 마세요**(3-4 참고).

작업 디렉터리를 `<ARCHIVE>` 밖으로 두고 실행합니다.
```
python <ARCHIVE>/tools/depth_all.py <ARCHIVE>
```
- 대상은 `training/image_2`의 **7,481 프레임 전체**입니다(train/val/test 모두. 분할은 depth 생성과 무관합니다).
- 결과는 `<ARCHIVE>/outputs/depth/<프레임번호>.png`에 저장됩니다. **16비트 PNG이고 KITTI depth 형식**입니다: `depth(m) = 픽셀값 / 256`, 0은 무효. 크기는 원본 이미지와 같습니다(대부분 1242×375).
- 이미 만들어진 프레임은 건너뛰므로, 중단되면 **같은 명령을 다시 실행**하면 이어서 진행됩니다. 파일은 임시 이름으로 쓴 뒤 이름을 바꿔 저장하므로 중간에 끊겨도 깨진 파일이 남지 않습니다.
- **디스크:** 약 3.5 GB(파일당 약 0.4 MB)가 필요합니다. 시작 전에 여유 공간을 확인하세요.
- **GPU로 실행합니다.** `depth_all.py`는 CUDA GPU가 없으면 오류로 종료합니다(CPU로 몰래 돌지 않음). 오류가 나면 2단계로 돌아가 CUDA 빌드 PyTorch를 설치하세요. 출력의 `device: cuda`를 확인합니다. **`--allow-cpu`는 쓰지 마세요**(프레임당 약 5초, 전체 약 10시간).
- **소요 시간:** RTX 5080에서 프레임당 약 0.34초, 전체 약 40분이었습니다. 진행 상황이 50프레임마다 `n/전체  초/프레임  ETA`로 출력됩니다. 오래 걸리는 작업이므로 백그라운드로 실행하고 진행 상황을 주기적으로 확인하세요.
- 일부만 먼저 만들어 보려면 `--ids <ARCHIVE>/splits/val.txt --limit 20`처럼 쓸 수 있습니다. 그래도 최종적으로는 인자 없이 전체를 완료해야 합니다.

**검증(모두 만족해야 함):**
1. `<ARCHIVE>/outputs/depth/`에 `000000.png`부터 `007480.png`까지 **7,481개**가 있고 `*.tmp.png`는 없어야 합니다.
2. 아무 프레임이나 다음처럼 읽어서 `uint16`, 이미지 크기와 같은 shape, 깊이 범위가 대략 3~80 m인지 확인합니다.
```python
import numpy as np
from PIL import Image
d = np.array(Image.open("<ARCHIVE>/outputs/depth/000010.png")).astype(np.float32) / 256.0   # meters
print(d.shape, d.min(), d.max())   # (375, 1242), 약 4.62 ~ 78.29
```
3. `000010`의 값이 5단계의 `outputs/check/000010_depth.npy`와 **평균 절대 차이 0.01 m 이하, 최대 0.1 m 이하**로 일치해야 합니다(PNG 양자화 오차 0.002 m 이하, GPU와 CPU의 부동소수점 차이로 최대 약 0.04 m까지 날 수 있음).
4. 여러 프레임을 무작위로 열어 RGB와 비교해서 depth가 구조에 맞는지(가까운 도로와 차량은 작은 값, 하늘과 먼 곳은 큰 값) 확인합니다.

## 7. 최종 보고

수강생에게 다음을 간단히 보고하세요.
1. 설치된 파이썬과 PyTorch 버전, GPU 사용 가능 여부
2. 분할 결과(프레임 수, MD5 일치 여부)
3. `000010` 검증 지표
4. 전체 depth 생성 결과(생성된 파일 수 7,481 여부, 걸린 시간, 사용한 장치, 모델 ID)
5. 문제가 있었던 단계와 해결 방법(없으면 "없음")
6. 생성한 파일 목록(`tools/`, `splits/`, `outputs/check/`, `outputs/depth/`)

## 8. 이후 실습 규칙 (모든 수강생 공통)

- **분할은 `splits/*.txt`만 사용합니다.** 이를 바꾸면 다른 수강생과 성능을 비교할 수 없습니다. 분할을 다시 만들거나 수정하지 마세요.
- **val은 모델 선택과 하이퍼파라미터 튜닝에, test는 최종 보고에 단 한 번만** 사용합니다. test 결과를 보고 모델을 고치지 않습니다.
- RGB만으로 detector를 학습합니다. depth는 4번째 채널(D) 입력으로 쓸 수 있도록 **`<ARCHIVE>/outputs/depth/<프레임번호>.png`(16비트, 값/256 = m)** 가 공통 규약입니다. 모든 수강생이 같은 모델(Depth Anything V2 Metric Outdoor Large)로 만든 같은 depth map을 사용해야 하므로, **depth를 다른 모델로 다시 만들어 쓰지 않습니다.**
- 검출 대상 영역은 차량(카메라) 좌표계의 라벨 `location`(x: 오른쪽, z: 전방, 단위 m)으로 정의합니다. 구체적인 영역은 실습 안내에 따릅니다.
- 평가 대상 클래스는 Car, Pedestrian, Cyclist입니다. `DontCare`는 학습과 평가에서 무시 영역으로 다룹니다.
- 이 분할은 공식 KITTI 분할(train 3,712 / val 3,769)과 다릅니다. KITTI 서버 점수와 직접 비교하지 마세요.

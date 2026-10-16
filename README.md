# Karate Club GNN 튜토리얼과 W&B Sweep

PyTorch와 PyTorch Geometric(PyG)으로 Karate Club 그래프의 노드 분류를 실습합니다. 입문 노트북, YAML 기반 학습, W&B 하이퍼파라미터 탐색, 선택한 설정의 재학습·평가로 구성됩니다.

## 파일 구성

| 파일 | 역할 |
| --- | --- |
| [Karate_club_gcn_tutorial.ipynb](./Karate_club_gcn_tutorial.ipynb) | 데이터 탐색, 2차원 임베딩, 학습, 미학습 노드 평가, GIF 생성 |
| [wandb_sweep_Karate_club_gnn.py](./wandb_sweep_Karate_club_gnn.py) | GCN·GAT·GraphSAGE·DNN 단일 학습과 W&B Sweep |
| [Karate_club_gnn_sweep_validation.ipynb](./Karate_club_gnn_sweep_validation.ipynb) | 선택한 Sweep 설정으로 재학습하고 미학습 노드를 평가 |
| [utils/model.py](./utils/model.py) | GCN, GAT, GraphSAGE와 DNN 기준 모델 |
| [utils/wandb.py](./utils/wandb.py) | YAML 로드, Sweep 설정 적용, 실행 이름 및 체크포인트 경로 처리 |
| [configs/config.yaml](./configs/config.yaml) | 최적화, 학습, 로그 및 체크포인트 기본 설정 |
| [configs/gnn_model_config.yaml](./configs/gnn_model_config.yaml) | 모델 기본 설정 |
| [configs/sweeps/gcn_config.yaml](./configs/sweeps/gcn_config.yaml), [gat_config.yaml](./configs/sweeps/gat_config.yaml), [graphsage_config.yaml](./configs/sweeps/graphsage_config.yaml), [dnn_config.yaml](./configs/sweeps/dnn_config.yaml) | 모델별 Sweep 명령과 탐색 범위 |
| [requirements-sweep.txt](./requirements-sweep.txt) | 학습 및 Sweep 의존성 |
| [launch_gnn.sh](./launch_gnn.sh), [launch_sweep.sh](./launch_sweep.sh) | Bash 실행 도우미 |
| [wanb_sweeps.md](./wanb_sweeps.md) | 상세 Sweep 실행 안내 |
| [loss_curve_vis/loss_curve_comp.ipynb](./loss_curve_vis/loss_curve_comp.ipynb) | 기본·튜닝 GCN, DNN·GCN 및 GAT·GraphSAGE 학습 손실 비교 |

## 데이터와 학습 목표

`torch_geometric.datasets.KarateClub()`으로 데이터를 불러오며 별도 데이터 파일은 필요하지 않습니다.

| 항목 | 값 / 설명 |
| --- | --- |
| 그래프 / 노드 수 | 1개 / 34명 |
| 무방향 연결 수 | 78개 |
| `data.edge_index` | `[2, 156]`; 각 연결을 양방향으로 저장 |
| `data.x` | `[34, 34]`; 노드별 원-핫 특징 |
| `data.y` | `[34]`; 0~3의 커뮤니티 레이블 |
| `data.train_mask` | `[34]`; 클래스당 1개, 총 4개 학습 노드 |
| `data.val_mask` | 4개; 스크립트·재학습은 클래스별 1개 |
| `data.test_mask` | 학습·검증을 제외한 26개 노드 |

전체 그래프의 특징과 연결을 사용하지만 학습 손실에는 4개 노드의 레이블만 사용합니다. 이미 주어진 그래프의 나머지 노드를 분류하는 반지도·전이적(transductive) 학습입니다. 학습 노드는 `[0, 4, 8, 24]`입니다. 학습 스크립트와 재학습 노트북은 각 클래스의 미학습 노드 중 인덱스가 가장 작은 노드를 검증용으로 1개씩 선택합니다. 검증 노드의 클래스가 균형을 이루며 무작위 선택이 아니므로 seed를 바꿔도 분할은 같습니다. 현재 검증 노드는 `[1, 5, 14, 25]`이며, 후보가 없는 클래스에는 오류를 발생시킵니다.

학습 목표는 PyG의 `Data`와 `edge_index`, `GCNConv`, 마스크 기반 손실, 임베딩 시각화, 하이퍼파라미터 탐색과 평가를 이해하는 것입니다.

## 실행 환경 준비

입문·재학습 노트북의 메타데이터에는 Python `3.10.21`, 커널 표시 이름 `ai_tutorial`이 기록되어 있습니다. 의존성 버전은 고정되어 있지 않습니다. 프로젝트 루트에서 실행하세요.

```powershell
conda create -n ai_tutorial python=3.10 -y
conda activate ai_tutorial
python -m pip install -r requirements-sweep.txt
python -m pip install jupyterlab ipykernel ipywidgets
python -m ipykernel install --user --name ai_tutorial --display-name ai_tutorial
```

기존 환경이 있으면 생성 단계는 생략합니다. requirements에는 `torch`, `torch-geometric`, `wandb`, `PyYAML`, `matplotlib`, `networkx`, `pillow`, `numpy`가 포함되며 Jupyter 도구는 별도 설치합니다. 추가 확장 설치는 [PyG 공식 설치 안내](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)를 참고하세요. 현재 모델과 데이터를 GPU로 이동하는 코드는 없어 CPU에서 학습합니다.

## 입문 노트북 실행

`python -m jupyterlab`으로 `Karate_club_gcn_tutorial.ipynb`를 열고 `ai_tutorial` 커널에서 위에서 아래로 실행합니다. VS Code에서도 같은 Python 환경을 선택한 뒤 Run All을 사용할 수 있습니다.

```text
노드 특징 [34, 34] + 그래프 연결 [2, 156]
    → GCNConv(34, 4) → ReLU
    → GCNConv(4, 4)  → ReLU
    → GCNConv(4, 2)  → ReLU
    → 임베딩 h [34, 2] → Linear(2, 4) → 로짓 out [34, 4]
```

`forward()`는 로짓과 임베딩을 반환합니다. 학습은 `CrossEntropyLoss`, Adam(`lr=0.01`), `range(500)`으로 0~499의 500회 업데이트를 수행합니다.

```python
out, h = model(data.x, data.edge_index)
loss = criterion(out[data.train_mask], data.y[data.train_mask])
```

학습 후 `model.eval()`에서 `data.test_mask`의 26개 노드에 대해 손실과 정확도를 계산합니다. 입문 노트북은 검증 마스크도 생성하지만 학습 중 검증이나 조기 종료에는 사용하지 않습니다. 테스트 결과를 반복적으로 확인하며 설정을 선택하면 독립적인 최종 평가로 해석하기 어렵습니다.

### 임베딩 애니메이션

매 epoch의 임베딩, epoch 번호, 손실을 기록합니다. 임베딩은 해당 업데이트 전 순전파 결과이며 산점도 색은 정답 레이블입니다. 저장된 좌표를 재생하므로 애니메이션 생성 중에는 추가 학습하지 않습니다.

| 항목 | 현재 설정 |
| --- | --- |
| 기록 수 | 500프레임 |
| 재생 속도 | `fps=50`; 명목상 한 번 재생에 10초 |
| 프레임 간격 | `interval=1000 / fps`; 20ms |
| 저장 위치 | `results/karate_club_embedding.gif` |
| 저장 크기 | `figsize=(7, 7)`, `dpi=100`; 700 × 700px |
| 노트북 표시 | `ani.to_jshtml(fps=fps)` 재생 컨트롤 |

![Karate Club GCN 임베딩 변화](./results/karate_club_embedding.gif)

## YAML 기반 단일 학습

```powershell
python wandb_sweep_Karate_club_gnn.py
python wandb_sweep_Karate_club_gnn.py --wandb --wandb_mode offline
# 그래프 연결을 사용하지 않는 기준 모델
python wandb_sweep_Karate_club_gnn.py --model_type DNN
# 온라인 기록: 최초 로그인 후 실행
wandb login
python wandb_sweep_Karate_club_gnn.py --wandb --wandb_entity kjsong
```

`--model_type GCN`, `GAT`, `GraphSAGE`, `DNN`을 지원합니다. GNN 세 모델은 모든 합성곱 계층에 같은 hidden 차원, ReLU와 dropout을 적용하고 로짓만 반환합니다. GAT는 `concat=False`로 attention head를 결합하고 GraphSAGE는 `aggr`로 이웃 집계 방법을 선택합니다. 입문 노트북의 2차원 임베딩 모델과 구조가 다릅니다. DNN은 GCN 계층 대신 Linear 계층을 쌓고 ReLU·dropout·선형 분류기를 사용합니다. 공통 호출을 위해 `edge_index` 인자를 받지만 사용하지 않고 노드 특징만 처리합니다.

업데이트 후 `model.eval()`에서 검증 노드 4개의 손실이 개선될 때 `best.pt`를 저장하고 80 epoch 연속 개선이 없으면 종료합니다. W&B에는 매 epoch `epoch`, `loss`(업데이트 전 학습 손실), `val_loss`, `val_accuracy`, `best_val_loss`를 기록합니다. 체크포인트에는 `best_val_loss`를 저장합니다.

`configs/config.yaml`의 로그·체크포인트 경로는 현재 KETI PC의 절대 경로입니다. 다른 PC에서는 먼저 `logging_dir: "logs"`, `checkpoint_dir: "checkpoints"` 등으로 변경하세요.

- W&B 없이 실행하면 기본 체크포인트 디렉터리의 `best.pt`를 사용해 이전 파일을 덮어쓸 수 있습니다.
- W&B 실행 시 기본 이름은 `<MODEL>-<run_id>`이고 경로는 `checkpoints/<MODEL>-<run_id>-<run_id>/best.pt`입니다. `<MODEL>`은 `GCN`, `GAT`, `GraphSAGE`, `DNN` 중 선택한 모델입니다. `--wandb_run_name NAME`이면 `checkpoints/NAME-<run_id>/best.pt`입니다.
- W&B 실행 로그는 `logging_dir` 아래 `wandb/`에 저장됩니다. Sweep agent 설정 파일은 프로젝트 루트의 `wandb/`에도 생성될 수 있습니다.

## W&B Sweep과 설정 재평가

```powershell
python -m wandb sweep --entity kjsong --project karate-club-gcn-sweep configs/sweeps/gcn_config.yaml
# 출력된 실제 SWEEP_ID로 교체
python -m wandb agent --count 1 kjsong/karate-club-gcn-sweep/SWEEP_ID
```

현재 네 모델의 Sweep은 Bayesian 탐색으로 epoch별 검증 손실 `val_loss`를 최소화합니다. 기존 `best_train_loss` 기준 Sweep은 새 YAML로 다시 등록해야 합니다. Windows에서 Python을 직접 실행하므로 Bash는 필요하지 않습니다. 탐색 범위와 Hyperband, 평가 노트북 사용법은 [Sweep 안내](./wanb_sweeps.md)를 참고하세요.

`Karate_club_gnn_sweep_validation.ipynb`는 선택한 로컬 W&B YAML을 읽고 모델을 새로 초기화해 500 epoch 재학습합니다.

## 손실 곡선 비교

[loss_curve_vis/loss_curve_comp.ipynb](./loss_curve_vis/loss_curve_comp.ipynb)는 저장된 NumPy 배열로 기본 GCN과 튜닝 GCN, 튜닝한 DNN과 GCN, 튜닝한 GCN·GAT·GraphSAGE의 학습 손실을 세 그래프로 비교합니다.

```powershell
conda activate ai_tutorial
Set-Location loss_curve_vis
python -m jupyterlab
```

노트북은 `./data/`를 상대 경로로 읽으므로 커널의 작업 디렉터리를 `loss_curve_vis`로 맞추세요. `data/`의 `gcn_train_loss_basic.npy`, `gcn_train_loss_hpt.npy`, `gat_train_loss_hpt.npy`, `graphsage_train_loss_hpt.npy`, `dnn_train_loss_hpt.npy`가 필요하며 현재 각 배열에는 500 epoch의 기록이 있습니다. 비교 노트북은 파일명에 따라 기본·튜닝 모델의 기록으로 표시합니다. 배열 자체에는 seed나 설정 메타데이터가 없으므로 엄밀한 동일 조건 비교로 해석하지 마세요.

입문 노트북은 `np.save`가 활성화되어 있어 실행 시 `loss_curve_vis/data/gcn_train_loss_basic.npy`를 덮어씁니다. 재학습 노트북의 NumPy import와 `dnn_train_loss_hpt.npy` 저장 예제는 주석 처리되어 있으므로 새 곡선이 필요하면 활성화하세요. 다른 모델을 선택할 때는 저장 파일명도 바꾸세요. 저장된 비교 배열은 서로 다른 실행·탐색 기준의 결과가 섞일 수 있으며 `.npy`만으로 생성 조건을 확인할 수 없습니다.

## 실행 시 참고

- 모듈을 찾지 못하면 설치한 Python과 노트북 커널이 같은 환경인지 확인하세요.
- `IProgress not found` 경고에는 같은 환경에 `ipywidgets`를 설치하고 커널을 재시작하세요.
- GIF 저장에는 Pillow가 필요하며 FFmpeg는 필요하지 않습니다.
- HTML 애니메이션으로 노트북 크기가 커질 수 있습니다. 기록 주기를 늘리거나 그림 크기를 줄이면 출력 크기를 줄일 수 있습니다.
- `logs/`, `wandb/`, `checkpoints/`, Python 캐시는 `.gitignore`로 제외됩니다. 검증 노트북이 참조하는 개인 Sweep 설정 파일은 저장소에 포함되지 않습니다.

## 참고 자료

- [PyTorch Geometric 설치 안내](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
- [KarateClub 데이터셋 문서](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.KarateClub.html)
- [GCNConv 문서](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html)
- [Matplotlib 애니메이션 안내](https://matplotlib.org/stable/users/explain/animations/animations.html)
- [W&B Sweep 설정 안내](https://docs.wandb.ai/models/sweeps/define-sweep-configuration)
- [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

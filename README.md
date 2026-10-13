# Karate Club GCN 튜토리얼

[Karate_club_gcn.ipynb](./Karate_club_gcn.ipynb)는 PyTorch와 PyTorch Geometric(PyG)을 사용해 그래프 신경망의 기본 원리를 실습하는 노트북입니다. Karate Club 소셜 네트워크에서 일부 노드의 커뮤니티 레이블만으로 GCN(Graph Convolutional Network)을 학습하고, 노드의 2차원 임베딩이 어떻게 변화하는지 애니메이션으로 시각화하고 GIF로 저장합니다.

## 학습 목표

- PyG의 `Data` 객체와 그래프 연결 정보인 `edge_index` 이해
- `GCNConv`를 이용한 그래프 합성곱 모델 구현
- 학습 노드에만 손실을 적용하는 반지도 학습 이해
- 학습 전후 노드 임베딩과 커뮤니티 구조 관찰
- epoch별 임베딩 기록을 이용한 애니메이션 생성 및 GIF 저장

## 데이터셋

노트북은 `torch_geometric.datasets.KarateClub()`으로 그래프를 불러옵니다. 별도의 데이터 파일을 준비할 필요는 없습니다. 아래 값은 노트북에 저장된 출력 기준입니다.

| 항목 | 값 / 설명 |
| --- | --- |
| 그래프 수 | 1 |
| 노드 수 | 34명 |
| 무방향 연결 수 | 78개 |
| `edge_index`에 저장된 간선 수 | 156개: 각 무방향 연결을 양방향으로 저장 |
| 노드 특징 | 34차원 원-핫 벡터 |
| 커뮤니티 클래스 수 | 4 |
| 학습 노드 수 | 4개: 클래스당 1개 |
| 평균 차수 | 약 4.59 |

주요 데이터 속성은 다음과 같습니다.

| 속성 | 형태 | 역할 |
| --- | --- | --- |
| `data.x` | `[34, 34]` | 노드 특징 행렬; 이 예제에서는 단위 행렬 |
| `data.edge_index` | `[2, 156]` | 각 열에 출발·도착 노드 번호를 저장하는 COO 형식 |
| `data.y` | `[34]` | 각 노드의 커뮤니티 레이블(0~3) |
| `data.train_mask` | `[34]` | 손실 계산에 사용할 노드를 선택하는 불리언 마스크 |

전체 노드의 특징과 그래프 연결을 사용하되, 4개 노드의 레이블만 학습에 사용합니다. 따라서 이미 주어진 그래프의 나머지 노드를 분류하는 **반지도·전이적(transductive) 노드 분류** 예제입니다.

## 실행 환경 및 설치

노트북 메타데이터에는 Python `3.10.21`, 커널 표시 이름 `ai_tutorial`이 기록되어 있습니다. 패키지 버전은 고정되어 있지 않습니다. 주요 의존성은 `torch`, `torch_geometric`, `networkx`, `matplotlib`, `IPython`, `Pillow`이며, 실행을 위해 Jupyter와 Python 커널이 필요합니다.

Conda를 사용하는 경우 아래 명령으로 환경을 준비할 수 있습니다.

```powershell
conda create -n ai_tutorial python=3.10 -y
conda activate ai_tutorial
python -m pip install torch
python -m pip install torch_geometric networkx matplotlib pillow jupyterlab ipykernel
python -m ipykernel install --user --name ai_tutorial --display-name ai_tutorial
```

PyG의 기본 사용은 PyTorch 설치 후 `pip install torch_geometric`으로 시작할 수 있습니다. 추가 확장 패키지 설치는 [PyG 공식 설치 문서](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)를 참고하세요. 현재 노트북에는 GPU로 모델과 데이터를 이동하는 코드가 없으며 CPU에서 실행합니다.

## 실행 방법

1. 위에서 준비한 `ai_tutorial` 환경을 활성화합니다.
2. 이 폴더에서 `python -m jupyterlab`을 실행하고 `Karate_club_gcn.ipynb`를 엽니다.
3. 커널을 `ai_tutorial`로 선택하고 위에서 아래로 모든 셀을 실행합니다.

VS Code에서도 노트북을 열고 같은 환경의 Python 커널을 선택한 뒤 **Run All**을 사용할 수 있습니다. `%matplotlib inline`이 포함되어 있으므로 Jupyter/IPython 노트북 환경에서 실행하세요.

## 코드 흐름

1. **시각화 함수 정의**: `visualize_graph()`는 NetworkX 그래프를, `visualize_embedding()`은 2차원 노드 임베딩을 그립니다.
2. **데이터 탐색**: 그래프 수, 특징 수, 클래스 수와 `Data` 속성 및 `edge_index`를 확인합니다.
3. **그래프 시각화**: `to_networkx(data, to_undirected=True)`로 변환한 그래프를 정답 커뮤니티별 색으로 표시합니다.
4. **GCN 모델 정의**: 세 개의 그래프 합성곱 계층과 선형 분류기를 구성합니다.
5. **학습 전 임베딩 확인**: 초기 모델이 생성한 `[34, 2]` 임베딩을 시각화합니다.
6. **학습 기록 저장**: 마스크로 선택한 노드의 손실을 최소화하면서 매 epoch의 임베딩 좌표, 반복 번호, 손실을 저장합니다.
7. **애니메이션 생성 및 저장**: `FuncAnimation`으로 산점도의 위치와 학습 정보를 갱신하고, `PillowWriter`로 GIF를 저장한 뒤 노트북에 재생 컨트롤을 표시합니다.

## 모델 구조

```text
노드 특징 [34, 34] + 그래프 연결 [2, 156]
    → GCNConv(34, 4) → tanh
    → GCNConv(4, 4)  → tanh
    → GCNConv(4, 2)  → tanh
    → 2차원 임베딩 h [34, 2]
    → Linear(2, 4)
    → 클래스별 로짓 out [34, 4]
```

각 `GCNConv`는 이웃 노드의 특징을 집계합니다. 세 계층을 거치면서 최대 3-hop 이웃 정보를 반영하고, 특징 차원을 `34 → 4 → 4 → 2`로 줄입니다. `forward()`는 분류 결과인 `out`과 시각화용 임베딩 `h`를 함께 반환합니다. `out`은 확률 변환 전의 로짓이며 `CrossEntropyLoss`에 그대로 전달합니다.

모델 생성 시 `torch.manual_seed(1234)`를 설정하고, 그래프 배치에는 `nx.spring_layout(..., seed=42)`를 사용합니다.

## 학습 설정

| 항목 | 설정 |
| --- | --- |
| 손실 함수 | `torch.nn.CrossEntropyLoss()` |
| 최적화 알고리즘 | `torch.optim.Adam` |
| 학습률 | `0.01` |
| 학습 반복 | `range(401)`: 0~400, 총 401회 |
| 임베딩 기록 주기 | `epoch % 1 == 0`: 매 epoch, 총 401프레임 |
| 애니메이션 재생 속도 | `fps = 50`: 프레임당 0.02초, 한 번 재생에 약 8.02초 |
| 프레임 전환 간격 | `interval=1000 / fps`: 20ms |
| GIF 저장 경로 | 현재 작업 디렉터리 기준 `./results/karate_club_embedding.gif` |
| 저장 해상도 | `figsize=(7, 7)`, `dpi=100`: 700 × 700px |

반지도 학습을 구현하는 핵심 코드는 다음과 같습니다.

```python
out, h = model(data.x, data.edge_index)
loss = criterion(out[data.train_mask], data.y[data.train_mask])
```

전체 그래프를 통과해 모든 노드의 출력을 계산하지만, 손실은 `train_mask`가 선택한 4개 노드에 대해서만 계산합니다. 이후 `loss.backward()`와 `optimizer.step()`으로 모델 파라미터를 갱신합니다.

## 임베딩 애니메이션

학습 셀은 기존의 개별 그림 출력과 `time.sleep(0.3)` 대신 다음 기록을 누적합니다.

```python
embedding_history.append(h.detach().cpu().numpy().copy())
epoch_history.append(epoch)
loss_history.append(loss.item())
```

`detach()`로 학습 그래프에서 분리하고, CPU의 NumPy 배열을 복사해 각 시점의 좌표를 독립적으로 보관합니다. 다음 셀의 `update(frame)`은 `scatter.set_offsets()`로 노드 위치를 바꾸고 epoch와 손실을 갱신합니다. 저장된 좌표를 재생하므로 애니메이션 생성 중에 추가로 학습하지 않습니다.

마지막 임베딩에 `tanh`가 적용되어 좌표는 `[-1, 1]` 범위에 있습니다. 애니메이션은 양쪽 축을 `[-1.1, 1.1]`로 고정하고 같은 축척을 사용하므로, 프레임마다 화면 크기가 바뀌지 않고 노드 이동을 비교할 수 있습니다.

```python
save_path = Path("./results/karate_club_embedding.gif")
ani.save(str(save_path), writer=PillowWriter(fps=fps), dpi=100)
plt.close(fig)
display(HTML(ani.to_jshtml(fps=fps)))
```

GIF는 `PillowWriter`로 저장하며, 노트북에는 `to_jshtml()`을 통한 재생 컨트롤이 표시됩니다. `plt.close(fig)`는 별도의 정적 그림이 함께 출력되지 않도록 합니다.

저장된 결과: [GIF 다운로드](./results/karate_club_embedding.gif)

![Karate Club GCN 학습 중 임베딩 변화](./results/karate_club_embedding.gif)

재생 속도는 애니메이션 셀의 `fps` 값으로 조절합니다. 기록 주기를 10 epoch로 바꾸려면 학습 셀의 조건을 `epoch % 10 == 0`으로 변경하세요. 이 경우 41프레임이 기록되며, `fps=4`이면 한 번 재생에 약 10.25초가 걸립니다. 기록 주기를 바꾼 후에는 `model = GCN()`을 포함한 학습 셀 전체를 다시 실행한 뒤 애니메이션 셀을 실행하세요.

## 출력 해석

- 데이터 통계와 특징·레이블·간선 텐서가 출력됩니다.
- 그래프 그림에서 노드 색은 정답 커뮤니티를 나타냅니다.
- 학습 전 임베딩의 형태는 `[34, 2]`입니다.
- 학습 종료 후 표시되는 애니메이션의 산점도에는 반복 번호와 학습 손실이 표시됩니다. 같은 커뮤니티 노드가 임베딩 공간에서 어떻게 모이는지 관찰할 수 있습니다.

임베딩 산점도의 색도 `data.y`에 따른 **정답 레이블**입니다. 예측 클래스의 색을 표시한 그림은 아닙니다. 현재 코드에는 별도의 검증·테스트 마스크, 정확도 계산, 모델 저장이 없으므로 시각화만으로 분류 성능을 수치적으로 판단할 수는 없습니다.

또한 학습 함수가 반환하는 `h`는 해당 반복에서 파라미터를 갱신하기 **전** 순전파로 계산한 임베딩입니다. 최종 갱신 후 임베딩이 필요하면 학습 종료 후 모델을 다시 호출해야 합니다.

## 실행 시 참고

- `ModuleNotFoundError`가 발생하면 선택한 노트북 커널과 패키지를 설치한 환경이 같은지 확인하세요.
- 저장된 출력에는 `IProgress not found` 경고가 있습니다. 이 경고가 나타나면 같은 환경에 `python -m pip install ipywidgets`로 설치한 뒤 커널을 재시작할 수 있습니다.
- GIF 저장 경로는 커널의 현재 작업 디렉터리를 기준으로 합니다. 저장 전에 `results` 폴더가 있는지 확인하세요. 폴더가 없으면 저장 호출 앞에 `save_path.parent.mkdir(parents=True, exist_ok=True)`를 추가할 수 있습니다.
- `Pillow`가 없으면 같은 커널에서 `%pip install pillow`를 실행하세요. GIF 저장에는 FFmpeg가 필요하지 않습니다.
- 노트북의 `fps = 50` 옆 주석에는 이전 설정인 4fps 설명이 남아 있습니다. 실제 재생 속도는 코드 값인 50fps입니다.
- HTML 애니메이션은 프레임 이미지를 출력에 포함하므로 저장된 노트북 크기가 커질 수 있습니다. `animation.embed_limit` 경고가 나타나면 기록 주기를 늘리거나 `figsize`를 줄여 프레임 수 또는 크기를 줄일 수 있습니다.
- 패키지 버전이 고정되어 있지 않아 실행 환경에 따라 손실이나 그림이 달라질 수 있습니다.

## 성능 비교 노트북

다음 두 노트북은 원본 GCN과 같은 데이터 및 학습 노드로 비교군을 평가합니다. 각 노트북에 원본과 동일한 구조의 GCN이 포함되어 있어 독립 실행할 수 있습니다.

| 노트북 | 비교군 | 사용 정보 |
| --- | --- | --- |
| [Karate_club_mlp.ipynb](./Karate_club_mlp.ipynb) | MLP: GCNConv를 Linear로 교체, 층 크기 및 tanh 유지 | 노드 특징만 사용 |
| [Karate_club_labelpropagation.ipynb](./Karate_club_labelpropagation.ipynb) | PyG LabelPropagation: alpha=0.9, 전파 100회 | 그래프 연결과 학습 노드 레이블 |

기존 `ai_tutorial` 커널에서 위에서 아래로 모두 실행하세요. 추가 패키지 설치는 필요하지 않습니다. GCN과 MLP는 Adam(lr=0.01)으로 401회 갱신하고, 기본 `train_mask`의 4개 노드만 학습에 사용합니다. 나머지 30개 노드에서 **Accuracy와 Macro-F1**을 계산합니다. LabelPropagation 입력은 학습 노드만 one-hot이며, 평가 노드의 정답은 전파에 사용하지 않습니다.

GCN과 MLP는 고정된 데이터 분할에서 초기화 시드 10개로 평균과 표본 표준편차를 보고합니다. 결정적인 LabelPropagation은 1회 실행합니다. 검증 집합이나 조기 종료 없이 미리 정한 설정과 마지막 모델을 평가하며, 이 결과로 테스트 집합에 맞춰 설정을 선택하지 않도록 합니다. KarateClub의 특징은 노드 ID를 나타내는 단위 행렬이므로, 특징만 사용하는 MLP의 미관측 노드 성능을 해석할 때 이 제약을 고려해야 합니다.

노트북에는 혼동 행렬, 정답/예측 그래프 및 추가 시각화가 포함됩니다. 실행 결과는 `results/mlp/`와 `results/labelpropagation/`에 저장합니다. 각각 `metrics.csv`(실행별 지표), `predictions_seed1234.csv`(대표 시드의 노드별 예측), `summary.json`(설정·버전·평균 성능)을 생성합니다. 재실행하면 해당 파일을 갱신합니다.

## 참고 자료

- [PyTorch Geometric 설치 안내](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
- [KarateClub 데이터셋 문서](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.KarateClub.html)
- [GCNConv 문서](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html)
- [Matplotlib 애니메이션 안내](https://matplotlib.org/stable/users/explain/animations/animations.html)
- [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

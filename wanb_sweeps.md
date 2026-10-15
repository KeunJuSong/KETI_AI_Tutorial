# Karate Club GNN: W&B Sweep 실행 안내

현재 `hpt_wandb` 브랜치의 코드와 YAML을 기준으로 2026-10-15에 점검했습니다. 프로젝트 루트에서 실행하세요. 파일명 `wanb_sweeps.md`는 기존 링크 호환을 위해 유지합니다.

## 1. 환경 준비

```powershell
conda activate ai_tutorial
python -m pip install -r requirements-sweep.txt
# 기존 로그인이 없을 때 실행
wandb login
```

환경 생성과 노트북 도구 설치는 [README](./README.md#실행-환경-준비)를 참고하세요. `configs/config.yaml`의 로그·체크포인트 경로는 KETI PC의 절대 경로이므로 다른 PC에서는 `logs`, `checkpoints` 같은 경로로 수정해야 합니다.

## 2. Sweep 등록

```powershell
python -m wandb sweep --entity kjsong --project karate-club-gcn-sweep configs/sweeps/gcn_config.yaml
```

출력된 `kjsong/karate-club-gcn-sweep/SWEEP_ID`를 사용합니다. 본인 계정·팀으로 실행하려면 등록 명령과 Sweep YAML의 `--wandb_entity`를 함께 변경하세요. 프로젝트를 바꾸면 등록 명령, YAML의 `--wandb_project`, `parameters.wandb_project.value`도 맞춰야 합니다.

등록된 Sweep은 등록 당시 설정을 사용합니다. YAML을 수정한 뒤에는 새로 등록해 새 ID를 사용하세요.

현재 명령은 `${env}`, `${interpreter}`, `${program}`으로 `wandb_sweep_Karate_club_gnn.py`를 직접 실행합니다. dotted key 파라미터는 `wandb.init()` 후 `wandb.config`에서 읽고 `utils/wandb.py`가 중첩 YAML 설정에 적용합니다. 파서가 받지 않는 dotted key를 CLI에 전달하지 않도록 `${args}`는 생략되어 있습니다.

등록할 YAML은 모델에 따라 선택합니다. 세 모델 모두 현재 W&B 프로젝트 `karate-club-gcn-sweep`을 사용합니다.

| 모델 | 등록할 설정 | CLI 모델명 |
| --- | --- | --- |
| GCN | `configs/sweeps/gcn_config.yaml` | `GCN` |
| GAT | `configs/sweeps/gat_config.yaml` | `GAT` |
| GraphSAGE | `configs/sweeps/graphsage_config.yaml` | `GraphSAGE` |

```powershell
python -m wandb sweep --entity kjsong --project karate-club-gcn-sweep configs/sweeps/gat_config.yaml
python -m wandb sweep --entity kjsong --project karate-club-gcn-sweep configs/sweeps/graphsage_config.yaml
```

공통 모델 기본값은 `configs/gnn_model_config.yaml`에 있습니다. 각 Sweep은 동일한 공통 스크립트에 해당 `--model_type`을 전달합니다.

## 3. Agent 실행

```powershell
# 실행 횟수 1회
python -m wandb agent --count 1 kjsong/karate-club-gcn-sweep/SWEEP_ID
# 실행 횟수 15회
python -m wandb agent --count 15 kjsong/karate-club-gcn-sweep/SWEEP_ID
```

먼저 `ai_tutorial` 환경을 활성화하세요. Windows에서 Bash나 SLURM은 필요하지 않습니다. `--count`는 해당 agent의 실행 수이며 전체 Sweep의 실행 수 제한과는 별개입니다.

Bash 환경에서는 다음 도우미를 사용할 수 있습니다.

```bash
bash launch_gnn.sh
bash launch_sweep.sh kjsong/karate-club-gcn-sweep/SWEEP_ID
```

`launch_gnn.sh`는 W&B를 활성화한 단일 학습이며 추가 CLI 인자를 받습니다. `launch_sweep.sh`는 첫 인자로 Sweep 경로를 받아 agent를 15회 실행합니다.

## 4. 현재 탐색 범위

탐색 방법은 `bayes`, 지표는 `best_train_loss`, 목표는 `minimize`입니다.

| 파라미터 | 범위 / 분포 |
| --- | --- |
| `models.GCN.num_layers` | 2, 3, 4 |
| `models.GCN.hidden_channels` | 4, 8, 16, 32, 64 |
| `models.GCN.dropout` | uniform: 0.0~0.3 |
| `optimizer.Adam.lr` | log_uniform_values: `1e-4`~`3e-3` |
| `optimizer.Adam.weight_decay` | log_uniform_values: `1e-6`~`1e-2` |
| `training.gradient_clip_val` | uniform: 0.5~2.0 |

기본 최대 학습은 500 epoch, 학습 코드의 patience는 80입니다. Gradient clipping은 기본 norm 방식이며 실제 역전파 후 적용됩니다.

Hyperband는 `min_iter: 500`입니다. 스크립트는 매 epoch 최적화 지표를 기록하므로 첫 판단 기준이 최대 학습 길이인 500회 기록에 해당합니다. 현재 설정에서는 학습 도중 자원 절감 효과가 제한적이며 patience는 별도로 먼저 종료를 유발할 수 있습니다. 더 이른 Sweep 종료를 원하면 `min_iter`를 줄이고 새 Sweep을 등록하세요. 실제 Hyperband 판단은 W&B가 수행합니다.

최적화 지표는 학습 노드 4개의 손실입니다. Sweep 스크립트는 검증/테스트 손실·정확도를 계산하지 않으므로 선택한 설정이 미학습 노드 분류 성능에서도 최적이라는 의미는 아닙니다.

세 모델의 계층 수·hidden 차원·dropout 탐색 범위는 위 GCN 표와 같으며 키의 모델명 부분만 달라집니다. optimizer와 clipping 범위도 공통입니다. 추가 탐색은 다음과 같습니다.

| 모델 | 추가 파라미터 | 후보 |
| --- | --- | --- |
| GAT | `models.GAT.heads` | 2, 4, 8, 16, 32 |
| GraphSAGE | `models.GraphSAGE.aggr` | `mean`, `sum`, `std`, `max`, `[mean, std]`, `[mean, sum]`, `[mean, sum, std]` |

## 5. 로그와 체크포인트

| 항목 | 현재 동작 |
| --- | --- |
| W&B epoch 로그 | `epoch`, `loss`, `train_loss`, `train_accuracy`, `best_train_loss` |
| Summary | `best_train_loss`, `best_epoch`, `epochs_trained`, `best_model_path`, `checkpoint_dir` |
| W&B 실행 로그 | 설정된 `logging_dir` 아래 `wandb/` |
| 기본 W&B 체크포인트 | `checkpoints/<MODEL>-<run_id>-<run_id>/best.pt` |
| 사용자 지정 실행 이름 | `checkpoints/<NAME>-<run_id>/best.pt` |
| W&B 없는 실행 | 기본 `checkpoint_dir`의 `best.pt` |

체크포인트에는 최적 epoch, 적용된 config, 모델 클래스·생성 인자, 모델 및 optimizer 상태, `best_train_loss`가 저장됩니다. 손실이 개선될 때 저장하며 `last.pt`는 생성하지 않습니다. 기본 실행 이름과 경로 처리에 각각 run ID가 들어가 기본 경로에는 ID가 두 번 포함됩니다.

`loss`는 dropout이 켜진 업데이트 전 손실입니다. `train_loss`와 체크포인트 선택 지표는 업데이트 후 `model.eval()` 상태의 손실입니다. YAML에 남아 있는 batch·샘플 수·검증·저장 주기 항목 중 현재 사용하지 않는 값은 [README](./README.md#yaml-기반-단일-학습)에 정리했습니다.

## 6. 선택한 설정으로 노트북 재학습

1. 사용할 실행의 로컬 W&B YAML을 준비합니다. 노트북 기본 경로 `wandb/sweep-utsldekv/config-uh7mds6z.yaml`는 개인 실행 결과입니다.
2. [Karate_club_gnn_sweep_validation.ipynb](./Karate_club_gnn_sweep_validation.ipynb)의 `path`를 실제 파일로 수정합니다.
3. 현재 선택된 GraphSAGE 외에 GCN·GAT를 사용하려면 모델 생성 셀에서 해당 `config_kwargs`와 생성 코드의 주석을 전환합니다.
4. `ai_tutorial` 커널에서 모든 셀을 순서대로 실행합니다.

노트북은 W&B YAML의 각 `{value: ...}` 항목을 추출해 선택한 모델의 `models.GCN.*`, `models.GAT.*` 또는 `models.GraphSAGE.*`와 `optimizer.Adam.*`, `training.gradient_clip_val`을 사용합니다. 모델을 새로 초기화해 500 epoch 학습하고 나머지 30개 노드(`~data.train_mask`)의 손실, 정확도, 예측 클래스와 실제 레이블을 확인합니다.

`best.pt`는 로드하지 않습니다. seed는 42(스크립트 기본값은 0), clipping 값은 반올림 없이 사용하며 patience 기반 종료는 적용하지 않습니다. 따라서 기존 실행의 최적 체크포인트 평가와 차이가 있습니다. 평가 노드의 결과를 설정 선택에 반복 사용하면 독립 테스트 결과로 해석하기 어렵습니다.

## 7. 단일 실행 및 문제 해결

```powershell
python wandb_sweep_Karate_club_gnn.py
python wandb_sweep_Karate_club_gnn.py --wandb --wandb_mode offline
python wandb_sweep_Karate_club_gnn.py --model_type GAT
python wandb_sweep_Karate_club_gnn.py --model_type GraphSAGE
python wandb_sweep_Karate_club_gnn.py --help
python -c "import sys; print(sys.executable)"
```

- 빠른 확인에는 `training.max_epochs`를 줄인 별도 YAML을 만들고 `--config`로 지정하세요.
- 이전 이름 `Karate_club_gcn.py` 또는 `wandb_sweep_Karate_club_gcn.py`가 포함된 오래된 Sweep은 현재 파일을 찾지 못합니다. 현재 YAML로 새로 등록하세요.
- `unrecognized arguments`에 dotted key가 보이면 등록된 명령에 `${args}`가 남아 있는지 확인하세요.
- `FileNotFoundError`가 나면 프로젝트 루트 실행 여부, YAML 경로, 평가 노트북의 로컬 설정 파일을 확인하세요.
- `logs/`, `wandb/`, `checkpoints/`, 캐시는 Git에서 제외됩니다. 새로 clone한 저장소에는 개인 Sweep 설정 파일이 없습니다.
- API key는 `wandb login`으로 관리하고 코드나 YAML에 직접 넣지 마세요.

참고: [W&B Sweep 설정 공식 문서](https://docs.wandb.ai/models/sweeps/define-sweep-configuration).

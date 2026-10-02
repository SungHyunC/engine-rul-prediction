# N-CMAPSS DS02 구현 명세와 완료 기록 / Implementation specification and completion record

작성자 / Author: **김성현 (2021271250)**  
실행일 / Execution date: **2026-10-02**  
시간대 / Time zone: Asia/Seoul

## 범위 / Scope

DS02 수집·검사, 메모리 제한 HDF5 읽기, 병렬 특징 추출, 회귀모델, 윈도·예측 정책 민감도, 소형 1D CNN, 저장 모델 추론과 한영 결과물을 구현한다. 구현을 제출일까지 미루지 않고 10월 2일에 실행한다.

Implement DS02 acquisition/inspection, bounded HDF5 reads, parallel features, regressors, window/prediction-policy sensitivity, a compact 1D CNN, saved-model inference and bilingual deliverables. Implement on 2 October rather than deferring work to submission deadlines.

## 데이터와 출처 / Data and provenance

- 원출처 / Original: [NASA PCoE repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/), entry 17, **Turbofan Engine Degradation Simulation-2**. 기존 FD001-FD004 자료와 구별한다. / Distinct from older FD001-FD004 data.
- 원배포 / Original archive: [NASA Dataset 17 ZIP](https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip), **15,760,443,389 bytes**. 전체 묶음이며 DS02 단독 크기가 아니다. / Full bundle, not DS02 alone.
- 실제 수집 / Acquired artifact: [Figshare third-party mirror v1](https://doi.org/10.6084/m9.figshare.20436504.v1), **2,450,472,504 bytes**, 제공 MD5 / verified published MD5 `61056251b36290e11371e017eed70eac`.
- SHA-256: `47971a68b239ecb756833218a95d68ded6eb7e63ee84e86671c8b188de1ca765`.
- 미러 체크섬 검증이며 NASA 배포본과 바이트 동일성을 주장하지 않는다. / Mirror checksum verification does not establish byte equivalence with NASA's distribution.
- 논문 / Paper: Arias Chao et al. (2021). *Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics*. Data 6(1), 5. [doi:10.3390/data6010005](https://doi.org/10.3390/data6010005).

출처·수집 시각·크기·체크섬은 `data/raw/source_manifest.json`, 구조는 `output/data_inspection.json`에 기록했다. 원시 파일은 버전 관리와 제출 소스에서 제외한다.

Provenance, time, size and checksums are in `data/raw/source_manifest.json`; structure is in `output/data_inspection.json`. Raw data are excluded from version control and the source submission.

| 분할 / Split | 엔진 / Engines | 행 / Rows | 사이클 / Cycles | 60/60 윈도 / Windows |
|---|---|---:|---:|---:|
| 개발 / Development | 2, 5, 10, 16, 18, 20 | 5,263,447 | 446 | 87,510 |
| 시험 / Test | 11, 14, 15 | 1,253,743 | 202 | 20,796 |
| 합계 / Total | 9 engines | 6,517,190 | 648 | 108,306 |

개발 엔진은 모두 비행 클래스 3이다. 시험 엔진 11은 클래스 3, 14는 클래스 1, 15는 클래스 2다. 클래스별 시험 엔진이 한 대뿐이므로 오차 차이를 클래스의 인과 효과로 해석하지 않는다.

All development engines have flight class 3. Test engines 11, 14 and 15 have classes 3, 1 and 2. One test engine per class cannot identify a causal effect of class on error.

## 공통 평가와 누수 방지 / Shared evaluation and leakage controls

1. 입력은 `W` 4채널과 `Xs` 14채널이다. `T`, `hs`, `Xv`, 엔진·사이클 ID, 비행 클래스와 `Y`는 특징으로 넣지 않는다. / Use only W and Xs; exclude degradation, health, virtual-sensor, ID, class and target metadata.
2. 엔진·비행 경계에서 윈도를 초기화한다. 미래 샘플과 다음 비행을 포함하지 않고 불완전 꼬리는 제외한다. / Reset at engine/flight boundaries; exclude future samples, the next flight and incomplete tails.
3. 개발 6개 엔진의 `GroupKFold(3)`를 사용하며 학습되는 변환은 각 학습 부분에서만 적합한다. / Use development-engine GroupKFold(3); fit transforms only on each training partition.
4. 상대 윈도 가중치는 `1 / (windows_in_cycle * cycles_in_engine)`이며 평균 1로 정규화한다. 엔진 총 가중치와 엔진 내 사이클 총 가중치를 각각 같게 한다. / Equal engine and within-engine cycle weight, normalized to mean-one window weights.
5. 같은 엔진·사이클의 예측을 평균하고 정답 일관성을 검사한다. **기록된 비행 종료 시점 추정**이며 순간 온라인 성능이 아니다. / Average within engine/cycle and verify targets; interpret at recorded flight end.
6. 엔진별 사이클 MAE·RMSE를 계산한 뒤 엔진 간 단순평균한다. 선택 기준은 개발 macro engine cycle RMSE다. / Select using the unweighted mean of development engine-level cycle RMSE.

### 시험 열람 이력 / Test inspection history

원래 60/60 실험에서 시험 엔진 11·14·15와 음수 예측을 이미 확인했다. 원래 결과는 `output/results`에 보존한다. 이후 확장은 개발 자료로 설정을 선택하지만 동일 시험의 재사용은 **기술적·탐색적 비교**다. 새로운 독립 시험 또는 확증적 일반화 검증으로 표현하지 않는다. 비음수 정책 추가 자체는 최초 시험을 본 뒤 결정했다.

The original 60/60 experiment already exposed the test engines and negative predictions. Preserve its results under `output/results`. Extensions select settings on development data, but the reused-test comparison is **descriptive and exploratory**, not a newly untouched or confirmatory evaluation. The decision to add a nonnegative policy followed the original test inspection.

## 구현 구성 / Components

### 기준 실험 / Baseline

- 60샘플, stride 60; 채널별 평균·표준편차(ddof=0)·최소·최대·샘플당 선형 기울기의 90개 특징. / 60-sample windows, stride 60 and 90 statistical features.
- 평균 기준모델, 가중 StandardScaler와 Ridge(alpha=10), HGB(max_iter=100, max_leaf_nodes=15, l2_regularization=1, early_stopping=False).
- 개발 CV로 Ridge 선택. 원래 시험 macro RMSE **11.276**, MAE **9.878** 사이클, 평균 기준모델 RMSE **20.104**. HGB 시험 RMSE **11.227**이 낮아도 시험 점수로 재선택하지 않았다. / Ridge selected by development CV; test rankings did not revise the selection.
- Ridge는 202개 시험 사이클 중 15개에서 음수를 예측했다. 원래 결과는 clipping 없이 보존한다. / Retain the 15 negative cycle predictions and original unmodified results.

### 윈도와 정책 민감도 / Window and policy sensitivity

- (window,stride): **(30,15), (30,30), (60,30), (60,60), (120,60), (120,120)**.
- Ridge/HGB × 6설정 × 원시/비음수 정책 = **24개 후보**. `max(0,prediction)`을 **윈도에 먼저 적용한 뒤 비행 평균**을 계산한다. / Two estimators × six settings × two policies; transform windows before aggregation.
- 원래 60/60의 엔진 폴드를 고정한다. 개발 CV로 선택한 후보만 6개 개발 엔진에 재적합하고 기존 시험을 기술적으로 평가한다. / Anchor original engine folds; refit the development-selected candidate and evaluate descriptively on reused test engines.
- 결과 / Outputs: `output/extended/sensitivity/`.

### 소형 1D CNN / Compact 1D CNN

- 18채널 원시 60샘플 윈도, stride 60; 학습 폴드에서 가중 채널 스케일링 적합. / Raw 18-channel windows with training-only weighted channel scaling.
- 개발 3폴드와 전체 재적합 각각 **12 epoch**, batch 512, learning rate 0.001, target scale 100. 조기 종료나 시험 점수 튜닝 없음. / Fixed training budget, no early stopping or test-score tuning.
- 실제 장치·구조·학습 이력·정규화와 체크포인트를 저장한다. / Save actual device, architecture, history, normalization and checkpoint.
- 결과 / Outputs: `output/extended/cnn/`.

### 병렬 처리 / Parallel processing

개발 전체 자료의 6개 고유 작업자/청크 설정을 각 3회 측정했다. 전체 warmup 후 설정 순서를 섞고 새 프로세스로 실행했다. 순서 있는 열 스키마·자료형·행 지문이 모두 같았다. 시간에는 HDF5 읽기·작업자 시작(import 포함)·특징 계산·조립을 포함하고 부모 최초 import·해시·저장은 제외한다. 메모리는 100ms 간격의 프로세스 트리 RSS 합계로 공유 페이지를 중복 집계할 수 있다. OS 캐시를 비우지 않았다.

Six worker/chunk settings ran three times each after a full warmup, with shuffled settings and fresh processes. Ordered schemas, dtypes and row fingerprints matched. Timing includes HDF5 reads, worker startup/imports, features and assembly; initial parent imports, hashing and saving are excluded. Process-tree RSS is sampled every 100 ms and can double-count shared pages. OS cache is not flushed.

최종 코드의 120,000행 청크 중앙값은 작업자 1개 **2.153초**, 4개 **1.473초**, 8개 **1.621초**다. 4개 속도비 **1.461**가 가장 높다. 4개 설정의 한 반복은 **2.285초**였으며 최소·최대·중앙값을 모두 보존한다. 특정 병목 원인이나 콜드 디스크 성능을 증명하는 실험은 아니다.

For the final code at 120,000-row chunks, median times were **2.153**, **1.473**, **1.621 s** for one, four and eight workers. Four gave the best speedup, **1.461**. One four-worker repeat took **2.285 s**; minimum, maximum and median are retained. These are not cold-disk or causal bottleneck measurements.

## 완료 체크리스트 / Completion checklist

- [x] 실제 수집·체크섬·구조 검사 / Acquisition, checksum and inspection.
- [x] 제한된 읽기·엔진/사이클 경계·직렬/병렬 동일성 / Bounded reads, boundaries and output agreement.
- [x] 통계 특징·기준모델·엔진 CV·원래 결과 보존 / Features, baselines, engine CV and preserved original results.
- [x] 6개 병렬 설정의 3회 반복·시간·메모리·처리량 / Repeated parallel measurements.
- [x] 24개 후보 CV와 선택 후보 평가 / CV of 24 candidates and selected evaluation.
- [x] CNN 3폴드·전체 재적합·체크포인트 / CNN folds, refit and checkpoint.
- [x] 저장 모델 추론 CLI·정답 없는 입력 검증 / Inference CLI and target-free input validation.
- [x] 오류 분석·전체 테스트·출처/의존성 / Error analysis, full tests, provenance and dependencies.
- [x] 한영 제안서·보고서 PDF 전체 페이지 검증 / Every-page validation of bilingual proposal and report PDFs.
- [x] 한영 발표자료 검증 / Bilingual slide validation.
- [x] 최종 소스·결과 묶음 검증 / Final source/result package validation.

실제 산출물 확인 후 체크를 갱신한다. / Update checks after examining actual artifacts.

확장 결과: 민감도 후보는 **Ridge 30/15 + 비음수 정책**이 선택됐으며 개발 macro RMSE **10.778887**, 재사용 시험 RMSE **10.591274**, MAE **9.021361**이다. CNN은 **4,081개 파라미터**, **RTX 4070 Ti SUPER**, 고정 12 epoch로 실행되었다. CNN 개발 macro RMSE **6.575258**, 재사용 시험 RMSE **5.695466**, MAE **4.520590**이며 전체 후보 중 개발 RMSE가 가장 낮아 최종 추론 모델로 선택했다. 총 CNN 실행은 **74.71초**, 전체 개발 적합은 **21.03초**였다.

Extension results: sensitivity selected **Ridge 30/15 with the nonnegative policy**, development macro RMSE **10.778887**, reused-test RMSE **10.591274**, MAE **9.021361**. The **4,081-parameter CNN** ran 12 fixed epochs on an **RTX 4070 Ti SUPER**. CNN development macro RMSE was **6.575258**, reused-test RMSE **5.695466**, MAE **4.520590**. Its lowest development RMSE selected it as the final inference model. Total CNN execution was **74.71 s**, including **21.03 s** for the full-development fit.

`output/final/model_card.json`은 개발 RMSE를 기준으로 CNN을 고정한다. `output/extended_verification.json`은 체크포인트 재로딩, 원래 예측 재현, Y를 읽지 않는 시험 20,796개 윈도·202개 사이클 추론, 정답 배열 없는 입력 476개 윈도·2개 사이클 검증을 기록한다. 선택용 CV는 중첩 CV의 독립 성능 추정이 아니다.

The final model card freezes CNN by development RMSE. Extended verification records checkpoint reload, matching predictions, target-free inference for 20,796 test windows/202 cycles, and an input without Y covering 476 windows/two cycles. Selection CV is not an independent nested-CV estimate.

최종 테스트 **114개 통과(17.61초)**. 최종 병렬 실험 19회는 열 스키마·자료형·순서 있는 특징 지문이 모두 일치했다. 사후 수명 구간별 오차와 최대 절대 오차 **16.008484** 사이클은 `output/final`에 보존한다.

Final tests: **114 passed in 17.61 s**. All 19 final benchmark passes matched ordered schemas, dtypes and feature fingerprints. Post-hoc life-stage errors and the maximum absolute error of **16.008484** cycles are retained under `output/final`.

## 제출 일정 / Submission dates

| 날짜 / Date | 내용 / Item |
|---|---|
| 2026-10-02 | 전체 구현과 실제 실험 실행일 / Implementation and experiment execution date |
| 2026-10-21 | 한국어·영어 각 1쪽 제안서 / One-page proposal per language |
| 2026-12-08 | 5분 발표 / Five-minute presentation |
| 2026-12-16 23:59 | 보고서와 소스 / Report and source |

이는 과제 마감이며 구현을 미루는 일정이 아니다. / These are assignment deadlines, not deferred implementation dates.

## 문서 생성 / Document generation

`tools/build_proposals.py`와 `tools/build_reports.py`는 Codex 번들 Python을 사용한다. 실제 metrics/CSV만 읽으며 수치를 만들지 않는다. ReportLab PDF는 번들 Poppler로 전체 페이지를 렌더링·시각 검증한다. 현재 번들에 LibreOffice가 없어 DOCX는 편집용 원본이며, PDF 검증이 DOCX의 페이지 배치를 보증하지 않는다. 설치된 데스크톱 LibreOffice를 대신 실행하지 않는다.

Builders use bundled document Python and measured metrics/CSV only. Every ReportLab PDF page is rendered with bundled Poppler and visually inspected. The bundle lacks LibreOffice; DOCX remains editable source, and PDF checks do not certify DOCX pagination. Installed desktop LibreOffice is not substituted.

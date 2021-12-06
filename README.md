## 1. Directory & Pre-processing

```
├── data
│   ├── train
│       └── train_#.json
│   ├── val
│       └── val_#.json
│   └── evidence
│       ├── ###_#_pre.json
│       └── ###_#_evidence.json
├── preproc
│   ├── train_#_pre.json
│   ├── val_#_pre.json
│   ├── test_#_pre.json
│   └── val_#_keyword.json
├── model
│   └── roberta_proposed
│        └── checkpoing-16883
├── models
│   └── #_#_model.pt
├── src
│   ├── main.py
│   ├── model.py
│   ├── processor.py
│   └── utils.py
├── requirements.txt
├── evidence_processing.py
├── evaluate_predictions.py
└── README.md
```

- data/train , data/val 에는 원본 논문 데이터를 랜덤샘플링한 파일들이 들어있음
- processor.py 에서 랜덤샘플링한 파일들을 가지고 데이터 전처리 → preproc 디렉토리에 ###_pre.json 파일 생성
- preproc 에는 전처리된 데이터인 ###_pre.json 파일들과 키워드 추출 모델의 결과를 포함한 ###_keyword.json 파일이 들어있음 
- data/evidence/###_pre.json 파일은 논문 데이터와 매칭되는 검색 모델 결과 데이터
- evidence_processing.py 에서 ###_#_pre.json 파일을 가지고 mrc 모델을 위한 데이터로 정제 시킴 → data/evidence 디렉토리에 ###_#_evidence.json 생성
- 학습/평가시 각 원본 데이터 파일과 해당하는 evidence.json 파일을 로드해서 사용

## 2. Model
### 2-1. 키워드 추출 모델
- 제안하는 방법으로 학습시키되, 제공받은 데이터의 10%를 랜덤샘플링한 데이터를 학습한 model
- 코드는 src 디렉토리에 저장
- 모델은 models 디렉토리에 저장
- models/1_2000_model.pt → epoch 1, step 2000

### 2-2. MRC 모델
- 제안하는 방법으로 학습시키되, 제공받은 데이터의 10%를 랜덤샘플링한 데이터를 학습한 model
- model/roberta_proposed/checkpoint-16883
- 모델 및 코드는 src 디렉토리에 저장

## 3. 실행
### 3-1. 키워드 추출 모델 
- main.py 파일 실행
- argument 설명


    - `-- train_flag` True: 모델 학습
    - `-- trian_flag` False: 모델 평가 / 데모
    - `-- data_dir` 모델의 결과 데이터 저장 위치
    - `-- save_dir` 학습된 모델 저장 위치
    - `-- load_dir` 학습된 모델 로드

- 모델 실행 결과로, 키워드 출력 레이블이 포함된 ###_keyword.json 파일 생성

### 3-2. MRC 모델
- run_mrc.py 파일 실행
- argument 설명
    
    
    - `-- output_dir` 학습된 모델 저장 위치
    - `— checkpoint` 학습된 모델 checkpoint 16883
    - `— from_init_weight` True: pre trained roberta weight 로드
    - `— do_train` True : 모델 학습
    - `-- do_eval` True : 모델 평가 / 데모
    - 현재는 데모용으로 평가됨, 기존 평가시 run_mrc.py 파일 내에 `load_and_evaluate` 함수 주석처리 필요

## 4. 데모

- 모델이 평가한 파일(`qa_predictions_##.json`), 원본 파일 `###_#.json`을 불러들여서 질문/정답/예측 순서로 저장하는 파일 생성
- 데모용 파일 `compare_roberta_proposed_0,json` 파일 저장

---

## 5줄요약
- `processor.py` 실행 : 랜덤샘플링한 데이터 파일로 ###_pre.json 파일 생성
- `main.py` 실행 : ###_pre.json 파일로 ###_keyword.json 파일 생성
- 검색 모델 실행 : 키워드 추출 모델이 드롭한 ###_keyword.json 파일로 ###_context.json 파일 생성
- `evidence_processing.py`실행 : 검색모델이 드롭한 ###_context.json 파일로 ###_evidence.json 파일 생성
- `run_mrc.py` 실행 : 학습 or 평가 or 데모

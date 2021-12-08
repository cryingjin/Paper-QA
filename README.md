## 키워드를 활용한 기계 독해 모델 (Machine Reading Comprehension using Keywords)
<img src = "https://user-images.githubusercontent.com/41279475/145188927-e6117cf1-8039-4686-86dd-b27164275d46.png" width="425"/> <img src = "https://user-images.githubusercontent.com/41279475/145188970-8d8a98f0-d6e5-4ec7-8892-77e5258d7e18.PNG" width="425"/>
<img src = "https://user-images.githubusercontent.com/41279475/145188988-8400b006-7a0f-4374-bfb7-c340e9f3fec3.PNG" width="425"/> <img src = "https://user-images.githubusercontent.com/41279475/145188998-8ff7082c-607b-4338-87f1-580b57c96c23.PNG" width="425"/>
- [1. Directory and Pre-processing](#1-directory-and-pre-processing)
  * [1-1. 키워드 추출(Keyword Extraction) 모델](#1-1--------keyword-extraction----)
  * [1-2. 기계 독해(Machine Reading Comprehension; MRC) 모델](#1-2-------machine-reading-comprehension--mrc----)
- [2. Model](#2-model)
  * [2-1. 키워드 추출(Keyword Extraction) 모델](#2-1--------keyword-extraction----)
  * [2-2. 기계 독해(MRC) 모델](#2-2-------mrc----)
- [3. How to use](#3-how-to-use)
  * [3-1. 키워드 추출(Keyword Extraction) 모델](#3-1--------keyword-extraction----)
  * [3-2. 기계 독해(MRC) 모델](#3-2-------mrc----)
***
## Training Enviroment
- 각 모델 설정값 참고
- RTX 8000 x 1
- CUDA 10.2
- [huggingface](https://huggingface.co/) 코드 참고
## 1. Directory and Pre-processing
### 1-1. 키워드 추출(Keyword Extraction) 모델
```
├── data
│   ├── train
│   │   └── train_#.json
│   ├── val
│   │   └── val_#.json
│   └── evidence
│       ├── ###_#_pre.json
│       └── ###_#_evidence.json
├── model
│   └── roberta_proposed
│        └── checkpoing-16883
├── src
│   ├── functions
│   │   ├── evaluate.py
│   │   ├── modules.py
│   │   ├── mrc_metrics.py
│   │   ├── processor_plus.py
│   │   └── utils.py
│   │
│   └── model
│       ├── main_functions.py
│       ├── models.py
│       └── roberta_model.py
│ 
├── ir
│   ├── evidence_sent_retrieval.py
│   └── processor.py
│
├── requirements.txt
├── evidence_processing.py
└── README.md
```
- [설명 추가]

### 1-2. 기계 독해(Machine Reading Comprehension; MRC) 모델
```
├── data
│   ├── train
│   │   └── train_#.json
│   ├── val
│   │   └── val_#.json
│   └── evidence
│       ├── ###_#_pre.json
│       └── ###_#_evidence.json
├── model
│   └── roberta_proposed
│        └── checkpoing-16883
├── src
│   ├── functions
│   │   ├── evaluate.py
│   │   ├── modules.py
│   │   ├── mrc_metrics.py
│   │   ├── processor_plus.py
│   │   └── utils.py
│   │
│   └── model
│       ├── main_functions.py
│       ├── models.py
│       └── roberta_model.py
│ 
├── ir
│   ├── evidence_sent_retrieval.py
│   └── processor.py
│
├── requirements.txt
├── evidence_processing.py
└── README.md
```
- data/train , data/val 에는 원본 논문 데이터를 랜덤샘플링(5%,10%)한 파일들이 들어있음
- processor.py 에서 랜덤샘플링한 파일들을 가지고 데이터 전처리 → preproc 디렉토리에 ###_pre.json 파일 생성 
- data/evidence/###_pre.json 파일은 논문 데이터와 매칭되는 검색 모델 결과 데이터
- evidence_processing.py 에서 ###_#_pre.json 파일을 가지고 mrc 모델을 위한 데이터로 정제 시킴 → data/evidence 디렉토리에 ###_#_evidence.json 생성
- 학습/평가시 각 원본 데이터 파일과 해당하는 evidence.json 파일을 로드해서 사용
***
## 2. Model
### 2-1. 키워드 추출(Keyword Extraction) 모델
- 제안하는 방법으로 학습시키되, 제공받은 데이터의 10%를 랜덤샘플링한 데이터를 학습한 model
- models/1_2000_model.pt
- 사용 코드는 src 디렉토리에 저장

### 2-2. 기계 독해(MRC) 모델
- 제안하는 방법으로 학습시키되, 제공받은 데이터의 10%를 랜덤샘플링한 데이터를 학습한 model
- model/roberta_proposed/checkpoint-16883
- 사용 코드는 src 디렉토리에 존재
***
## 3. How to use
### 3-1. 키워드 추출(Keyword Extraction) 모델
```
python main.py
```
- argument 설명

    - `-- train_flag` True: 모델 학습
    - `-- trian_flag` False: 모델 평가 / 데모
    - `-- data_dir` 모델의 결과 데이터 저장 위치
    - `-- save_dir` 학습된 모델 저장 위치
    - `-- load_dir` 학습된 모델 로드

- 모델 실행 결과로, 키워드 출력 레이블이 포함된 ###_keyword.json 파일 생성

### 3-2. 기계 독해(MRC) 모델
**데모영상**
![데모영상_0 (1)](https://user-images.githubusercontent.com/41279475/145186877-aa09ec79-2cb3-4b82-bdfa-f27522c3d864.gif)
```
pip install -r requirements.txt
python run_mrc.py
```
- argument 설명
    - 자세한 설정은 run_mrc.py 파일 참고
    - `-- output_dir` 학습된 모델 저장 위치
    - `-- checkpoint` 학습된 모델 checkpoint 16883
    - `-- from_init_weight` True: pre trained roberta weight 로드
    - `-- do_train` True : 모델 학습
    - `-- do_eval` True : 모델 평가 
    - `-- do_predict` 데모 실행, 현재 디폴트 값
---

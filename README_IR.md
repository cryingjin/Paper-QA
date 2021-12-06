## 1. Directory & Indexing

```
├── data
│   └── ###_keyword.json
├── documents
├── index
│   ├── document_index
│   ├── paragraph_index
│   └── title_index
├── src
│   └── lucene
│        ├── document_index.py
│        ├── evidence_sent_retrieval.py
│        └── paper_documents.py
├── config.py
└── README.md
```

- data에는 키워드 추출 모델의 결과를 반영한 ###_keyword.json 파일이 들어있음
- paper_documents.py에서 ###_keyword.json 파일을 가지고 논문의 본문 내용을 세 문장씩(현재 문장 + 다음 두 문장) 문장 구분 및 형태소 분석하여 문서화 → documents 디렉토리에 저장
- index 디렉토리에는 document_index.py를 실행한 결과 파일이 들어있음

## 2. Model
- 논문 데이터의 본문 내용을 문서화한 후, 색인하여 검색하는 model
- 코드는 src 디렉토리에 저장

## 3. 실행
- paper_documents.py 실행 → 문서 색인
- 실행 결과로, documents 디렉토리 내 문서들을 색인한 결과 파일들이 index 디렉토리에 저장

- evidence_sent_retrieval.py 실행 → paragraph 검색
- query를 입력하여 검색된 paragraph들 중, 키워드가 포함되어 있는 상위 10개의 paragraph 출력
- 실행 결과로, filtered context를 포함하는 ###_context.json 파일 생성

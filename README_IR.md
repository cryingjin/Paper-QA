## 1. Directory & Indexing

```
├── data
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

- data 디렉토리에는 원본 논문 데이터가 들어있음
- documents 디렉토리에는 paper_documents.py를 실행한 결과인 세 문장씩(현재 문장 + 다음 두 문장) 구분 및 형태소 분석한 파일이 들어있음
```
셋째, 본 연구에서는 온라인과 오프라인 방식을 모두 사용하여 설문조사를 하였다.
셋째/NR+,/SP 보/VV+ㄴ/ETM 연구/NNG+에서/JKB+는/JX 온라인/NNG+과/JC 오프라인/NNG 방식/NNG+을/JKO 모두/MAG 사용/NNG+하/NNG+여/XSN 설문/NNG+조사/NNG+를/JKO 하/VV+았/EP+다/EF+./SF

기존 많은 연구들이 오프라인과 온라인 설문결과 차이가 발생할 수도 있다고 함에도 불구하고 본 연구에서는 이를 고려하지 않았다.
기존/NNG 많/VA+은/ETM 연구/NNG+들/XSN+이/JKS 오프라인/NNG+과/JC 온라인/NNG 설문/NNG+결과/NNG 차이/NNG+가/JKS 발생/NNG+하/XSV+ㄹ/ETM 수/NNB+도/JX 있/VV+다고/EC 함/NNG+에/JKB+도/JX 불구/XR+하/XSA+고/EC 보/VV+ㄴ/ETM 연구/NNG+에서/JKB+는/JX 이/NP+를/JKO 고려/NNG+하/XSV+지/EC 않/VX+았/EP+다/EF+./SF

또한 설문 샘플링을 서울과 경기지역에서 했는데 조사된 성별, 나이, 교육수준, 가구형태가 대표성을 갖는지에 대한 검정이 부족하였다.
또한/MAJ 설문/NNG 샘플링/NNG+을/JKO 서울/NNP+과/JKB 경기/NNG+지역/NNG+에서/JKB 하/VV+었/EP+는데/EC 조사/NNG+되/XSV+ㄴ/ETM 성별/NNG+,/SP 나이/NNG+,/SP 교육/NNG+수준/NNG+,/SP 가구/NNG+형태/NNG+가/JKS 대표/NNG+성/XSN+을/JKO 갖/VV+는지/EC+에/JKB 대하/VV+ㄴ/ETM 검정/NNG+이/JKS 부족/NNG+하/XSV+았/EP+다/EF+./SF
```
- index 디렉토리에는 document_index.py를 실행한 결과 파일이 들어있음

## 2. Model
- 논문 데이터의 본문 내용을 문서화한 후, 색인하여 검색하는 model
- 코드는 src 디렉토리에 저장

## 3. 실행
### 3-1. 전처리
- paper_documents.py 실행
- 실행 결과로, documents 디렉토리에 Passage로 나누어진 파일들 저장
### 3-2. 문서 색인
- paper_documents.py 실행
- 실행 결과로, documents 디렉토리 내 문서들을 색인한 결과 파일들이 index 디렉토리에 저장
### 3-3. paragraph 검색
- evidence_sent_retrieval.py 실행 
- query를 입력하여 검색된 Passage 중, 키워드가 포함되어 있는 상위 10개의 Passage 출력
- 실행 결과로, data 디렉토리에 filtered context를 포함하는 ###_context.json 파일 생성

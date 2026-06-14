# LLM

LangChain 기반으로 LLM 활용 기술(프롬프트·캐싱·멀티모달·LCEL·RAG)을 학습한 기록.

## 목차

1. [LLM 기초](#01-llm-기초)
2. [프롬프트 캐싱](#02-프롬프트-캐싱)
3. [멀티모달](#03-멀티모달)
4. [LCEL](#04-lcel)
5. [이미지 생성](#05-이미지-생성)
6. [RAG](#06-rag)
7. [RAG-Anything](#07-rag-anything)
8. [Google Gemini](#08-google-gemini)

---

## 01. LLM 기초

### ChatOpenAI

OpenAI의 대화형 모델과 상호작용하는 표준 인터페이스. 단순 텍스트 완성이 아닌 **대화(Chat) 형식**에 최적화된 모델을 호출한다.

- `model_name`: 사용할 모델명.
- `temperature` (0.0~2.0): **낮으면(0.1)** 가장 확률 높은 답을 골라 일관·정확 (코딩·사실 확인), **높으면(0.9)** 다양한 단어로 창의적 (소설·아이디어).

### LogProb

토큰이 예측될 확률을 로그로 표현한 지표. 확률 1.0은 `log(1.0)=0`, 낮아질수록 `-1, -5`처럼 음수가 된다. **0에 가까울수록 모델이 그 토큰에 확신**을 가졌다는 뜻으로, 답변의 확신도·망설임을 분석할 때 쓴다.

### 스트리밍 출력

| 구분 | `invoke` (일괄) | `stream` (스트리밍) |
|------|----------------|---------------------|
| 작동 | 100% 완성 후 한꺼번에 반환 | 첫 조각 생성 즉시 실시간 전송 |
| 체감 속도 | 느림 | 매우 빠름 |
| 적합 | 결과 후처리 내부 로직 | 챗봇 UI, 긴 글 생성 |
| 캐싱 | 쉬움 (질문-답변 매칭) | 까다로움 (조각 단위) |

> `stream`이 뱉는, 길이가 일정하지 않은 토큰 조각을 **청크(chunk)**라고 한다.

## 02. 프롬프트 캐싱

반복하여 동일하게 들어가는 입력 토큰의 **비용을 아끼는** 기법. 점점 정교해지는 순서로 학습했다.

| 방식 | 매칭 | 특징 |
|------|------|------|
| **InMemoryCache** | 정확히 동일한 key | 로컬 RAM, `dict`와 유사. 유사도 분석 불가 |
| **Dict Stream Caching** | 동일 key | stream은 조각이라 다 합친 뒤 직접 저장 |
| **RedisCache** | 동일 key | Redis에서 key 조회 → 없으면 API 호출 후 캐싱 |
| **RedisSemanticCache** | **의미 일치** | 질문을 임베딩한 벡터로 유사도 검색. 단어가 달라도 의도가 같으면 캐시 히트 |

**RedisSemanticCache 흐름**: 질문 → HuggingFace 임베딩(벡터 A) → Redis 내 기존 벡터들과 거리 비교 → `score_threshold` 이내면 **캐시 히트**(LLM 호출 없이 즉시 반환), 아니면 API 호출 후 저장. (벡터 검색을 위해 `RediSearch` 모듈이 탑재된 `redis-stack` 필요)

- **`score_threshold`**: 낮을수록 더 유사해야 히트 (0 = 완전 동일). **영어는 0.2**, **한국어는 0.35~0.4**가 유사 영역이다.
- **Redis `db` 분리**: Redis는 `db=0~15`(16개)를 제공하며, 번호가 다르면 데이터가 섞이지 않는다. `decode_responses=True`로 바이트를 UTF-8 문자열로 자동 변환한다.
- **회차별 동작**: 1회차는 API 호출(느림) 후 저장 → 2회차는 동일 질문이라 캐시 히트 → 3회차 이후는 **문장이 조금 달라도 의미가 유사하면** 캐시 히트(매우 빠름).

> **직접 구현 stream semantic cache**: `RedisSemanticCache`는 `invoke`엔 잘 맞지만 `stream`은 결과가 조각조각이라 자동 캐시와 궁합이 애매하다. 그래서 스트리밍 답변을 `full_content`로 다 모은 뒤 직접 Redis에 저장하는 구조를 만들어, 검색·히트·호출·저장을 전부 직접 제어했다.

## 03. 멀티모달

이미지를 텍스트와 함께 입력해 분석한다. 이미지를 **base64로 인코딩**해 `HumanMessage`의 `image_url`로 전달한다.

```python
{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}
```

예) `톰.jpg`를 넣고 *"고양이와 강아지 중 어디에 더 가까운가?"* 를 스트리밍으로 질의.

## 04. LCEL

**LCEL(LangChain Expression Language)** — 여러 구성요소(프롬프트·모델·출력 파싱)를 리눅스 파이프(`|`)처럼 연결해 하나의 체인을 만드는 선언적 언어.

```python
chain = prompt_template | llm | StrOutputParser()
# 입력 → 프롬프트 완성 → llm 전달 → 문자열만 추출
chain.invoke({"country": "아이슬란드"})
```

### 프롬프트 템플릿

- **PromptTemplate**: 문자열 하나를 만드는 템플릿. (`"한국에서 {country}까지 비행시간 알려줘"`)
- **ChatPromptTemplate**: `system`/`human`/`ai` 채팅 메시지 목록을 만드는 템플릿.
- **MessagesPlaceholder**: 이전 대화 기록을 나중에 끼워 넣는 자리.

### OutputParser

- **StrOutputParser**: 문자열로 리턴.
- **JsonOutputParser**: JSON을 `dict`로 리턴.
- **PydanticOutputParser**: 사용자가 정의한 데이터 모델 규격에 맞춰 리턴.

### 실행·흐름 제어

- **Batch**: 여러 입력을 list로 받아 일괄 처리. (하나의 체인 ← 여러 데이터). 입력이 4개면 1·2·3을 동시에 처리하고 하나가 끝나면 4번을 처리한다.
- **Parallel**: 하나의 데이터를 여러 템플릿에 넣어 병렬 처리.
- **Async**: `ainvoke`(비동기 단일 호출, 결과는 `await`로 꺼냄), `async for`(비동기 stream)를 사용한다. 동기는 한 요청을 처리하는 동안 다른 요청이 대기하지만, 비동기는 응답을 기다리는 사이 다른 요청을 처리할 수 있다. **LLM 호출은 네트워크 대기 시간이 많아 비동기와 잘 맞는다.**
- **RunnablePassthrough**: 한 방향으로 흐르는 파이프라인에서 **기존 입력을 보존하며 새 값을 추가**하는 통로.
- **RunnableLambda**: LCEL 체인 중간에 **내 함수**를 넣어 실행.

## 05. 이미지 생성

OpenAI 이미지 모델로 이미지를 생성하고, 응답의 `b64_json`을 디코딩해 파일로 저장한다.

```python
response = openAI.images.generate(model="gpt-image-1-mini", ...)
image_bytes = base64.b64decode(response.data[0].b64_json)
```

영상 생성 모델(`sora-2`)도 함께 다뤘다.

## 06. RAG

**RAG(Retrieval-Augmented Generation, 검색 증강 생성)** — 생성형 AI가 학습하지 않은 최신 정보나 내부 데이터를 바탕으로 답변하게 돕는 기술.

- **R (Retrieval)**: 질문과 관련된 문서를 DB에서 검색.
- **A (Augmentation)**: 검색된 정보를 질문과 결합해 프롬프트 보강.
- **G (Generation)**: 보강된 맥락으로 최종 답변 생성.

### 개발 8단계

**① 문서 불러오기** — 형식별 로더로 `Document` 객체 리스트를 만든다. `Document`는 실제 텍스트인 `page_content`와 부가정보인 `metadata`로 구성된다.

```python
loader = PyMuPDFLoader("경로.pdf")   # TextLoader / CSVLoader / UnstructuredExcelLoader 등 형식별
docs = loader.load()
```

**② 문서 분할** — `RecursiveCharacterTextSplitter`로 청크 단위로 쪼갠다.

- **`chunk_size`** (조각 하나의 크기, 보통 500~800자): LLM의 기억력과 검색 정확도를 좌우한다. 너무 작으면 맥락이 잘려 이해도가 떨어지고, 너무 크면 불필요한 정보가 섞여 답변의 초점이 흐려진다.
- **`chunk_overlap`** (조각 간 겹침, `chunk_size`의 10~20%): 경계에 걸친 문장이 끊기지 않게 앞뒤를 겹쳐 자른다.
  > "피자는 느끼해 난 그래서 싫어"를 겹침 없이 자르면 `[피자는 느끼해]` / `[그래서 싫어]`로 맥락이 끊긴다. 겹쳐 자르면 `[느끼해 난 그래서]`가 공통으로 남아 문맥 손실이 줄어든다.

**③ 임베딩** — `OpenAIEmbeddings`로 텍스트를 숫자 벡터로 변환.

**④ Vector DB 생성·저장** — 분할된 문서를 전부 임베딩해 **FAISS**에 저장한다. 이후 질문도 벡터로 바꿔 유사도 검색(`similarity_search`)이 가능해진다.

**⑤ 검색기 생성** — `vectorstore.as_retriever(search_kwargs={"k": n})`

| `k` | 동작 | 트레이드오프 |
|-----|------|--------------|
| `k=1` | 핵심 정보 하나만 | 답변이 집중되나 주변 맥락 부족 |
| `k=5` | 관련 조각 5개 | 정보가 풍부하나 불필요한 문서가 섞일 수 있음 |
| 기본 `k=4` | — | — |

**⑥ 프롬프트 생성** — `Context`(검색된 문서) + `Question`(사용자 질문) + `Answer` 틀로 구성.

**⑦ LLM 생성** — 문서 기반 답변이라 창의성을 막으려 `temperature=0`으로 둔다.

**⑧ 체인 실행** — LCEL로 검색기 → 프롬프트 → LLM → 출력 파서를 연결한다.

```python
chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
```

## 07. RAG-Anything

다양한 형식의 문서를 분석해 **그래프 기반 멀티모달 지식 베이스**를 구축하는 차세대 RAG 프레임워크.

1. **Multi-modal Parsing**: 문서 안의 여러 형태 정보를 파싱.
   - 계층적 텍스트 추출(제목·본문 구조 유지), 이미지 캡션·메타데이터, **LaTeX 수식 인식**, 표 구조 파싱.
2. **Graph 기반 지식 그라운딩**: 텍스트·이미지·표·수식에서 뽑은 정보를 **지식 그래프(노드·엣지)**로 연결. VLM으로 본문↔이미지/표의 논리적 연결을 만들고, 여러 문서의 그래프를 하나로 병합(Merged KG).
3. **Hybrid Retrieval**: 한 가지 검색만 쓰지 않고 **그래프 검색(개념 관계망) + 벡터 검색(유사 청크)**을 결합해 LLM에 전달.

> 일반 RAG는 벡터 검색 결과만 전달하지만, RAG-Anything은 **벡터 + 그래프 관계 + 이미지/표/수식 해석**까지 전달한다. (단, 표 방향이 역방향이면 분석력이 떨어지고 이미지보다 텍스트를 우선하는 경향이 있다)

![RAG-Anything 프레임워크](images/rag_anything_framework.png)

## 08. Google Gemini

OpenAI 대신 **Google의 Gemini**를 LangChain으로 사용한다. `ChatGoogleGenerativeAI`로 모델을 호출하며, LCEL 체인은 동일하게 적용된다.

```python
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
chain = prompt | model
```

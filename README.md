# langchain_practice
RAG, Langchain practice
RAG→TTS Prompt Demo (LangChain + FastAPI)
========================================

이 프로젝트는 소설 TXT/MD를 업로드하면 토큰 한도를 넘지 않도록 자동으로 분할하고,
참고 자료(RAG)와 결합하여 **TTS용 프롬프트 JSON**을 생성합니다.
UI로 업로드/생성/다운로드가 가능하며, API 엔드포인트도 제공합니다.

핵심 특징
- 토큰 기준 분할(가능하면 tiktoken 사용, 불가 시 문자 기준 분할로 폴백)
- Chroma(Vector DB) 영구 저장소 사용, 텔레메트리 비활성화
- 단일 대사 생성(/tts) + 전체 TXT 배치 생성(/tts_from_txt_upload)
- 업로드 UI 제공(index.html)

--------------------------------------------------------------------------------
1) 필수 요건
--------------------------------------------------------------------------------
- Python 3.10+ 권장
- pip 최신 버전

패키지 설치는 제공된 requirements.txt로 진행합니다.

--------------------------------------------------------------------------------
2) 설치 (가상환경 권장)
--------------------------------------------------------------------------------
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# macOS / Linux (bash/zsh)
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

※ 만약 설치 중 충돌이 나면 pip를 최신으로 올리고 다시 시도하세요.
  (추가로 회사 프록시/방화벽 환경이면 pip 인덱스 접근 권한을 확인)

--------------------------------------------------------------------------------
3) .env 설정
--------------------------------------------------------------------------------
프로젝트 루트에 .env 파일을 만들고 아래 항목을 필요에 따라 채웁니다.

OPENAI_API_KEY=sk-...            # OpenAI LLM을 사용할 때 필요
OPENAI_MODEL=gpt-4o-mini         # 기본값. 원하는 모델명으로 변경 가능
USE_OPENAI_EMBEDDINGS=false      # 기본 false (로컬 e5-small-v2 임베딩 사용, 비용 0원)
CHROMA_DIR=.chroma_store         # 벡터DB 저장 폴더
ANONYMIZED_TELEMETRY=FALSE       # (권장) Chroma 텔레메트리 로그 억제

* OPENAI_API_KEY 가 없으면 /tts, /tts_from_txt_upload 호출 시 LLM 단계에서 실패(429/401)가 날 수 있습니다.
  먼저 소량(max_items로 제한)으로 동작 확인을 하거나, 로컬 LLM으로 대체 적용을 고려하세요.

--------------------------------------------------------------------------------
4) 서버 실행
--------------------------------------------------------------------------------
# 개발 서버
python -m uvicorn main1:app --reload --port 8000

브라우저에서:
- UI:   http://localhost:8000/
- 문서: http://localhost:8000/docs

업로드된 파일과 생성된 JSON은 ./uploads/sess-xxxx/ 하위에 저장되며,
응답의 download_url 또는 /download/<session>/tts_prompts.json 로 내려받을 수 있습니다.

--------------------------------------------------------------------------------
5) 사용 방법 (UI 기준)
--------------------------------------------------------------------------------
(1) 참고 자료 업로드(인덱싱)
- 최소 "소설 TXT"를 업로드하고 "인덱싱 실행"을 클릭합니다.
- 캐릭터 카드(.txt/.md), 스타일 가이드(.txt/.md), 발음 사전(.json)은 선택입니다.

(2) 단일 대사 → TTS JSON
- Speaker/Scene을 입력하고 대사를 넣은 뒤 "TTS JSON 생성" 클릭.
- 응답 창에서 생성된 JSON을 확인합니다.

(3) 원문 TXT 전체 → 묶음 TTS JSON
- "원문 TXT 업로드"에 전체 텍스트를 올리고 "전체 TTS JSON 생성" 클릭.
- 토큰 제한을 넘지 않도록 자동 분할 → 각 발화에 대해 TTS 프롬프트 생성 →
  합쳐진 JSON을 반환합니다.
- "JSON 다운로드" 버튼으로 tts_prompts.json 파일 저장.

Tip:
- 처음에는 "최대 처리 발화 수(max items)"를 10~20으로 설정해 속도/비용을 확인하세요.
- also_ingest=true (기본)로 하면 업로드한 전체 텍스트를 벡터DB에 즉시 반영하여 검색 품질을 높입니다.

--------------------------------------------------------------------------------
6) 사용 방법 (API 예시)
--------------------------------------------------------------------------------
# PowerShell: /tts_from_txt_upload (배치)
$fd = New-Object System.Net.Http.MultipartFormDataContent
$bytes = [System.IO.File]::ReadAllBytes("novel.txt")
$file = New-Object System.Net.Http.ByteArrayContent($bytes)
$file.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::Parse("text/plain")
$fd.Add($file, "novel", "novel.txt")
$fd.Add([System.Net.Http.StringContent]::new("true"), "also_ingest")
$fd.Add([System.Net.Http.StringContent]::new("20"), "max_items")  # 테스트용
Invoke-RestMethod -Uri "http://localhost:8000/tts_from_txt_upload" -Method Post -Body $fd

# curl (macOS/Linux)
curl -X POST http://localhost:8000/tts_from_txt_upload \
  -F "novel=@novel.txt" \
  -F "also_ingest=true" \
  -F "max_items=20"

# 단일 대사 생성
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"utterance":"안녕하세요? 다음 일정 공유 부탁해요","speaker_id":"S1","scene":"회의"}'

--------------------------------------------------------------------------------
7) 구성요소 요약
--------------------------------------------------------------------------------
- main1.py
  * FastAPI 서버, /ingest_upload, /tts, /tts_from_txt_upload, /download 등 엔드포인트 구현
  * tiktoken(가능 시) 기반 토큰 분할 → 폴백 문자 분할
  * Chroma PersistentClient + anonymized_telemetry=False (텔레메트리 비활성)
- index.html
  * Step1: 업로드/인덱싱
  * Step2: 단일 대사 → TTS JSON
  * Step3: 전체 TXT → 묶음 TTS JSON + 다운로드 버튼
- requirements.txt
  * FastAPI, Uvicorn, LangChain, ChromaDB, sentence-transformers 등 프로젝트 의존성

--------------------------------------------------------------------------------
8) 자주 묻는 문제/트러블슈팅
--------------------------------------------------------------------------------
● 429 insufficient_quota / 401 Unauthorized
  - OpenAI 키/결제/할당량을 확인하세요(.env의 OPENAI_API_KEY).
  - 먼저 Step 3에서 max_items로 소량만 생성해 테스트하세요.
  - 레이트리밋이 잦으면 .env에서 OPENAI_MAX_RETRIES, OPENAI_TIMEOUT 같은 값(코드 확장 시)을 조정.

● 텔레메트리 무한 로그 (posthog capture() error)
  - 본 프로젝트는 코드상 Chroma PersistentClient에 anonymized_telemetry=False를 주입하여 차단합니다.
  - 그래도 남는 환경에서는 .env에 ANONYMIZED_TELEMETRY=FALSE 를 추가하고, 서버 재시작.

● 모듈 누락/버전 충돌
  - pip install -r requirements.txt로 재설치, pip 업그레이드.
  - 사내 프록시/방화벽이면 pip 인덱스 접근 권한 확인.

● 포트 충돌 / 서버 중복 실행
  - Windows: netstat -ano | findstr :8000 → taskkill /PID <PID> /F
  - macOS/Linux: lsof -i :8000 → kill -9 <PID>

● LangChain Deprecation Warning (get_relevant_documents)
  - 향후 langchain-core 1.0에서 제거 예정 경고입니다. 추후 retreiver.invoke()로 이식하면 사라집니다.

--------------------------------------------------------------------------------
9) 폴더 구조 및 산출물
--------------------------------------------------------------------------------
/project-root
  ├─ main1.py
  ├─ index.html
  ├─ requirements.txt
  ├─ .env                      # (사용자 생성)
  ├─ .chroma_store/            # (자동) 벡터 DB 영구 저장
  └─ uploads/
       └─ sess-xxxx/           # 업로드/결과 세션 폴더
            ├─ novel_...txt
            └─ tts_prompts.json  # Step 3 결과 (UI에서 다운로드도 가능)

--------------------------------------------------------------------------------
10) 라이선스/주의
--------------------------------------------------------------------------------
- 업로드하는 텍스트의 저작권/개인정보에 유의하세요.
- OpenAI API를 사용할 경우 과금 및 정책을 사전에 확인하세요.

끝. 문제/로그가 있으면 해당 줄을 복사해서 이슈로 공유해 주세요!

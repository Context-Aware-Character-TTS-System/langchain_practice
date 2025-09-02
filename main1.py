import os
import json
import uuid
import re
import asyncio
import difflib
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from collections import Counter, defaultdict

from fastapi import FastAPI, Body, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv

# --- LangChain & VectorStore ---
try:
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    from langchain.schema import Document  # type: ignore

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# VectorStore
from langchain_community.vectorstores import Chroma

# Prompting
from langchain_core.prompts import ChatPromptTemplate

# Chroma client (텔레메트리 차단용)
import chromadb
from chromadb.config import Settings

# -------------------------------------------------
# Env & Settings
# -------------------------------------------------
load_dotenv()
BASE_DIR = Path(__file__).parent
PERSIST_DIR = Path(os.getenv("CHROMA_DIR", ".chroma_store"))
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

USE_OPENAI_EMB = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_CORPUS = os.getenv("DEFAULT_CORPUS_ID", "default")
BATCH_CONCURRENCY = int(os.getenv("BATCH_CONCURRENCY", "6"))

app = FastAPI(title="RAG→TTS Prompt Service", version="0.6.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Embeddings & LLM
# -------------------------------------------------
def get_embedding():
    if USE_OPENAI_EMB:
        return OpenAIEmbeddings(model="text-embedding-3-large")
    # 로컬 임베딩(토큰 비용 0원)
    return HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")

def get_llm():
    # OpenAI LLM만 예시 (키는 .env에)
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0)

EMB = get_embedding()
LLM = get_llm()

# -------------------------------------------------
# VectorStore (단일 컬렉션 + kind + corpus_id 메타데이터)
# -------------------------------------------------
def get_vs() -> Chroma:
    PERSIST_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(PERSIST_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    return Chroma(
        collection_name="corpus",
        embedding_function=EMB,
        persist_directory=str(PERSIST_DIR),
        client=client,
    )

# -------------------------------------------------
# In-memory corpus data
# -------------------------------------------------
# 작품별 발음 사전, 캐릭터 이름 목록
LEXICONS: Dict[str, Dict[str, str]] = defaultdict(dict)      # corpus_id -> {term: pron}
CHAR_NAMES: Dict[str, Set[str]] = defaultdict(set)            # corpus_id -> { "개구리 왕자", ... }
CASTS: Dict[str, Dict[str, str]] = defaultdict(dict)  # corpus_id -> { character_name: voice_actor }

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def load_text_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _make_splitter():
    # token 기반 → 없으면 char 기반
    try:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=900,        # ⬅ ingest 속도↑ (청크 수↓)
            chunk_overlap=120,
        )
        return splitter
    except Exception:
        return RecursiveCharacterTextSplitter(
            chunk_size=1200, chunk_overlap=150, separators=["\n\n", "\n", " "]
        )

def _sanitize_corpus_id(raw: Optional[str]) -> str:
    if not raw:
        return DEFAULT_CORPUS
    s = re.sub(r"[^\w\-]+", "_", raw.strip())
    return s or DEFAULT_CORPUS

def _meta(kind: str, corpus_id: Optional[str], extra: Optional[Dict] = None):
    m = {"kind": kind, "corpus_id": _sanitize_corpus_id(corpus_id)}
    if extra:
        m.update(extra)
    return m

def _eq(field: str, value: str) -> Dict:
    # Chroma 최신 where 문법 호환 ($and + $eq)
    return {field: {"$eq": value}}

def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "").lower()

def _best_match(name: str, corpus_id: str) -> str:
    """화자 이름을 카드의 정식 이름으로 정규화."""
    if not name or name in ("Narrator", "Unknown"):
        return name or "Unknown"
    names = list(CHAR_NAMES.get(corpus_id, set()))
    if not names:
        return name

    n = _norm(name)
    # 완전/포함 일치 우선
    for cand in names:
        cn = _norm(cand)
        if cn == n or cn in n or n in cn:
            return cand

    # 유사도 매칭(완화)
    best, score = None, 0.0
    for cand in names:
        s = difflib.SequenceMatcher(None, n, _norm(cand)).ratio()
        if s > score:
            best, score = cand, s
    if score >= 0.6:   # 0.72 → 0.6으로 완화(고전 텍스트 단축형 커버)
        return best
    return name

# -------------------------------------------------
# Doc builders
# -------------------------------------------------
def to_docs_from_text(text: str, kind: str, corpus_id: Optional[str], extra_meta: Optional[Dict] = None) -> List[Document]:
    splitter = _make_splitter()
    chunks = splitter.split_text(text)
    docs = [
        Document(page_content=c, metadata=_meta(kind, corpus_id, extra_meta))
        for c in chunks if c.strip()
    ]
    return docs

def to_docs_character_cards(raw: str, corpus_id: Optional[str]) -> List[Document]:
    """
    카드 포맷 예시:
      홍길동(성우=김OO): 정의감 강함, 존댓말 / 격정적일 땐 속도↑
      홍판서: 위압적, 낮고 느림
      길동|성우=박OO: 청년, 공손체
    """
    cid = _sanitize_corpus_id(corpus_id)
    docs: List[Document] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue

        # name_part : desc 로 분리
        if ":" not in s:
            continue
        name_part, desc = s.split(":", 1)
        name_part = name_part.strip()
        desc = desc.strip()
        if not name_part:
            continue

        # 성우=... 추출 (괄호/파이프 모두 지원)
        va = None
        m = re.search(r"성우\s*=\s*([^)|\s]+)", name_part)
        if m:
            va = m.group(1).strip()

        # 이름 정제(괄호/파이프 뒤 내용 제거)
        name = re.sub(r"\(.*?\)", "", name_part)
        name = re.sub(r"\|.*$", "", name).strip()
        if not name:
            continue

        CHAR_NAMES[cid].add(name)
        if va:
            CASTS[cid][name] = va

        docs.append(
            Document(
                page_content=desc or name,
                metadata=_meta("character_card", cid, {"character": name, "voice_actor": va} if va else {"character": name}),
            )
        )

    # 전체 텍스트도 하나 보관(검색)
    if raw.strip():
        docs.append(Document(page_content=raw, metadata=_meta("character_card", cid)))
    return docs


def to_docs_style_guides(raw: str, corpus_id: Optional[str]) -> List[Document]:
    return [Document(page_content=raw, metadata=_meta("style_guide", corpus_id))]

# -------------------------------------------------
# Core ingest
# -------------------------------------------------
def ingest_corpus_from_paths(
    novel_path: Path,
    character_cards_path: Optional[Path] = None,
    style_guides_path: Optional[Path] = None,
    lexicon_path: Optional[Path] = None,
    corpus_id: Optional[str] = None,
):
    corpus_id = _sanitize_corpus_id(corpus_id or novel_path.stem)
    vs = get_vs()

    # Novel
    novel_text = load_text_file(novel_path)
    novel_docs = to_docs_from_text(novel_text, kind="novel_chunk", corpus_id=corpus_id)
    if novel_docs:
        vs.add_documents(novel_docs)

    # Character cards
    cc_docs = []
    if character_cards_path and character_cards_path.exists():
        cc_text = load_text_file(character_cards_path)
        cc_docs = to_docs_character_cards(cc_text, corpus_id=corpus_id)
        if cc_docs:
            vs.add_documents(cc_docs)

    # Style guides
    sg_docs = []
    if style_guides_path and style_guides_path.exists():
        sg_text = load_text_file(style_guides_path)
        sg_docs = to_docs_style_guides(sg_text, corpus_id=corpus_id)
        if sg_docs:
            vs.add_documents(sg_docs)

    # Lexicon
    if lexicon_path and lexicon_path.exists():
        try:
            data = json.loads(load_text_file(lexicon_path))
            if isinstance(data, dict):
                LEXICONS[corpus_id].update(data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid lexicon JSON: {e}")

    vs.persist()
    return {
        "status": "ok",
        "corpus_id": corpus_id,
        "added": {
            "novel_chunks": len(novel_docs),
            "character_cards": len(cc_docs),
            "style_guides": len(sg_docs),
            "lexicon_terms": len(LEXICONS[corpus_id]),
            "characters": sorted(list(CHAR_NAMES.get(corpus_id, set()))),
        },
    }

# -------------------------------------------------
# Retrieval & generation
# -------------------------------------------------
def retrieve_context(utterance: str, speaker_id: str, scene: str, corpus_id: str,
                     k_novel=1, k_card=1, k_style=1) -> str:
    vs = get_vs()
    filt_novel = {"$and": [_eq("kind", "novel_chunk"),   _eq("corpus_id", corpus_id)]}
    filt_card  = {"$and": [_eq("kind", "character_card"), _eq("corpus_id", corpus_id)]}
    filt_style = {"$and": [_eq("kind", "style_guide"),   _eq("corpus_id", corpus_id)]}

    ret_novel = vs.as_retriever(search_kwargs={"k": k_novel, "filter": filt_novel})
    ret_card  = vs.as_retriever(search_kwargs={"k": k_card,  "filter": filt_card})
    ret_style = vs.as_retriever(search_kwargs={"k": k_style, "filter": filt_style})

    ctx_parts = []
    cards  = ret_card.invoke(f"{speaker_id} {utterance}")
    styles = ret_style.invoke(f"{scene} {utterance}")
    novels = ret_novel.invoke(utterance)

    ctx_parts += [d.page_content for d in cards]
    ctx_parts += [d.page_content for d in styles]
    ctx_parts += [d.page_content for d in novels]
    return "\n---\n".join(ctx_parts)

def subset_lexicon(utterance: str, corpus_id: str) -> Dict[str, str]:
    m = LEXICONS.get(corpus_id, {})
    return {k: v for k, v in m.items() if k in utterance}

PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a TTS prompt generator. Use the CONTEXT and LEXICON to set tone, style, and pronunciations. "
        "Return valid JSON with fields: speaker_id, voice_hint(gender,age), "
        "style(mood,formality,energy), tts_params(rate,pitch,volume), "
        "pronunciation_overrides, text. Output JSON only. No comments.",
    ),
    (
        "human",
        "CONTEXT:\n{context}\n\nLEXICON(JSON):\n{lexicon}\n\n"
        "INPUT:\nSpeaker={speaker_id}\nSceneTag={scene}\nUtterance={utterance}\n\n"
        "Generate the JSON.",
    ),
])

def _postprocess_speaker_id(raw_json: Dict, inferred: str, corpus_id: str) -> Dict:
    fixed = _best_match(inferred, corpus_id)
    raw_json["speaker_id"] = fixed
    # 캐스팅 정보 동봉(있으면)
    va = CASTS.get(corpus_id, {}).get(fixed)
    if va:
        raw_json["voice_actor"] = va
    return raw_json

def generate_tts_prompt_sync(utterance: str, speaker_id: str, scene: str, corpus_id: str) -> Dict:
    context = retrieve_context(utterance, speaker_id, scene, corpus_id=corpus_id)
    overrides = subset_lexicon(utterance, corpus_id=corpus_id)

    chain = PROMPT | LLM
    result = chain.invoke(
        {
            "context": context,
            "lexicon": json.dumps(overrides, ensure_ascii=False),
            "speaker_id": speaker_id,
            "scene": scene,
            "utterance": utterance,
        }
    ).content

    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        data = {
            "speaker_id": speaker_id,
            "voice_hint": {"gender": "unknown", "age": "unknown"},
            "style": {"mood": "neutral", "formality": "기본", "energy": "medium"},
            "tts_params": {"rate": 1.0, "pitch": 0, "volume": 0},
            "pronunciation_overrides": overrides,
            "text": utterance,
            "_raw": result,
        }
    return _postprocess_speaker_id(data, speaker_id, corpus_id)

async def generate_tts_prompt_async(utterance: str, speaker_id: str, scene: str, corpus_id: str) -> Dict:
    # 동시성용(속도↑)
    context = retrieve_context(utterance, speaker_id, scene, corpus_id=corpus_id)
    overrides = subset_lexicon(utterance, corpus_id=corpus_id)
    res = await (PROMPT | LLM).ainvoke({
        "context": context,
        "lexicon": json.dumps(overrides, ensure_ascii=False),
        "speaker_id": speaker_id,
        "scene": scene,
        "utterance": utterance,
    })
    try:
        data = json.loads(res.content)
    except Exception:
        data = {
            "speaker_id": speaker_id,
            "voice_hint": {"gender": "unknown", "age": "unknown"},
            "style": {"mood": "neutral", "formality": "기본", "energy": "medium"},
            "tts_params": {"rate": 1.0, "pitch": 0, "volume": 0},
            "pronunciation_overrides": overrides,
            "text": utterance,
            "_raw": res.content,
        }
    return _postprocess_speaker_id(data, speaker_id, corpus_id)

# -------------------------------------------------
# Advanced utterance parser: split quotes vs narration, infer speaker
# -------------------------------------------------
# 따옴표 쌍 정의
_QUOTE_PAIRS = {
    "“": "”", '"': '"', "‘": "’", "'": "'", "「": "」", "『": "』",
}

def _detect_speaker_name(pre_ctx: str, post_ctx: str) -> Optional[str]:
    _SAID_VERB = r'(?:말했|말하|중얼|외쳤|물었|아뢰었|아뢰였|대답했|속삭였|소리쳤|되물었|덧붙였|응수했|부르|불렀)'
    # "…"(라고) <이름> (가|이|는|은) <말했…>
    m = re.search(rf'(?:라고|라며|하며|하고)?\s*([가-힣A-Za-z0-9 ]{{1,20}}?)(?:가|이|는|은)\s*{_SAID_VERB}', post_ctx)
    if m:
        cand = m.group(1).strip()
        if cand not in {"그", "그녀", "누군가", "사람", "아이"}:
            return cand
    # <이름> (가|이|는|은) <말했…> "…"
    m = re.search(rf'([가-힣A-Za-z0-9 ]{{1,20}}?)(?:가|이|는|은)\s*{_SAID_VERB}\s*(?:며|라고)?\s*$', pre_ctx)
    if m:
        cand = m.group(1).strip()
        if cand not in {"그", "그녀", "누군가", "사람", "아이"}:
            return cand
    return None

def _merge_adjacent(items: List[Dict]) -> List[Dict]:
    """연속된 나레이션은 붙여서 덩어리 줄이기."""
    out: List[Dict] = []
    for it in items:
        if out and it["speaker_id"] == "Narrator" and out[-1]["speaker_id"] == "Narrator" and it["scene"] == out[-1]["scene"]:
            out[-1]["utterance"] += (" " if out[-1]["utterance"] and not out[-1]["utterance"].endswith("\n") else "") + it["utterance"]
        else:
            out.append(it)
    return out

def _parse_with_fsm(text: str) -> List[Dict]:
    """줄바꿈을 포함해 따옴표 내부는 '한 화자 한 덩어리'로 수집하는 FSM 파서."""
    lines = text.splitlines()
    scene = "default"
    items: List[Dict] = []

    in_quote = False
    q_close = ""
    quote_buf: List[str] = []
    # 나레이션은 라인 단위로 모으되, 연속이면 merge 단계에서 합쳐짐
    for raw in lines:
        line = raw.rstrip("\n")
        s = line.strip()

        # 따옴표 밖에서만 씬/명시화자 처리
        if not in_quote:
            # 씬 헤더
            m = re.match(r'^(#+)\s+(.+)', s)
            if m:
                scene = re.sub(r'^(#+)\s+', '', s)
                continue
            if re.match(r'^(\d+\s*장\.?|CHAPTER\s+\d+)', s, flags=re.I):
                scene = s
                continue
            # 명시화자: S1: ...
            m = re.match(r'^(S[\w\d]+)\s*:\s*(.+)$', s)
            if m:
                spk, utt = m.group(1), m.group(2).strip()
                if utt:
                    items.append({"speaker_id": spk, "scene": scene, "utterance": utt})
                continue

        i = 0
        pre_ctx_tail = ""  # 따옴표 직전 컨텍스트
        while i < len(line):
            ch = line[i]
            # 따옴표 시작
            if not in_quote and ch in _QUOTE_PAIRS:
                q_close = _QUOTE_PAIRS[ch]
                in_quote = True
                pre_ctx_tail = line[max(0, i-40):i]
                i += 1
                quote_buf = []
                continue
            # 따옴표 내부
            if in_quote:
                if ch == q_close:
                    # 따옴표 닫힘 → 한 화자 한 덩어리
                    qtext = "".join(quote_buf).strip()
                    post_ctx = line[i+1:i+1+40]
                    spk = _detect_speaker_name(pre_ctx_tail, post_ctx) or "Unknown"
                    items.append({"speaker_id": spk, "scene": scene, "utterance": qtext})
                    in_quote = False
                    q_close = ""
                    quote_buf = []
                    i += 1
                    continue
                else:
                    quote_buf.append(ch)
                    i += 1
                    continue
            # 따옴표 밖(나레이션)
            i += 1
        # 라인 종료시 처리
        if not in_quote and s:
            items.append({"speaker_id": "Narrator", "scene": scene, "utterance": s})
        if in_quote:
            quote_buf.append("\n")  # 멀티라인 대사 유지

    # 파일 끝에 따옴표가 닫히지 않은 경우(예외)도 한 덩어리로 수집
    if in_quote and quote_buf:
        items.append({"speaker_id": "Unknown", "scene": scene, "utterance": "".join(quote_buf).strip()})

    return _merge_adjacent(items)

def parse_utterances_from_text(text: str) -> List[Dict]:
    """외부에서 호출되는 파서: FSM 기반."""
    return _parse_with_fsm(text)

# -------------------------------------------------
# LLM Bootstrap (Option A): TXT만으로 카드/가이드/사전 생성
# -------------------------------------------------
def _top_speakers(items, top_n=8):
    cnt = Counter([it["speaker_id"] for it in items])
    return [spk for spk, _ in cnt.most_common(top_n)]

def _speaker_samples(items, top_n=8, max_lines_per_speaker=40):
    tops = set(_top_speakers(items, top_n=top_n))
    by_spk = {}
    for it in items:
        spk = it["speaker_id"]
        if spk not in tops:
            continue
        by_spk.setdefault(spk, [])
        if len(by_spk[spk]) < max_lines_per_speaker:
            by_spk[spk].append(it["utterance"])
    return {spk: "\n".join(lines) for spk, lines in by_spk.items()}

CHAR_CARD_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "너는 대사 말투 분석가다. 각 화자의 말투/성격/호칭/금지·권장 표현/감정 경향을 1~2문장으로 요약한다. "
     "출력은 각 줄에 '이름: 요약...' 형식만 사용한다. 불명확하면 추정값을 표시하라."),
    ("human",
     "다음은 화자별 대사 샘플이다(키=화자ID/이름, 값=샘플 대사들).\n"
     "{samples}\n\n"
     "각 화자에 대해 한 줄씩 '이름: ...'로 출력해라.")
])

STYLE_GUIDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "너는 편집자다. 작품의 전체 톤 가이드와 나레이션 규칙, 감정→TTS(rate, pitch, volume) 매핑표를 마크다운으로 만든다. "
     "간결하지만 실무적으로 바로 쓸 수 있도록 작성하라."),
    ("human",
     "다음은 작품의 일부 대사/지문이다:\n{snippet}\n\n"
     "마크다운 형식으로 스타일 가이드를 작성해라. 표 하나에 감정별 권장 파라미터를 포함하라.")
])

LEXICON_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "너는 TTS 발음 사전을 만든다. 입력 용어 리스트에 대해 JSON {{원표기: '발음'}} 형식만 출력한다. "
     "모르면 해당 표기를 그대로 값으로 둔다. 주석이나 설명은 금지."),
    ("human",
     "용어 리스트:\n{terms}\n\nJSON만 출력하라.")
])

def _extract_terms_for_lexicon(text: str, limit=200):
    toks = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]+", text))
    toks = [t for t in toks if len(t) > 1]
    toks.sort()
    return toks[:limit]

@app.post("/bootstrap_from_txt_upload")
async def bootstrap_from_txt_upload(
    novel: UploadFile = File(...),
    also_ingest: Optional[bool] = Form(True),
    top_speakers: Optional[int] = Form(8),
    max_lines_per_speaker: Optional[int] = Form(40),
    corpus_id: Optional[str] = Form(None),
):
    """TXT만으로 캐릭터 카드/스타일 가이드/발음 사전을 생성(A)하고 (옵션) 즉시 인덱싱."""
    try:
        session_id = f"sess-{uuid.uuid4().hex[:8]}"
        session_dir = UPLOAD_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        novel_path = session_dir / f"novel_{novel.filename}"
        with open(novel_path, "wb") as f:
            f.write(novel.file.read())

        text = load_text_file(novel_path)
        corpus_id = _sanitize_corpus_id(corpus_id or novel_path.stem)

        # 파싱 → 샘플
        items = parse_utterances_from_text(text)
        samples = _speaker_samples(items, top_n=int(top_speakers or 8),
                                   max_lines_per_speaker=int(max_lines_per_speaker or 40))

        # 캐릭터 카드
        cc_text = (CHAR_CARD_PROMPT | LLM).invoke({"samples": json.dumps(samples, ensure_ascii=False)}).content
        # 스타일 가이드
        snippet = "\n".join([it["utterance"] for it in items[:200]])
        sg_md = (STYLE_GUIDE_PROMPT | LLM).invoke({"snippet": snippet}).content
        # 발음 사전
        terms = _extract_terms_for_lexicon(text, limit=200)
        lx_json = (LEXICON_PROMPT | LLM).invoke({"terms": "\n".join(terms)}).content
        try:
            lex_map = json.loads(lx_json)
            if not isinstance(lex_map, dict):
                raise ValueError("lexicon is not a dict")
        except Exception:
            lex_map = {t: t for t in terms}

        # 저장
        cc_path = session_dir / "character_cards.txt"
        sg_path = session_dir / "style_guide.md"
        lx_path = session_dir / "lexicon.json"
        cc_path.write_text(cc_text, encoding="utf-8")
        sg_path.write_text(sg_md, encoding="utf-8")
        lx_path.write_text(json.dumps(lex_map, ensure_ascii=False, indent=2), encoding="utf-8")

        # 캐릭터명 수집(업로드 카드와 동일한 파서 재사용)
        _ = to_docs_character_cards(cc_text, corpus_id=corpus_id)  # CHAR_NAMES에 반영됨(메모리)

        # (옵션) 인덱싱
        added = None
        if also_ingest:
            res = ingest_corpus_from_paths(novel_path, cc_path, sg_path, lx_path, corpus_id=corpus_id)
            added = res.get("added", {})

        return JSONResponse({
            "downloads": {
                "character_cards": f"/download/{session_id}/character_cards.txt",
                "style_guide": f"/download/{session_id}/style_guide.md",
                "lexicon": f"/download/{session_id}/lexicon.json",
            },
            "added": added,
            "session": session_id,
            "corpus_id": corpus_id
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"bootstrap_from_txt_upload failed: {e}")

# -------------------------------------------------
# API: ingest / ingest_upload / tts / batch / ingest_session
# -------------------------------------------------
@app.post("/ingest")
def ingest_json(payload: Dict = Body(...)):
    try:
        novel_path_str = payload.get("novel_path")
        if not novel_path_str:
            raise HTTPException(status_code=400, detail="novel_path is required")
        novel_path = Path(novel_path_str)
        if not novel_path.exists():
            raise HTTPException(status_code=400, detail="novel_path not found")

        corpus_id = _sanitize_corpus_id(payload.get("corpus_id") or novel_path.stem)
        cc = payload.get("character_cards_path")
        sg = payload.get("style_guides_path")
        lx = payload.get("lexicon_path")
        return ingest_corpus_from_paths(
            novel_path,
            Path(cc) if cc else None,
            Path(sg) if sg else None,
            Path(lx) if lx else None,
            corpus_id=corpus_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest_upload")
async def ingest_upload(
    novel: UploadFile = File(...),
    character_cards: Optional[UploadFile] = File(None),
    style_guide: Optional[UploadFile] = File(None),
    lexicon: Optional[UploadFile] = File(None),
    corpus_id: Optional[str] = Form(None),
):
    """브라우저에서 txt/md/json 파일 업로드 → 즉시 인덱싱."""
    session_dir = UPLOAD_DIR / f"sess-{uuid.uuid4().hex[:8]}"
    session_dir.mkdir(parents=True, exist_ok=True)

    def _save(upload: UploadFile, fname: str) -> Path:
        path = session_dir / fname
        with open(path, "wb") as f:
            f.write(upload.file.read())
        return path

    try:
        novel_path = _save(novel, f"novel_{novel.filename}")
        cc_path = _save(character_cards, f"characters_{character_cards.filename}") if character_cards else None
        sg_path = _save(style_guide, f"style_{style_guide.filename}") if style_guide else None
        lx_path = _save(lexicon, f"lexicon_{lexicon.filename}") if lexicon else None

        res = ingest_corpus_from_paths(
            novel_path, cc_path, sg_path, lx_path,
            corpus_id=_sanitize_corpus_id(corpus_id or novel_path.stem)
        )
        return JSONResponse(res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ingest_upload failed: {e}")

@app.post("/tts")
async def tts(payload: Dict = Body(...)):
    try:
        utterance = payload.get("utterance")
        speaker_id = payload.get("speaker_id", "S1")
        scene = payload.get("scene", "default")
        corpus_id = _sanitize_corpus_id(payload.get("corpus_id") or DEFAULT_CORPUS)
        if not utterance:
            raise HTTPException(status_code=400, detail="utterance is required")
        # 화자명도 카드 기준으로 정규화
        speaker_id = _best_match(speaker_id, corpus_id)
        return JSONResponse(generate_tts_prompt_sync(utterance, speaker_id, scene, corpus_id))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"tts failed: {e}")

@app.post("/tts_from_txt_upload")
async def tts_from_txt_upload(
    novel: UploadFile = File(...),
    also_ingest: Optional[bool] = Form(True),
    max_items: Optional[int] = Form(None),
    corpus_id: Optional[str] = Form(None),
):
    """TXT 전체 업로드 → (옵션) 인덱싱 → 발화별 TTS JSON(동시 처리) → 묶어서 반환"""
    try:
        session_id = f"sess-{uuid.uuid4().hex[:8]}"
        session_dir = UPLOAD_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        novel_path = session_dir / f"novel_{novel.filename}"
        with open(novel_path, "wb") as f:
            f.write(novel.file.read())

        text = load_text_file(novel_path)
        corpus_id = _sanitize_corpus_id(corpus_id or novel_path.stem)

        if also_ingest:
            ingest_corpus_from_paths(novel_path, corpus_id=corpus_id)

        items = parse_utterances_from_text(text)
        if max_items is not None:
            try:
                items = items[: int(max_items)]
            except Exception:
                pass

        # 🔹 화자명 정규화(캐릭터 카드와 매칭)
        for it in items:
            it["speaker_id"] = _best_match(it["speaker_id"], corpus_id)

        # 🔹 동시 처리로 LLM 호출 속도↑
        sem = asyncio.Semaphore(BATCH_CONCURRENCY)
        async def one(it):
            async with sem:
                return await generate_tts_prompt_async(it["utterance"], it["speaker_id"], it["scene"], corpus_id)

        outputs = await asyncio.gather(*[one(it) for it in items])

        out_path = session_dir / "tts_prompts.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"count": len(outputs), "items": outputs, "corpus_id": corpus_id}, f, ensure_ascii=False, indent=2)

        return JSONResponse({
            "count": len(outputs),
            "items": outputs,
            "corpus_id": corpus_id,
            "download_url": f"/download/{session_id}/tts_prompts.json"
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"tts_from_txt_upload failed: {e}")

@app.post("/ingest_session")
def ingest_session(payload: Dict = Body(...)):
    """Bootstrap 결과 세션 폴더를 받아 즉시 인덱싱.
    입력: {"session": "sess-xxxx", "corpus_id": "myWork"}"""
    try:
        session = payload.get("session")
        if not session:
            raise HTTPException(status_code=400, detail="session is required")
        session_dir = UPLOAD_DIR / session
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail="session not found")

        novel_candidates = list(session_dir.glob("novel_*"))
        novel_path = novel_candidates[0] if novel_candidates else None
        if not (novel_path and novel_path.exists()):
            raise HTTPException(status_code=400, detail="novel file not found in session")

        corpus_id = _sanitize_corpus_id(payload.get("corpus_id") or novel_path.stem)
        cc_path = session_dir / "character_cards.txt"
        sg_path = session_dir / "style_guide.md"
        lx_path = session_dir / "lexicon.json"

        res = ingest_corpus_from_paths(
            novel_path,
            cc_path if cc_path.exists() else None,
            sg_path if sg_path.exists() else None,
            lx_path if lx_path.exists() else None,
            corpus_id=corpus_id,
        )
        return JSONResponse(res)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ingest_session failed: {e}")

# -------------------------------------------------
# Serve UI + downloads
# -------------------------------------------------
@app.get("/")
def root():
    index_path = BASE_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"tip": "Place index.html next to main1.py, or call /docs"})

@app.get("/download/{session}/{filename}")
def download_file(session: str, filename: str):
    target = UPLOAD_DIR / session / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(target)

if __name__ == "__main__":
    print("Run API server:\n  python -m uvicorn main1:app --reload --port 8000")

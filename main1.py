import os
import json
import uuid
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict

from fastapi import FastAPI, Body, UploadFile, File, HTTPException
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

app = FastAPI(title="RAG→TTS Prompt Service", version="0.5.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# -------------------------------------------------
# Embeddings & LLM
# -------------------------------------------------
def get_embedding():
    if USE_OPENAI_EMB:
        return OpenAIEmbeddings(model="text-embedding-3-large")
    return HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")

def get_llm():
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
# Lexicon (작품별)
# -------------------------------------------------
LEXICONS: Dict[str, Dict[str, str]] = defaultdict(dict)  # corpus_id -> {term: pron}

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def load_text_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _make_splitter():
    try:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=400, chunk_overlap=80
        )
        return splitter
    except Exception:
        return RecursiveCharacterTextSplitter(
            chunk_size=900, chunk_overlap=150, separators=["\n\n", "\n", " "]
        )

def _meta(kind: str, corpus_id: Optional[str], extra: Optional[Dict] = None):
    m = {"kind": kind, "corpus_id": corpus_id or DEFAULT_CORPUS}
    if extra:
        m.update(extra)
    return m

def _sanitize_corpus_id(raw: str) -> str:
    # 파일명/입력에서 안전한 corpus id 생성
    s = re.sub(r"[^\w\-]+", "_", raw.strip())
    return s or DEFAULT_CORPUS

def _resolve_corpus_id(novel_path: Optional[Path], provided: Optional[str]) -> str:
    if provided:
        return _sanitize_corpus_id(provided)
    if novel_path:
        return _sanitize_corpus_id(novel_path.stem)
    return DEFAULT_CORPUS

# -------------------------------------------------
# Doc builders
# -------------------------------------------------
def to_docs_from_text(text: str, kind: str, corpus_id: Optional[str], extra_meta: Optional[Dict] = None) -> List[Document]:
    splitter = _make_splitter()
    chunks = splitter.split_text(text)
    docs = [
        Document(
            page_content=c, 
            metadata=_meta(kind, corpus_id, extra=extra_meta)
        )
        for c in chunks if c.strip()
    ]
    return docs

def to_docs_character_cards(raw: str, corpus_id: Optional[str]) -> List[Document]:
    docs = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            name, desc = line.split(":", 1)
            docs.append(
                Document(
                    page_content=desc.strip(),
                    metadata=_meta("character_card", corpus_id, {"character": name.strip()}),
                )
            )
    # 전체 카드 텍스트도 의미검색용으로 1건 추가
    docs.append(Document(page_content=raw, metadata=_meta("character_card", corpus_id)))
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
    corpus_id = _resolve_corpus_id(novel_path, corpus_id)
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
        },
    }

# -------------------------------------------------
# Retrieval & generation
# -------------------------------------------------
def retrieve_context(utterance: str, speaker_id: str, scene: str, corpus_id: str,
                     k_novel=2, k_card=2, k_style=1) -> str:
    vs = get_vs()
    filt_novel = {"kind": "novel_chunk",   "corpus_id": corpus_id}
    filt_card  = {"kind": "character_card","corpus_id": corpus_id}
    filt_style = {"kind": "style_guide",   "corpus_id": corpus_id}

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

def generate_tts_prompt(utterance: str, speaker_id: str, scene: str, corpus_id: str) -> Dict:
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
        return json.loads(result)
    except json.JSONDecodeError:
        return {
            "speaker_id": speaker_id,
            "voice_hint": {"gender": "unknown", "age": "unknown"},
            "style": {"mood": "neutral", "formality": "기본", "energy": "medium"},
            "tts_params": {"rate": 1.0, "pitch": 0, "volume": 0},
            "pronunciation_overrides": overrides,
            "text": utterance,
            "_raw": result,
        }

# -------------------------------------------------
# Utterance parsing
# -------------------------------------------------
# 원래 파서
# def parse_utterances_from_text(text: str):
#     lines = text.splitlines()
#     current_scene = "default"
#     items = []
#     for raw in lines:
#         line = raw.strip()
#         if not line:
#             continue
#         if re.match(r'^(#+)\s+(.+)', line):
#             current_scene = re.sub(r'^(#+)\s+', '', line);  continue
#         if re.match(r'^(\d+\s*장\.?|CHAPTER\s+\d+)', line, flags=re.I):
#             current_scene = line;  continue
#         m = re.match(r'^(S[\w\d]+)\s*:\s*(.+)$', line)
#         if m:
#             speaker = m.group(1);  utt = m.group(2).strip()
#             if utt:
#                 items.append({"speaker_id": speaker, "scene": current_scene, "utterance": utt})
#             continue
#         items.append({"speaker_id": "Narrator", "scene": current_scene, "utterance": line})
#     return items
# 수정 후 파서
# --- Advanced utterance parser: split quotes vs narration, infer speaker from reporting clause ---

# 따옴표 패턴들(순서 유지: 먼저 매칭되는 걸 사용)
_QUOTE_REGEXES = [
    re.compile(r'“([^”]+)”'),
    re.compile(r'"([^"]+)"'),
    re.compile(r'『([^』]+)』'),
    re.compile(r'「([^」]+)」'),
    re.compile(r'‘([^’]+)’'),
    re.compile(r"'([^']+)'"),
]

# 보고(말하다) 동사(어간/활용 일부 포함)
_SAID_VERB = r'(?:말했|말하|중얼|외쳤|물었|대답했|속삭였|소리쳤|되물었|덧붙였|응수했|부르|불렀|말한다|말하다)'

# 화자 후보를 추정(따옴표 앞/뒤 컨텍스트)
def _detect_speaker_name(_pre: str, _post: str) -> Optional[str]:
    pre = _pre[-40:] if _pre else ""
    post = _post[:40] if _post else ""

    # "…"(라고) <이름>(가/이/는/은) <말했…>
    m = re.search(
        rf'(?:라고|라며|하며|하고)?\s*([가-힣A-Za-z0-9 ]{{1,20}}?)(?:가|이|는|은)\s*{_SAID_VERB}',
        post
    )
    if m:
        name = m.group(1).strip()
        if name not in {"그", "그녀", "누군가", "사람", "아이"}:
            return name

    # <이름>(가/이/는/은) <말했…> "…"
    m = re.search(
        rf'([가-힣A-Za-z0-9 ]{{1,20}}?)(?:가|이|는|은)\s*{_SAID_VERB}\s*(?:며|라고)?\s*$',
        pre
    )
    if m:
        name = m.group(1).strip()
        if name not in {"그", "그녀", "누군가", "사람", "아이"}:
            return name

    return None  # 못 찾으면 None

# 한 줄에서 따옴표 기준으로 [나레이션] — [대사] — [나레이션] … 분리
def _split_line_by_quotes(line: str):
    parts = []
    pos = 0
    while pos < len(line):
        earliest = None
        for pat in _QUOTE_REGEXES:
            m = pat.search(line, pos)
            if m and (earliest is None or m.start() < earliest.start()):
                earliest = m
        if not earliest:
            tail = line[pos:].strip()
            if tail:
                parts.append(("narration", tail, "", ""))  # (type, text, pre_ctx, post_ctx)
            break

        # 따옴표 전 나레이션
        pre_text = line[pos:earliest.start()]
        if pre_text.strip():
            parts.append(("narration", pre_text.strip(), "", ""))

        # 따옴표 안 대사
        quote_text = earliest.group(1).strip()
        pre_ctx = line[max(0, earliest.start()-40):earliest.start()]
        post_ctx = line[earliest.end():min(len(line), earliest.end()+40)]
        parts.append(("quote", quote_text, pre_ctx, post_ctx))

        pos = earliest.end()

    return parts  # [("narration" | "quote", text, pre_ctx, post_ctx), ...]

def parse_utterances_from_text(text: str):
    """
    고급 파서:
      - 장(씬) 헤더 인식(#, ##, '1장.' / 'CHAPTER 1')
      - 'Sx: ...' 형식은 그대로 화자 지정
      - 같은 줄에 따옴표 안 대사 + 보고문이 섞여 있으면
        * 따옴표 안은 화자(보고문 추정)가 말한 것으로,
        * 따옴표 밖은 Narrator로 분리
    반환: [{speaker_id, scene, utterance}, ...]
    """
    lines = text.splitlines()
    current_scene = "default"
    items = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # 씬 헤더
        if re.match(r'^(#+)\s+(.+)', line):
            current_scene = re.sub(r'^(#+)\s+', '', line)
            continue
        if re.match(r'^(\d+\s*장\.?|CHAPTER\s+\d+)', line, flags=re.I):
            current_scene = line
            continue

        # 명시적 화자: S1: ...
        m = re.match(r'^(S[\w\d]+)\s*:\s*(.+)$', line)
        if m:
            speaker = m.group(1)
            utt = m.group(2).strip()
            if utt:
                items.append({"speaker_id": speaker, "scene": current_scene, "utterance": utt})
            continue

        # 따옴표 기반 분리
        parts = _split_line_by_quotes(line)

        # 따옴표가 없으면 그냥 나레이션
        if not parts or all(t != "quote" for t, *_ in parts):
            items.append({"speaker_id": "Narrator", "scene": current_scene, "utterance": line})
            continue

        # 따옴표가 있으면, 각 조각을 역할에 따라 추가
        for (ptype, text_piece, pre_ctx, post_ctx) in parts:
            if ptype == "narration":
                if text_piece:
                    items.append({"speaker_id": "Narrator", "scene": current_scene, "utterance": text_piece})
            else:  # quote
                speaker_name = _detect_speaker_name(pre_ctx, post_ctx) or "Unknown"
                items.append({"speaker_id": speaker_name, "scene": current_scene, "utterance": text_piece})

    return items

# -------------------------------------------------
# LLM Bootstrap (Option A)
# -------------------------------------------------
CHAR_CARD_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "너는 대사 말투 분석가다. 각 화자의 말투/성격/호칭/금지·권장 표현/감정 경향을 1~2문장으로 요약한다. "
     "출력은 각 줄에 'Sx: 요약...' 형식만 사용한다. 불명확하면 추정값을 표시하라."),
    ("human",
     "다음은 화자별 대사 샘플이다(키=화자ID, 값=샘플 대사들).\n"
     "{samples}\n\n"
     "각 화자에 대해 한 줄씩 'Sx: ...'로 출력해라.")
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

def _extract_terms_for_lexicon(text: str, limit=200):
    toks = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]+", text))
    toks = [t for t in toks if len(t) > 1]
    toks.sort()
    return toks[:limit]

@app.post("/bootstrap_from_txt_upload")
async def bootstrap_from_txt_upload(
    novel: UploadFile = File(...),
    also_ingest: Optional[bool] = True,
    top_speakers: Optional[int] = 8,
    max_lines_per_speaker: Optional[int] = 40,
    corpus_id: Optional[str] = None,
):
    """TXT만으로 캐릭터 카드/스타일 가이드/발음 사전을 LLM으로 자동 생성(A 방법)."""
    try:
        session_id = f"sess-{uuid.uuid4().hex[:8]}"
        session_dir = UPLOAD_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        novel_path = session_dir / f"novel_{novel.filename}"
        with open(novel_path, "wb") as f:
            f.write(novel.file.read())

        text = load_text_file(novel_path)
        corpus_id = _resolve_corpus_id(novel_path, corpus_id)

        # 1) 파싱 → 샘플
        items = parse_utterances_from_text(text)
        samples = _speaker_samples(items, top_n=int(top_speakers or 8), max_lines_per_speaker=int(max_lines_per_speaker or 40))

        # 2) 캐릭터 카드
        cc_text = (CHAR_CARD_PROMPT | LLM).invoke({"samples": json.dumps(samples, ensure_ascii=False)}).content
        # 3) 스타일 가이드
        snippet = "\n".join([it["utterance"] for it in items[:200]])
        sg_md = (STYLE_GUIDE_PROMPT | LLM).invoke({"snippet": snippet}).content
        # 4) 발음 사전
        terms = _extract_terms_for_lexicon(text, limit=200)
        lx_json = (LEXICON_PROMPT | LLM).invoke({"terms": "\n".join(terms)}).content
        try:
            lex_map = json.loads(lx_json)
            if not isinstance(lex_map, dict):
                raise ValueError("lexicon is not a dict")
        except Exception:
            lex_map = {t: t for t in terms}

        # 5) 저장
        cc_path = session_dir / "character_cards.txt"
        sg_path = session_dir / "style_guide.md"
        lx_path = session_dir / "lexicon.json"
        cc_path.write_text(cc_text, encoding="utf-8")
        sg_path.write_text(sg_md, encoding="utf-8")
        lx_path.write_text(json.dumps(lex_map, ensure_ascii=False, indent=2), encoding="utf-8")

        # 6) (옵션) 인덱싱
        added = None
        if also_ingest:
            res = ingest_corpus_from_paths(novel_path, cc_path, sg_path, lx_path, corpus_id=corpus_id)
            added = res.get("added", {})
        else:
            # also_ingest가 false여도, 서버 메모리 Lexicon에는 미리 반영(선택 사항)
            LEXICONS[corpus_id].update(lex_map)

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

        corpus_id = payload.get("corpus_id")
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
    corpus_id: Optional[str] = None,
):
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

        corpus_id = _resolve_corpus_id(novel_path, corpus_id)
        res = ingest_corpus_from_paths(novel_path, cc_path, sg_path, lx_path, corpus_id=corpus_id)
        return JSONResponse(res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ingest_upload failed: {e}")

@app.post("/tts")
async def tts(payload: Dict = Body(...)):
    try:
        utterance = payload.get("utterance")
        speaker_id = payload.get("speaker_id", "S1")
        scene = payload.get("scene", "default")
        corpus_id = _sanitize_corpus_id(payload.get("corpus_id", DEFAULT_CORPUS))
        if not utterance:
            raise HTTPException(status_code=400, detail="utterance is required")
        return JSONResponse(generate_tts_prompt(utterance, speaker_id, scene, corpus_id))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"tts failed: {e}")

@app.post("/tts_from_txt_upload")
async def tts_from_txt_upload(
    novel: UploadFile = File(...),
    also_ingest: Optional[bool] = True,
    max_items: Optional[int] = None,
    corpus_id: Optional[str] = None,
):
    """TXT 전체를 업로드 → (옵션) 인덱싱 → 발화별 TTS JSON → 합치기"""
    try:
        session_id = f"sess-{uuid.uuid4().hex[:8]}"
        session_dir = UPLOAD_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        novel_path = session_dir / f"novel_{novel.filename}"
        with open(novel_path, "wb") as f:
            f.write(novel.file.read())

        text = load_text_file(novel_path)
        corpus_id = _resolve_corpus_id(novel_path, corpus_id)

        if also_ingest:
            ingest_corpus_from_paths(novel_path, corpus_id=corpus_id)

        items = parse_utterances_from_text(text)
        if max_items is not None:
            try:
                items = items[: int(max_items)]
            except Exception:
                pass

        outputs = []
        for it in items:
            out = generate_tts_prompt(it["utterance"], it["speaker_id"], it["scene"], corpus_id)
            outputs.append(out)

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

        corpus_id = _resolve_corpus_id(novel_path, payload.get("corpus_id"))
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

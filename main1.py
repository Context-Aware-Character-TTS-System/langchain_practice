import os
import json
import uuid
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import Counter

from fastapi import FastAPI, Body, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv

# --- LangChain & VectorStore ---
# Try new import path first; fallback for older installs
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

app = FastAPI(title="RAG→TTS Prompt Service", version="0.4.0")

# CORS (로컬 파일에서 열어도 호출되도록)
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
    # OpenAI LLM만 예시로 붙임 (키는 .env에)
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0)

EMB = get_embedding()
LLM = get_llm()

# -------------------------------------------------
# VectorStore (단일 컬렉션 + kind 메타데이터)
# -------------------------------------------------
def get_vs() -> Chroma:
    PERSIST_DIR.mkdir(exist_ok=True)
    # 텔레메트리 완전 차단 + 영구 저장 경로 지정
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
# Schemas (간단 내부 용)
# -------------------------------------------------
@dataclass
class Lexicon:
    mapping: Dict[str, str]

LEXICON = Lexicon(mapping={})

# -------------------------------------------------
# Helpers: loaders
# -------------------------------------------------
def load_text_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _make_splitter():
    # Prefer token-based splitter; fallback to char-based if tiktoken not available
    try:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=400,
            chunk_overlap=80,
        )
        return splitter
    except Exception:
        return RecursiveCharacterTextSplitter(
            chunk_size=900, chunk_overlap=150, separators=["\n\n", "\n", " "]
        )

def to_docs_from_text(text: str, kind: str, extra_meta: Optional[Dict] = None) -> List[Document]:
    splitter = _make_splitter()
    chunks = splitter.split_text(text)
    meta = extra_meta or {}
    docs = [
        Document(page_content=c, metadata={"kind": kind, **meta})
        for c in chunks
        if c.strip()
    ]
    return docs

def to_docs_character_cards(raw: str) -> List[Document]:
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
                    metadata={"kind": "character_card", "character": name.strip()},
                )
            )
    # 전체 텍스트도 하나로 보관(의미 검색용)
    docs.append(Document(page_content=raw, metadata={"kind": "character_card"}))
    return docs

def to_docs_style_guides(raw: str) -> List[Document]:
    return [Document(page_content=raw, metadata={"kind": "style_guide"})]

# -------------------------------------------------
# Core ingest
# -------------------------------------------------
def ingest_corpus_from_paths(
    novel_path: Path,
    character_cards_path: Optional[Path] = None,
    style_guides_path: Optional[Path] = None,
    lexicon_path: Optional[Path] = None,
):
    vs = get_vs()

    # Novel
    novel_text = load_text_file(novel_path)
    novel_docs = to_docs_from_text(novel_text, kind="novel_chunk")
    if novel_docs:
        vs.add_documents(novel_docs)

    # Character cards
    cc_docs = []
    if character_cards_path and character_cards_path.exists():
        cc_text = load_text_file(character_cards_path)
        cc_docs = to_docs_character_cards(cc_text)
        if cc_docs:
            vs.add_documents(cc_docs)

    # Style guides
    sg_docs = []
    if style_guides_path and style_guides_path.exists():
        sg_text = load_text_file(style_guides_path)
        sg_docs = to_docs_style_guides(sg_text)
        if sg_docs:
            vs.add_documents(sg_docs)

    # Lexicon
    if lexicon_path and lexicon_path.exists():
        try:
            data = json.loads(load_text_file(lexicon_path))
            if isinstance(data, dict):
                LEXICON.mapping.update(data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid lexicon JSON: {e}")

    vs.persist()
    return {
        "status": "ok",
        "added": {
            "novel_chunks": len(novel_docs),
            "character_cards": len(cc_docs),
            "style_guides": len(sg_docs),
            "lexicon_terms": len(LEXICON.mapping),
        },
    }

# -------------------------------------------------
# Retrieval & generation
# -------------------------------------------------
def retrieve_context(utterance: str, speaker_id: str, scene: str, k_novel=2, k_card=2, k_style=1) -> str:
    vs = get_vs()
    ret_novel = vs.as_retriever(search_kwargs={"k": k_novel, "filter": {"kind": "novel_chunk"}})
    ret_card  = vs.as_retriever(search_kwargs={"k": k_card,  "filter": {"kind": "character_card"}})
    ret_style = vs.as_retriever(search_kwargs={"k": k_style, "filter": {"kind": "style_guide"}})

    ctx_parts = []
    cards  = ret_card.invoke(f"{speaker_id} {utterance}")
    styles = ret_style.invoke(f"{scene} {utterance}")
    novels = ret_novel.invoke(utterance)

    ctx_parts += [d.page_content for d in cards]
    ctx_parts += [d.page_content for d in styles]
    ctx_parts += [d.page_content for d in novels]
    return "\n---\n".join(ctx_parts)

def subset_lexicon(utterance: str) -> Dict[str, str]:
    return {k: v for k, v in LEXICON.mapping.items() if k in utterance}

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

def generate_tts_prompt(utterance: str, speaker_id: str, scene: str) -> Dict:
    context = retrieve_context(utterance, speaker_id, scene)
    overrides = subset_lexicon(utterance)

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
# Utterance parsing (auto split full TXT into speaker/scene/utterance)
# -------------------------------------------------
def parse_utterances_from_text(text: str):
    """Return list of {speaker_id, scene, utterance} parsed from full text.
    Rules:
      - Scene tag changes on markdown headers (#, ##) or lines like '1장. 제목' / 'CHAPTER 1'
      - Dialogue lines like 'S1: 내용' set speaker_id to that token
      - Non-empty lines without speaker prefix are treated as Narrator
    """
    lines = text.splitlines()
    current_scene = "default"
    items = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        # Scene headers
        if re.match(r'^(#+)\s+(.+)', line):
            current_scene = re.sub(r'^(#+)\s+', '', line)
            continue
        if re.match(r'^(\d+\s*장\.?|CHAPTER\s+\d+)', line, flags=re.I):
            current_scene = line
            continue
        # Dialogue: S1: hello
        m = re.match(r'^(S[\w\d]+)\s*:\s*(.+)$', line)
        if m:
            speaker = m.group(1)
            utt = m.group(2).strip()
            if utt:
                items.append({"speaker_id": speaker, "scene": current_scene, "utterance": utt})
            continue
        # Otherwise: narrator
        items.append({"speaker_id": "Narrator", "scene": current_scene, "utterance": line})
    return items

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

def _extract_terms_for_lexicon(text: str, limit=200):
    # 영문/숫자/하이픈/언더스코어 포함 토큰만 추출 (주로 약어, 제품명 등)
    toks = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]+", text))
    # 길이 2 이상만
    toks = [t for t in toks if len(t) > 1]
    toks.sort()
    return toks[:limit]

@app.post("/bootstrap_from_txt_upload")
async def bootstrap_from_txt_upload(
    novel: UploadFile = File(...),
    also_ingest: Optional[bool] = True,
    top_speakers: Optional[int] = 8,
    max_lines_per_speaker: Optional[int] = 40,
):
    """TXT만으로 캐릭터 카드/스타일 가이드/발음 사전을 LLM으로 자동 생성(A 방법).
    생성 결과 파일을 다운로드 링크로 제공하고, 요청 시 바로 인덱싱/반영한다."""
    try:
        # Save upload to session dir
        session_id = f"sess-{uuid.uuid4().hex[:8]}"
        session_dir = UPLOAD_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        novel_path = session_dir / f"novel_{novel.filename}"
        with open(novel_path, "wb") as f:
            f.write(novel.file.read())

        text = load_text_file(novel_path)

        # 1) 파싱 → 화자 샘플 만들기
        items = parse_utterances_from_text(text)
        samples = _speaker_samples(items, top_n=int(top_speakers or 8), max_lines_per_speaker=int(max_lines_per_speaker or 40))

        # 2) 캐릭터 카드 생성
        cc_chain = CHAR_CARD_PROMPT | LLM
        cc_text = cc_chain.invoke({"samples": json.dumps(samples, ensure_ascii=False)}).content

        # 3) 스타일 가이드 생성
        snippet = "\n".join([it["utterance"] for it in items[:200]])
        sg_chain = STYLE_GUIDE_PROMPT | LLM
        sg_md = sg_chain.invoke({"snippet": snippet}).content

        # 4) 발음 사전 생성
        terms = _extract_terms_for_lexicon(text, limit=200)
        lx_chain = LEXICON_PROMPT | LLM
        lx_json = lx_chain.invoke({"terms": "\n".join(terms)}).content
        try:
            lex_map = json.loads(lx_json)
            if not isinstance(lex_map, dict):
                raise ValueError("lexicon is not a dict")
        except Exception:
            # 폴백: 동일 매핑
            lex_map = {t: t for t in terms}

        # 5) 파일 저장
        cc_path = session_dir / "character_cards.txt"
        sg_path = session_dir / "style_guide.md"
        lx_path = session_dir / "lexicon.json"
        cc_path.write_text(cc_text, encoding="utf-8")
        sg_path.write_text(sg_md, encoding="utf-8")
        lx_path.write_text(json.dumps(lex_map, ensure_ascii=False, indent=2), encoding="utf-8")

        # 6) (옵션) 인덱싱 & 반영
        added = None
        if also_ingest:
            res = ingest_corpus_from_paths(novel_path, cc_path, sg_path, lx_path)
            added = res.get("added", {})

        # 7) 응답
        return JSONResponse({
            "downloads": {
                "character_cards": f"/download/{session_id}/character_cards.txt",
                "style_guide": f"/download/{session_id}/style_guide.md",
                "lexicon": f"/download/{session_id}/lexicon.json",
            },
            "added": added,
            "session": session_id
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"bootstrap_from_txt_upload failed: {e}")

# -------------------------------------------------
# API: JSON ingest / 파일 업로드 ingest / TXT→batch TTS / 세션 인덱싱
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
        cc = payload.get("character_cards_path")
        sg = payload.get("style_guides_path")
        lx = payload.get("lexicon_path")
        return ingest_corpus_from_paths(
            novel_path,
            Path(cc) if cc else None,
            Path(sg) if sg else None,
            Path(lx) if lx else None,
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
):
    """브라우저에서 txt/md/json 파일을 직접 업로드하는 경로."""
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

        res = ingest_corpus_from_paths(novel_path, cc_path, sg_path, lx_path)
        return JSONResponse(res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ingest_upload failed: {e}")

@app.post("/tts")
async def tts(payload: Dict = Body(...)):
    try:
        utterance = payload.get("utterance")
        speaker_id = payload.get("speaker_id", "S1")
        scene = payload.get("scene", "default")
        if not utterance:
            raise HTTPException(status_code=400, detail="utterance is required")
        return JSONResponse(generate_tts_prompt(utterance, speaker_id, scene))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"tts failed: {e}")

@app.post("/tts_from_txt_upload")
async def tts_from_txt_upload(
    novel: UploadFile = File(...),
    also_ingest: Optional[bool] = True,
    max_items: Optional[int] = None
):
    """Upload a full TXT, auto-split into utterances, optionally ingest as corpus,
    then generate TTS prompts for each utterance and return a combined JSON array."""
    try:
        # Save upload to session dir
        session_id = f"sess-{uuid.uuid4().hex[:8]}"
        session_dir = UPLOAD_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        novel_path = session_dir / f"novel_{novel.filename}"
        with open(novel_path, "wb") as f:
            f.write(novel.file.read())

        text = load_text_file(novel_path)

        # Optionally ingest to corpus (so retrieval has context)
        if also_ingest:
            ingest_corpus_from_paths(novel_path)

        # Parse utterances
        items = parse_utterances_from_text(text)
        if max_items is not None:
            try:
                cap = int(max_items)
                items = items[:cap]
            except Exception:
                pass

        # Generate prompts per item
        outputs = []
        for it in items:
            out = generate_tts_prompt(it["utterance"], it["speaker_id"], it["scene"])
            outputs.append(out)

        # Save combined json for download/debug
        out_path = session_dir / "tts_prompts.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"count": len(outputs), "items": outputs}, f, ensure_ascii=False, indent=2)

        return JSONResponse({"count": len(outputs), "items": outputs, "download_url": f"/download/{session_id}/tts_prompts.json"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"tts_from_txt_upload failed: {e}")

@app.post("/ingest_session")
def ingest_session(payload: Dict = Body(...)):
    """Bootstrap 결과가 저장된 세션 폴더를 받아 즉시 인덱싱한다.
    입력: {"session": "sess-xxxx"}"""
    try:
        session = payload.get("session")
        if not session:
            raise HTTPException(status_code=400, detail="session is required")
        session_dir = UPLOAD_DIR / session
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail="session not found")

        # 파일 경로 탐색
        novel_candidates = list(session_dir.glob("novel_*"))
        novel_path = novel_candidates[0] if novel_candidates else None
        cc_path = session_dir / "character_cards.txt"
        sg_path = session_dir / "style_guide.md"
        lx_path = session_dir / "lexicon.json"

        if not (novel_path and novel_path.exists()):
            raise HTTPException(status_code=400, detail="novel file not found in session")

        res = ingest_corpus_from_paths(
            novel_path,
            cc_path if cc_path.exists() else None,
            sg_path if sg_path.exists() else None,
            lx_path if lx_path.exists() else None,
        )
        return JSONResponse(res)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ingest_session failed: {e}")

# -------------------------------------------------
# Serve UI (index.html in same folder) + downloads
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

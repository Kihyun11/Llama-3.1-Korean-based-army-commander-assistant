# news_spcify.py
# DeepSearch News API -> fetch -> HARD GATE by MUST_TERMS -> (optional) watchlist -> TF-IDF on STRICT only

import argparse
import math
import re
import time
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import requests


# =============================
# 0) CONFIG
# =============================
BASE_URL = "https://api-v2.deepsearch.com"
ENDPOINT = "/v1/global-articles"   # ✅ 끝 슬래시 제거 권장

API_KEY = "9b41b3ef6f784e2a8ac87ee8dffebdbe"  # ✅ Bearer token (CLI --api_key로 덮어쓰기 가능)

DEBUG = False
PAUSE_SEC = 0.35


# =============================
# 1) TOKENIZE + TF-IDF
# =============================
TOKEN_RE = re.compile(r"[^0-9A-Za-z가-힣]+")

KOREAN_STOPWORDS = {
    "그리고", "하지만", "또는", "그러나", "때문에", "대한", "관련", "이번", "오늘", "현재", "지난",
    "위해", "이후", "확인", "가능", "기자", "보도", "발표", "입장", "정도", "정말", "이런", "저런",
    "등", "중", "더", "및", "또", "수", "것",
}
EN_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with", "as",
    "is", "are", "was", "were", "be", "been", "it", "that", "this", "by", "from",
    "at", "about", "into", "over", "after", "before", "will", "may", "might",
}


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    text = TOKEN_RE.sub(" ", text).strip().lower()
    raw = [t for t in text.split() if len(t) >= 2]
    out: List[str] = []
    for t in raw:
        if t.isdigit():
            continue
        if t in EN_STOPWORDS or t in KOREAN_STOPWORDS:
            continue
        out.append(t)
    return out


def tfidf_keywords(
    docs_tokens: List[List[str]],
    top_k: int = 12
) -> Tuple[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
    N = len(docs_tokens)
    if N == 0:
        return [], []

    df = Counter()
    for toks in docs_tokens:
        df.update(set(toks))

    idf = {w: math.log((N + 1) / (df_w + 1)) + 1.0 for w, df_w in df.items()}

    per_doc_top: List[List[Tuple[str, float]]] = []
    corpus_scores = defaultdict(float)

    for toks in docs_tokens:
        tf = Counter(toks)
        doc_len = max(1, sum(tf.values()))
        scores: Dict[str, float] = {}
        for w, c in tf.items():
            s = (c / doc_len) * idf.get(w, 0.0)
            scores[w] = s
            corpus_scores[w] += s
        per_doc_top.append(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k])

    corpus_top = sorted(corpus_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return corpus_top, per_doc_top


# =============================
# 2) API CALL
# =============================
def deepsearch_news_search(
    api_key: str,
    page: int = 1,
    page_size: int = 10,
    keyword: Optional[str] = None,
    company_name: Optional[str] = None,
    symbols: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    highlight: Optional[str] = None,
    clustering: bool = False,
    uniquify: bool = True,
    order: Optional[str] = None,  # "score" | "published_at"
    research_insight: bool = False,
    allow_redirects: bool = True,
) -> Dict[str, Any]:
    url = f"{BASE_URL.rstrip('/')}{ENDPOINT}"
    headers = {"Authorization": f"Bearer {api_key}"}

    params: Dict[str, Any] = {
        "page": page,
        "page_size": page_size,
        "clustering": str(clustering).lower(),
        "uniquify": str(uniquify).lower(),
        "research_insight": str(research_insight).lower(),
    }

    if keyword:
        params["keyword"] = keyword
    if company_name:
        params["company_name"] = company_name
    if symbols:
        params["symbols"] = symbols
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    if highlight:
        params["highlight"] = highlight
    if order:
        params["order"] = order

    # 디버그용: 실제 요청 URL/HTTPS 확인
    if DEBUG:
        print("[DEBUG] FINAL URL:", url)
        assert url.startswith("https://"), f"NOT HTTPS: {url}"

    r = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=35,
        allow_redirects=allow_redirects
    )

    if DEBUG:
        print("\n[DEBUG] status:", r.status_code)
        print("[DEBUG] url:", r.url)
        if not allow_redirects:
            print("[DEBUG] Location:", r.headers.get("Location"))
        print("[DEBUG] body head:", r.text[:900])

    r.raise_for_status()
    return r.json()


def extract_articles(payload: Any) -> List[Dict[str, Any]]:
    """Wide parsing for variable response schemas."""
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []

    # 흔한 케이스
    for k in ("articles", "items", "results", "docs", "data"):
        v = payload.get(k)
        if isinstance(v, list):
            return v

    # 중첩 케이스
    for outer in ("data", "result"):
        o = payload.get(outer)
        if isinstance(o, dict):
            for k in ("articles", "items", "results", "docs"):
                v = o.get(k)
                if isinstance(v, list):
                    return v

    return []


def normalize_article(a: Dict[str, Any]) -> Dict[str, Any]:
    title = a.get("title") or a.get("headline") or ""
    desc = a.get("description") or a.get("summary") or ""
    content = a.get("content") or a.get("body") or ""
    url = a.get("url") or a.get("link") or a.get("content_url") or ""
    publisher = a.get("publisher") or (a.get("source") or {}).get("name") or a.get("source") or ""
    published = a.get("published_at") or a.get("publishedAt") or a.get("created_at") or a.get("date") or ""
    return {
        "title": str(title),
        "description": str(desc),
        "content": str(content),
        "url": str(url),
        "publisher": str(publisher),
        "published_at": str(published),
        "_raw": a,
    }


# =============================
# 3) HARD GATE (MUST TERMS)
# =============================
DEFAULT_MUST_TERMS = [
    # Military / kinetic
    "미사일", "탄도미사일", "순항미사일", "발사", "요격", "공습", "포격", "충돌", "교전",
    # Force posture / readiness
    "군사훈련", "합동훈련", "기동훈련", "전력", "병력", "배치", "증강", "동원", "출격", "해군", "공군",
    # Strategic / nuclear
    "핵", "핵실험", "ICBM", "전략자산",
    # Diplomacy escalation
    "제재", "봉쇄", "최후통첩", "안보리", "유엔",
    # Cyber / hybrid
    "사이버 공격", "해킹", "침해", "랜섬웨어", "ddos", "gps 교란", "정보전",
    # Geography / core actors
    "북한", "평양", "한미", "주한미군", "대만", "남중국해",
]


def blob(a: Dict[str, Any]) -> str:
    # 게이트에서 사용할 텍스트 (제목+요약+본문)
    return " ".join([
        a.get("title", ""),
        a.get("description", ""),
        a.get("content", ""),
    ]).strip()


def pass_must_terms(a: Dict[str, Any], must_terms: List[str]) -> bool:
    t = blob(a).lower()
    if not t:
        return False
    return any(k.lower() in t for k in must_terms)


def load_terms_from_file(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    terms: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            terms.append(s)
    return terms if terms else None


# =============================
# 4) OUTPUT HELPERS
# =============================
def print_articles(articles: List[Dict[str, Any]], limit: int = 20, title: str = "Articles"):
    print(f"\n=== {title} (top {min(limit, len(articles))}) ===")
    for i, a in enumerate(articles[:limit], 1):
        print(f"{i:02d}. [{a.get('publisher','')}] {a.get('published_at','')} | {a.get('title','')}")
        print(f"    {a.get('url','')}")


# =============================
# 5) MAIN
# =============================
def main():
    ap = argparse.ArgumentParser(description="DeepSearch News Research + Hard Keyword Gate + TF-IDF")
    ap.add_argument("--api_key", default=None, help="DeepSearch API key (Bearer). Overrides in-file API_KEY.")

    ap.add_argument("--keyword", default=None,
                    help='keyword query (supports AND/OR/NOT and field search e.g. title:(삼성전자 AND 구글))')
    ap.add_argument("--page_size", type=int, default=20)
    ap.add_argument("--max_pages", type=int, default=3)
    ap.add_argument("--date_from", default=None, help="YYYY-MM-DD")
    ap.add_argument("--date_to", default=None, help="YYYY-MM-DD")
    ap.add_argument("--clustering", action="store_true")
    ap.add_argument("--order", default="score", choices=["score", "published_at"])
    ap.add_argument("--highlight", action="store_true")
    ap.add_argument("--research_insight", action="store_true")
    ap.add_argument("--company_name", default=None)
    ap.add_argument("--symbols", default=None)

    # gate
    ap.add_argument("--strict_gate", action="store_true",
                    help="Enable HARD GATE: drop articles without any MUST term.")
    ap.add_argument("--terms_file", default=None,
                    help="Text file of MUST terms (one per line). Overrides defaults.")
    ap.add_argument("--keep_watchlist", action="store_true",
                    help="Keep non-matching articles as WATCHLIST (printed separately).")

    ap.add_argument("--tfidf_topk", type=int, default=12)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no_redirects", action="store_true",
                    help="Disable redirects to diagnose unexpected http/port80 issues.")
    args = ap.parse_args()

    global DEBUG
    DEBUG = bool(args.debug)

    # 어떤 파일을 실행 중인지 확인(디버그에 도움)
    if DEBUG:
        print("[DEBUG] RUNNING FILE:", os.path.abspath(__file__))
        print("[DEBUG] BASE_URL:", BASE_URL)
        print("[DEBUG] ENDPOINT:", ENDPOINT)

    api_key = (args.api_key or API_KEY or "").strip()
    if not api_key:
         raise RuntimeError("API_KEY가 비어있거나 placeholder입니다. 파일/--api_key로 넣어주세요.")

    must_terms = load_terms_from_file(args.terms_file) or DEFAULT_MUST_TERMS

    # 1) Fetch pages
    fetched: List[Dict[str, Any]] = []
    for page in range(1, args.max_pages + 1):
        payload = deepsearch_news_search(
            api_key=api_key,
            page=page,
            page_size=args.page_size,
            keyword=args.keyword,
            company_name=args.company_name,
            symbols=args.symbols,
            date_from=args.date_from,
            date_to=args.date_to,
            clustering=bool(args.clustering),
            uniquify=True,  # 기본 true
            order=args.order,
            highlight="true" if args.highlight else None,
            research_insight=bool(args.research_insight),
            allow_redirects=not args.no_redirects,
        )

        raw = extract_articles(payload)
        if not raw:
            break

        fetched.extend([normalize_article(x) for x in raw])
        time.sleep(PAUSE_SEC)

    print(f"\nFetched (raw): {len(fetched)}")

    if not fetched:
        print("❌ 0 results from API. Try:")
        print("- remove date_from/date_to")
        print("- simpler keyword (or no keyword)")
        print("- run with --debug --no_redirects to inspect redirects")
        return

    # 2) HARD GATE
    strict: List[Dict[str, Any]] = []
    watch: List[Dict[str, Any]] = []

    if args.strict_gate:
        for a in fetched:
            if pass_must_terms(a, must_terms):
                strict.append(a)
            else:
                watch.append(a)
    else:
        strict = fetched[:]

    print(f"STRICT (after hard gate): {len(strict)}")
    if args.strict_gate and args.keep_watchlist:
        print(f"WATCHLIST (non-matching): {len(watch)}")

    if not strict:
        print("\n❌ STRICT가 0개입니다.")
        print("해결 팁:")
        print("1) MUST_TERMS를 넓히기(동의어 추가: '훈련/연습', '발사/시험', '요격/격추' 등)")
        print("2) 기사 텍스트에 content가 안 들어오면 title/description 위주로만 판단됩니다.")
        print("3) --strict_gate 끄고 먼저 API 결과를 확인하세요.")
        if args.keep_watchlist and watch:
            print_articles(watch, limit=10, title="WATCHLIST sample")
        return

    # 3) Print STRICT articles
    print_articles(strict, limit=20, title="STRICT Articles")

    # 4) TF-IDF only on STRICT set
    docs_tokens = [tokenize(blob(a)) for a in strict]
    corpus_top, per_doc_top = tfidf_keywords(docs_tokens, top_k=args.tfidf_topk)

    print("\n=== Corpus Top Keywords (TF-IDF over STRICT) ===")
    for w, s in corpus_top:
        print(f"{w:20s} {s:.4f}")

    print("\n=== Per-Article Keywords (top 3) ===")
    for i, a in enumerate(strict[:10], 1):
        print(f"\n[{i}] {a.get('title','')}")
        for w, s in per_doc_top[i - 1][:3]:
            print(f"  - {w:18s} {s:.4f}")

    # 5) Optional WATCHLIST
    if args.strict_gate and args.keep_watchlist and watch:
        print_articles(watch, limit=10, title="WATCHLIST (non-matching)")


if __name__ == "__main__":
    main()

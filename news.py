# news.py
# DeepSearch API v2: /v1/global-articles -> 기사 수집 -> TF-IDF 키워드 추출
# 인증: (권장) Authorization: Bearer <API_KEY>
#       (대안) ?api_key=<API_KEY>

import re
import math
import time
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any, Optional

import requests


# =============================
# 0) 설정
# =============================
BASE_URL = "https://api-v2.deepsearch.com"
API_KEY = "9b41b3ef6f784e2a8ac87ee8dffebdbe"

DEBUG = False  # True면 응답 일부 출력

# 기본 쿼리/필터
QUERY = "김정일 OR 북한 OR 미사일 OR 전쟁"        # 필요하면 안보 키워드로 확장: "북한 OR 미사일 OR 제재 ..."
LANG = "ko"          # API가 지원할 수도/안 할 수도 있음 (지원 안 하면 무시될 수 있음)
LIMIT = 20           # 한 번에 가져올 개수
MAX_PAGES = 2        # 페이지가 지원되면 사용
PAUSE_SEC = 0.4

# 메이저 언론 필터 (옵션: 실제 publisher 표기 확인 후 맞춰야 정확히 걸림)
USE_PUBLISHER_FILTER = False
MAJOR_PUBLISHERS = {
    "연합뉴스", "MBC", "KBS", "SBS", "JTBC", "YTN",
    "조선일보", "중앙일보", "동아일보", "한겨레", "경향신문",
    "매일경제", "한국경제",
}


# =============================
# 1) 전처리/키워드(TF-IDF)
# =============================
TOKEN_RE = re.compile(r"[^0-9A-Za-z가-힣]+")

KOREAN_STOPWORDS = {
    "그리고","하지만","또는","그러나","때문에","대한","관련","이번","오늘","현재","지난",
    "위해","이후","확인","가능","기자","보도","발표","입장","정도","정말","이런","저런",
    "등","중","더","및","또","수","것",
}
EN_STOPWORDS = {
    "the","a","an","and","or","but","to","of","in","on","for","with","as","is","are",
    "was","were","be","been","it","that","this","by","from","at","about","into","over",
    "after","before","will","may","might",
}

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    text = TOKEN_RE.sub(" ", text).strip().lower()
    raw = [t for t in text.split() if len(t) >= 2]
    out = []
    for t in raw:
        if t.isdigit():
            continue
        if t in EN_STOPWORDS or t in KOREAN_STOPWORDS:
            continue
        out.append(t)
    return out

def tfidf_keywords(docs_tokens: List[List[str]], top_k: int = 15):
    N = len(docs_tokens)
    if N == 0:
        return [], []

    df = Counter()
    for toks in docs_tokens:
        df.update(set(toks))

    idf = {w: math.log((N + 1) / (df_w + 1)) + 1.0 for w, df_w in df.items()}

    per_doc_top = []
    corpus_scores = defaultdict(float)

    for toks in docs_tokens:
        tf = Counter(toks)
        doc_len = max(1, sum(tf.values()))
        scores = {}
        for w, c in tf.items():
            s = (c / doc_len) * idf.get(w, 0.0)
            scores[w] = s
            corpus_scores[w] += s
        per_doc_top.append(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k])

    corpus_top = sorted(corpus_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return corpus_top, per_doc_top


# =============================
# 2) DeepSearch /v1/global-articles 호출
# =============================
def fetch_global_articles(
    query: Optional[str] = None,
    lang: Optional[str] = None,
    limit: int = 20,
    page: int = 1,
    use_bearer: bool = True,
) -> Dict[str, Any]:
    """
    엔드포인트:
      GET /v1/global-articles

    인증:
      - Authorization: Bearer <API_KEY>  (권장)
      - 또는 ?api_key=<API_KEY>

    파라미터 스키마는 문서에 따라 다를 수 있어
    (query/q, page/offset, limit/size 등).
    여기서는 흔한 이름으로 보내고, 필요하면 쉽게 바꿀 수 있게 작성.
    """
    url = f"{BASE_URL}/v1/articles"

    headers = {}
    params: Dict[str, Any] = {}

    # 인증
    if use_bearer:
        headers["Authorization"] = f"Bearer {API_KEY}"
    else:
        params["api_key"] = API_KEY

    # 검색/필터 (API가 해당 키를 지원해야 동작)
    if query:
        # 후보 키: query / q
        params["query"] = query
    if lang:
        params["lang"] = lang

    # 페이지네이션 후보
    params["page"] = page
    params["limit"] = limit

    r = requests.get(url, headers=headers, params=params, timeout=30)

    if DEBUG:
        print("\n[DEBUG] status:", r.status_code)
        print("[DEBUG] url:", r.url)
        print("[DEBUG] body head:", r.text[:800])

    # 401/403이면 인증 방식 변경해볼 수 있게 호출부에서 처리
    r.raise_for_status()
    return r.json()


def extract_articles(payload: Any) -> List[Dict[str, Any]]:
    """
    응답 구조가 확실치 않아서 폭넓게 파싱.
    흔한 경우:
      - payload["data"] 가 리스트
      - payload["articles"] / ["items"] / ["results"] / ["docs"]
      - payload 자체가 리스트
    """
    if payload is None:
        return []

    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        return []

    # 1) direct list keys
    for k in ("articles", "items", "results", "docs", "data"):
        v = payload.get(k)
        if isinstance(v, list):
            return v

    # 2) nested dict
    for outer in ("data", "result"):
        o = payload.get(outer)
        if isinstance(o, dict):
            for k in ("articles", "items", "results", "docs"):
                v = o.get(k)
                if isinstance(v, list):
                    return v

    return []


def normalize_article(a: Dict[str, Any]) -> Dict[str, Any]:
    """
    출력/분석용으로 필드를 표준화.
    """
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


def article_to_text(a: Dict[str, Any]) -> str:
    return " ".join([a.get("title", ""), a.get("description", ""), a.get("content", "")]).strip()


def print_articles(articles: List[Dict[str, Any]], limit: int = 20):
    print(f"\n=== 기사 목록(상위 {min(limit, len(articles))}) ===")
    for i, a in enumerate(articles[:limit], 1):
        print(f"{i:02d}. [{a.get('publisher','')}] {a.get('published_at','')} | {a.get('title','')}")
        print(f"    {a.get('url','')}")


def main():
    if not API_KEY or API_KEY == "PUT_YOUR_API_KEY_HERE":
        raise RuntimeError("API_KEY를 본인 키로 교체하세요.")

    all_articles: List[Dict[str, Any]] = []

    # 1) 우선 Bearer로 시도
    for page in range(1, MAX_PAGES + 1):
        try:
            payload = fetch_global_articles(query=QUERY, lang=LANG, limit=LIMIT, page=page, use_bearer=True)
        except requests.HTTPError as e:
            # 401/403이면 api_key 방식으로 폴백 시도
            status = e.response.status_code if e.response is not None else None
            if status in (401, 403):
                payload = fetch_global_articles(query=QUERY, lang=LANG, limit=LIMIT, page=page, use_bearer=False)
            else:
                raise

        raw_articles = extract_articles(payload)
        if not raw_articles:
            break

        normed = [normalize_article(a) for a in raw_articles]

        # (옵션) 발행사 필터
        if USE_PUBLISHER_FILTER:
            normed = [a for a in normed if a.get("publisher") in MAJOR_PUBLISHERS]

        all_articles.extend(normed)
        time.sleep(PAUSE_SEC)

    print(f"\n가져온 기사 수: {len(all_articles)}")
    if not all_articles:
        print("\n❌ 기사 0개입니다.")
        print("체크:")
        print("- /v1/global-articles 응답에 실제 기사 리스트가 어떤 키에 담기는지 (DEBUG=True로 확인)")
        print("- query 파라미터명이 query가 아니라 q일 수 있음 → fetch_global_articles에서 params 키 변경")
        print("- page/limit 대신 offset/size 방식일 수 있음")
        return

    print_articles(all_articles, limit=20)

    # TF-IDF 키워드
    docs_tokens = [tokenize(article_to_text(a)) for a in all_articles]
    corpus_top, per_doc_top = tfidf_keywords(docs_tokens, top_k=12)

    print("\n=== 전체 코퍼스 Top 키워드 ===")
    for w, s in corpus_top:
        print(f"{w:20s} {s:.4f}")

    print("\n=== 기사별 Top 키워드(상위 3개) ===")
    for i, a in enumerate(all_articles[:10], 1):
        print(f"\n[{i}] {a.get('title','')}")
        for w, s in per_doc_top[i-1][:3]:
            print(f"  - {w:18s} {s:.4f}")


if __name__ == "__main__":
    main()





# """
# NewsAPI.org -> 기사 수집 -> (옵션) 한국어 필터 -> TF-IDF 키워드 추출

# ✅ 기능
# 1) NewsAPI /top-headlines 또는 /everything로 기사 가져오기
# 2) 한국 기사 위주: top-headlines는 country="kr" 사용 가능
# 3) everything은 country 파라미터가 없어서 q/language/from/to 등으로 제한
# 4) 기사 목록(제목/출처/시간/URL) 출력
# 5) 전체 Top 키워드 + 기사별 Top 키워드 출력

# ⚠️ 보안 주의
# - API 키를 코드에 직접 넣으면 유출 위험이 있습니다.
# - 절대 깃허브/공유 폴더에 그대로 올리지 마세요.
# """

# import re
# import math
# import time
# from collections import Counter, defaultdict
# from typing import List, Tuple, Optional, Dict, Any

# import requests


# # =============================
# # 0) 설정
# # =============================

# # ✅ 여기에 본인 NewsAPI.org 키를 넣으세요.
# NEWSAPI_KEY = "c4da76d8bb104fe182c7a2a8a3fcee87"

# NEWSAPI_BASE = "https://newsapi.org/v2"

# DEBUG = True

# DEFAULT_PAGE_SIZE = 30
# DEFAULT_MAX_PAGES = 2
# DEFAULT_PAUSE_SEC = 0.8

# # 한국어/한글 필터(혼입 방지)
# APPLY_HANGUL_FILTER = True


# # =============================
# # 1) NewsAPI 호출 유틸
# # =============================

# def _check_api_key():
#     if not NEWSAPI_KEY or NEWSAPI_KEY == "YOUR_NEWSAPI_KEY_HERE":
#         raise RuntimeError("NEWSAPI_KEY를 본인 키로 교체하세요.")


# def _request_newsapi(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     NewsAPI는 apiKey를 파라미터로 보내거나 헤더 X-Api-Key로 보낼 수 있음.
#     인증/프록시 이슈를 줄이기 위해 헤더 방식 사용.
#     """
#     _check_api_key()
#     url = f"{NEWSAPI_BASE}/{endpoint.lstrip('/')}"
#     headers = {"X-Api-Key": NEWSAPI_KEY}

#     r = requests.get(url, params=params, headers=headers, timeout=20)

#     # 디버그 출력
#     if DEBUG:
#         print("\n[DEBUG] REQUEST URL:", r.url)
#         print("[DEBUG] Status:", r.status_code)

#     try:
#         data = r.json()
#     except Exception:
#         data = {"_raw_text": r.text}

#     if DEBUG and r.status_code != 200:
#         print("[DEBUG] Response:", str(data)[:1200])

#     r.raise_for_status()
#     return data


# def fetch_top_headlines(
#     country: str = "kr",
#     category: Optional[str] = None,
#     q: Optional[str] = None,
#     page_size: int = DEFAULT_PAGE_SIZE,
#     max_pages: int = 1,
#     pause_sec: float = DEFAULT_PAUSE_SEC,
# ) -> List[dict]:
#     """
#     NewsAPI /top-headlines
#     - 한국 기사 위주로는 country="kr" 사용 가능
#     - category: business/technology/... (NewsAPI 정의)
#     - q: 키워드(선택)
#     """
#     all_articles: List[dict] = []

#     for page in range(1, max_pages + 1):
#         params: Dict[str, Any] = {
#             "country": country,
#             "pageSize": page_size,
#             "page": page,
#         }
#         if category:
#             params["category"] = category
#         if q:
#             params["q"] = q

#         data = _request_newsapi("/top-headlines", params)

#         if data.get("status") != "ok":
#             raise RuntimeError(f"NewsAPI error: {data}")

#         articles = data.get("articles", [])
#         if not articles:
#             break

#         all_articles.extend(articles)
#         time.sleep(pause_sec)

#     return all_articles


# def fetch_everything(
#     q: str,
#     language: str = "ko",
#     sort_by: str = "publishedAt",
#     from_date: Optional[str] = None,  # "YYYY-MM-DD" 또는 ISO8601
#     to_date: Optional[str] = None,
#     domains: Optional[str] = None,    # 예: "yna.co.kr,joongang.co.kr"
#     sources: Optional[str] = None,    # 예: "bbc-news,reuters" (NewsAPI source id)
#     page_size: int = DEFAULT_PAGE_SIZE,
#     max_pages: int = DEFAULT_MAX_PAGES,
#     pause_sec: float = DEFAULT_PAUSE_SEC,
# ) -> List[dict]:
#     """
#     NewsAPI /everything
#     - country 파라미터는 없음
#     - 대신 language/from/to/domains/sources/q로 제한
#     """
#     all_articles: List[dict] = []

#     for page in range(1, max_pages + 1):
#         params: Dict[str, Any] = {
#             "q": q,
#             "language": language,
#             "sortBy": sort_by,
#             "pageSize": page_size,
#             "page": page,
#         }
#         if from_date:
#             params["from"] = from_date
#         if to_date:
#             params["to"] = to_date
#         if domains:
#             params["domains"] = domains
#         if sources:
#             params["sources"] = sources

#         data = _request_newsapi("/everything", params)

#         if data.get("status") != "ok":
#             raise RuntimeError(f"NewsAPI error: {data}")

#         articles = data.get("articles", [])
#         if not articles:
#             break

#         all_articles.extend(articles)
#         time.sleep(pause_sec)

#     return all_articles


# # =============================
# # 2) 전처리 + 토큰화
# # =============================

# TOKEN_RE = re.compile(r"[^0-9A-Za-z가-힣]+")

# KOREAN_STOPWORDS = {
#     "그리고", "하지만", "또는", "그러나", "때문에", "대한", "관련", "이번", "오늘",
#     "현재", "지난", "위해", "이후", "확인", "가능", "기자", "보도", "발표", "입장",
#     "정도", "정말", "이런", "저런", "등", "중", "더", "및", "또", "수", "것",
# }
# EN_STOPWORDS = {
#     "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with", "as",
#     "is", "are", "was", "were", "be", "been", "it", "that", "this", "by", "from",
#     "at", "about", "into", "over", "after", "before", "will", "may", "might",
# }


# def tokenize(text: str) -> List[str]:
#     if not text:
#         return []
#     text = TOKEN_RE.sub(" ", text).strip().lower()
#     raw = [t for t in text.split() if len(t) >= 2]

#     tokens: List[str] = []
#     for t in raw:
#         if t.isdigit():
#             continue
#         if t in EN_STOPWORDS or t in KOREAN_STOPWORDS:
#             continue
#         tokens.append(t)
#     return tokens


# def article_to_text(a: dict) -> str:
#     # NewsAPI: title/description/content(종종 ...로 잘림)
#     parts = [a.get("title") or "", a.get("description") or "", a.get("content") or ""]
#     return " ".join(parts)


# def contains_hangul(s: str) -> bool:
#     if not s:
#         return False
#     return any("가" <= ch <= "힣" for ch in s)


# def filter_korean_articles(articles: List[dict]) -> List[dict]:
#     """
#     language='ko'라도 간혹 영문이 섞일 수 있어서,
#     title/description에 한글이 있는 기사만 남김.
#     """
#     out: List[dict] = []
#     for a in articles:
#         text = ((a.get("title") or "") + " " + (a.get("description") or "")).strip()
#         if contains_hangul(text):
#             out.append(a)
#     return out


# # =============================
# # 3) TF-IDF 키워드 추출
# # =============================

# def tfidf_keywords(
#     docs_tokens: List[List[str]],
#     top_k: int = 15
# ) -> Tuple[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
#     N = len(docs_tokens)
#     if N == 0:
#         return [], []

#     # DF
#     df = Counter()
#     for toks in docs_tokens:
#         df.update(set(toks))

#     # IDF
#     idf = {w: math.log((N + 1) / (df_w + 1)) + 1.0 for w, df_w in df.items()}

#     per_doc_top: List[List[Tuple[str, float]]] = []
#     corpus_scores = defaultdict(float)

#     for toks in docs_tokens:
#         tf = Counter(toks)
#         doc_len = max(1, sum(tf.values()))

#         scores: Dict[str, float] = {}
#         for w, c in tf.items():
#             tf_norm = c / doc_len
#             s = tf_norm * idf.get(w, 0.0)
#             scores[w] = s
#             corpus_scores[w] += s

#         per_doc_top.append(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k])

#     corpus_top = sorted(corpus_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
#     return corpus_top, per_doc_top


# # =============================
# # 4) 출력 유틸
# # =============================

# def print_articles_list(articles: List[dict], limit: int = 20):
#     print(f"\n=== 가져온 기사 목록(상위 {min(limit, len(articles))}) ===")
#     for i, a in enumerate(articles[:limit], start=1):
#         src = (a.get("source") or {}).get("name", "")
#         pub = a.get("publishedAt", "")
#         title = a.get("title", "")
#         url = a.get("url", "")
#         print(f"{i:02d}. [{src}] {pub} | {title}\n    {url}")


# def print_keywords(
#     corpus_top: List[Tuple[str, float]],
#     per_doc_top: List[List[Tuple[str, float]]],
#     articles: List[dict],
#     per_doc_show: int = 10,
#     per_doc_topn: int = 5,
# ):
#     print("\n=== 전체 코퍼스 Top 키워드 ===")
#     for w, s in corpus_top:
#         print(f"{w:20s} {s:.4f}")

#     print(f"\n=== 기사별 Top 키워드(상위 {per_doc_topn}개) ===")
#     for i, a in enumerate(articles[:per_doc_show], start=1):
#         title = (a.get("title") or "").strip()
#         src = (a.get("source") or {}).get("name", "")
#         print(f"\n[{i}] {title}  ({src})")
#         for w, s in per_doc_top[i-1][:per_doc_topn]:
#             print(f"  - {w:18s} {s:.4f}")


# # =============================
# # 5) 메인: 한국 기사 수집 (fallback 포함)
# # =============================

# def get_korean_news_with_fallback(
#     query_ko: str,
#     page_size: int = 30,
#     max_pages: int = 2,
# ) -> List[dict]:
#     """
#     우선순위:
#     1) top-headlines (country=kr)  -- 한국 헤드라인에 가장 강함
#     2) everything (language=ko, q=query_ko) -- 쿼리 기반 확장
#     3) everything (language=en, q=Korea OR ...) -- 응급: 한국 관련 영문 기사

#     그리고 마지막에 한글 필터(옵션) 적용.
#     """
#     # 1) 한국 헤드라인
#     try:
#         articles = fetch_top_headlines(country="kr", page_size=page_size, max_pages=1)
#     except Exception:
#         articles = []

#     # 2) 쿼리 기반 확장 (한국어)
#     if not articles:
#         try:
#             articles = fetch_everything(
#                 q=query_ko,
#                 language="ko",
#                 page_size=page_size,
#                 max_pages=max_pages,
#             )
#         except Exception:
#             articles = []

#     # 3) 응급: 영문으로라도 한국 관련
#     if not articles:
#         try:
#             articles = fetch_everything(
#                 q="Korea OR South Korea OR Seoul",
#                 language="en",
#                 page_size=page_size,
#                 max_pages=1,
#             )
#         except Exception:
#             articles = []

#     if APPLY_HANGUL_FILTER and articles:
#         filtered = filter_korean_articles(articles)
#         if DEBUG:
#             print(f"\n[DEBUG] Hangul filter: {len(filtered)}/{len(articles)} kept")
#         articles = filtered

#     return articles


# def main():
#     # ✅ 한국어 쿼리 예시(정세/안보 중심)
#     # 너무 빡세게 하면 0개가 나올 수 있으니 적당히 넓게 시작하는 게 좋아요.
#     query_ko = "국방부 OR 합동훈련 OR 미사일 OR 제재 OR 드론 OR 사이버"

#     articles = get_korean_news_with_fallback(
#         query_ko=query_ko,
#         page_size=30,
#         max_pages=2,
#     )

#     print(f"\n가져온 기사 수: {len(articles)}")
#     if not articles:
#         print("\n❌ 기사 0개입니다.")
#         print("체크리스트:")
#         print("- API 키 유효/플랜 제한(401/429 등)")
#         print("- DEBUG=True 로그에서 status/response 확인")
#         print("- top-headlines는 country=kr 가능, everything은 country 불가")
#         return

#     print_articles_list(articles, limit=20)

#     # 키워드 추출
#     docs = [article_to_text(a) for a in articles]
#     docs_tokens = [tokenize(t) for t in docs]

#     corpus_top, per_doc_top = tfidf_keywords(docs_tokens, top_k=12)

#     print_keywords(corpus_top, per_doc_top, articles, per_doc_show=10, per_doc_topn=5)


# if __name__ == "__main__":
#     main()

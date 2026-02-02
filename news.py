import re
import math
from collections import Counter, defaultdict
from typing import List, Tuple
import requests

# ✅ GNews API 키
GNEWS_API_KEY = "0cd6c9d70da4f67aea02eb283dee1206"

GNEWS_BASE = "https://gnews.io/api/v4"


# -----------------------------
# 1) GNews API 호출 (Search)
# -----------------------------
def fetch_gnews_search(
    q: str,
    lang: str = "en",
    country: str = None,
    max_results: int = 10,
) -> List[dict]:
    """
    GNews Search Endpoint:
    https://gnews.io/api/v4/search?q=...&lang=en&max=5&apikey=YOUR_API_KEY
    """
    if not GNEWS_API_KEY or GNEWS_API_KEY == "YOUR_GNEWS_KEY_HERE":
        raise RuntimeError("GNEWS_API_KEY를 본인 키로 교체하세요.")

    params = {
        "q": q,
        "lang": lang,
        "max": max_results,
        "apikey": GNEWS_API_KEY,  # ✅ GNews는 apikey
    }
    if country:
        params["country"] = country

    r = requests.get(f"{GNEWS_BASE}/search", params=params, timeout=20)
    if r.status_code != 200:
        print("Status:", r.status_code)
        print("Response:", r.text[:500])
        r.raise_for_status()

    data = r.json()
    return data.get("articles", [])


# -----------------------------
# 2) 전처리 + 토큰화
# -----------------------------
TOKEN_RE = re.compile(r"[^0-9A-Za-z가-힣]+")

KOREAN_STOPWORDS = {
    "그리고", "하지만", "또는", "그러나", "때문에", "대한", "관련", "이번", "오늘",
    "현재", "지난", "위해", "이후", "확인", "가능", "기자", "보도", "발표", "입장",
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

    tokens = []
    for t in raw:
        if t.isdigit():
            continue
        if t in EN_STOPWORDS or t in KOREAN_STOPWORDS:
            continue
        tokens.append(t)
    return tokens


def article_to_text(a: dict) -> str:
    # GNews: title / description / content (플랜에 따라 content가 짧을 수 있음)
    parts = [a.get("title") or "", a.get("description") or "", a.get("content") or ""]
    return " ".join(parts)


# -----------------------------
# 3) TF-IDF 키워드 추출
# -----------------------------
def tfidf_keywords(
    docs_tokens: List[List[str]],
    top_k: int = 15
) -> Tuple[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
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
            tf_norm = c / doc_len
            s = tf_norm * idf.get(w, 0.0)
            scores[w] = s
            corpus_scores[w] += s

        per_doc_top.append(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k])

    corpus_top = sorted(corpus_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return corpus_top, per_doc_top


# -----------------------------
# 4) 실행
# -----------------------------
def main():
    #기사 검색 키워드 쿼리
    query = "(border OR missile OR sanctions OR naval OR drone OR cyber)"
    articles = fetch_gnews_search(q=query, lang="en", max_results=20)

    print(f"가져온 기사 수: {len(articles)}\n")

    docs = [article_to_text(a) for a in articles]
    docs_tokens = [tokenize(t) for t in docs]

    corpus_top, per_doc_top = tfidf_keywords(docs_tokens, top_k=12)

    print("=== 전체 코퍼스 Top 키워드 ===")
    for w, s in corpus_top:
        print(f"{w:20s} {s:.4f}")

    print("\n=== 기사별 Top 키워드(상위 5개) ===")
    for i, a in enumerate(articles[:10], start=1):
        title = (a.get("title") or "").strip()
        src = (a.get("source") or {}).get("name", "")
        print(f"\n[{i}] {title}  ({src})")
        for w, s in per_doc_top[i-1][:5]:
            print(f"  - {w:18s} {s:.4f}")

    print("\n=== 가져온 기사 목록(상위 20) ===")
    for i, a in enumerate(articles, start=1):
        src = (a.get("source") or {}).get("name", "")
        pub = a.get("publishedAt", "")
        title = a.get("title", "")
        url = a.get("url", "")
        print(f"{i:02d}. [{src}] {pub} | {title}\n    {url}")



if __name__ == "__main__":
    main()

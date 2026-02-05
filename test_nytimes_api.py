#!/usr/bin/env python3
"""
Test NY Times Most Popular API - /emailed/1.json endpoint
With tokenization, TF-IDF analysis, and keyword filtering
"""
#EMAIL: wjlee619@gmail.com
#PW: Qazwsx123@
#API KEY: acEPq0IEyXyVLZb10uBSJ2DOFduwy8fYb5G6tHZXW9ZqxL2Q


import argparse
import math
import re
import json
import requests
from collections import Counter, defaultdict
from typing import Dict, Any, Optional, List, Tuple

# NY Times API Configuration
NYTIMES_BASE_URL = "https://api.nytimes.com/svc/search/v2"
NYTIMES_API_KEY = "acEPq0IEyXyVLZb10uBSJ2DOFduwy8fYb5G6tHZXW9ZqxL2Q"

DEBUG = False

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
def get_most_emailed_articles(
    api_key: str,
    period: int = 1,
    keyword: str = "south korea"
) -> Dict[str, Any]:
    """
    Fetch news articles from NY Times Article Search API v2.
    
    Args:
        api_key: NY Times API key
        period: Time period (not used for v2 search API)
        keyword: Search query (will be passed as-is to the API)
    
    Returns:
        API response as dictionary
    """
    url = f"{NYTIMES_BASE_URL}/articlesearch.json"
    
    params = {
        "q": keyword,
        "api-key": api_key.strip(),
        "sort": "newest",
        "page": 0
    }
    
    if DEBUG:
        print(f"[DEBUG] Fetching: {url}")
        print(f"[DEBUG] Query: {keyword}")
        print(f"[DEBUG] API Key (first 10 chars): {api_key[:10]}...")
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
        if DEBUG:
            print(f"[DEBUG] Full error details: {e}")
        return {}

def filter_by_keyword(articles: List[Dict[str, Any]], keyword: str = "war") -> List[Dict[str, Any]]:
    """Articles are already filtered by server-side search, so return all."""
    return articles

def print_articles(articles: List[Dict[str, Any]], limit: int = 20, title: str = "Articles"):
    """Pretty print articles in clean format."""
    count = min(limit, len(articles))
    print(f"\n{'='*80}")
    print(f"{title.upper()} ({count} results)")
    print(f"{'='*80}\n")
    for i, article in enumerate(articles[:limit], 1):
        title_text = article.get('title', 'N/A')[:75]
        section = article.get('section', 'N/A')
        date = article.get('published_date', 'N/A').split('T')[0]
        abstract = article.get('abstract', 'N/A')[:110].strip()
        
        print(f"{i}. {title_text}")
        print(f"   📰 {section} | 📅 {date}")
        print(f"   {abstract}...")
        print()

def main():
    ap = argparse.ArgumentParser(description="NY Times Most Popular API + TF-IDF Analysis (War keyword filtering)")
    ap.add_argument("--period", type=int, default=1, choices=[1, 7, 30],
                    help="Time period (1, 7, or 30 days)")
    ap.add_argument("--keyword", type=str, default="south korea or korea or korean or seoul or pyongyang",
                    help="Keywords to filter articles. Separate with comma, 'or', or '|' (default: south korea or korea or korean or seoul or pyongyang)")
    ap.add_argument("--limit", type=int, default=20,
                    help="Number of articles to display")
    ap.add_argument("--tfidf_topk", type=int, default=12,
                    help="Top K keywords for TF-IDF")
    ap.add_argument("--debug", action="store_true",
                    help="Enable debug mode")
    args = ap.parse_args()
    
    global DEBUG
    DEBUG = bool(args.debug)
    
    api_key = NYTIMES_API_KEY
    
    if not api_key:
        print("❌ API key not set!")
        return
    
    # Fetch most emailed articles
    print(f"\n⏳ Fetching articles from NY Times (searching for: '{args.keyword}')...")
    data = get_most_emailed_articles(api_key, keyword=args.keyword)
    
    if not data or "response" not in data:
        print("❌ No results from API")
        return
    
    response = data.get("response", {})
    articles_raw = response.get("docs", [])
    
    # Normalize article structure (v2 API uses different field names)
    articles = []
    for doc in articles_raw:
        article = {
            "title": doc.get("headline", {}).get("main", "N/A"),
            "abstract": doc.get("abstract", "N/A"),
            "section": doc.get("section_name", "N/A"),
            "published_date": doc.get("pub_date", "N/A"),
            "url": doc.get("web_url", "N/A"),
        }
        articles.append(article)
    
    print(f"✅ Retrieved {len(articles)} articles\n")
    
    if DEBUG:
        try:
            if articles_raw:
                print("[DEBUG] First raw result (truncated):")
                print(json.dumps(articles_raw[0], indent=2, ensure_ascii=False)[:10000])
        except Exception as e:
            print(f"[DEBUG] Could not dump first result: {e}")
    
    # Filter by keyword (already filtered by API, so this is a no-op)
    filtered_articles = filter_by_keyword(articles, keyword=args.keyword)
    
    if not filtered_articles:
        print(f"❌ No articles found for: '{args.keyword}'")
        print(f"\n💡 Try different keywords or broader terms.")
        return
    
    print(f"✓ Found {len(filtered_articles)} article(s) for: '{args.keyword}'\n")
    
    # Print filtered articles
    print_articles(filtered_articles, limit=args.limit, title=f"Articles matching '{args.keyword}'")
    
    # TF-IDF Analysis on filtered articles
    print(f"{'='*80}")
    print(f"TF-IDF KEYWORD ANALYSIS")
    print(f"{'='*80}\n")
    
    docs_tokens = [tokenize(" ".join([
        article.get("title", ""),
        article.get("abstract", "")
    ])) for article in filtered_articles]
    
    corpus_top, per_doc_top = tfidf_keywords(docs_tokens, top_k=args.tfidf_topk)
    
    print("📊 TOP KEYWORDS (across all articles):")
    if corpus_top:
        for i, (w, s) in enumerate(corpus_top, 1):
            bar = "█" * int(s * 30)
            print(f"  {i:2d}. {w:20s} {bar} {s:.3f}")
    else:
        print("  (no significant keywords found)")
    
    if len(filtered_articles) <= 10:
        print(f"\n📄 PER-ARTICLE KEYWORDS:\n")
        for i, article in enumerate(filtered_articles, 1):
            title = article.get('title', 'N/A')[:60]
            print(f"  [{i}] {title}")
            if i - 1 < len(per_doc_top) and per_doc_top[i - 1]:
                for w, s in per_doc_top[i - 1][:3]:
                    print(f"      • {w:20s} (score: {s:.3f})")
            else:
                print(f"      (no keywords)")
            print()
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()

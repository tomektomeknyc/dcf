# news_fetcher.py

import feedparser
import re

# ─── Helper Function to Clean HTML Tags from Headlines ─────────────────────────
def clean_html_tags(text: str) -> str:
    return re.sub(r'<.*?>', '', text or '')

# ─── Main Function: Fetch Live News for a Given Stock Ticker ──────────────────
def fetch_live_news(stock_ticker: str, region: str = "US", lang: str = "en-US"):
    """
    Fetches recent headlines for a stock from Yahoo Finance RSS feed.
    :param stock_ticker: Stock ticker symbol (e.g., 'AAPL')
    :param region: Region code for localization (default is 'US')
    :param lang: Language code (default is 'en-US')
    :return: List of dictionaries with title, link, published
    """
    feed_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={stock_ticker}&region={region}&lang={lang}"
    try:
        feed = feedparser.parse(feed_url)
    except Exception as e:
        return [{"title": f"Error fetching news for {stock_ticker}: {e}", "link": "", "published": ""}]

    headlines = []
    for entry in feed.entries[:5]:  # Limit to 5 most recent
        title = clean_html_tags(entry.get("title", ""))
        link = entry.get("link", "")
        published = entry.get("published", "")
        headlines.append({
            "title": title,
            "link": link,
            "published": published
        })

    return headlines

# ─── Optional: General Market Headlines (No Ticker) ────────────────────────────
def get_headlines():
    """
    Returns general finance headlines (no specific stock).
    """
    feed_url = "https://finance.yahoo.com/news/rssindex"
    try:
        feed = feedparser.parse(feed_url)
    except Exception as e:
        return [{"title": f"Error fetching general headlines: {e}", "link": "", "published": ""}]

    headlines = []
    for entry in feed.entries[:5]:
        title = clean_html_tags(entry.get("title", ""))
        link = entry.get("link", "")
        published = entry.get("published", "")
        headlines.append({
            "title": title,
            "link": link,
            "published": published
        })

    return headlines

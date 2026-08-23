import math
import numpy as np
import pandas as pd
import requests
import streamlit as st
import xml.etree.ElementTree as ET
from urllib.parse import quote
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# CONFIG
# =====================================================
DATASET_PATH = r"D:\Project(mtech)\dataset.xlsx"
RESULTS_PER_PAGE = 8
API_RESULTS_PER_PAGE = 7
DOMAIN_CARD_HEIGHT = 300

# =====================================================
# DOMAIN EXPLORER
# =====================================================
DOMAIN_EXPLORER = {
    "Artificial Intelligence": {
        "emoji": "🤖",
        "what": "Build systems that reason, learn, and solve problems intelligently.",
        "uses": ["Virtual assistants", "Decision support", "Smart automation"],
        "search_hint": "artificial intelligence"
    },
    "Machine Learning": {
        "emoji": "📊",
        "what": "Learn patterns from data and make predictions or decisions.",
        "uses": ["Prediction", "Classification", "Fraud detection"],
        "search_hint": "machine learning"
    },
    "Deep Learning": {
        "emoji": "🧠",
        "what": "Use multi-layer neural networks for complex text, image, and audio tasks.",
        "uses": ["Image recognition", "Speech analysis", "Medical imaging"],
        "search_hint": "deep learning"
    },
    "Natural Language Processing": {
        "emoji": "💬",
        "what": "Help computers understand, analyze, and generate human language.",
        "uses": ["Chatbots", "Sentiment analysis", "Translation"],
        "search_hint": "nlp"
    },
    "Computer Vision": {
        "emoji": "👁️",
        "what": "Help machines understand images and videos.",
        "uses": ["Face recognition", "Object detection", "Medical imaging"],
        "search_hint": "cv"
    },
    "Data Science": {
        "emoji": "📈",
        "what": "Extract insights from data using analysis, visualization, and models.",
        "uses": ["Dashboards", "Forecasting", "Business intelligence"],
        "search_hint": "data science"
    },
    "Recommendation Systems": {
        "emoji": "🎯",
        "what": "Suggest relevant items, content, or products based on patterns.",
        "uses": ["Movie recommendations", "Product suggestions", "Course recommendation"],
        "search_hint": "recommendation systems"
    },
    "IoT": {
        "emoji": "📡",
        "what": "Connect sensors and devices to collect and exchange data in real time.",
        "uses": ["Smart homes", "Health monitoring", "Smart parking"],
        "search_hint": "iot"
    },
    "Cybersecurity": {
        "emoji": "🛡️",
        "what": "Protect systems, networks, and data from attacks and misuse.",
        "uses": ["Threat detection", "Phishing detection", "Secure systems"],
        "search_hint": "cybersecurity"
    },
    "Cloud Computing": {
        "emoji": "☁️",
        "what": "Deploy and manage scalable services on cloud platforms.",
        "uses": ["Deployment", "Serverless services", "Cloud automation"],
        "search_hint": "cloud computing"
    },
    "Robotics": {
        "emoji": "🦾",
        "what": "Combine hardware and software to build intelligent machines.",
        "uses": ["Autonomous bots", "Industrial automation", "Drones"],
        "search_hint": "robotics"
    },
}

# =====================================================
# ALIASES / QUERY NORMALIZATION
# =====================================================
ALIASES = {
    "ai": "artificial intelligence",
    "artificial intelligence": "artificial intelligence",
    "ml": "machine learning",
    "machine learning": "machine learning",
    "dl": "deep learning",
    "deep learning": "deep learning",
    "nlp": "natural language processing",
    "natural language processing": "natural language processing",
    "cv": "computer vision",
    "computer vision": "computer vision",
    "recommendation": "recommendation systems",
    "recommendation system": "recommendation systems",
    "recommendation systems": "recommendation systems",
    "iot": "iot",
    "cyber security": "cybersecurity",
    "cybersecurity": "cybersecurity",
    "cloud": "cloud computing",
    "cloud computing": "cloud computing",
    "robot": "robotics",
    "robotics": "robotics",
}

APPLICATION_KEYWORDS = {
    "education": ["education", "student", "learning", "course", "academic"],
    "finance": ["finance", "financial", "banking", "fintech", "fraud", "stock"],
    "healthcare": ["healthcare", "health care", "medical", "disease", "patient", "hospital"],
}

CORE_DOMAINS = {
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "natural language processing",
    "computer vision",
    "data science",
    "recommendation systems",
    "iot",
    "cybersecurity",
    "cloud computing",
    "robotics",
}

DOMAIN_PATTERNS = {
    "artificial intelligence": ["artificial intelligence", " ai ", ",ai", "ai,", "(ai)", "/ai"],
    "machine learning": ["machine learning", " ml ", ",ml", "ml,", "(ml)", "/ml"],
    "deep learning": ["deep learning", " dl ", ",dl", "dl,", "(dl)", "/dl", "neural network"],
    "natural language processing": ["natural language processing", "nlp", "text processing", "text classification"],
    "computer vision": ["computer vision", "cv", "image processing", "image classification", "vision"],
    "data science": ["data science", "analytics", "data analysis"],
    "recommendation systems": ["recommendation systems", "recommendation system", "recommender systems", "recommender"],
    "iot": ["iot", "internet of things", "embedded systems", "sensor systems"],
    "cybersecurity": ["cybersecurity", "cyber security", "security", "network security"],
    "cloud computing": ["cloud computing", "cloud", "aws", "azure", "gcp"],
    "robotics": ["robotics", "robot", "autonomous systems", "drone"],
}

# =====================================================
# SESSION STATE
# =====================================================
defaults = {
    "query_text": "",
    "active_query": "",
    "results": [],
    "api_results": [],
    "page": 1,
    "api_page": 1,
    "bookmarks": {},
    "show_domains": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =====================================================
# PAGE CONFIG + CSS
# =====================================================
st.set_page_config(page_title="Project Recommender", page_icon="📚", layout="wide")

st.markdown(f"""
<style>
.main {{
    padding-top: 1.4rem;
}}

.block-container {{
    max-width: 1400px;
}}

.hero-box {{
    padding: 2rem;
    border-radius: 24px;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: white;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 12px 28px rgba(0,0,0,0.22);
    margin-bottom: 1rem;
}}

.hero-title {{
    font-size: 2.65rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
    letter-spacing: -0.02em;
}}

.hero-subtitle {{
    color: #cbd5e1;
    font-size: 1rem;
    line-height: 1.7;
    max-width: 900px;
}}

.section-title {{
    font-size: 1.45rem;
    font-weight: 800;
    margin-top: 0.7rem;
    margin-bottom: 0.15rem;
}}

.section-sub {{
    color: #94a3b8;
    margin-bottom: 0.65rem;
    font-size: 0.94rem;
}}

.small-note {{
    color: #94a3b8;
    font-size: 0.91rem;
    margin-top: -0.05rem;
    margin-bottom: 0.65rem;
}}

.domain-card {{
    height: {DOMAIN_CARD_HEIGHT}px;
    padding: 1rem;
    border-radius: 18px;
    background: rgba(15,23,42,0.82);
    border: 1px solid rgba(255,255,255,0.08);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}}

.domain-title {{
    font-size: 1.04rem;
    font-weight: 800;
    margin-bottom: 0.55rem;
}}

.domain-body {{
    color: #dbe4f0;
    font-size: 0.93rem;
    line-height: 1.55;
}}

.domain-muted {{
    color: #a8b3c7;
    font-size: 0.88rem;
    margin-top: 0.45rem;
}}

.difficulty-badge {{
    display: inline-block;
    padding: 0.24rem 0.65rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    margin-right: 0.4rem;
}}

.beginner {{
    background-color: rgba(34,197,94,0.17);
    color: #22c55e;
    border: 1px solid rgba(34,197,94,0.35);
}}

.intermediate {{
    background-color: rgba(245,158,11,0.18);
    color: #f59e0b;
    border: 1px solid rgba(245,158,11,0.35);
}}

.advanced {{
    background-color: rgba(239,68,68,0.18);
    color: #ef4444;
    border: 1px solid rgba(239,68,68,0.35);
}}

.domain-badge {{
    display: inline-block;
    padding: 0.24rem 0.65rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    color: #60a5fa;
    background-color: rgba(96,165,250,0.16);
    border: 1px solid rgba(96,165,250,0.35);
    margin-right: 0.4rem;
}}

.source-badge {{
    display: inline-block;
    padding: 0.24rem 0.65rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    color: #c084fc;
    background-color: rgba(192,132,252,0.16);
    border: 1px solid rgba(192,132,252,0.35);
}}

div[data-testid="stExpander"] {{
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    overflow: hidden;
    margin-bottom: 0.75rem;
    background: rgba(15,23,42,0.85);
}}

hr {{
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 1rem 0 0.8rem 0;
}}
</style>
""", unsafe_allow_html=True)

# =====================================================
# RECOMMENDATION QUALITY METRICS
# =====================================================
def compute_relevance(results, query):
    """
    Treat a result as 'relevant' if its domain matches any of the query parts.
    Returns a list of 1/0 relevance labels.
    """
    query_parts = parse_multi_domain_query(query)
    relevance = []
    for r in results:
        domain_text = str(r.get('domain', '')).lower()
        title_text = clean_text(r.get('title', ''))
        combined_text = clean_text(r.get('description', '') + " " + r.get('title', ''))

        matched = False
        for part in query_parts:
            if part in CORE_DOMAINS and domain_match(domain_text, part):
                matched = True
                break
            if part in domain_text or part in title_text:
                matched = True
                break
            if part in APPLICATION_KEYWORDS:
                keywords = APPLICATION_KEYWORDS[part]
                if any(k in combined_text for k in keywords):
                    matched = True
                    break
        relevance.append(1 if matched else 0)
    return relevance

def precision_at_k(relevance, k):
    """Fraction of top-k results that are relevant."""
    if k == 0:
        return 0.0
    top_k = relevance[:k]
    return sum(top_k) / k

def recall_at_k(relevance, k, total_relevant):
    """Fraction of all relevant items that appear in top-k."""
    if total_relevant == 0:
        return 0.0
    top_k = relevance[:k]
    return sum(top_k) / total_relevant

def mean_reciprocal_rank(relevance):
    """Reciprocal rank of the first relevant result."""
    for i, rel in enumerate(relevance):
        if rel == 1:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(relevance, k):
    """Normalized Discounted Cumulative Gain at k."""
    def dcg(rels):
        return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))

    top_k = relevance[:k]
    actual_dcg = dcg(top_k)

    ideal = sorted(relevance, reverse=True)[:k]
    ideal_dcg = dcg(ideal)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg

def compute_all_metrics(results, query, k_values=None):
    """
    Compute Precision@K, Recall@K, MRR, and NDCG@K for given results.
    Returns a dict of metric_name -> {k: value}.
    """
    if k_values is None:
        k_values = [3, 5, 10]

    relevance = compute_relevance(results, query)
    total_relevant = sum(relevance)
    n = len(relevance)

    metrics = {
        "Precision@K": {},
        "Recall@K": {},
        "NDCG@K": {},
        "MRR": mean_reciprocal_rank(relevance),
        "Total Results": n,
        "Total Relevant": total_relevant,
        "Avg Cosine Score": np.mean([r.get("score", 0) for r in results]) if results else 0,
    }

    for k in k_values:
        effective_k = min(k, n)
        metrics["Precision@K"][k] = precision_at_k(relevance, effective_k)
        metrics["Recall@K"][k] = recall_at_k(relevance, effective_k, total_relevant)
        metrics["NDCG@K"][k] = ndcg_at_k(relevance, effective_k)

    return metrics

# =====================================================
# HELPERS
# =====================================================
def clean_text(text):
    return str(text).strip().lower()

def normalize_difficulty(value):
    v = str(value).strip().lower()
    if "beginner" in v:
        return "Beginner"
    if "advanced" in v:
        return "Advanced"
    if "intermediate" in v:
        return "Intermediate"
    return "Intermediate"

def preview_text(text, max_chars=220):
    text = str(text).strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."

def fallback_description(title, domain, difficulty):
    title = str(title).strip()
    domain = str(domain).strip() if str(domain).strip() else "general"
    difficulty = str(difficulty).strip() if str(difficulty).strip() else "intermediate"
    return f"{title} is a {difficulty.lower()} level project in the {domain} domain. It can be explored as a practical academic project with implementation and evaluation."

def infer_skills(title, desc, domain, existing_skills=""):
    if str(existing_skills).strip():
        return str(existing_skills).strip()

    text = f"{title} {desc} {domain}".lower()
    skills = []

    rules = {
        "Python": ["python"],
        "Machine Learning": ["machine learning", "classification", "regression", "prediction"],
        "Deep Learning": ["deep learning", "cnn", "lstm", "gan", "transformer", "neural"],
        "NLP": ["nlp", "text", "sentiment", "language", "chatbot", "translation"],
        "Computer Vision": ["image", "vision", "face", "object detection", "video", "gesture"],
        "OpenCV": ["opencv"],
        "Pandas": ["pandas", "data analysis", "analytics", "dashboard"],
        "IoT": ["iot", "sensor", "smart home", "embedded"],
        "Cloud Computing": ["cloud", "aws", "azure", "deployment"],
        "Cybersecurity": ["security", "malware", "phishing", "intrusion"],
        "Robotics": ["robot", "robotics", "autonomous", "drone"],
        "Finance": ["finance", "banking", "stock", "fraud"],
        "Healthcare": ["health", "medical", "patient", "disease"],
        "Education": ["education", "student", "learning", "course"],
    }

    for skill, keys in rules.items():
        if any(k in text for k in keys):
            skills.append(skill)

    if not skills:
        d = str(domain).lower()
        if "natural language" in d or "nlp" in d:
            skills = ["Python", "NLP", "Text Processing"]
        elif "vision" in d or "cv" in d:
            skills = ["Python", "Computer Vision", "OpenCV"]
        elif "iot" in d:
            skills = ["Python", "IoT", "Sensors"]
        elif "robot" in d:
            skills = ["Python", "Robotics"]
        elif "cloud" in d:
            skills = ["Cloud Computing", "Deployment"]
        elif "cyber" in d:
            skills = ["Python", "Cybersecurity"]
        elif "deep learning" in d:
            skills = ["Python", "Deep Learning"]
        elif "data science" in d:
            skills = ["Python", "Pandas", "Data Analysis"]
        else:
            skills = ["Python", "Machine Learning"]

    return ", ".join(dict.fromkeys(skills))

def project_specific_roadmap(title, domain):
    t = str(title).lower()
    d = str(domain).lower()

    if "chatbot" in t:
        return "Define chatbot use-case → Collect intents/data → Build conversation logic or NLP model → Test responses → Deploy demo"
    if "recommend" in t:
        return "Collect user/item data → Preprocess data → Build recommendation model → Evaluate relevance → Deploy demo"
    if "spam" in t or "fake news" in t or "sentiment" in t:
        return "Collect text dataset → Clean and preprocess text → Train classification model → Evaluate accuracy → Build demo"
    if "mask" in t or "face" in t or "object detection" in t or "image" in t:
        return "Collect image dataset → Preprocess images → Train vision model → Evaluate results → Build demo"
    if "stock" in t or "forecast" in t:
        return "Collect time-series data → Clean and visualize trends → Train forecasting model → Evaluate predictions → Build demo"
    if "iot" in d or "sensor" in t or "smart home" in t:
        return "Select sensors/devices → Build data collection flow → Process data → Create alerts/logic → Test prototype"
    if "robot" in d or "drone" in t:
        return "Define robot task → Integrate sensors/control → Build navigation or task logic → Test safely → Improve performance"
    if "cyber" in d or "malware" in t or "phishing" in t:
        return "Collect security data/logs → Extract features → Build detection model → Evaluate security performance → Build demo"
    if "cloud" in d:
        return "Design architecture → Build service → Deploy on cloud → Monitor usage → Demonstrate workflow"
    if "natural language" in d or "nlp" in d:
        return "Define NLP problem → Collect text data → Preprocess text → Train NLP model → Evaluate → Build demo"
    if "vision" in d or "cv" in d:
        return "Define vision task → Collect image/video data → Preprocess → Train model → Evaluate → Build demo"
    if "deep learning" in d:
        return "Prepare dataset → Build deep model → Train → Tune parameters → Evaluate → Build demo"
    if "data science" in d:
        return "Collect data → Clean dataset → Perform analysis → Build model/dashboard → Evaluate findings"
    if "artificial intelligence" in d or "machine learning" in d:
        return "Define problem → Collect dataset → Preprocess → Train model → Evaluate → Build demo"

    return "Define problem → Collect data → Build solution → Evaluate results → Present demo"

def expand_query(query: str):
    q = clean_text(query)
    canonical = ALIASES.get(q, q)

    expanded_terms = [q, canonical]
    if q in APPLICATION_KEYWORDS:
        expanded_terms.extend(APPLICATION_KEYWORDS[q])

    seen = set()
    final_terms = []
    for term in expanded_terms:
        if term not in seen:
            seen.add(term)
            final_terms.append(term)

    return canonical, " ".join(final_terms)

def parse_multi_domain_query(query):
    """
    Supports inputs like:
    ai + ml, ai/ml, nlp, deep learning, iot + cloud
    Converts shortcuts into canonical domain/application names.
    """
    q = clean_text(query)

    for sep in ["+", ",", "/", "&"]:
        q = q.replace(sep, "|")

    raw_parts = [part.strip() for part in q.split("|") if part.strip()]

    canonical_parts = []
    for part in raw_parts:
        canonical_parts.append(ALIASES.get(part, part))

    # remove duplicates while preserving order
    return list(dict.fromkeys(canonical_parts))

def build_search_text_for_query(query):
    canonical, expanded_query = expand_query(query)
    parts = parse_multi_domain_query(query)

    expanded_terms = []
    for part in parts:
        expanded_terms.append(part)
        if part in DOMAIN_PATTERNS:
            expanded_terms.extend(DOMAIN_PATTERNS[part])
        if part in APPLICATION_KEYWORDS:
            expanded_terms.extend(APPLICATION_KEYWORDS[part])

    if not expanded_terms:
        expanded_terms = [expanded_query]

    return " ".join(dict.fromkeys(expanded_terms))

def domain_match(domain_text, canonical_domain):
    domain_text = f" {str(domain_text).lower()} "
    patterns = DOMAIN_PATTERNS.get(canonical_domain, [canonical_domain])
    return any(p in domain_text for p in patterns)

@st.cache_data
def load_dataset(path):
    df = pd.read_excel(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    out = pd.DataFrame()

    if "title" not in df.columns:
        raise ValueError("No title column found in dataset.")

    out["title"] = df["title"].fillna("").astype(str).str.strip()
    out["description"] = df["description"].fillna("").astype(str).str.strip() if "description" in df.columns else ""
    out["domain"] = df["domain"].fillna("general").astype(str).str.strip().str.lower() if "domain" in df.columns else "general"
    out["skills"] = df["skills"].fillna("").astype(str).str.strip() if "skills" in df.columns else ""
    out["difficulty"] = df["difficulty"].fillna("Intermediate").astype(str).apply(normalize_difficulty) if "difficulty" in df.columns else "Intermediate"

    out = out[out["title"] != ""].copy()

    out["description"] = out.apply(
        lambda row: fallback_description(row["title"], row["domain"], row["difficulty"])
        if not str(row["description"]).strip() else row["description"],
        axis=1
    )

    out["skills"] = out.apply(
        lambda row: infer_skills(row["title"], row["description"], row["domain"], row["skills"]),
        axis=1
    )

    out["roadmap"] = out.apply(
        lambda row: project_specific_roadmap(row["title"], row["domain"]),
        axis=1
    )

    out["combined"] = (
        out["title"] + " " +
        out["description"] + " " +
        out["domain"] + " " +
        out["skills"] + " " +
        out["difficulty"]
    ).apply(clean_text)

    out = out.drop_duplicates(subset=["title"]).reset_index(drop=True)
    return out

@st.cache_resource
def build_model(data):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vec.fit_transform(data["combined"])
    return vec, X

def search_projects(query, df, vectorizer, tfidf_matrix):
    canonical, expanded_query = expand_query(query)
    query_parts = parse_multi_domain_query(query)
    final_query = build_search_text_for_query(query)

    q_vec = vectorizer.transform([final_query])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()

    df_copy = df.copy()
    df_copy["score"] = scores
    df_copy["source"] = "Local Dataset"

    core_parts = [p for p in query_parts if p in CORE_DOMAINS]
    app_parts = [p for p in query_parts if p in APPLICATION_KEYWORDS]

    # MULTI-DOMAIN SEARCH
    # Example: ai + ml, nlp/deep learning, iot, cloud
    if len(core_parts) >= 2:
        def count_core_matches(domain_text):
            return sum(1 for p in core_parts if domain_match(domain_text, p))

        df_copy["domain_match_count"] = df_copy["domain"].apply(count_core_matches)

        # First preference: projects that match all selected core domains.
        strict_df = df_copy[df_copy["domain_match_count"] >= len(core_parts)].copy()

        # If dataset has very few exact combination rows, allow projects matching at least 2 domains.
        if len(strict_df) < RESULTS_PER_PAGE:
            relaxed_df = df_copy[df_copy["domain_match_count"] >= 2].copy()
            df_copy = relaxed_df if not relaxed_df.empty else strict_df
        else:
            df_copy = strict_df

        df_copy["score"] += df_copy["domain_match_count"] * 0.45

    elif len(core_parts) == 1:
        single = core_parts[0]
        df_copy = df_copy[df_copy["domain"].apply(lambda x: domain_match(x, single))]
        df_copy["score"] += df_copy["domain"].apply(lambda x: 0.50 if domain_match(x, single) else 0)

    elif canonical in APPLICATION_KEYWORDS:
        keywords = APPLICATION_KEYWORDS[canonical]
        pattern = "|".join(keywords)
        df_copy = df_copy[df_copy["combined"].str.contains(pattern, na=False)]
        df_copy["score"] += df_copy["combined"].apply(
            lambda x: 0.15 if any(k in x for k in keywords) else 0
        )

    # Application-area boost inside combination searches, e.g., cv + healthcare
    for app in app_parts:
        keywords = APPLICATION_KEYWORDS[app]
        df_copy["score"] += df_copy["combined"].apply(
            lambda x: 0.25 if any(k in x for k in keywords) else 0
        )

    # Title/domain boosts for every query part
    for part in query_parts:
        df_copy["score"] += df_copy["title"].apply(lambda x: 0.25 if part in clean_text(x) else 0)
        if part in CORE_DOMAINS:
            df_copy["score"] += df_copy["domain"].apply(lambda x: 0.35 if domain_match(x, part) else 0)

    # Keep only meaningful matches unless there are no filtered rows
    df_copy = df_copy.sort_values("score", ascending=False).reset_index(drop=True)
    return df_copy.to_dict("records")

# =====================================================
# arXiv API
# =====================================================
def make_arxiv_query(query):
    parts = parse_multi_domain_query(query)
    if parts:
        # arXiv supports AND queries. This makes multi-domain searches like ai+ml meaningful.
        query_terms = []
        for p in parts:
            if p == "artificial intelligence":
                query_terms.append('all:"artificial intelligence"')
            elif p == "machine learning":
                query_terms.append('all:"machine learning"')
            elif p == "deep learning":
                query_terms.append('all:"deep learning"')
            elif p == "natural language processing":
                query_terms.append('(all:nlp OR all:"natural language processing")')
            elif p == "computer vision":
                query_terms.append('all:"computer vision"')
            elif p == "cloud computing":
                query_terms.append('all:"cloud computing"')
            elif p in APPLICATION_KEYWORDS:
                query_terms.append('all:"' + p + '"')
            else:
                query_terms.append('all:"' + p + '"')
        return " AND ".join(query_terms)

    _, expanded_query = expand_query(query)
    return 'all:"' + expanded_query + '"'

def fetch_arxiv(query, limit=API_RESULTS_PER_PAGE * 3):
    arxiv_query = make_arxiv_query(query)

    urls = [
        "https://export.arxiv.org/api/query",
        "http://export.arxiv.org/api/query",
    ]

    params = {
        "search_query": arxiv_query,
        "start": 0,
        "max_results": limit,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    headers = {
        "User-Agent": "ProjectRecommender/1.0 (student academic project; contact: local-demo)"
    }

    for url in urls:
        try:
            response = requests.get(url, params=params, headers=headers, timeout=20)
            response.raise_for_status()
            if response.text.strip():
                return response.text
        except Exception:
            continue

    # Final fallback: broader simple query if strict AND query fails
    try:
        _, expanded_query = expand_query(query)
        fallback_params = {
            "search_query": "all:" + expanded_query,
            "start": 0,
            "max_results": limit,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        response = requests.get("http://export.arxiv.org/api/query", params=fallback_params, headers=headers, timeout=20)
        response.raise_for_status()
        return response.text
    except Exception:
        return None

def get_api_results(query):
    xml_text = fetch_arxiv(query)
    if not xml_text:
        return []

    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    results = []

    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        id_el = entry.find("atom:id", ns)
        published_el = entry.find("atom:published", ns)

        title = title_el.text.strip().replace("\n", " ") if title_el is not None and title_el.text else "Research Idea"
        summary = summary_el.text.strip().replace("\n", " ") if summary_el is not None and summary_el.text else "No description available."
        url = id_el.text.strip() if id_el is not None and id_el.text else ""
        published = published_el.text[:10] if published_el is not None and published_el.text else ""

        if published:
            summary = f"Published: {published}. " + summary

        results.append({
            "title": title,
            "description": summary,
            "domain": "research",
            "difficulty": "Advanced",
            "skills": "Research, Literature Survey, Paper Reading",
            "roadmap": "Read paper → understand core idea → identify gap → implement simplified version",
            "source": "arXiv",
            "url": url
        })

    return results

def why_recommended_text(query, row):
    canonical, _ = expand_query(query)
    domain = str(row["domain"]).strip()
    difficulty = str(row["difficulty"]).strip()

    if canonical in clean_text(row["title"]):
        return f"Strong title match for your search and suitable for {difficulty.lower()} level."
    if canonical in clean_text(domain):
        return f"Directly related to the {domain} domain and suitable for {difficulty.lower()} exploration."
    return f"Relevant to your search and grouped under {difficulty.lower()} difficulty."

def difficulty_badge_html(difficulty):
    d = str(difficulty).strip().lower()
    if d == "beginner":
        return '<span class="difficulty-badge beginner">Beginner</span>'
    if d == "advanced":
        return '<span class="difficulty-badge advanced">Advanced</span>'
    return '<span class="difficulty-badge intermediate">Intermediate</span>'

def add_bookmark(item_dict):
    st.session_state.bookmarks[item_dict["title"]] = item_dict

def render_expander_card(row_dict, idx, query):
    source = row_dict.get("source", "Local Dataset")

    with st.expander(f"{idx}. {row_dict['title']}", expanded=False):
        st.markdown(
            f"""
            <div style="margin-bottom:0.6rem;">
                {difficulty_badge_html(row_dict['difficulty'])}
                <span class="domain-badge">{row_dict['domain']}</span>
                <span class="source-badge">{source}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption(why_recommended_text(query, row_dict))
        st.write(row_dict["description"])
        st.markdown(f"**Skills:** {row_dict['skills']}")
        st.markdown(f"**Roadmap:** {row_dict['roadmap']}")

        c1, c2 = st.columns([1, 1])
        with c1:
            payload = {
                "title": row_dict["title"],
                "description": row_dict["description"],
                "domain": row_dict["domain"],
                "difficulty": row_dict["difficulty"],
                "skills": row_dict["skills"],
                "roadmap": row_dict["roadmap"],
                "source": source,
                "url": row_dict.get("url", ""),
                "notes": st.session_state.bookmarks.get(row_dict["title"], {}).get("notes", "")
            }
            st.button(
                "🔖 Bookmark",
                key=f"bookmark_{source}_{idx}_{row_dict['title']}",
                on_click=add_bookmark,
                args=(payload,),
                use_container_width=True
            )
        with c2:
            if row_dict.get("url", ""):
                st.markdown(f"[Open link]({row_dict['url']})")

def render_bookmark_card(item, key):
    st.markdown(f"### {item['title']}")
    st.markdown(
        f"""
        {difficulty_badge_html(item['difficulty'])}
        <span class="domain-badge">{item['domain']}</span>
        <span class="source-badge">{item['source']}</span>
        """,
        unsafe_allow_html=True
    )
    st.write(preview_text(item["description"], 240))
    st.markdown(f"**Skills:** {item['skills']}")
    st.markdown(f"**Roadmap:** {item['roadmap']}")

    note_key = f"note_{key}"
    updated_note = st.text_area(
        f"Notes for {item['title']}",
        value=st.session_state.bookmarks[key].get("notes", ""),
        key=note_key,
        height=80
    )
    st.session_state.bookmarks[key]["notes"] = updated_note

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Remove Bookmark", key=f"remove_{key}"):
            del st.session_state.bookmarks[key]
            st.rerun()
    with c2:
        if item.get("url", ""):
            st.markdown(f"[Open link]({item['url']})")

def run_recommendation(query, df, vectorizer, tfidf_matrix):
    st.session_state.active_query = query
    # Clear previous results FIRST so stale data never shows on error
    st.session_state.results = []
    st.session_state.api_results = []
    try:
        st.session_state.results = search_projects(query, df, vectorizer, tfidf_matrix)
        st.session_state.api_results = get_api_results(query)
    except Exception as e:
        st.error(f"Search failed: {e}")
    st.session_state.page = 1
    st.session_state.api_page = 1

# =====================================================
# LOAD DATA
# =====================================================
try:
    df = load_dataset(DATASET_PATH)
    vectorizer, tfidf_matrix = build_model(df)
except Exception as e:
    st.error(f"Dataset loading failed: {e}")
    st.stop()

# =====================================================
# HERO
# =====================================================
st.markdown("""
<div class="hero-box">
    <div class="hero-title">Project Recommender</div>
    <div class="hero-subtitle">
        Understand domains, explore project ideas, compare difficulty levels, and shortlist the best ideas with notes.
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# SEARCH + ACTIONS
# =====================================================
st.markdown('<div class="section-title">Search Projects</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Search by interest, use-case, domain, or idea.</div>', unsafe_allow_html=True)

def on_query_change():
    q = st.session_state.query_text.strip()
    if q:
        run_recommendation(q, df, vectorizer, tfidf_matrix)

query_col, rec_col, exp_col = st.columns([7.4, 1.4, 1.2])
with query_col:
    st.text_input(
        "Enter your interest / domain / idea",
        placeholder='Examples: "ai + ml", "nlp/deep learning", "iot, cloud", "cv", "healthcare"',
        key="query_text",
        on_change=on_query_change
    )
with rec_col:
    st.write("")
    st.write("")
    if st.button("Recommend", type="primary", use_container_width=True):
        q = st.session_state.query_text.strip()
        if q:
            run_recommendation(q, df, vectorizer, tfidf_matrix)
        else:
            st.warning("Please enter a query.")
with exp_col:
    st.write("")
    st.write("")
    if st.button("Explore", use_container_width=True):
        st.session_state.show_domains = not st.session_state.show_domains
        st.rerun()

st.markdown('<div class="small-note">Try single domains like ai, ml, nlp, cv or combinations like ai + ml, iot/cloud, nlp,deep learning. Application searches: education, finance, healthcare.</div>', unsafe_allow_html=True)

# =====================================================
# DOMAIN EXPLORER
# =====================================================
if st.session_state.show_domains:
    st.markdown('<div class="section-title">Explore Domains</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Read a quick description and explore related ideas directly.</div>', unsafe_allow_html=True)

    domain_names = list(DOMAIN_EXPLORER.keys())
    for row_start in range(0, len(domain_names), 3):
        cols = st.columns(3, gap="medium")
        for j in range(3):
            idx = row_start + j
            if idx < len(domain_names):
                domain_name = domain_names[idx]
                info = DOMAIN_EXPLORER[domain_name]
                with cols[j]:
                    uses_html = "".join([f"<li>{u}</li>" for u in info["uses"]])
                    st.markdown(
                        f"""
                        <div class="domain-card">
                            <div>
                                <div class="domain-title">{info['emoji']} {domain_name}</div>
                                <div class="domain-body">{info['what']}</div>
                                <div class="domain-muted"><b>Used in:</b></div>
                                <div class="domain-muted">
                                    <ul style="margin-top:0.15rem; padding-left:1.1rem;">
                                        {uses_html}
                                    </ul>
                                </div>
                                <div class="domain-muted">Try searching: "{info["search_hint"]}"</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    if st.button(f"Explore {domain_name}", key=f"explore_{domain_name}", use_container_width=True):
                        st.session_state.show_domains = False
                        run_recommendation(info["search_hint"], df, vectorizer, tfidf_matrix)
                        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

# =====================================================
# TWO-COLUMN RESULTS
# =====================================================
if st.session_state.active_query:
    left_col, right_col = st.columns([1.08, 0.92], gap="large")

    with left_col:
        st.markdown('<div class="section-title">Project Ideas</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">Ranked project ideas from your curated dataset for <strong>"{st.session_state.active_query}"</strong>.</div>', unsafe_allow_html=True)

        results = st.session_state.results
        active_query = st.session_state.active_query

        start = (st.session_state.page - 1) * RESULTS_PER_PAGE
        end = start + RESULTS_PER_PAGE
        page_data = results[start:end]

        if page_data:
            for i, row in enumerate(page_data, start=1):
                render_expander_card(row, start + i, active_query)

            p1, p2 = st.columns(2)
            with p1:
                if st.button("Previous", key="prev_local", use_container_width=True):
                    if st.session_state.page > 1:
                        st.session_state.page -= 1
                        st.rerun()
            with p2:
                if st.button("Next", key="next_local", use_container_width=True):
                    if end < len(results):
                        st.session_state.page += 1
                        st.rerun()

            if end >= len(results):
                st.info("You have reached the end of local project ideas for this search.")
        else:
            st.info("No strong local matches found for this query.")

    with right_col:
        st.markdown('<div class="section-title">Trending Research Ideas</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">Fresh research directions from arXiv related to <strong>"{st.session_state.active_query}"</strong>.</div>', unsafe_allow_html=True)

        api_results = st.session_state.api_results
        active_query = st.session_state.active_query

        api_start = (st.session_state.api_page - 1) * API_RESULTS_PER_PAGE
        api_end = api_start + API_RESULTS_PER_PAGE
        api_page_data = api_results[api_start:api_end]

        if api_page_data:
            for i, row in enumerate(api_page_data, start=1):
                render_expander_card(row, api_start + i, active_query)

            r1, r2 = st.columns(2)
            with r1:
                if st.button("Previous Research", key="prev_api", use_container_width=True):
                    if st.session_state.api_page > 1:
                        st.session_state.api_page -= 1
                        st.rerun()
            with r2:
                if st.button("Next Research", key="next_api", use_container_width=True):
                    if api_end < len(api_results):
                        st.session_state.api_page += 1
                        st.rerun()

            if api_end >= len(api_results):
                st.info("No more research ideas are available right now.")
        else:
            st.info("Live research ideas are temporarily unavailable.")

# =====================================================
# SHORTLIST
# =====================================================
st.markdown('<div class="section-title">My Shortlisted Projects</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Bookmark interesting ideas and write quick notes while deciding.</div>', unsafe_allow_html=True)

if not st.session_state.bookmarks:
    st.info("No bookmarked projects yet.")
else:
    with st.expander("View My Shortlist", expanded=True):
        export_rows = []

        for key, item in list(st.session_state.bookmarks.items()):
            with st.container(border=True):
                render_bookmark_card(item, key)
                export_rows.append({
                    "title": item["title"],
                    "description": item["description"],
                    "domain": item["domain"],
                    "difficulty": item["difficulty"],
                    "skills": item["skills"],
                    "roadmap": item["roadmap"],
                    "source": item["source"],
                    "notes": st.session_state.bookmarks[key]["notes"]
                })

        if export_rows:
            export_df = pd.DataFrame(export_rows)
            csv_data = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Shortlist as CSV",
                data=csv_data,
                file_name="shortlisted_projects.csv",
                mime="text/csv",
                use_container_width=True
            )

# =====================================================
# RECOMMENDATION QUALITY METRICS PANEL
# =====================================================
if st.session_state.active_query and st.session_state.results:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Recommendation Quality Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How well the recommender performed for your last search query.</div>', unsafe_allow_html=True)

    all_metrics = compute_all_metrics(st.session_state.results, st.session_state.active_query)

    # Top-level KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Results", all_metrics["Total Results"])
    k2.metric("Relevant Found", all_metrics["Total Relevant"])
    k3.metric("MRR", f"{all_metrics['MRR']:.3f}")
    k4.metric("Avg Cosine Score", f"{all_metrics['Avg Cosine Score']:.4f}")
    relevancy_rate = (all_metrics['Total Relevant'] / all_metrics['Total Results'] * 100) if all_metrics['Total Results'] > 0 else 0
    k5.metric("Relevancy Rate", f"{relevancy_rate:.1f}%")

    # Precision / Recall / NDCG at different K values
    st.markdown('**Metrics @ K**', unsafe_allow_html=True)
    k_col1, k_col2, k_col3 = st.columns(3)

    with k_col1:
        st.markdown("**Precision@K**")
        for k, val in all_metrics["Precision@K"].items():
            st.progress(min(val, 1.0), text=f"K={k}: {val:.3f}")

    with k_col2:
        st.markdown("**Recall@K**")
        for k, val in all_metrics["Recall@K"].items():
            st.progress(min(val, 1.0), text=f"K={k}: {val:.3f}")

    with k_col3:
        st.markdown("**NDCG@K**")
        for k, val in all_metrics["NDCG@K"].items():
            st.progress(min(val, 1.0), text=f"K={k}: {val:.3f}")

    # Cosine similarity score display
    if st.session_state.results:
        st.markdown('**Similarity Scores**', unsafe_allow_html=True)
        st.markdown('<div class="small-note">Relative similarity strength for the top results.</div>', unsafe_allow_html=True)

        top_scores = st.session_state.results[:15]
        max_score = max((r.get("score", 0) for r in top_scores), default=1) or 1

        # Header row
        h1, h2, h3 = st.columns([0.4, 3.4, 1])
        with h1:
            st.markdown('<span style="color:#94a3b8; font-size:0.78rem; font-weight:600;">#</span>', unsafe_allow_html=True)
        with h2:
            st.markdown('<span style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Project</span>', unsafe_allow_html=True)
        with h3:
            st.markdown('<div style="text-align:right;"><span style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Score</span></div>', unsafe_allow_html=True)

        st.markdown('<hr style="margin:0.2rem 0 0.6rem 0; border-color:rgba(255,255,255,0.06);">', unsafe_allow_html=True)

        for i, r in enumerate(top_scores, 1):
            score = round(r.get("score", 0), 4)
            pct = (score / max_score * 100) if max_score > 0 else 0

            c1, c2, c3 = st.columns([0.4, 3.4, 1])
            with c1:
                st.markdown(f'<span style="color:#64748b; font-size:0.88rem;">{i}</span>', unsafe_allow_html=True)
            with c2:
                st.progress(min(pct / 100, 1.0), text=r["title"])
            with c3:
                st.markdown(f'<div style="text-align:right; color:#60a5fa; font-size:0.88rem; font-weight:700;">{score:.4f}</div>', unsafe_allow_html=True)
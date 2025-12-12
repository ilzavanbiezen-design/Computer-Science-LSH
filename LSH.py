import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any
import re
import hashlib
import random
from itertools import combinations
import matplotlib.pyplot as plt
import math
from typing import Tuple
from collections import defaultdict

# ---------- Data structure ----------
NUM_HASHES = 180  # length of the MinHash signature
MAX_HASH = (1 << 61) - 1  # large prime-ish modulus (2^61 - 1)
STOP_TRIGRAMS: Set[str] = set()
IDF: Dict[str, float] = {}

@dataclass
class Product:
    pid: int
    shop: str
    title: str
    model_id: str              # gold label (evaluation only)
    features: Dict[str, Any]
    tokens: Set[str] = field(default_factory=set)
    model_tokens: Set[str] = field(default_factory=set)
    title_tokens: Set[str] = field(default_factory=set)

    # normalized fields extracted later
    brand: str = ""
    size_inch: float | None = None
    resolution: str | None = None

    # MinHash signature (to be filled later)
    signature: List[int] | None = None


def load_products_from_json(json_path: str) -> List[Product]:
    """
    Reads TVs-all-merged.json (modelID -> list of offers) and returns a flat list of Product objects.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    products: List[Product] = []
    pid = 0

    for model_id, offers in raw.items():
        for offer in offers:
            shop = offer.get("shop", "")
            title = offer.get("title", "")
            features = offer.get("featuresMap", {})

            products.append(Product(
                pid=pid,
                shop=shop,
                title=title,
                model_id=model_id,  # DO NOT use in matching
                features=features
            ))
            pid += 1

    return products

# Load the data
#json_path = "/Users/ilzav/Downloads/TVs-all-merged/TVs-all-merged.json"
#print("Loading data from JSON...")
#products = load_products_from_json(json_path)
#print(f"Loaded {len(products)} product offers.")

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

IMPORTANT_TECH_WORDS = {"led", "lcd", "plasma", "hdtv", "uhd", "smart", "3d", "oled"}
def extract_informative_title_tokens(title: str, brand: str | None = None) -> set[str]:
    title_lower = title.lower().replace('"', ' ').replace("'", " ")
    raw_tokens = re.split(r'[^a-z0-9]+', title_lower)

    tokens: set[str] = set()
    for w in raw_tokens:
        if not w:
            continue

        # model-like: letters AND digits, length >= 3
        if re.search(r'[a-z]', w) and re.search(r'\d', w) and len(w) >= 3:
            tokens.add(w)
            continue

        # tech words
        if w in IMPORTANT_TECH_WORDS:
            tokens.add(w)
            continue

        # brand from title if matches
        if brand and w == brand.lower():
            tokens.add(f"brand={brand.lower()}")
            continue

    return tokens

def extract_char_trigrams(title: str) -> set[str]:
    """
    Return a set of character 3-grams from a normalized title.
    E.g. "samsung 46 inch" -> {"sam", "ams", "msu", ...}
    """
    norm = normalize_text(title)       # already in your code
    norm = norm.replace(" ", "_")      # keep word boundaries but as a char
    grams: set[str] = set()
    for i in range(len(norm) - 2):
        g = norm[i:i+3]
        # skip ultra-common stop-grams
        if g in STOP_TRIGRAMS:
            continue
        grams.add(g)
    return grams

def compute_token_df(products: List[Product]) -> dict[str, int]:
    df = defaultdict(int)
    for p in products:
        seen: set[str] = set(p.tokens)
        for t in seen:
            df[t] += 1
    return df

def compute_token_idf(df: dict[str, int], n_docs: int) -> dict[str, float]:
    idf = {}
    for t, c in df.items():
        # classic smooth IDF
        idf[t] = math.log((n_docs + 1) / (c + 1)) + 1.0
    return idf

def compute_trigram_document_frequencies(products: List[Product]) -> Dict[str, int]:
    """
    For each product, compute the set of 3-grams in its title (no stop-gram filter yet)
    and count in how many products each 3-gram appears.
    """
    df: Dict[str, int] = defaultdict(int)
    for p in products:
        norm = normalize_text(p.title).replace(" ", "_")
        grams_for_doc: set[str] = set()
        for i in range(len(norm) - 2):
            grams_for_doc.add(norm[i:i+3])
        for g in grams_for_doc:
            df[g] += 1
    return df

def init_stop_trigrams(products: List[Product], df_ratio: float) -> None:
    """
    Initialize the global STOP_TRIGRAMS set with 3-grams that appear in more than
    df_ratio fraction of all products (e.g. 0.6 = 60%).
    """
    global STOP_TRIGRAMS

    df = compute_trigram_document_frequencies(products)
    n_docs = len(products)
    cutoff = df_ratio * n_docs

    STOP_TRIGRAMS = {g for g, c in df.items() if c >= cutoff}

    print(f"Identified {len(STOP_TRIGRAMS)} stop-grams (df_ratio >= {df_ratio})")

def build_tokens_for_product(p: Product, brand_lexicon: set[str]) -> None:
    tokens: Set[str] = set()

    # --- NORMALIZED FEATURES (as you already do) ---
    brand = parse_brand(p.features)
    title_brand = parse_brand_from_title(p.title, brand_lexicon)
    if not brand and title_brand:
        brand = title_brand
    size_inch = parse_size_inch(p.features)
    if size_inch is None:
        size_inch = parse_size_from_title(p.title)
    resolution = parse_resolution(p.features)

    p.brand = brand
    p.size_inch = size_inch
    p.resolution = resolution

    # --- TITLE TOKENS: NOW CHAR TRIGRAMS, NOT MODEL WORDS ---
    core_title = extract_core_title(p.title, brand)
    title_3grams = extract_char_trigrams(core_title)
    tokens |= title_3grams

    title_tokens = extract_informative_title_tokens(p.title)
    p.title_tokens = title_tokens

    # --- FEATURE TOKENS (keep these, they’re good and different from prior work) ---
    if brand:
        tokens.add(f"{brand.lower()}")
    if size_inch is not None:
        tokens.add(f"{round(size_inch)}in")
    if resolution:
        tokens.add(f"{resolution}")

    # UPC etc.
    upc = p.features.get("UPC") or p.features.get("Upc") or p.features.get("upc")
    if upc:
        upc_norm = re.sub(r'\D+', '', str(upc))
        if upc_norm:
            tokens.add(f"upc={upc_norm}")

    #p.model_tokens = extract_model_tokens(p.title)
    p.tokens = tokens

def build_brand_lexicon(products: List[Product]) -> set[str]:
    brands = set()
    for p in products:
        b = parse_brand(p.features)
        if b:
            brands.add(b.lower())
    return brands

def parse_brand_from_title(title: str, brand_lexicon: set[str]) -> str | None:
    words = re.split(r'[^a-z0-9]+', title.lower())
    for w in words:
        if w in brand_lexicon:
            return w
    return None

def parse_brand(features: Dict[str, Any]) -> str:
    val = features.get("Brand") or features.get("brand") or features.get("Brand name") or features.get("brand name")
    if not val:
        return ""
    norm = normalize_text(str(val))
    return norm.split()[0] if norm else ""


def parse_size_inch(features: Dict[str, Any]) -> float | None:
    keys = [
        "Screen Size Class",
        "Screen Size (Measured Diagonally)",
        "screen size",
        "Screen Size",
        "Size",
        "size"
    ]
    for key in keys:
        if key in features:
            raw = str(features[key])
            m = re.search(r'(\d+(\.\d+)?)', raw)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
    return None

def parse_size_from_title(title: str) -> float | None:
    # Look for something like 32" , 70" , 69.5" with optional words after
    m = re.search(r'(\d+(\.\d+)?)\s*("?)(\s*(class|diag|diagonal|inch|in))?', title.lower())
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None

def parse_resolution(features: Dict[str, Any]) -> str | None:
    val = features.get("Maximum Resolution") or features.get("max resolution")
    if not val:
        return None
    s = normalize_text(str(val))
    if "3840" in s or "2160" in s or "4k" in s:
        return "4k"
    if "1920" in s or "1080" in s:
        return "1080p"
    if "1366" in s or "768" in s or "720" in s:
        return "720p"
    # fallback
    digits = re.findall(r'\d+', s)
    return "x".join(digits) if digits else None

def build_tokens_for_all(products: List[Product], brand_lexicon: set[str]) -> None:
    for p in products:
        build_tokens_for_product(p, brand_lexicon)

def extract_core_title(title: str, brand: str | None) -> str:
    norm = title.lower()
    if brand:
        idx = norm.find(brand.lower())
        if idx != -1:
            return title[idx:]  # keep original casing, from brand onwards
    # fallback: if no brand or not found, maybe strip prefix up to first colon
    if ':' in title:
        return title.split(':', 1)[1]
    return title
#---------------------MinHashing---------------------

def create_hash_functions(num_hashes: int, max_hash: int) -> List[tuple[int, int]]:
    """
    Create 'num_hashes' random linear hash functions of the form:
    h(x) = (a * x + b) mod max_hash
    """
    # random.seed(42)  # for reproducibility
    funcs: List[tuple[int, int]] = []
    for _ in range(num_hashes):
        a = random.randrange(1, max_hash)
        b = random.randrange(0, max_hash)
        funcs.append((a, b))
    return funcs

def token_to_int(token: str) -> int:
    """
    Stable mapping from a token string to an integer in [0, max_hash).
    """
    h = hashlib.md5(token.encode("utf-8")).hexdigest()
    x = int(h, 16) % MAX_HASH
    return x

def compute_minhash_signature(tokens: Set[str],
                              hash_funcs: List[tuple[int, int]],
                              max_hash: int) -> List[int]:
    """
    Compute the MinHash signature for a given set of tokens.
    """
    num_hashes = len(hash_funcs)
    # initialize all positions with max_hash (acts like +∞)
    signature = [max_hash] * num_hashes

    for t in tokens:
        x = token_to_int(t)
        for i, (a, b) in enumerate(hash_funcs):
            h = (a * x + b) % max_hash
            if h < signature[i]:
                signature[i] = h

    return signature

def build_minhash_signatures(products: List[Product],
                             num_hashes: int = NUM_HASHES) -> List[tuple[int, int]]:
    """
    Build MinHash signatures for all products.
    Returns the list of hash functions used (so you can reuse them later).
    """
    hash_funcs = create_hash_functions(num_hashes, MAX_HASH)

    for p in products:
        p.signature = compute_minhash_signature(p.tokens, hash_funcs, MAX_HASH)

    return hash_funcs

#---------------------------compute actual pairs------------------------------

def compute_true_duplicate_groups(products: List[Product]) -> List[List[int]]:
    """
    Gold *groups* of duplicates based on model_id.
    Each group is a list of product indices that share the same model_id.
    We only keep groups that involve at least two different shops
    (to mirror your original cross-shop pair definition).
    """
    by_model: Dict[str, List[int]] = defaultdict(list)

    for idx, p in enumerate(products):
        mid = p.model_id
        if not mid:
            continue
        by_model[mid].append(idx)

    gold_groups: List[List[int]] = []
    for indices in by_model.values():
        if len(indices) < 2:
            continue
        shops = {products[i].shop for i in indices}
        if len(shops) >= 2:
            gold_groups.append(indices)

    return gold_groups


def compute_true_duplicate_pairs(products: List[Product]) -> Set[Tuple[int, int]]:
    """
    Backwards-compatible: derive pair-level gold from gold groups.
    Only include cross-shop pairs (same logic as before).
    """
    gold_groups = compute_true_duplicate_groups(products)
    true_dup_pairs: Set[Tuple[int, int]] = set()

    for group in gold_groups:
        for i, j in combinations(group, 2):
            if products[i].shop != products[j].shop:
                true_dup_pairs.add((i, j))

    return true_dup_pairs

def count_cross_shop_pairs(products: List[Product]) -> int:
    n = len(products)
    total = 0
    for i, j in combinations(range(n), 2):
        if products[i].shop != products[j].shop:
            total += 1
    return total

#-----------------LSH 2 ----------------------------

def get_b_r_for_threshold(n: int, target_t: float) -> Tuple[int, int, float]:
    """
    Find integers (b, r) such that:
      - n = b * r
      - t_hat = (1 / b) ** (1 / r) is as close as possible to target_t

    Returns (b, r, t_hat).
    """
    if not (0.0 < target_t < 1.0):
        raise ValueError("target_t must be in (0,1)")

    candidates: List[Tuple[int, int, float]] = []

    # enumerate all factor pairs (b, r) with b * r = n
    for r in range(1, n + 1):
        if n % r != 0:
            continue
        b = n // r
        t_hat = (1.0 / b) ** (1.0 / r)
        candidates.append((b, r, t_hat))

    # choose the (b, r) whose t_hat is closest to target_t (in log-space for stability)
    best_b, best_r, best_t_hat = min(
        candidates,
        key=lambda x: abs(math.log(x[2]) - math.log(target_t))
    )

    return best_b, best_r, best_t_hat

def lsh_candidate_pairs_br(products: List[Product],
                           bands: int,
                           rows_per_band: int,
                           only_cross_shop: bool = True) -> Set[Tuple[int, int]]:
    """
    LSH with b bands, r rows per band.
    - bands = b
    - rows_per_band = r
    Signature length n must satisfy: n = bands * rows_per_band.
    Returns pairs of (i, j) where i,j are indices in `products`.
    """
    if not products:
        return set()

    sig_len = len(products[0].signature)
    if sig_len != bands * rows_per_band:
        raise ValueError(
            f"Signature length {sig_len} != bands * rows_per_band "
            f"({bands} * {rows_per_band} = {bands * rows_per_band})"
        )

    # buckets: (band_idx, band_hash) -> list of product indices
    buckets: Dict[Tuple[int, int], List[int]] = {}

    for band_idx in range(bands):
        start = band_idx * rows_per_band
        end = start + rows_per_band

        for idx, p in enumerate(products):
            band_slice = tuple(p.signature[start:end])
            key = (band_idx, hash(band_slice))
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(idx)

    candidate_pairs: Set[Tuple[int, int]] = set()

    for key, idx_list in buckets.items():
        if len(idx_list) < 2:
            continue
        for i, j in combinations(idx_list, 2):
            if only_cross_shop and products[i].shop == products[j].shop:
                continue
            if i == j:
                continue
            pair = (min(i, j), max(i, j))
            candidate_pairs.add(pair)

    return candidate_pairs

def passes_hard_filters(p1: Product, p2: Product) -> bool:
    # Brand filter
    if p1.brand and p2.brand and p1.brand != p2.brand:
        return False

    # Resolution filter
    if p1.resolution and p2.resolution and p1.resolution != p2.resolution:
        return False

    # Size filter (allow some tolerance)
    if p1.size_inch and p2.size_inch:
        if abs(p1.size_inch - p2.size_inch) > 1.0:
            return False

    return True

def jaccard(a: Set[str], b: Set[str]) -> float:
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0

def weighted_jaccard(a: Set[str], b: Set[str], idf: dict[str, float]) -> float:
    inter = a & b
    union = a | b

    if not union:
        return 0.0

    num = sum(idf.get(t, 0.0) for t in inter)
    den = sum(idf.get(t, 0.0) for t in union)

    return num / den if den > 0 else 0.0

def signature_similarity(sig1: List[int], sig2: List[int]) -> float:
    """
    Estimate Jaccard similarity from two MinHash signatures.
    """
    assert len(sig1) == len(sig2)
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)

def get_same_model_pairs(products: List[Product], max_pairs: int = 5) -> List[tuple[int, int]]:
    """
    Return up to max_pairs pairs (i, j) where model_id is the same
    and shop is different (likely true duplicates).
    """
    pairs: List[tuple[int, int]] = []
    n = len(products)
    for i, j in combinations(range(n), 2):
        if products[i].model_id == products[j].model_id and products[i].shop != products[j].shop:
            pairs.append((i, j))
            if len(pairs) >= max_pairs:
                break
    return pairs


def get_different_model_pairs(products: List[Product], max_pairs: int = 5) -> List[tuple[int, int]]:
    """
    Return up to max_pairs pairs (i, j) where model_id is different
    and shop is different (likely non-duplicates).
    """
    pairs: List[tuple[int, int]] = []
    n = len(products)
    for i, j in combinations(range(n), 2):
        if products[i].model_id != products[j].model_id and products[i].shop != products[j].shop:
            pairs.append((i, j))
            if len(pairs) >= max_pairs:
                break
    return pairs

def is_duplicate(p1: Product, p2: Product, theta: float) -> bool:
    if not passes_hard_filters(p1, p2):
        return False
    #if p1.model_tokens and p2.model_tokens:
        if not (p1.model_tokens & p2.model_tokens):
            return False
    simTitle = jaccard(p1.title_tokens, p2.title_tokens)
    sim3Gram = jaccard(p1.tokens, p2.tokens)
    weightedSim = weighted_jaccard(p1.tokens, p2.tokens, IDF)
    return weightedSim >= theta

def cluster_duplicates_union_find(products: List[Product],
                                  candidate_pairs: Set[Tuple[int, int]],
                                  theta: float) -> List[List[int]]:
    """
    Build duplicate clusters (groups) using single-linkage-style clustering:
      - For every LSH candidate pair (i, j), compute similarity via is_duplicate.
      - If is_duplicate(products[i], products[j], theta) is True, we connect i and j.
      - Clusters are the connected components under these edges.

    Returns:
        A list of clusters, each cluster is a list of product indices (len >= 2).
    """

    n = len(products)
    if n == 0:
        return []

    # --- Union–find (Disjoint Set) structure ---
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    # --- Add edges for "strong enough" pairs ---
    for i, j in candidate_pairs:
        # guard just in case indices are out of range
        if 0 <= i < n and 0 <= j < n:
            if is_duplicate(products[i], products[j], theta=theta):
                union(i, j)

    # --- Collect clusters by root representative ---
    clusters_by_root: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        root = find(i)
        clusters_by_root[root].append(i)

    # Keep only true clusters (size >= 2)
    clusters: List[List[int]] = [
        sorted(idxs) for idxs in clusters_by_root.values() if len(idxs) >= 2
    ]

    return clusters

def clusters_to_pairs(clusters: List[List[int]]) -> Set[Tuple[int, int]]:
    """
    Convert clusters of indices into a set of all intra-cluster pairs.
    """
    pairs: Set[Tuple[int, int]] = set()
    for idxs in clusters:
        for i, j in combinations(sorted(idxs), 2):
            pairs.add((i, j))
    return pairs

def f1_score(precision: float, recall: float) -> float:
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def f1_star(pair_quality: float, pair_completeness: float) -> float:
    """
    F1* = harmonic mean of pair quality and pair completeness.
    (blocking quality measure)
    """
    return (2 * pair_quality * pair_completeness /
            (pair_quality + pair_completeness)) if (pair_quality + pair_completeness) > 0 else 0.0

def evaluate_baseline_cjs_all_pairs(products: List[Product], theta: float) -> Dict[str, float]:
    """
    Baseline: constrained Jaccard on ALL cross-shop pairs (no blocking).
    Returns pair_quality, pair_completeness, F1_star, precision, recall, F1.
    """
    gold = compute_true_duplicate_pairs(products)
    n = len(products)

    # --- 1) Candidate pairs = ALL cross-shop pairs ---
    candidate_pairs: Set[tuple[int, int]] = set()
    for i, j in combinations(range(n), 2):
        if products[i].shop != products[j].shop:
            candidate_pairs.add((i, j))

    # Blocking metrics (even though there's no blocking, you can still compute them)
    dup_in_candidates = candidate_pairs & gold

    pair_completeness = len(dup_in_candidates) / len(gold) if gold else 0.0
    pair_quality = len(dup_in_candidates) / len(candidate_pairs) if candidate_pairs else 0.0
    F1_star_val = f1_star(pair_quality, pair_completeness)

    # For baseline (no blocking), pair_completeness should be 1.0,
    # because we include ALL cross-shop pairs.
    # pair_quality is just proportion of true duplicates among all cross-shop pairs.

    # --- 2) Classification with CJS on all candidate pairs ---
    predicted_pairs: Set[tuple[int, int]] = set()
    for i, j in candidate_pairs:
        if is_duplicate(products[i], products[j], theta=theta):
            predicted_pairs.add((i, j))

    print(len(predicted_pairs))
    # --- 3) Classic precision / recall / F1 ---
    TP = len(predicted_pairs & gold)
    FP = len(predicted_pairs - gold)
    FN = len(gold - predicted_pairs)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1_val = f1_score(precision, recall)

    return {
        "theta": theta,
        "num_gold_pairs": len(gold),
        "num_candidate_pairs": len(candidate_pairs),
        "num_predicted_pairs": len(predicted_pairs),
        "pair_quality": pair_quality,
        "pair_completeness": pair_completeness,
        "F1_star": F1_star_val,
        "precision": precision,
        "recall": recall,
        "F1": F1_val,
    }

def evaluate_lsh_cjs(products: List[Product],
                     bands: int,
                     rows_per_band: int,
                     theta: float = 0.7) -> Dict[str, float]:
    """
    Evaluate LSH + CJS on a given product list (no bootstrapping here).
    - products: list of Product with tokens, brand/size/res, and signature filled
    - bands, rows_per_band: LSH banding parameters (b, r)
    - theta: Jaccard threshold for CJS

    Returns a dict with pair_quality, pair_completeness, F1*, precision, recall, F1,
    plus frac_comparisons and some counts.
    """
    n = len(products)
    if n < 2:
        return {
            "num_gold_pairs": 0,
            "num_candidate_pairs": 0,
            "num_predicted_pairs": 0,
            "pair_quality": 0.0,
            "pair_completeness": 0.0,
            "F1_star": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "F1": 0.0,
            "frac_comparisons": 0.0,
        }

    gold = compute_true_duplicate_pairs(products)
    total_cross = count_cross_shop_pairs(products)

    # 1) Blocking with LSH
    candidate_pairs = lsh_candidate_pairs_br(
        products,
        bands=bands,
        rows_per_band=rows_per_band,
        only_cross_shop=True
    )

    dup_in_candidates = candidate_pairs & gold

    pair_completeness = len(dup_in_candidates) / len(gold) if gold else 0.0
    pair_quality = len(dup_in_candidates) / len(candidate_pairs) if candidate_pairs else 0.0
    F1_star_val = f1_star(pair_quality, pair_completeness)

    frac_comparisons = (
        len(candidate_pairs) / total_cross
        if total_cross > 0 else 0.0
    )

    # 2) Clustering-based prediction: build duplicate groups, then pairs
    clusters = cluster_duplicates_union_find(products, candidate_pairs, theta=theta)
    predicted_pairs = clusters_to_pairs(clusters)

    # 3) Final precision/recall/F1 wrt gold
    TP = len(predicted_pairs & gold)
    FP = len(predicted_pairs - gold)
    FN = len(gold - predicted_pairs)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1_val    = f1_score(precision, recall)

    return {
        "num_gold_pairs": len(gold),
        "num_candidate_pairs": len(candidate_pairs),
        "num_predicted_pairs": len(predicted_pairs),
        "pair_quality": pair_quality,
        "pair_completeness": pair_completeness,
        "F1_star": F1_star_val,
        "precision": precision,
        "recall": recall,
        "F1": F1_val,
        "frac_comparisons": frac_comparisons,
    }

def bootstrap_for_threshold(products: List[Product],
                            target_t: float,
                            theta: float,
                            B: int) -> Dict[str, float]:
    """
    For a given LSH threshold t (0.05..0.95), run B bootstraps:
      - build out-of-bag test sets
      - recompute tokens/signatures for test sets
      - run LSH + CJS with (b, r) derived from target_t
    Returns average metrics over bootstraps.
    """
    n_total = len(products)
    all_indices = list(range(n_total))

    # Precompute brand_lexicon from full data (no model_id usage)
    brand_lexicon = build_brand_lexicon(products)

    bands, rows_per_band, t_hat = get_b_r_for_threshold(NUM_HASHES, target_t)
    print(f"t={target_t:.2f} -> bands={bands}, rows_per_band={rows_per_band}, t̂={t_hat:.4f}")

    metrics_list = []

    best_F1 = 0
    best_frac = 0
    for b in range(B):
        # 1) Bootstrap sample indices WITH replacement
        boot_indices = [random.randrange(n_total) for _ in range(n_total)]
        boot_set = set(boot_indices)

        # 2) Out-of-bag test indices
        test_indices = [i for i in all_indices if i not in boot_set]
        if len(test_indices) < 2:
            # Nothing to evaluate on, skip this replicate
            continue

        # 3) Build test product list (new objects, but keep same content)
        test_products: List[Product] = []
        for new_pid, i in enumerate(test_indices):
            orig = products[i]
            p = Product(
                pid=new_pid,
                shop=orig.shop,
                title=orig.title,
                model_id=orig.model_id,
                features=orig.features
            )
            test_products.append(p)

        # 4) Tokens & signatures for test set
        build_tokens_for_all(test_products, brand_lexicon)
        build_minhash_signatures(test_products, num_hashes=NUM_HASHES)

        # 5) Evaluate LSH + CJS on this test set
        res = evaluate_lsh_cjs(
            test_products,
            bands=bands,
            rows_per_band=rows_per_band,
            theta=theta
        )
        metrics_list.append(res)

        if res["F1"] > best_F1:
            best_F1 = res["F1"]
            best_frac = res["frac_comparisons"]

        print(f"  Bootstrap {b+1}/{B}: test size={len(test_products)}, F1={res['F1']:.3f}, "
              f"PC={res['pair_completeness']:.3f}, PQ={res['pair_quality']:.6f}, "
              f"frac={res['frac_comparisons']:.5f}")

    # Average metrics over bootstraps
    if not metrics_list:
        return {}

    avg: Dict[str, float] = {}
    keys = metrics_list[0].keys()

    for k in keys:
        vals = [m[k] for m in metrics_list if isinstance(m[k], (int, float))]
        if vals:
            avg[k] = sum(vals) / len(vals)
        else:
            avg[k] = metrics_list[0][k]

    avg["t"] = target_t
    avg["bands"] = bands
    avg["rows_per_band"] = rows_per_band
    return avg

#------------------------------Main-------------------------
if __name__ == "__main__":
    json_path = r"/Users/ilzav/Downloads/TVs-all-merged/TVs-all-merged.json"

    # ---- LOAD PRODUCTS (full set, used as population for bootstraps) ----
    products = load_products_from_json(json_path)
    print(f"Loaded {len(products)} product records")

     # NEW: compute STOP_TRIGRAMS before building tokens
    init_stop_trigrams(products, df_ratio=0.6)
    print(STOP_TRIGRAMS)

    # ---- BUILD TOKENS ----
    brand_lexicon = build_brand_lexicon(products)
    build_tokens_for_all(products, brand_lexicon)

    df = compute_token_df(products)
    IDF = compute_token_idf(df, len(products))

    print("Computing MinHash signatures...")
    hash_funcs = build_minhash_signatures(products)
    print("Done computing MinHash signatures.")

    # You don't strictly need to build tokens/signatures here, because
    # bootstraps rebuild them for each test set. This is fine.

    # ---- BASELINE on full data (optional, already done) ----
    for theta in [0.3, 0.35, 0.4, 0.45]:
        res = evaluate_baseline_cjs_all_pairs(products, theta=theta)
        print(f"\nBaseline CJS (all pairs) with theta={theta}")
        print(f"  #gold pairs:        {res['num_gold_pairs']}")
        print(f"  #candidate pairs:   {res['num_candidate_pairs']}")
        print(f"  #predicted pairs:   {res['num_predicted_pairs']}")
        print(f"  Pair quality (PQ):  {res['pair_quality']:.4f}")
        print(f"  Pair completeness:  {res['pair_completeness']:.4f}")
        print(f"  F1*:                {res['F1_star']:.4f}")
        print(f"  Precision:          {res['precision']:.4f}")
        print(f"  Recall:             {res['recall']:.4f}")
        print(f"  F1:                 {res['F1']:.4f}")

    # ---- BOOTSTRAP LSH + CJS over t = 0.05..0.95 ----
    theta = 0.4  # Jaccard threshold for CJS
    B = 50        # number of bootstraps

    t_values = [i / 100 for i in range(5, 100, 5)]  # 0.05, 0.10, ..., 0.95
    t_values.insert(0, 0.001) 
    lsh_results = []

    for t in t_values:
        print(f"\n=== LSH bootstrapping for t={t:.2f} ===")
        avg_metrics = bootstrap_for_threshold(products, target_t=t, theta=theta, B=B)
        lsh_results.append(avg_metrics)
        print(f"AVERAGED over {B} bootstraps:")
        print(f"  bands={avg_metrics['bands']}, rows_per_band={avg_metrics['rows_per_band']}")
        print(f"  F1:   {avg_metrics['F1']:.4f}")
        print(f"  Prec: {avg_metrics['precision']:.4f}")
        print(f"  Rec:  {avg_metrics['recall']:.4f}")
        print(f"  PQ:   {avg_metrics['pair_quality']:.6f}")
        print(f"  PC:   {avg_metrics['pair_completeness']:.4f}")
        print(f"  F1*:  {avg_metrics['F1_star']:.6f}")
        print(f"  frac: {avg_metrics['frac_comparisons']:.5f}")

    results_sorted = sorted(lsh_results, key=lambda r: r["frac_comparisons"])

    fractions = [r["frac_comparisons"] for r in results_sorted]
    pc_values = [r["pair_completeness"] for r in results_sorted]
    pq_values = [r["pair_quality"] for r in results_sorted]
    f1_values = [r["F1"] for r in results_sorted]
    f1_star_values = [r["F1_star"] for r in results_sorted]

    # Figure 2: Pair Completeness vs Fraction of Comparisons
    plt.plot(fractions, pc_values, color='black')
    plt.xlabel("Fraction of comparisons")
    plt.ylabel("Pair completeness")
    plt.title("Pair completeness vs Fraction of comparisons")
    plt.grid(True)
    plt.xlim(-0.01, 1.0)
    plt.show()

    # Figure 3: Pair Quality vs Fraction of Comparisons
    plt.plot(fractions, pq_values, color='black')
    plt.xlabel("Fraction of comparisons")
    plt.ylabel("Pair quality")
    plt.title("Pair quality vs Fraction of comparisons")
    plt.grid(True)
    plt.xlim(-0.01, 0.25)
    plt.show()

    # Figure 4: F1 vs Fraction of Comparisons
    plt.plot(fractions, f1_values, color='black')
    plt.axhline(y=0.3126, color='red', linestyle='--', label='Baseline F1 = 0.3126')
    plt.xlabel("Fraction of comparisons")
    plt.ylabel("F1")
    plt.title("F1 vs Fraction of comparisons")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Figure 5: F1 star vs Fraction of Comparisons
    plt.plot(fractions, f1_star_values, color='black')
    plt.xlabel("Fraction of comparisons")
    plt.ylabel("F1 star")
    plt.title("F1 star vs Fraction of comparisons")
    plt.grid(True)
    plt.legend()
    plt.show()
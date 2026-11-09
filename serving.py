"""
STEP2-2. 서빙 모듈 — 저장한 모델 4개를 불러와 예측만 합니다 (학습 없음).
PART 10 노트북의 셀 8 · 10 · 11과 같은 계산입니다.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
MODEL_DIR = BASE / "models"
DATA_DIR = BASE / "data"

MODEL_FILES = {
    "recommend": "part6_recommend.pkl",
    "sentiment": "part7_1_sentiment_ml.pkl",
    "segment": "part8_seller_segment.pkl",
    "churn": "part9_seller_churn.pkl",
}
RISK_BINS = [0, 0.3, 0.5, 0.7, 1.0]
RISK_LABELS = ["낮음", "보통", "높음", "매우 높음"]
SEGMENT_ORDER = ["핵심 우량", "신규 성장", "휴면", "저평점 위험"]


def load_models() -> dict:
    """models/ 폴더의 .pkl 4개를 불러옵니다. 없는 파일은 알려 줍니다."""
    missing = [f for f in MODEL_FILES.values() if not (MODEL_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(f"models 폴더에 없는 파일: {', '.join(missing)}")
    return {k: joblib.load(MODEL_DIR / f) for k, f in MODEL_FILES.items()}


# ── PART 6 : 유사 상품 추천 ──────────────────────────────────────
def recommend(models: dict, product_id: str) -> pd.DataFrame:
    """기준 상품 1개 + 같은 카테고리에서 특징이 가장 가까운 상품 k개"""
    art = models["recommend"]
    prods = art["products"]
    hit = prods.index[prods["product_id"] == product_id]
    if len(hit) == 0:
        return pd.DataFrame()
    i = hit[0]
    x = art["scaler"].transform(np.log1p(prods.loc[[i], art["features"]]))
    nn, cat_idx = art["knn_by_category"][prods.at[i, "category"]]
    dist, pos = nn.kneighbors(x, n_neighbors=art["k"] + 1)
    rec = cat_idx[pos[0]]
    keep = rec != i
    rec, dist = rec[keep][: art["k"]], dist[0][keep][: art["k"]]
    out = prods.loc[[i] + list(rec)].copy()
    out.insert(0, "구분", ["기준 상품"] + [f"추천 {n + 1}" for n in range(len(rec))])
    out["거리"] = [0.0] + list(np.round(dist, 3))
    return out.reset_index(drop=True)


# ── PART 7-1 : 리뷰 감성 ─────────────────────────────────────────
def sentiment(models: dict, texts) -> np.ndarray:
    """리뷰 글 → 부정 확률 (모델의 0번 클래스 = 부정)"""
    art = models["sentiment"]
    texts = pd.Series(texts).fillna("").astype(str)
    return art["model"].predict_proba(art["tfidf"].transform(texts))[:, 0]


def score_recent_reviews(models: dict, reviews: pd.DataFrame) -> pd.DataFrame:
    r = reviews.copy()
    r["neg_prob"] = sentiment(models, r["review_comment_message"])
    r["pred"] = np.where(r["neg_prob"] >= 0.5, "부정", "긍정")
    return r


# ── PART 8 · 9 : 판매자 유형 + 이탈 확률 ──────────────────────────
def segment_input(prof: pd.DataFrame, review_fill: float) -> pd.DataFrame:
    return pd.DataFrame({
        "log_frequency": np.log1p(prof["frequency"]),
        "log_monetary": np.log1p(prof["monetary"]),
        "recency_days": prof["recency_days"],
        "tenure_days": prof["tenure_days"],
        "avg_review_score": prof["avg_review_score"].fillna(review_fill),
    })


def score_sellers(models: dict, sellers: pd.DataFrame) -> pd.DataFrame:
    """판매자 특징표 → 유형 · 이탈 확률 · 위험 단계 · 예상 손실"""
    seg, churn = models["segment"], models["churn"]
    s = sellers.copy()
    xs = segment_input(s, seg["review_fill"])[seg["features"]]
    s["segment"] = pd.Series(seg["kmeans"].predict(seg["scaler"].transform(xs))).map(
        {int(k): v for k, v in seg["names"].items()}).values
    s["churn_prob"] = churn["model"].predict_proba(s[churn["features"]])[:, 1]
    s["risk"] = pd.cut(s["churn_prob"], RISK_BINS, labels=RISK_LABELS, include_lowest=True)
    s["expected_loss"] = s["churn_prob"] * s["sales_last_3m"]
    return s


def model_info(models: dict) -> pd.DataFrame:
    rows = []
    for key, f in MODEL_FILES.items():
        p = MODEL_DIR / f
        rows.append({"파일": f, "용도": {"recommend": "유사 상품 추천 (PART 6)",
                                          "sentiment": "리뷰 감성 (PART 7-1)",
                                          "segment": "판매자 유형 (PART 8)",
                                          "churn": "판매자 이탈 확률 (PART 9)"}[key],
                     "크기(KB)": round(p.stat().st_size / 1024)})
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════
# 새 입력값으로 예측하기 (학습한 모델에 처음 보는 값을 넣는 함수들)
# ════════════════════════════════════════════════════════════
SEGMENT_GUIDE = {
    "핵심 우량": ("주문이 많고 최근에도 꾸준히 파는 판매자",
              "전담 매니저 배정, 성수기 물량 · 재고 사전 협의, 수수료 혜택으로 유지"),
    "신규 성장": ("시작한 지 얼마 안 된 판매자",
              "온보딩 가이드 · 첫 3개월 노출 지원, 첫 리뷰 관리 교육"),
    "휴면": ("오랫동안 판매가 없는 판매자",
           "재활성 쿠폰 · 프로모션 참여 제안, 반응이 없으면 관리 대상에서 제외"),
    "저평점 위험": ("평점이 낮고 출고 · 배송이 늦은 판매자",
               "출고 지연 경고, 배송 기한 재설정, 개선 없으면 노출 제한 검토"),
}

POR_KO = {
    "não": "아니다", "nao": "아니다", "recebi": "받았다", "chegou": "도착했다", "atraso": "지연",
    "atrasado": "늦은", "produto": "상품", "entrega": "배송", "prazo": "기한", "antes": "전에",
    "bom": "좋다", "ótimo": "최고", "otimo": "최고", "excelente": "훌륭하다", "recomendo": "추천한다",
    "rápido": "빠르다", "rapido": "빠르다", "rápida": "빠르다", "perfeito": "완벽하다", "gostei": "마음에 든다",
    "péssimo": "최악", "pessimo": "최악", "ruim": "나쁘다", "defeito": "불량", "quebrado": "파손",
    "ainda": "아직", "veio": "왔다", "errado": "잘못된", "comprei": "샀다", "aguardando": "기다리는 중",
    "nunca": "절대 안", "devolver": "반품", "dinheiro": "돈", "qualidade": "품질", "certo": "맞게",
    "tudo": "모두", "bem": "잘", "muito": "매우", "satisfeito": "만족", "parabéns": "축하", "amei": "아주 좋다",
    "correto": "정확하다", "dentro": "안에", "loja": "가게", "faltando": "빠진", "falta": "부족",
    "contato": "연락", "resposta": "답변", "nenhum": "아무것도 없음", "nenhuma": "아무것도 없음", "sem": "~없이",
    "pedido": "주문", "agora": "지금", "dias": "날(일)", "entregue": "배송됨", "embalagem": "포장",
    "embalado": "포장된", "lindo": "예쁘다", "adorei": "아주 좋다", "super": "아주", "recebido": "받음",
    "diferente": "다른", "comprado": "산", "nada": "아무것도", "problema": "문제", "atendimento": "응대",
    "aguardo": "기다리는 중", "ótima": "최고", "excelentes": "훌륭하다", "obrigado": "고맙다", "obrigada": "고맙다",
    "prazo,": "기한", "rapidez": "빠름", "cumpriu": "지켰다", "chegaram": "도착했다", "esperava": "기대했다",
}


def sentiment_explain(models: dict, text: str, top: int = 6) -> dict:
    """리뷰 1개 → 부정 확률 + 판단에 영향을 준 단어 (TF-IDF 점수 × 모델 가중치)"""
    art = models["sentiment"]
    x = art["tfidf"].transform([text])
    neg = float(art["model"].predict_proba(x)[0, 0])
    coef = art["model"].coef_[0]                       # + 는 긍정(1) 쪽, - 는 부정(0) 쪽
    vocab = art["tfidf"].get_feature_names_out()
    idx = x.nonzero()[1]
    contrib = pd.DataFrame({"단어": vocab[idx], "영향": x.toarray()[0, idx] * coef[idx]})
    contrib["뜻"] = contrib["단어"].map(lambda w: " ".join(POR_KO.get(t, "") for t in w.split()).strip())
    return {
        "neg_prob": neg,
        "neg_words": contrib[contrib["영향"] < 0].nsmallest(top, "영향"),
        "pos_words": contrib[contrib["영향"] > 0].nlargest(top, "영향"),
        "known": len(idx),
    }


def seller_from_input(v: dict) -> pd.DataFrame:
    """화면에서 입력한 판매자 값 → 모델이 학습한 특징 이름으로 변환 (1행)"""
    freq = max(float(v["frequency"]), 1.0)
    row = {
        "frequency": freq, "monetary": float(v["monetary"]),
        "recency_days": float(v["recency_days"]), "tenure_days": float(v["tenure_days"]),
        "active_months": float(v["active_months"]), "avg_delivery_days": float(v["avg_delivery_days"]),
        "late_rate": float(v["late_rate"]), "ship_late_rate": float(v["ship_late_rate"]),
        "avg_review_score": float(v["avg_review_score"]), "low_review_rate": float(v["low_review_rate"]),
        "orders_last_3m": float(v["orders_last_3m"]), "orders_prev_3m": float(v["orders_prev_3m"]),
    }
    row["avg_order_value"] = row["monetary"] / freq
    for col in ["frequency", "monetary", "avg_order_value", "orders_last_3m", "orders_prev_3m"]:
        row[f"log_{col}"] = np.log1p(row[col])
    row["log_order_trend"] = row["log_orders_last_3m"] - row["log_orders_prev_3m"]
    return pd.DataFrame([row])


def predict_seller(models: dict, v: dict) -> dict:
    """입력값 1명 → 유형 + 이탈 확률 + 위험 단계 + 추천 관리"""
    seg, churn = models["segment"], models["churn"]
    df = seller_from_input(v)
    xs = seg["scaler"].transform(segment_input(df, seg["review_fill"])[seg["features"]])
    label = int(seg["kmeans"].predict(xs)[0])
    name = {int(k): n for k, n in seg["names"].items()}[label]
    dists = seg["kmeans"].transform(xs)[0]               # 각 유형 중심까지의 거리
    near = {int(k): n for k, n in seg["names"].items()}
    closeness = pd.Series({near[i]: d for i, d in enumerate(dists)}).sort_values()
    prob = float(churn["model"].predict_proba(df[churn["features"]])[0, 1])
    risk = pd.cut([prob], RISK_BINS, labels=RISK_LABELS, include_lowest=True)[0]
    desc, action = SEGMENT_GUIDE.get(name, ("", ""))
    return {"segment": name, "desc": desc, "action": action, "churn_prob": prob,
            "risk": str(risk), "closeness": closeness}


def recommend_from_features(models: dict, category_kr: str, values: dict, k: int = 5) -> pd.DataFrame:
    """아직 없는 새 상품의 특징 → 같은 카테고리에서 가장 비슷한 기존 상품 k개"""
    art = models["recommend"]
    prods = art["products"]
    cat = prods.loc[prods["category_kr"] == category_kr, "category"].iloc[0]
    x = pd.DataFrame([{f: float(values[f]) for f in art["features"]}])
    xs = art["scaler"].transform(np.log1p(x))
    nn, cat_idx = art["knn_by_category"][cat]
    dist, pos = nn.kneighbors(xs, n_neighbors=min(k, len(cat_idx)))
    out = prods.loc[cat_idx[pos[0]]].copy()
    out.insert(0, "순위", range(1, len(out) + 1))
    out["거리"] = np.round(dist[0], 3)
    return out.reset_index(drop=True)


# ════════════════════════════════════════════════════════════
# 사용자 친화 표현 · 한국어 리뷰 지원 · 판정 근거
# ════════════════════════════════════════════════════════════
SEGMENT_FRIENDLY = {
    "핵심 우량": "잘 팔리는 주력 판매자",
    "신규 성장": "막 시작한 새내기 판매자",
    "휴면": "판매가 뜸해진 판매자",
    "저평점 위험": "평점이 낮아 관리가 필요한 판매자",
}

CHURN_FEATURE_KO = {
    "recency_days": "마지막 판매 후", "tenure_days": "첫 판매 후 지난 날",
    "active_months": "판매가 있었던 달 수", "log_frequency": "지금까지 받은 주문",
    "log_monetary": "지금까지 판매액", "log_avg_order_value": "주문 1건당 금액",
    "avg_delivery_days": "평균 배송 기간", "late_rate": "늦게 도착한 비율",
    "ship_late_rate": "출고 마감을 넘긴 비율", "avg_review_score": "평균 별점",
    "low_review_rate": "1·2점 리뷰 비율", "log_orders_last_3m": "최근 3개월 주문",
    "log_orders_prev_3m": "그 전 3개월 주문", "log_order_trend": "주문이 늘고 줄어든 추세",
}

CATEGORY_KO = {
    "health_beauty": "건강·뷰티", "computers_accessories": "컴퓨터 주변기기", "auto": "자동차용품",
    "bed_bath_table": "침구·욕실", "furniture_decor": "가구·장식", "sports_leisure": "스포츠·레저",
    "perfumery": "향수", "housewares": "생활용품", "telephony": "휴대폰·액세서리", "watches_gifts": "시계·선물",
    "food_drink": "식품·음료", "baby": "유아용품", "stationery": "문구", "tablets_printing_image": "태블릿·인쇄",
    "toys": "완구", "fixed_telephony": "유선전화", "garden_tools": "정원용품", "fashion_bags_accessories": "가방·잡화",
    "small_appliances": "소형가전", "consoles_games": "게임기·게임", "audio": "음향기기", "fashion_shoes": "신발",
    "cool_stuff": "아이디어 상품", "luggage_accessories": "여행가방", "air_conditioning": "냉난방기",
    "construction_tools_construction": "건축 공구", "kitchen_dining_laundry_garden_furniture": "주방·세탁·정원 가구",
    "costruction_tools_garden": "정원 공구", "fashion_male_clothing": "남성 의류", "pet_shop": "반려동물용품",
    "office_furniture": "사무용 가구", "market_place": "마켓플레이스", "electronics": "전자제품",
    "home_appliances": "생활가전", "party_supplies": "파티용품", "home_confort": "홈 편의용품",
    "costruction_tools_tools": "공구", "agro_industry_and_commerce": "농업·산업", "furniture_mattress_and_upholstery": "매트리스·소파",
    "books_technical": "전문 서적", "home_construction": "집수리", "musical_instruments": "악기",
    "furniture_living_room": "거실 가구", "construction_tools_lights": "조명", "industry_commerce_and_business": "산업·비즈니스",
    "food": "식품", "art": "미술품", "furniture_bedroom": "침실 가구", "books_general_interest": "일반 서적",
    "construction_tools_safety": "안전용품", "fashion_underwear_beach": "속옷·수영복", "fashion_sport": "스포츠 의류",
    "signaling_and_security": "보안·표지", "computers": "컴퓨터", "christmas_supplies": "크리스마스용품",
    "fashio_female_clothing": "여성 의류", "home_appliances_2": "생활가전 2", "books_imported": "수입 서적",
    "drinks": "음료", "cine_photo": "카메라·사진", "la_cuisine": "주방용품", "music": "음악", "home_comfort_2": "홈 편의용품 2",
    "small_appliances_home_oven_and_coffee": "오븐·커피머신", "cds_dvds_musicals": "CD·DVD", "dvds_blu_ray": "DVD·블루레이",
    "flowers": "꽃", "arts_and_craftmanship": "공예", "diapers_and_hygiene": "기저귀·위생용품",
    "fashion_childrens_clothes": "아동 의류", "security_and_services": "보안 서비스", "unknown": "분류 없음",
}

# 번역이 안 될 때 쓰는 간이 사전 : 한국어 표현 → 모델이 학습한 포르투갈어 단어
KO_FALLBACK = [
    (r"(못\s*받|안\s*왔|안\s*와|도착하지|오지\s*않|배송\s*안)", "ainda não recebi o produto"),
    (r"(늦|지연|느리)", "atraso entrega atrasada"),
    (r"(깨진|깨져|파손|부서)", "veio quebrado"),
    (r"(불량|고장|작동\s*안|안\s*돼)", "produto com defeito"),
    (r"(최악|별로|실망|나쁘)", "péssimo ruim"),
    (r"(다른\s*상품|잘못\s*왔|다른\s*색|다른\s*게)", "veio errado diferente"),
    (r"(환불|반품|교환)", "quero devolver"),
    (r"(연락.{0,2}없|답변.{0,2}없|응답.{0,2}없)", "sem resposta nenhum contato"),
    (r"(빠르|빨리|빨랐|빠른|일찍|금방)", "chegou rápido antes do prazo"),
    (r"(꼼꼼|포장.{0,4}잘|잘\s*포장)", "bem embalado"),
    (r"(좋|만족|훌륭|최고|마음에\s*들)", "ótimo produto excelente"),
    (r"(추천)", "recomendo"),
    (r"(완벽|딱\s*맞)", "perfeito"),
]


def has_korean(text: str) -> bool:
    import re
    return bool(re.search(r"[가-힣]", text or ""))


def korean_fallback(text: str) -> str:
    import re
    words = [pt for pat, pt in KO_FALLBACK if re.search(pat, text)]
    neg = re.search(r"(안\s*좋|좋지\s*않|별로)", text)
    if neg:
        words = [w for w in words if not w.startswith("ótimo")] + ["não gostei"]
    return " ".join(words)


def to_portuguese(text: str):
    """한국어 리뷰 → 포르투갈어. 반환: (모델에 넣을 문장, 방식)"""
    if not has_korean(text):
        return text, "원문"
    try:
        from deep_translator import GoogleTranslator
        out = GoogleTranslator(source="ko", target="pt").translate(text)
        if out and not has_korean(out):
            return out, "자동 번역"
    except Exception:
        pass
    return korean_fallback(text), "간이 사전"


REASON_GROUPS = {
    "판매 규모": ["log_frequency", "log_monetary", "log_avg_order_value"],
    "최근 주문 흐름": ["log_orders_last_3m", "log_orders_prev_3m", "log_order_trend"],
    "판매 경력": ["tenure_days", "active_months"],
    "마지막 판매 후": ["recency_days"],
    "배송": ["avg_delivery_days", "late_rate", "ship_late_rate"],
    "고객 평가": ["avg_review_score", "low_review_rate"],
}


def churn_reasons(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    """이탈 확률을 올리고 내린 요인 = (표준화한 입력값 × 모델 가중치)를 관련 항목끼리 합침.
    주문 수와 판매액처럼 함께 움직이는 값은 따로 보면 방향이 엇갈려 보이므로 묶어서 봅니다."""
    pipe = models["churn"]["model"]
    feats = models["churn"]["features"]
    z = pipe[:-1].transform(df[feats])[0]
    contrib = dict(zip(feats, z * pipe[-1].coef_[0]))
    rows = [{"요인": g, "영향": sum(contrib.get(f, 0) for f in fs)} for g, fs in REASON_GROUPS.items()]
    out = pd.DataFrame(rows)
    out["방향"] = np.where(out["영향"] > 0, "위험을 높임", "위험을 낮춤")
    return out.sort_values("영향", key=abs, ascending=False).reset_index(drop=True)


# 브라질 27개 주(州) 약어 → 한국어 이름
STATE_KO = {
    "AC": "아크리", "AL": "알라고아스", "AP": "아마파", "AM": "아마조나스", "BA": "바이아",
    "CE": "세아라", "DF": "연방구 (브라질리아)", "ES": "이스피리투산투", "GO": "고이아스",
    "MA": "마라냥", "MT": "마투그로수", "MS": "마투그로수두술", "MG": "미나스제라이스",
    "PA": "파라", "PB": "파라이바", "PR": "파라나", "PE": "페르남부쿠", "PI": "피아우이",
    "RJ": "리우데자네이루", "RN": "히우그란지두노르치", "RS": "히우그란지두술", "RO": "혼도니아",
    "RR": "호라이마", "SC": "산타카타리나", "SP": "상파울루", "SE": "세르지페", "TO": "토칸칭스",
}


def state_name(code: str) -> str:
    """'SP' → '상파울루 (SP)'"""
    return f"{STATE_KO.get(code, code)} ({code})" if code in STATE_KO else str(code)

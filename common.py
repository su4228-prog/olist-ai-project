"""세 페이지가 함께 쓰는 데이터 · 모델 불러오기 (한 번 읽고 기억)"""
from pathlib import Path

import pandas as pd
import streamlit as st

import serving

DATA_DIR = Path(__file__).parent / "data"
BLUE, RED, GREY, GREEN = "#2E70F2", "#FF6D2A", "#C9CDD3", "#2E70F2"
LIME = "#FF6D2A"   # 화면 강조용 (그래프 밖)

# ── 그래프 색 규칙 : 파랑 그라데이션 = 기본 · 좋음 / 주황 그라데이션 = 주목 · 나쁨 ──
MAIN = "#2E70F2"         # 기본 데이터 (파랑)
MAIN_DEEP = "#1C56E2"    # 파랑 그라데이션 진한 끝
MAIN_LIGHT = "#BFD1FB"   # 파랑 그라데이션 연한 끝
HIGHLIGHT = "#FF6D2A"    # 주목할 값 : 최고 달 · 새 상품 (주황)
HIGH_DEEP = "#E3490A"    # 주황 그라데이션 진한 끝
HIGH_LIGHT = "#FDC3A6"   # 주황 그라데이션 연한 끝
GOOD = MAIN              # 좋음 = 파랑
WARN = "#FFA270"         # 주의 = 연한 주황
BAD = HIGHLIGHT          # 나쁨 · 위험 = 주황
MUTE = "#C9CDD3"         # 비교용 나머지 (회색)
HATCH_BG, HATCH_FG = "#F1F2F4", "#BCC1C9"   # 나머지 막대의 빗금 무늬
HEAT = [[0, "#F3F3F5"], [0.35, "#D8E2FD"], [0.7, "#6F95F2"], [1, "#1C56E2"]]   # 적음 → 많음 (파랑 한 가지로 진해짐)
GRAD_GOOD = "linear-gradient(90deg, #BFD1FB, #1C56E2)"   # HTML 막대용
GRAD_BAD = "linear-gradient(90deg, #FDC3A6, #E3490A)"


def shade(values, light, deep):
    """값이 클수록 진한 색 : 막대마다 그라데이션 색을 계산"""
    import numpy as np
    v = np.asarray(list(values), dtype=float)
    t = (v - v.min()) / (v.max() - v.min()) if len(v) and v.max() > v.min() else np.ones(len(v))
    t = 0.35 + 0.65 * t
    def mix(c1, c2, k):
        a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
        b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
        return "#" + "".join(f"{round(x + (y - x) * k):02X}" for x, y in zip(a, b))
    return [mix(light, deep, k) for k in t]
SOURCE = "데이터: Olist Brazilian E-Commerce Public Dataset (Kaggle, CC BY-NC-SA 4.0) · 교육용 분석 대시보드"


@st.cache_data
def load_data():
    orders = pd.read_csv(DATA_DIR / "orders.csv.gz", parse_dates=["purchase_ts"])
    orders["month"] = orders["purchase_ts"].dt.to_period("M").astype(str)
    items = pd.read_csv(DATA_DIR / "items.csv.gz")
    sellers = pd.read_csv(DATA_DIR / "sellers_latest.csv.gz")
    reviews = pd.read_csv(DATA_DIR / "reviews_recent.csv.gz", parse_dates=["review_creation_date"])
    return orders, items, sellers, reviews


@st.cache_resource
def get_models():
    return serving.load_models()


@st.cache_data
def scored_sellers(_models, sellers):
    return serving.score_sellers(_models, sellers)


@st.cache_data
def scored_reviews(_models, reviews):
    return serving.score_recent_reviews(_models, reviews)


def models_or_stop():
    try:
        return get_models()
    except FileNotFoundError as e:
        st.error(f"{e}. Colab에서 내려받은 .pkl 4개를 models 폴더에 넣어 주세요.")
        st.stop()


@st.cache_data
def _load_products(catalog_mtime: float):
    info = pd.read_csv(DATA_DIR / "products_info.csv.gz")
    catalog_file = DATA_DIR / "product_catalog.csv"
    catalog = pd.read_csv(catalog_file) if catalog_mtime else None
    return info, catalog


def load_products():
    """추천 탭용 상품표 = 판매 실적 + (있으면) 내 쇼핑몰 상품 정보. 상품 정보표를 넣거나 고치면 바로 반영"""
    f = DATA_DIR / "product_catalog.csv"
    return _load_products(f.stat().st_mtime if f.exists() else 0.0)


@st.cache_data(show_spinner="리뷰를 포르투갈어로 옮기는 중…")
def translate(text: str):
    return serving.to_portuguese(text)


def friendly(segment: str) -> str:
    return serving.SEGMENT_FRIENDLY.get(segment, segment)


@st.cache_data
def load_word_stats():
    """단어별 불만 리뷰 · 만족 리뷰 등장 비율 (make_word_stats.py로 만든 파일)"""
    f = DATA_DIR / "word_stats.csv.gz"
    return pd.read_csv(f).set_index("word") if f.exists() else None


@st.cache_data(show_spinner=False)
def word_meaning(word: str) -> str:
    """포르투갈어 단어 → 한국어 뜻 (사전에 없으면 자동 번역, 안 되면 빈칸)"""
    known = serving.POR_KO.get(word)
    if known:
        return known
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="pt", target="ko").translate(word) or ""
    except Exception:
        return ""

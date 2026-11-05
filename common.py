"""세 페이지가 함께 쓰는 데이터 · 모델 불러오기 (Streamlit 캐시로 한 번 읽고 재사용)"""
from pathlib import Path

import pandas as pd
import streamlit as st

import serving

DATA_DIR = Path(__file__).parent / "data"
CATALOG_FILE = DATA_DIR / "product_catalog.csv"   # 선택 : 내 쇼핑몰 상품 이름 · 사진 · 링크
SOURCE = "데이터: Olist Brazilian E-Commerce Public Dataset (Kaggle, CC BY-NC-SA 4.0) · 교육용 분석 대시보드"


# ── 데이터 ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    """서빙용 데이터 4종 → (주문, 판매 상품, 최신 판매자 특징, 최근 리뷰)"""
    orders = pd.read_csv(DATA_DIR / "orders.csv.gz", parse_dates=["purchase_ts"])
    orders["month"] = orders["purchase_ts"].dt.to_period("M").astype(str)
    items = pd.read_csv(DATA_DIR / "items.csv.gz")
    sellers = pd.read_csv(DATA_DIR / "sellers_latest.csv.gz")
    reviews = pd.read_csv(DATA_DIR / "reviews_recent.csv.gz", parse_dates=["review_creation_date"])
    return orders, items, sellers, reviews


@st.cache_data
def _load_products(catalog_mtime: float):
    # catalog_mtime 이 캐시 키 역할 : 상품 정보표 파일이 바뀌면 다시 읽음 (0 = 파일 없음)
    info = pd.read_csv(DATA_DIR / "products_info.csv.gz")
    catalog = pd.read_csv(CATALOG_FILE) if catalog_mtime else None
    return info, catalog


def load_products():
    """추천 탭용 상품표 = 판매 실적 + (있으면) 내 쇼핑몰 상품 정보. 상품 정보표를 넣거나 고치면 바로 반영"""
    return _load_products(CATALOG_FILE.stat().st_mtime if CATALOG_FILE.exists() else 0.0)


@st.cache_data
def load_word_stats():
    """단어별 불만 리뷰 · 만족 리뷰 등장 비율 (make_word_stats.py로 만든 파일, 없으면 None)"""
    f = DATA_DIR / "word_stats.csv.gz"
    return pd.read_csv(f).set_index("word") if f.exists() else None


# ── 모델 · 예측 결과 ──────────────────────────────────────────
@st.cache_resource
def get_models():
    """models/ 의 .pkl 4개를 앱 전체에서 한 번만 불러옴"""
    return serving.load_models()


def models_or_stop():
    """모델을 불러오고, 파일이 없으면 안내 메시지를 띄운 뒤 페이지 실행을 멈춤"""
    try:
        return get_models()
    except FileNotFoundError as e:
        st.error(f"{e}. Colab에서 내려받은 .pkl 4개를 models 폴더에 넣어 주세요.")
        st.stop()


@st.cache_data
def scored_sellers(_models, sellers):
    """판매자 표에 유형 · 이탈 확률 · 위험 단계를 붙인 결과 (_models 는 캐시 키에서 제외)"""
    return serving.score_sellers(_models, sellers)


@st.cache_data
def scored_reviews(_models, reviews):
    """최근 리뷰에 부정 확률 · 판정을 붙인 결과"""
    return serving.score_recent_reviews(_models, reviews)


# ── 번역 · 표시용 문구 ────────────────────────────────────────
@st.cache_data(show_spinner="리뷰를 포르투갈어로 옮기는 중…")
def translate(text: str):
    """리뷰 글 → (모델에 넣을 포르투갈어 문장, 번역 방식)"""
    return serving.to_portuguese(text)


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


def friendly(segment: str) -> str:
    """판매자 유형 이름 → 화면에 보여 줄 쉬운 표현"""
    return serving.SEGMENT_FRIENDLY.get(segment, segment)

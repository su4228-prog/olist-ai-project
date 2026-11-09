import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import serving
import ui
from common import (BAD, HATCH_BG, HATCH_FG, HIGHLIGHT, MAIN, MAIN_DEEP, MAIN_LIGHT, MUTE, SOURCE, shade, friendly, load_data, models_or_stop, scored_sellers)

orders, items, sellers_raw, reviews = load_data()
models = models_or_stop()
sellers = scored_sellers(models, sellers_raw)

ui.header("홈", "브라질 마켓플레이스 Olist의 실제 주문 데이터를 분석하고, 미리 학습한 AI 모델로 새로운 값의 결과를 바로 예측합니다.",
          right="2017-01 ~ 2018-08 데이터")

# ── 1줄 : 핵심 숫자 4개 ──
k1, k2, k3, k4 = st.columns(4, gap="medium")
k1.metric(":material/receipt_long: 분석한 주문", f"{len(orders):,}건")
k2.metric(":material/payments: 총매출", f"R$ {orders['payment_value'].sum() / 1e6:,.2f}M")
k3.metric(":material/storefront: 최근 3개월 활동 판매자", f"{len(sellers):,}명")
k4.metric(":material/star: 평균 별점", f"{orders['review_score'].mean():.2f}점")

# ── 2줄 : 월별 매출 + 판매자 유형 ──
with ui.row("sales"):
    c1, c2 = st.columns([7, 5], gap="medium")
    with c1, ui.card("home-sales"):
        st.markdown("### 월별 매출")
        mon = orders.groupby(pd.Grouper(key="purchase_ts", freq="MS")).agg(
            매출=("payment_value", "sum"), 주문=("order_id", "size")).reset_index()
        best = mon["매출"].idxmax()
        colors = shade(mon["매출"], MAIN_LIGHT, MAIN_DEEP)   # 높을수록 진한 파랑
        colors[best] = HIGHLIGHT
        fig = go.Figure(go.Bar(x=mon["purchase_ts"], y=mon["매출"], marker_color=colors,
                               hovertemplate="%{x|%Y-%m}<br>R$ %{y:,.0f}<extra></extra>"))
        fig.update_layout(height=300, margin={"t": 8, "b": 8}, yaxis_title=None, xaxis_title=None,
                          bargap=0.35)
        ui.chart(fig)
        st.caption(f"주황 막대가 가장 높은 달 ({mon.at[best, 'purchase_ts']:%Y-%m}, 블랙프라이데이)입니다. "
                   "성장은 주문 1건당 금액보다 주문 수가 늘어서 생겼습니다.")

    with c2, ui.card("home-seller"):
        st.markdown("### 판매자 유형")
        cnt = sellers["segment"].value_counts().reindex(serving.SEGMENT_ORDER, fill_value=0)
        palette = [MAIN_DEEP, "#8FB0F7", HATCH_BG, BAD]   # 주력 · 새내기 · 뜸해짐(빗금) · 관리 필요(주황)
        fig = go.Figure(go.Pie(labels=[friendly(s) for s in cnt.index], values=cnt.values, hole=.68,
                               marker={"colors": palette, "line": {"color": "#FFFFFF", "width": 2},
                                   "pattern": {"shape": ["", "", "/", ""], "fgcolor": HATCH_FG, "solidity": 0.3}},
                               sort=False, textinfo="none", hovertemplate="%{label}<br>%{value:,}명<extra></extra>"))
        fig.update_layout(height=255, margin={"t": 4, "b": 4, "l": 4, "r": 4}, showlegend=False,
                          annotations=[{"text": f"<b>{cnt.sum():,}명</b><br><span style='font-size:11px'>최근 3개월</span>",
                                        "showarrow": False, "font": {"size": 16}}])
        ui.chart(fig)
        for (seg, n), col in zip(cnt.items(), [MAIN_DEEP, "#8FB0F7", MUTE, BAD]):
            st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:.88rem;padding:2px 0'>"
                        f"<span><span style='display:inline-block;width:10px;height:10px;border-radius:3px;"
                        f"background:{col};margin-right:8px'></span>{friendly(seg)}</span>"
                        f"<b>{n:,}명 · {n / cnt.sum() * 100:.0f}%</b></div>", unsafe_allow_html=True)

# ── 3줄 : 분석 → 모델 흐름 + 바로 가기 ──
with ui.row("flow"):
    c3, c4 = st.columns([7, 5], gap="medium")
    with c3, ui.card("home-flow"):
        st.markdown("### 분석에서 모델까지")
        st.markdown("""
    | 데이터에서 발견한 것 | 그래서 만든 모델 |
    | --- | --- |
    | 고객 97%가 한 번만 구매 → 고객의 구매 기록으로는 추천할 수 없음 | 상품 특징이 닮은 상품을 찾는 **비슷한 상품 찾기** |
    | 배송이 늦으면 평균 별점 4.30점 → 2.57점 | 리뷰 글로 불만을 골라내는 **리뷰 판정** |
    | 늦은 배송 · 낮은 별점이 일부 판매자에게 몰림 | 판매자를 4가지로 나누는 **판매자 유형** |
    | 판매자가 떠나면 상품 수와 매출이 줄어듦 | 3개월 안에 판매를 멈출지 미리 알려 주는 **판매 중단 예측** |
    """)
    with c4:
        with ui.card("home-ai"):
            st.markdown("### 🤖 AI 모델")
            st.markdown("새 값을 넣으면 바로 판정합니다.  \n**리뷰 판정** · **판매자 진단** · **비슷한 상품 찾기**")
            st.page_link("views/ai_models.py", label="AI 모델 써 보기", icon=":material/arrow_forward:")
        with ui.card("home-analysis"):
            st.markdown("### 📊 데이터 분석")
            st.markdown("월별 흐름 · 요일×시간 패턴 · 카테고리 · 배송과 별점 · 브라질 주별 비교")
            st.page_link("views/analysis.py", label="데이터 분석 보기", icon=":material/arrow_forward:")

st.caption(SOURCE)

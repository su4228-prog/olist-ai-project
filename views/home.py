"""홈 페이지 : 핵심 숫자 · 월별 매출 · 판매자 유형 비율 · 분석→모델 흐름 · 바로 가기"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import serving
import ui
from common import SOURCE, friendly, load_data, models_or_stop, scored_sellers

SEGMENT_PALETTE = [ui.BLUE, ui.YELLOW, ui.LAVENDER, ui.RED]   # serving.SEGMENT_ORDER 순서

orders, items, sellers_raw, reviews = load_data()
models = models_or_stop()
sellers = scored_sellers(models, sellers_raw)

ui.header("홈", "브라질 마켓플레이스 Olist의 실제 주문 데이터를 분석하고, 미리 학습한 AI 모델로 새로운 값의 결과를 바로 예측합니다.",
          right="2017-01 ~ 2018-08 데이터")

# ── 1줄 : 핵심 숫자 4개 ──────────────────────────────────────
k1, k2, k3, k4 = st.columns(4, gap="medium")
k1.metric(":material/receipt_long: 분석한 주문", f"{len(orders):,}건")
k2.metric(":material/payments: 총매출", f"R$ {orders['payment_value'].sum() / 1e6:,.2f}M")
k3.metric(":material/storefront: 최근 3개월 활동 판매자", f"{len(sellers):,}명")
k4.metric(":material/star: 평균 별점", f"{orders['review_score'].mean():.2f}점")

# ── 2줄 : 월별 매출 + 판매자 유형 ────────────────────────────
with ui.row("sales"):
    c1, c2 = st.columns([7, 5], gap="medium")

    # 월별 매출 막대 (가장 높은 달만 노란색)
    with c1, ui.card("home-sales"):
        st.markdown("### 월별 매출")
        monthly = orders.groupby(pd.Grouper(key="purchase_ts", freq="MS")).agg(
            매출=("payment_value", "sum"), 주문=("order_id", "size")).reset_index()
        best = monthly["매출"].idxmax()
        colors = [ui.YELLOW if i == best else ui.BLUE for i in range(len(monthly))]
        fig = go.Figure(go.Bar(x=monthly["purchase_ts"], y=monthly["매출"], marker_color=colors,
                               hovertemplate="%{x|%Y-%m}<br>R$ %{y:,.0f}<extra></extra>"))
        fig.update_layout(height=300, margin={"t": 8, "b": 8}, yaxis_title=None, xaxis_title=None,
                          bargap=0.35)
        ui.chart(fig)
        st.caption(f"노란 막대가 가장 높은 달 ({monthly.at[best, 'purchase_ts']:%Y-%m}, 블랙프라이데이)입니다. "
                   "성장은 주문 1건당 금액보다 주문 수가 늘어서 생겼습니다.")

    # 판매자 유형 도넛 + 유형별 인원 · 비율 범례
    with c2, ui.card("home-seller"):
        st.markdown("### 판매자 유형")
        counts = sellers["segment"].value_counts().reindex(serving.SEGMENT_ORDER, fill_value=0)
        total = counts.sum()
        fig = go.Figure(go.Pie(labels=[friendly(s) for s in counts.index], values=counts.values, hole=.68,
                               marker={"colors": SEGMENT_PALETTE, "line": {"color": ui.WHITE, "width": 2}},
                               sort=False, textinfo="none", hovertemplate="%{label}<br>%{value:,}명<extra></extra>"))
        fig.update_layout(height=255, margin={"t": 4, "b": 4, "l": 4, "r": 4}, showlegend=False,
                          annotations=[{"text": f"<b>{total:,}명</b><br><span style='font-size:11px'>최근 3개월</span>",
                                        "showarrow": False, "font": {"size": 16}}])
        ui.chart(fig)
        for (seg, n), color in zip(counts.items(), SEGMENT_PALETTE):
            st.markdown(f'<div class="legend-row"><span><i style="background:{color}"></i>{friendly(seg)}</span>'
                        f'<b>{n:,}명 · {n / total * 100:.0f}%</b></div>', unsafe_allow_html=True)

# ── 3줄 : 분석 → 모델 흐름 + 바로 가기 ───────────────────────
with ui.row("flow"):
    c3, c4 = st.columns([7, 5], gap="medium")

    # 데이터에서 발견한 문제와, 그 문제를 풀기 위해 만든 모델 대응표
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

    # 다른 페이지로 가는 바로 가기 카드 2개
    with c4:
        with ui.card("home-ai", accent=True):
            st.markdown("### 🤖 AI 모델")
            st.markdown("새 값을 넣으면 바로 판정합니다.  \n**리뷰 판정** · **판매자 진단** · **비슷한 상품 찾기**")
            st.page_link("views/ai_models.py", label="AI 모델 써 보기", icon=":material/arrow_forward:")
        with ui.card("home-analysis"):
            st.markdown("### 📊 데이터 분석")
            st.markdown("월별 흐름 · 요일×시간 패턴 · 카테고리 · 배송과 별점 · 브라질 주별 비교")
            st.page_link("views/analysis.py", label="데이터 분석 보기", icon=":material/arrow_forward:")

st.caption(SOURCE)

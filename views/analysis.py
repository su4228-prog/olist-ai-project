"""데이터 분석 페이지 : 지역 · 기간 필터 → 핵심 지표 · 인사이트 · 주제별 탭 6개"""
import pandas as pd
import plotly.express as px
import streamlit as st

import serving
import ui
from common import SOURCE, load_data

ALL = "전체"
WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]
DELIVERY_BINS = [0, 7, 14, 21, 30, 60, 1000]
DELIVERY_LABELS = ["~7일", "8~14일", "15~21일", "22~30일", "31~60일", "60일+"]
HEATMAP_SCALE = ["#EFF7FD", ui.BLUE_DEEP]   # 주문 적음 → 많음
# 원본 데이터 열 이름 → 화면 표 머리글
ORDER_COLUMNS_KO = {
    "order_id": "주문번호", "customer_unique_id": "고객번호", "purchase_ts": "주문 시각",
    "customer_state": "주", "payment_value": "결제 금액(R$)", "payment_type": "결제 수단",
    "delivery_days": "배송 일수", "is_late": "늦게 도착 (1=예)", "review_score": "별점",
}

orders, items, _, _ = load_data()

# ── 사이드바 필터 : 고객 지역 + 기간 ──────────────────────────
codes = sorted(orders["customer_state"].unique(), key=lambda c: serving.STATE_KO.get(c, c))
state = st.sidebar.selectbox("고객 지역 (브라질의 주)", [ALL] + codes,
                             format_func=lambda c: "전체 지역" if c == ALL else serving.state_name(c),
                             help="Olist는 브라질 쇼핑몰이라 지역은 브라질의 27개 주(州)입니다. 한국의 도(道)와 비슷한 단위입니다.")
state_label = "전체 지역" if state == ALL else serving.state_name(state)
months = sorted(orders["month"].unique())
m_from, m_to = st.sidebar.select_slider("기간", options=months, value=(months[0], months[-1]))

# 필터 적용 : f = 선택한 주문, fi = 그 주문에 포함된 상품 (카테고리는 한국어로)
mask = orders["month"].between(m_from, m_to)
if state != ALL:
    mask &= orders["customer_state"] == state
f = orders[mask]
fi = items[items["order_id"].isin(f["order_id"])]
fi = fi.assign(category=fi["category"].map(serving.CATEGORY_KO).fillna(fi["category"]))
if f.empty:
    st.warning("선택한 조건에 맞는 주문이 없습니다. 기간이나 지역을 넓혀 주세요.")
    st.stop()

# ── 제목 · 핵심 지표 6개 ─────────────────────────────────────
ui.header("데이터 분석", "배송 완료 주문 기준입니다. 왼쪽에서 지역과 기간을 바꾸면 모든 숫자가 함께 바뀝니다.",
          right=f"{m_from} ~ {m_to} · {state_label}")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("주문 수", f"{len(f):,}건")
k2.metric("매출", f"R$ {f['payment_value'].sum() / 1e6:,.2f}M")
k3.metric("주문 1건당 금액", f"R$ {f['payment_value'].mean():,.0f}",
          help=f"평균(객단가)입니다. 가운데 값(중앙값)은 R$ {f['payment_value'].median():,.0f} (큰 주문 몇 건이 평균을 끌어올림)")
k4.metric("배송 기간 (보통)", f"{f['delivery_days'].median():.1f}일",
          help="주문부터 도착까지 걸린 날. 아주 늦은 몇 건에 끌려가지 않도록 가운데 값(중앙값)을 씁니다.")
k5.metric("늦게 도착한 비율", f"{f['is_late'].mean() * 100:.1f}%")
k6.metric("평균 별점", f"{f['review_score'].mean():.2f}점")

# ── 핵심 인사이트 : 재구매율 · 지연과 별점 · 지역 집중도 ─────────
purchases_per_customer = f.groupby("customer_unique_id").size()
repeat = (purchases_per_customer >= 2).mean() * 100
on_time = f.loc[f["is_late"] == 0, "review_score"].mean()
late = f.loc[f["is_late"] == 1, "review_score"].mean()

lines = [f"🔁 재구매 고객은 **{repeat:.1f}%** → 두 번째 구매를 유도하는 것이 가장 큰 기회"]
if pd.notna(late):
    lines.append(f"🚚 늦게 도착하면 별점이 **{on_time:.2f}점 → {late:.2f}점**으로 떨어짐")
if state == ALL:
    share = f["customer_state"].value_counts(normalize=True).head(3)
    lines.append(f"🗺️ 주문 상위 3개 주 ({', '.join(serving.STATE_KO.get(c, c) for c in share.index)})가 전체의 **{share.sum() * 100:.1f}%**")
else:
    all_median = orders.loc[orders["month"].between(m_from, m_to), "delivery_days"].median()
    lines.append(f"🗺️ {state_label}의 배송 기간 중앙값은 **{f['delivery_days'].median():.1f}일** "
                 f"(전체 {all_median:.1f}일)")

with ui.card("insight", accent=True):
    st.markdown("### 💡 핵심 인사이트")
    st.markdown("\n".join(f"- {t}" for t in lines))

# ── 주제별 탭 ────────────────────────────────────────────────
with ui.card("tabs"):
    tab_sales, tab_time, tab_cat, tab_delivery, tab_region, tab_data = st.tabs(
        ["📈 매출 추이", "🕒 주문 시간", "🏷️ 카테고리", "🚚 배송·별점", "🗺️ 지역", "📄 데이터"])

    # 월별 매출 · 주문 수 · 객단가 추이
    with tab_sales:
        metric = st.radio("지표", ["매출", "주문 수", "주문 1건당 금액"], horizontal=True)
        monthly = f.groupby(pd.Grouper(key="purchase_ts", freq="MS")).agg(
            매출=("payment_value", "sum"), 주문수=("order_id", "size")).rename(columns={"주문수": "주문 수"})
        monthly["주문 1건당 금액"] = monthly["매출"] / monthly["주문 수"]
        monthly = monthly[monthly["주문 수"] > 0].reset_index()
        fig = px.bar(monthly, x="purchase_ts", y=metric, color_discrete_sequence=[ui.BLUE],
                     labels={"purchase_ts": "월"}, title=f"월별 {metric}")
        ui.chart(fig)
        best = monthly.loc[monthly[metric].idxmax(), "purchase_ts"].strftime("%Y-%m")
        note = " (2017-11-24 블랙프라이데이)" if best == "2017-11" else ""
        st.info(f"가장 높은 달은 **{best}** 입니다.{note}")

    # 요일 × 시간대 주문 수 히트맵
    with tab_time:
        heat = f.assign(요일=f["purchase_ts"].dt.dayofweek, 시=f["purchase_ts"].dt.hour).pivot_table(
            index="요일", columns="시", values="order_id", aggfunc="count", fill_value=0)
        heat = heat.reindex(index=range(7), columns=range(24), fill_value=0)
        heat.index = WEEKDAYS
        fig = px.imshow(heat, color_continuous_scale=HEATMAP_SCALE, aspect="auto",
                        labels={"x": "시", "y": "요일", "color": "주문 수"}, title="요일 × 시간대 주문 수")
        ui.chart(fig)
        peak_day = heat.sum(axis=1).idxmax()
        peak_hour = heat.sum(axis=0).idxmax()
        st.info(f"주문이 가장 많은 요일은 **{peak_day}요일**, 시간대는 **{peak_hour}시**입니다. "
                "알림 · 타임 세일 시간을 정할 때 씁니다.")

    # 카테고리 순위 (매출 또는 판매 수량 기준 TOP N)
    with tab_cat:
        c1, c2 = st.columns([1, 3])
        basis = c1.radio("순위 기준", ["매출", "판매 수량"])
        top_n = c1.slider("표시할 개수", 5, 30, 15)
        cat = fi.groupby("category").agg(매출=("price", "sum"), 수량=("price", "size")).rename(
            columns={"수량": "판매 수량"}).sort_values(basis, ascending=False).head(top_n).reset_index()
        fig = px.bar(cat, x=basis, y="category", orientation="h", color_discrete_sequence=[ui.BLUE],
                     title=f"카테고리 {basis} TOP {top_n}")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title=None)
        ui.chart(fig, c2)

    # 배송 기간별 평균 별점 + 월별 지연 비율
    with tab_delivery:
        c1, c2 = st.columns(2)
        bins = pd.cut(f["delivery_days"], DELIVERY_BINS, labels=DELIVERY_LABELS)
        by_days = f.groupby(bins, observed=True)["review_score"].mean().reset_index()
        fig = px.bar(by_days, x="delivery_days", y="review_score", text_auto=".2f", range_y=[1, 5],
                     color_discrete_sequence=[ui.BLUE], labels={"delivery_days": "배송 기간", "review_score": "평균 별점"},
                     title="배송이 길어질수록 별점이 떨어집니다")
        ui.chart(fig, c1)
        late_rate = (f.groupby(pd.Grouper(key="purchase_ts", freq="MS"))["is_late"].mean() * 100).dropna().reset_index()
        fig = px.line(late_rate, x="purchase_ts", y="is_late", markers=True, color_discrete_sequence=[ui.RED],
                      labels={"purchase_ts": "월", "is_late": "늦게 도착한 비율 (%)"}, title="월별 늦게 도착한 비율")
        ui.chart(fig, c2)
        if pd.notna(late):
            st.info(f"제때 도착한 주문의 평균 리뷰는 **{on_time:.2f}점**, 지연된 주문은 **{late:.2f}점**입니다.")

    # 브라질 주별 지표 비교 (막대 + 표)
    with tab_region:
        region = f.groupby("customer_state").agg(
            주문수=("order_id", "size"), 매출=("payment_value", "sum"), 배송기간=("delivery_days", "median"),
            지연율=("is_late", "mean"), 평균리뷰=("review_score", "mean")).reset_index()
        region["지연율"] *= 100
        region["customer_state"] = region["customer_state"].map(serving.state_name)
        region.columns = ["주", "주문 수", "매출", "배송 기간 (보통, 일)", "늦게 도착한 비율 (%)", "평균 별점"]
        pick = st.selectbox("비교할 지표", region.columns[1:])
        fig = px.bar(region.sort_values(pick, ascending=False), x="주", y=pick, color_discrete_sequence=[ui.BLUE],
                     title=f"주별 {pick}")
        fig.update_layout(xaxis_title=None, xaxis_tickangle=-45)
        ui.chart(fig)
        st.dataframe(region.round(2), hide_index=True, width="stretch")

    # 필터가 적용된 주문 원본 표 + CSV 내려받기
    with tab_data:
        show = (f.drop(columns=["month"])
                 .assign(customer_state=f["customer_state"].map(serving.state_name))
                 .rename(columns=ORDER_COLUMNS_KO))
        st.caption(f"{len(show):,}행 · 필터가 적용된 주문 표")
        st.dataframe(show.head(1000), hide_index=True, width="stretch")
        st.download_button("CSV 내려받기", show.to_csv(index=False).encode("utf-8-sig"),
                           "olist_orders_filtered.csv", "text/csv")

st.caption(SOURCE)

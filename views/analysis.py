import pandas as pd
import plotly.express as px
import streamlit as st

import serving
import ui
import plotly.graph_objects as go

from common import (BAD, GOOD, HATCH_BG, HATCH_FG, HEAT, HIGH_DEEP, HIGH_LIGHT, HIGHLIGHT, MAIN, MAIN_DEEP,
                    MAIN_LIGHT, MUTE, SOURCE, WARN, load_data, shade)

orders, items, _, _ = load_data()

# ── 사이드바 필터 ─────────────────────────────────────────
codes = sorted(orders["customer_state"].unique(), key=lambda c: serving.STATE_KO.get(c, c))
state = st.sidebar.selectbox("고객 지역 (브라질의 주)", ["전체"] + codes,
                             format_func=lambda c: "전체 지역" if c == "전체" else serving.state_name(c),
                             help="Olist는 브라질 쇼핑몰이라 지역은 브라질의 27개 주(州)입니다. 한국의 도(道)와 비슷한 단위입니다.")
state_label = "전체 지역" if state == "전체" else serving.state_name(state)
months = sorted(orders["month"].unique())
m_from, m_to = st.sidebar.select_slider("기간", options=months, value=(months[0], months[-1]))

mask = orders["month"].between(m_from, m_to)
if state != "전체":
    mask &= orders["customer_state"] == state
f = orders[mask]
fi = items[items["order_id"].isin(f["order_id"])]
fi = fi.assign(category=fi["category"].map(serving.CATEGORY_KO).fillna(fi["category"]))
if f.empty:
    st.warning("선택한 조건에 맞는 주문이 없습니다. 기간이나 지역을 넓혀 주세요.")
    st.stop()

# ── 제목 · 핵심 지표 ──────────────────────────────────────
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

# ── 핵심 인사이트 (필터에 따라 숫자가 바뀜) ─────────────────
n_buy = f.groupby("customer_unique_id").size()
repeat = (n_buy >= 2).mean() * 100
on_time = f.loc[f["is_late"] == 0, "review_score"].mean()
late = f.loc[f["is_late"] == 1, "review_score"].mean()
lines = [f"🔁 재구매 고객은 **{repeat:.1f}%** → 두 번째 구매를 유도하는 것이 가장 큰 기회"]
if pd.notna(late):
    lines.append(f"🚚 늦게 도착하면 별점이 **{on_time:.2f}점 → {late:.2f}점**으로 떨어짐")
if state == "전체":
    share = f["customer_state"].value_counts(normalize=True).head(3)
    lines.append(f"🗺️ 주문 상위 3개 주 ({', '.join(serving.STATE_KO.get(c, c) for c in share.index)})가 전체의 **{share.sum() * 100:.1f}%**")
else:
    all_med = orders.loc[orders["month"].between(m_from, m_to), "delivery_days"].median()
    lines.append(f"🗺️ {state_label}의 배송 기간 중앙값은 **{f['delivery_days'].median():.1f}일** "
                 f"(전체 {all_med:.1f}일)")
with ui.card("insight"):
    st.markdown("### 💡 핵심 인사이트")
    st.markdown("\n".join(f"- {t}" for t in lines))

with ui.card("tabs"):
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📈 매출 추이", "🕒 주문 시간", "🏷️ 카테고리", "🚚 배송·별점", "🗺️ 지역", "📄 데이터"])

    with tab1:
        metric = st.radio("지표", ["매출", "주문 수", "주문 1건당 금액"], horizontal=True)
        mon = f.groupby(pd.Grouper(key="purchase_ts", freq="MS")).agg(
            매출=("payment_value", "sum"), 주문수=("order_id", "size")).rename(columns={"주문수": "주문 수"})
        mon["주문 1건당 금액"] = mon["매출"] / mon["주문 수"]
        mon = mon[mon["주문 수"] > 0].reset_index()
        bi = mon[metric].idxmax()
        colors = shade(mon[metric], MAIN_LIGHT, MAIN_DEEP)          # 높을수록 진한 파랑
        colors[bi] = HIGHLIGHT
        fig = go.Figure(go.Bar(x=mon["purchase_ts"], y=mon[metric], marker_color=colors,
                               hovertemplate="%{x|%Y-%m}<br>%{y:,.0f}<extra></extra>"))
        avg = mon[metric].mean()
        fig.add_hline(y=avg, line_dash="dot", line_color="#8A9099", line_width=1.5,
                      annotation_text=f"월평균 {avg:,.0f}", annotation_position="top left",
                      annotation_font_color="#5B616B")
        fig.add_annotation(x=mon.at[bi, "purchase_ts"], y=mon.at[bi, metric], text="<b>최고</b>", showarrow=False,
                           yshift=12, font={"color": HIGH_DEEP})
        fig.update_layout(title=f"월별 {metric}", xaxis_title=None, yaxis_title=None, bargap=0.3)
        ui.chart(fig)
        best = mon.at[bi, "purchase_ts"].strftime("%Y-%m")
        note = " (2017-11-24 블랙프라이데이)" if best == "2017-11" else ""
        st.info(f"주황 막대가 가장 높은 달 **{best}** 입니다.{note} 파랑은 높을수록 진해지고, 점선은 월평균입니다.")

    with tab2:
        days = ["월", "화", "수", "목", "금", "토", "일"]
        heat = f.assign(요일=f["purchase_ts"].dt.dayofweek, 시=f["purchase_ts"].dt.hour).pivot_table(
            index="요일", columns="시", values="order_id", aggfunc="count", fill_value=0)
        heat = heat.reindex(index=range(7), columns=range(24), fill_value=0)
        heat.index = days
        fig = px.imshow(heat, color_continuous_scale=HEAT, aspect="auto",
                        labels={"x": "시", "y": "요일", "color": "주문 수"}, title="요일 × 시간대 주문 수")
        fig.update_traces(xgap=2, ygap=2)
        fig.update_xaxes(dtick=2)
        # 가장 붐비는 3칸을 주황 테두리로 표시 (주황 = 주목)
        top3 = heat.stack().nlargest(3)
        for (d, h), _ in top3.items():
            yi = days.index(d)
            fig.add_shape(type="rect", x0=h - 0.5, x1=h + 0.5, y0=yi - 0.5, y1=yi + 0.5,
                          line={"color": HIGHLIGHT, "width": 3})
        ui.chart(fig)
        peak_d = heat.sum(axis=1).idxmax()
        peak_h = heat.sum(axis=0).idxmax()
        st.info(f"파랑이 진할수록 주문이 많고, **주황 테두리 3칸이 가장 붐비는 시간**입니다. 가장 많은 요일은 **{peak_d}요일**, "
                f"시간대는 **{peak_h}시**입니다. 알림 · 타임 세일 시간을 정할 때 씁니다.")

    with tab3:
        c1, c2 = st.columns([1, 3])
        basis = c1.radio("순위 기준", ["매출", "판매 수량"])
        top_n = c1.slider("표시할 개수", 5, 30, 15)
        c1.caption("파랑 = 상위 3개 · 빗금 = 나머지")
        cat = fi.groupby("category").agg(매출=("price", "sum"), 수량=("price", "size")).rename(
            columns={"수량": "판매 수량"}).sort_values(basis, ascending=False).head(top_n).reset_index()
        cat = cat.iloc[::-1]                                  # 위에서부터 1위가 오도록
        n = len(cat)
        top_cols = shade(cat[basis].tail(3), MAIN_LIGHT, MAIN_DEEP)
        colors = [HATCH_BG] * (n - 3) + top_cols if n > 3 else top_cols[-n:]
        shapes = ["/"] * max(n - 3, 0) + [""] * min(3, n)
        fmt = (lambda v: f"R$ {v / 1e3:,.0f}K") if basis == "매출" else (lambda v: f"{v:,.0f}개")
        fig = go.Figure(go.Bar(x=cat[basis], y=cat["category"], orientation="h",
                               marker={"color": colors, "pattern": {"shape": shapes, "fgcolor": HATCH_FG, "size": 6,
                                                                    "solidity": 0.25},
                                       "line": {"width": 0}},
                               text=[fmt(v) for v in cat[basis]], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>"))
        fig.update_layout(title=f"카테고리 {basis} TOP {top_n}", yaxis_title=None, xaxis_title=None,
                          height=max(330, 24 * n + 80), margin={"r": 70})
        ui.chart(fig, c2)

    with tab4:
        c1, c2 = st.columns(2)
        bins = pd.cut(f["delivery_days"], [0, 7, 14, 21, 30, 60, 1000],
                      labels=["~7일", "8~14일", "15~21일", "22~30일", "31~60일", "60일+"])
        by = f.groupby(bins, observed=True)["review_score"].mean().reset_index()
        color = [MAIN_DEEP if v >= 4.3 else MAIN if v >= 4 else (WARN if v >= 3 else HIGH_DEEP) for v in by["review_score"]]
        fig = go.Figure(go.Bar(x=by["delivery_days"].astype(str), y=by["review_score"], marker_color=color,
                               text=[f"{v:.2f}" for v in by["review_score"]], textposition="outside",
                               cliponaxis=False, hovertemplate="%{x}<br>평균 별점 %{y:.2f}<extra></extra>"))
        fig.update_layout(title="배송이 길어질수록 별점이 떨어집니다", yaxis={"range": [1, 5.4], "title": "평균 별점"},
                          xaxis_title="배송 기간")
        ui.chart(fig, c1)
        c1.caption("파랑 = 4점 이상 · 연주황 = 3점대 · 진주황 = 3점 미만")
        lr = (f.groupby(pd.Grouper(key="purchase_ts", freq="MS"))["is_late"].mean() * 100).dropna().reset_index()
        avg_late = f["is_late"].mean() * 100
        fig = go.Figure(go.Scatter(
            x=lr["purchase_ts"], y=lr["is_late"], mode="lines+markers", fill="tozeroy",
            fillgradient={"type": "vertical", "colorscale": [[0, "rgba(255,109,42,0)"], [1, "rgba(255,109,42,0.38)"]]},
            line={"color": HIGHLIGHT, "width": 2.5, "shape": "spline", "smoothing": 0.6},
            marker={"size": [11 if v >= avg_late * 1.5 else 6 for v in lr["is_late"]], "color": HIGH_DEEP,
                    "line": {"color": "#FFFFFF", "width": 1.5}},
            hovertemplate="%{x|%Y-%m}<br>%{y:.1f}%<extra></extra>"))
        fig.add_hline(y=avg_late, line_dash="dot", line_color="#8A9099", line_width=1.5,
                      annotation_text=f"평균 {avg_late:.1f}%", annotation_position="top left",
                      annotation_font_color="#5B616B")
        fig.update_layout(title="월별 늦게 도착한 비율", yaxis_title="늦게 도착한 비율 (%)", xaxis_title=None)
        ui.chart(fig, c2)
        c2.caption("큰 점 = 평균의 1.5배를 넘은 달 (주문이 몰린 시기)")
        if pd.notna(late):
            st.info(f"제때 도착한 주문의 평균 별점은 **{on_time:.2f}점**, 늦게 도착한 주문은 **{late:.2f}점**입니다.")

    with tab5:
        reg = f.groupby("customer_state").agg(
            주문수=("order_id", "size"), 매출=("payment_value", "sum"), 배송기간=("delivery_days", "median"),
            지연율=("is_late", "mean"), 평균리뷰=("review_score", "mean")).reset_index()
        reg["지연율"] *= 100
        reg["customer_state"] = reg["customer_state"].map(serving.state_name)
        reg.columns = ["주", "주문 수", "매출", "배송 기간 (보통, 일)", "늦게 도착한 비율 (%)", "평균 별점"]
        pick = st.selectbox("비교할 지표", reg.columns[1:])
        worse_high = pick in ["배송 기간 (보통, 일)", "늦게 도착한 비율 (%)"]   # 높을수록 나쁜 지표
        srt = reg.sort_values(pick, ascending=False).reset_index(drop=True)
        n = len(srt)
        top = shade(srt[pick].head(3), *((HIGH_LIGHT, HIGH_DEEP) if worse_high else (MAIN_LIGHT, MAIN_DEEP)))
        colors = top + [HATCH_BG] * max(n - 3, 0)
        shapes = [""] * min(3, n) + ["/"] * max(n - 3, 0)
        fig = go.Figure(go.Bar(x=srt["주"], y=srt[pick],
                               marker={"color": colors[:n], "pattern": {"shape": shapes, "fgcolor": HATCH_FG, "size": 6,
                                                                        "solidity": 0.25}},
                               hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>"))
        avg_r = reg[pick].mean()
        fig.add_hline(y=avg_r, line_dash="dot", line_color="#5B616B", line_width=1.2,
                      annotation_text=f"주 평균 {avg_r:,.1f}", annotation_position="top right",
                      annotation_font_color="#5B616B")
        fig.update_layout(title=f"주별 {pick}", xaxis_title=None, xaxis_tickangle=-45, yaxis_title=None)
        ui.chart(fig)
        st.caption(("주황 = 가장 나쁜 3개 주 (높을수록 나쁜 지표)" if worse_high else "파랑 = 상위 3개 주")
                   + " · 빗금 = 나머지 · 점선 = 주 평균")
        st.dataframe(reg.round(2), hide_index=True, width="stretch")

    with tab6:
        show = f.drop(columns=["month"]).assign(customer_state=f["customer_state"].map(serving.state_name)).rename(columns={
            "order_id": "주문번호", "customer_unique_id": "고객번호", "purchase_ts": "주문 시각",
            "customer_state": "주", "payment_value": "결제 금액(R$)", "payment_type": "결제 수단",
            "delivery_days": "배송 일수", "is_late": "늦게 도착 (1=예)", "review_score": "별점"})
        st.caption(f"{len(show):,}행 · 필터가 적용된 주문 표")
        st.dataframe(show.head(1000), hide_index=True, width="stretch")
        st.download_button("CSV 내려받기", show.to_csv(index=False).encode("utf-8-sig"),
                           "olist_orders_filtered.csv", "text/csv")


st.caption(SOURCE)

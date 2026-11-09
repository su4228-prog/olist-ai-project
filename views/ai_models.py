import html

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import serving
import ui
from common import (BAD, BLUE, GOOD, GRAD_BAD, GRAD_GOOD, GREY, HIGHLIGHT, LIME, MAIN, MAIN_DEEP, MUTE, RED, SOURCE, friendly, load_data, load_products, load_word_stats,
                    models_or_stop, scored_reviews, scored_sellers, translate, word_meaning)

_, _, sellers_raw, reviews_raw = load_data()
models = models_or_stop()
prod_info, catalog = load_products()
ss = st.session_state

SEG_COLOR = {"핵심 우량": ("#E8EFFE", "#1C56E2"), "신규 성장": ("#EEF3FE", "#6F95F2"),
             "휴면": ("#F3F3F5", "#8A9099"), "저평점 위험": ("#FFEDE4", "#E3490A")}
word_stats = load_word_stats()


def why_block(word: str, side: str) -> str:
    """단어 하나가 왜 불만/만족 쪽인지 : 실제 리뷰에서의 등장 비율로 설명"""
    meaning = word_meaning(word)
    head = f'<span class="w">{word}</span>' + (f' <span class="m">· {meaning}</span>' if meaning else "")
    if word_stats is None or word not in word_stats.index:
        say = "불만 리뷰에서 더 자주 쓰인 말입니다." if side == "neg" else "만족 리뷰에서 더 자주 쓰인 말입니다."
        return f'<div class="why">{head}<div class="say">{say}</div></div>'
    n, p = word_stats.at[word, "neg_rate"] * 100, word_stats.at[word, "pos_rate"] * 100
    big, small = (n, p) if side == "neg" else (p, n)
    ratio = big / small if small > 0.05 else None
    who = "불만" if side == "neg" else "만족"
    if ratio is not None and ratio < 1.3:
        say = "불만 · 만족 리뷰 모두에 비슷하게 나오는 말이라 판정에 주는 영향은 작습니다."
    elif ratio is None:
        say = f"거의 {who} 리뷰에서만 쓰이는 말입니다."
    else:
        say = f"{who} 리뷰에서 <b>{ratio:.0f}배</b> 더 자주 쓰이는 말입니다." if ratio >= 2 else \
              f"{who} 리뷰에서 조금 더 자주 쓰이는 말입니다."
    rows = "".join(
        f'<div class="row"><span>{label}</span><div class="bar"><i style="width:{min(v, 100):.0f}%;background:{c}"></i></div>'
        f'<span>{v:.0f}%</span></div>'
        for label, v, c in [("불만 리뷰", n, GRAD_BAD), ("만족 리뷰", p, GRAD_GOOD)])
    return f'<div class="why">{head}{rows}<div class="say">{say}</div></div>'

ui.header("AI 모델", "실제 판매 데이터로 미리 학습해 둔 모델이, 왼쪽에 넣은 새 값을 보고 오른쪽에 바로 판정합니다.",
          right="모델 4개 · 기준일 2018-08-23")
tab1, tab2, tab3, tab4 = st.tabs(["💬 리뷰 판정", "🏪 판매자 진단", "🧭 비슷한 상품 찾기", "📋 모델 설명"])

# ── 1. 리뷰 판정 ─────────────────────────────────────────
with tab1:
    left, right = st.columns([5, 7], gap="medium")
    with left, ui.card("sent-in"):
        st.markdown("### 리뷰 입력")
        st.caption("불만 리뷰인지 판정합니다. 한국어로 써도 됩니다.")
        examples = {
            "직접 입력": "",
            "아직 못 받음": "아직 상품을 못 받았어요. 배송 기한이 지났는데 연락도 없어요.",
            "파손": "상품이 깨진 채로 왔어요. 품질이 최악이에요.",
            "빠른 배송 칭찬": "예정보다 빨리 왔고 상품도 아주 좋아요. 추천합니다!",
            "다른 색상 도착": "검은색을 주문했는데 다른 색이 왔어요. 반품하고 싶어요.",
            "포르투갈어 원문": "Ainda não recebi o produto e o prazo já passou",
        }
        ex = st.selectbox("예시 문장 불러오기", list(examples), index=1)
        text = st.text_area("리뷰 글", examples[ex], height=120, placeholder="예: 포장이 꼼꼼하고 배송도 빨랐어요")
        if st.button("판정하기", type="primary", key="sent_btn", width="stretch"):
            if text.strip():
                pt, how = translate(text)
                ss["sent"] = {"text": text, "pt": pt, "how": how, **serving.sentiment_explain(models, pt)}
            else:
                ss.pop("sent", None)
                st.warning("리뷰 글을 입력해 주세요.")

    with right, ui.card("sent-out"):
        st.markdown("### 판정 결과")
        r = ss.get("sent")
        if not r:
            ui.empty("왼쪽에 리뷰를 넣고 <b>판정하기</b>를 누르면 결과가 여기에 나타납니다.")
        elif r["known"] == 0:
            st.warning("판정에 쓸 수 있는 표현을 찾지 못했습니다. 배송 · 상품 상태 · 만족도에 대한 문장으로 써 주세요.")
        else:
            p = r["neg_prob"]
            if p >= 0.5:
                ui.result("불만 리뷰", f"불만일 가능성 {p * 100:.0f}% · 고객센터가 먼저 확인할 리뷰", "#FFEDE4", "#E3490A")
            else:
                ui.result("만족 리뷰", f"불만일 가능성 {p * 100:.0f}%", "#E8EFFE", "#1C56E2")
            if r["how"] != "원문":
                note = ("자동 번역 (문장 전체를 옮김)" if r["how"] == "자동 번역"
                        else "간이 사전 (번역 연결이 안 되어 핵심 표현만 옮김)")
                st.markdown(
                    f'<div class="readbox">'
                    f'<div class="rb-row"><span class="rb-k">입력한 글</span><span>{html.escape(r["text"])}</span></div>'
                    f'<div class="rb-row"><span class="rb-k">모델이 읽은<br>포르투갈어</span><span class="rb-pt">{html.escape(r["pt"])}</span></div>'
                    f'<div class="rb-row"><span class="rb-k">방식</span><span class="rb-note">{note}</span></div>'
                    f'</div>', unsafe_allow_html=True)
            def agrees(word, side):
                """실제 리뷰 통계도 같은 쪽을 가리키는 단어만 근거로 보여 줌 (모델 · 데이터가 엇갈리는 말은 숨김)"""
                if word_stats is None or word not in word_stats.index:
                    return True
                n, p_ = word_stats.at[word, "neg_rate"], word_stats.at[word, "pos_rate"]
                return n > p_ * 1.3 if side == "neg" else p_ > n * 1.3

            neg_w = r["neg_words"][r["neg_words"]["단어"].map(lambda w: agrees(w, "neg"))].head(4)
            pos_w = r["pos_words"][r["pos_words"]["단어"].map(lambda w: agrees(w, "pos"))].head(4)
            neg_sum, pos_sum = -r["neg_words"]["영향"].sum(), r["pos_words"]["영향"].sum()
            lean = "불만" if r["neg_prob"] >= 0.5 else "만족"
            st.markdown(f"**왜 {lean} 리뷰로 봤나요?** 아래 말들이 실제 리뷰 3만여 건에서 어느 쪽에 더 자주 나왔는지를 "
                        f"보고 판단했습니다. " + ("불만 쪽 단서가 더 강했습니다." if neg_sum > pos_sum else "만족 쪽 단서가 더 강했습니다."))
            c1, c2 = st.columns(2)
            for col, words, side, head in [(c1, neg_w, "neg", "😟 불만으로 기울게 한 말"), (c2, pos_w, "pos", "🙂 만족으로 기울게 한 말")]:
                col.markdown(f"**{head}**")
                if words.empty:
                    col.markdown(f'<div class="why"><div class="say" style="margin:0;color:#7C8783">'
                                 f'{"불만" if side == "neg" else "만족"} 쪽으로 뚜렷하게 기울게 한 말은 없습니다.</div></div>',
                                 unsafe_allow_html=True)
                else:
                    col.markdown("".join(why_block(w, side) for w in words["단어"]), unsafe_allow_html=True)
            st.caption("막대는 실제 리뷰 중 그 말이 들어간 리뷰의 비율입니다 (불만 = 1·2점, 만족 = 4·5점).")

    with ui.card("sent-many"), st.expander("여러 리뷰 한 번에 판정하기"):
        many = st.text_area("한 줄에 리뷰 하나씩", "상품이 아주 좋아요\n주문한 지 한 달인데 아직 안 왔어요\nVeio com defeito",
                            height=110)
        rows = [t for t in many.splitlines() if t.strip()]
        if rows:
            prob = serving.sentiment(models, [translate(t)[0] for t in rows])
            st.dataframe(pd.DataFrame({"리뷰": rows, "불만일 가능성": [f"{x * 100:.0f}%" for x in prob],
                                       "판정": np.where(prob >= 0.5, "불만", "만족")}),
                         hide_index=True, width="stretch")

# ── 2. 판매자 진단 ───────────────────────────────────────
with tab2:
    sellers = scored_sellers(models, sellers_raw)
    groups = {
        "판매 규모": [("frequency", "받은 주문 (건)", 1, 1, "%d", False),
                  ("monetary", "판매액 (R$)", 0.0, 100.0, "%.0f", False),
                  ("orders_last_3m", "최근 3개월 주문", 0, 1, "%d", False),
                  ("orders_prev_3m", "그 전 3개월 주문", 0, 1, "%d", False)],
        "활동 기간": [("recency_days", "마지막 판매 후 (일)", 0, 1, "%d", False),
                  ("tenure_days", "첫 판매 후 (일)", 0, 1, "%d", False),
                  ("active_months", "판매한 달 (개월)", 1, 1, "%d", False)],
        "고객 만족 · 배송": [("avg_review_score", "평균 별점 (1~5)", 1.0, 0.1, "%.1f", False),
                        ("low_review_rate", "1·2점 리뷰 (%)", 0.0, 1.0, "%.0f", True),
                        ("avg_delivery_days", "평균 배송 (일)", 0.0, 0.5, "%.1f", False),
                        ("late_rate", "늦게 도착 (%)", 0.0, 1.0, "%.0f", True),
                        ("ship_late_rate", "출고 마감 넘김 (%)", 0.0, 1.0, "%.0f", True)],
    }
    keys = [f[0] for g in groups.values() for f in g]
    medians = sellers.groupby("segment")[keys].median()
    default = {"frequency": 150, "monetary": 15000.0, "recency_days": 3, "tenure_days": 540, "active_months": 18,
               "avg_review_score": 4.2, "low_review_rate": 0.08, "avg_delivery_days": 11.0, "late_rate": 0.05,
               "ship_late_rate": 0.03, "orders_last_3m": 30, "orders_prev_3m": 28}
    options = {"직접 입력": None} | {f"{friendly(s)}의 보통 모습": s for s in serving.SEGMENT_ORDER}

    left, right = st.columns([5, 7], gap="medium")
    with left, ui.card("seller-in"):
        st.markdown("### 판매 기록 입력")
        preset = st.selectbox("예시 판매자로 채우기", list(options),
                              help="실제 판매자 1,817명 중 그 유형에 속한 판매자들의 가운데 값(중앙값)을 채워 넣습니다.")
        base = default if options[preset] is None else medians.loc[options[preset]].to_dict()
        vals = {}
        for gname, fields in groups.items():
            st.markdown(f"**{gname}**")
            cols = st.columns(2)
            for i, (k, name, lo, step, fmt, pct) in enumerate(fields):
                v = base[k] * 100 if pct else base[k]
                v = int(round(v)) if isinstance(step, int) else float(round(v, 2))
                vmax = 100.0 if pct else (5.0 if k == "avg_review_score" else None)
                got = cols[i % 2].number_input(name, min_value=lo, max_value=vmax, value=max(v, lo), step=step,
                                               format=fmt, key=f"{k}_{preset}")
                vals[k] = got / 100 if pct else got
        if st.button("진단하기", type="primary", key="seller_btn", width="stretch"):
            ss["seller"] = {"vals": dict(vals), **serving.predict_seller(models, vals)}

    with right, ui.card("seller-out"):
        st.markdown("### 진단 결과")
        r = ss.get("seller")
        if not r:
            ui.empty("왼쪽에 판매 기록을 넣고 <b>진단하기</b>를 누르면 결과가 여기에 나타납니다.<br>"
                     "처음이라면 <b>예시 판매자로 채우기</b>에서 하나를 골라 보세요.")
        else:
            if r["vals"] != vals:
                st.caption("입력값이 바뀌었습니다. 새 값으로 보려면 진단하기를 다시 누르세요.")
            bg, fg = SEG_COLOR.get(r["segment"], ("#F1F1F5", "#333"))
            ui.result(friendly(r["segment"]), r["desc"], bg, fg)
            st.markdown(f"**이렇게 관리하세요** : {r['action']}")
            m1, m2 = st.columns(2)
            m1.metric("3개월 안에 판매를 멈출 가능성", f"{r['churn_prob'] * 100:.0f}%")
            same = (sellers["segment"] == r["segment"]).sum()
            m2.metric("실제 판매자 중 같은 유형", f"{same:,}명")
            risk_color = {"낮음": "green", "보통": "orange", "높음": "red", "매우 높음": "red"}.get(r["risk"], "gray")
            st.markdown(f"위험 단계 : :{risk_color}[**{r['risk']}**] · 50% 이상이면 미리 연락할 대상으로 봅니다.")
            reasons = serving.churn_reasons(models, serving.seller_from_input(r["vals"]))
            fig = px.bar(reasons, x="영향", y="요인", orientation="h", color="방향",
                         color_discrete_map={"위험을 높임": BAD, "위험을 낮춤": GOOD},
                         title="왜 이렇게 판정했나요?")
            fig.update_layout(yaxis={"categoryorder": "total ascending", "title": None}, height=290,
                              xaxis_title="← 위험을 낮춤 · 위험을 높임 →", legend_title=None)
            ui.chart(fig)
            sim = 1 / (r["closeness"] + 1e-6)
            sim = (sim / sim.sum() * 100).round(0)
            short = {"핵심 우량": "주력 판매자", "신규 성장": "새내기 판매자", "휴면": "뜸해진 판매자", "저평점 위험": "관리 필요 판매자"}
            near = pd.DataFrame({"유형": [short.get(s, s) for s in sim.index], "닮은 정도 (%)": sim.values})
            fig = px.bar(near, x="닮은 정도 (%)", y="유형", orientation="h", title="어느 유형과 가장 닮았나요?",
                         color="유형", color_discrete_map={short[k]: v[1] for k, v in SEG_COLOR.items()})
            fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending", "title": None}, height=230)
            ui.chart(fig)

# ── 3. 비슷한 상품 찾기 ──────────────────────────────────
with tab3:
    prods = models["recommend"]["products"]
    feats = models["recommend"]["features"]
    left, right = st.columns([4, 8], gap="medium")
    with left, ui.card("rec-in"):
        st.markdown("### 새 상품 정보")
        st.caption("같은 카테고리에서 가장 비슷한 기존 상품 5개를 찾습니다.")
        cat = st.selectbox("카테고리", sorted(prods["category_kr"].unique()))
        med = prods[prods["category_kr"] == cat][feats].median()
        pf = [("price", "가격 (R$)", 1.0, 5.0, "%.0f"), ("weight_g", "무게 (g)", 1.0, 50.0, "%.0f"),
              ("photos_qty", "사진 수 (장)", 1.0, 1.0, "%.0f"), ("avg_review_score", "예상 별점", 1.0, 0.1, "%.1f"),
              ("length_cm", "가로 (cm)", 1.0, 1.0, "%.0f"), ("width_cm", "세로 (cm)", 1.0, 1.0, "%.0f"),
              ("height_cm", "높이 (cm)", 1.0, 1.0, "%.0f")]
        pv = {}
        cols = st.columns(2)
        for i, (k, name, lo, step, fmt) in enumerate(pf):
            pv[k] = cols[i % 2].number_input(name, min_value=lo, max_value=5.0 if k == "avg_review_score" else None,
                                             value=float(max(round(med[k], 1), lo)), step=step, format=fmt,
                                             key=f"p_{k}_{cat}")
        st.caption("처음 값은 이 카테고리 상품들의 가운데 값입니다.")
        if st.button("비슷한 상품 찾기", type="primary", key="rec_btn", width="stretch"):
            ss["rec"] = {"cat": cat, "pv": dict(pv), "df": serving.recommend_from_features(models, cat, pv)}

    with right, ui.card("rec-out"):
        st.markdown("### 추천 결과")
        r = ss.get("rec")
        if not r:
            ui.empty("왼쪽에 새 상품 정보를 넣고 <b>비슷한 상품 찾기</b>를 누르면 비슷한 상품 5개가 카드로 나타납니다.")
        else:
            rec = r["df"].merge(prod_info, on="product_id", how="left")
            if catalog is not None:
                rec = rec.merge(catalog.drop(columns=["category_kr"], errors="ignore"), on="product_id", how="left")
            rpv, rcat = r["pv"], r["cat"]
            if (rcat, rpv) != (cat, pv):
                st.caption("입력값이 바뀌었습니다. 새 값으로 보려면 비슷한 상품 찾기를 다시 누르세요.")
            cat_sold = prod_info.loc[prod_info["product_id"].isin(prods.loc[prods["category_kr"] == rcat, "product_id"]),
                                     "sold_count"]
            many = cat_sold.quantile(0.75) if len(cat_sold) else np.inf

            def size_label(x):
                vol = x["length_cm"] * x["width_cm"] * x["height_cm"]
                return "작은 상품" if vol < 2000 else ("중간 크기" if vol < 15000 else "큰 상품")

            def sim_label(d):
                return "매우 비슷" if d < 0.5 else ("비슷" if d < 1.0 else ("조금 비슷" if d < 1.5 else "차이 있음"))

            def diff(a, b):
                return f"{(a - b) / b * 100:+.0f}%" if b else "–"

            def tag_list(x):
                t = []
                if pd.notna(x.get("sold_count")) and x["sold_count"] >= many:
                    t.append("🔥 잘 팔림")
                if x["avg_review_score"] >= 4.5:
                    t.append("⭐ 별점 높음")
                if x["price"] <= rpv["price"] * 0.8:
                    t.append("💸 더 저렴")
                elif x["price"] >= rpv["price"] * 1.2:
                    t.append("💎 더 비쌈")
                if x["weight_g"] <= rpv["weight_g"] * 0.7:
                    t.append("🪶 더 가벼움")
                return t

            cols = st.columns(3)
            for i, x in rec.iterrows():
                with cols[i % 3].container(border=True):
                    name = x["name"] if "name" in rec.columns and pd.notna(x["name"]) and str(x["name"]).strip() else None
                    if "image_url" in rec.columns and pd.notna(x["image_url"]) and str(x["image_url"]).strip():
                        st.image(x["image_url"], width="stretch")
                    st.caption(f"{i + 1}위 · {sim_label(x['거리'])}")
                    if name:
                        link = x["link"] if "link" in rec.columns and pd.notna(x["link"]) else None
                        st.markdown(f"**[{name}]({link})**" if link else f"**{name}**")
                    st.markdown(f'<div class="price">R$ {x["price"]:,.0f}</div>', unsafe_allow_html=True)
                    sold = f"{int(x['sold_count']):,}건" if pd.notna(x.get("sold_count")) else "–"
                    st.markdown(f"{x['category_kr']} · {size_label(x)}  \n★ {x['avg_review_score']:.1f} · 판매 {sold}  \n"
                                f"{x['weight_g'] / 1000:,.1f}kg · {x['length_cm']:.0f}×{x['width_cm']:.0f}×{x['height_cm']:.0f}cm")
                    tg = tag_list(x)
                    if tg:
                        st.markdown(ui.tags(tg), unsafe_allow_html=True)
                    st.caption(f"가격 {diff(x['price'], rpv['price'])} · 무게 {diff(x['weight_g'], rpv['weight_g'])}"
                               f" · #{x['product_id'][:8]}")
            with cols[len(rec) % 3]:
                pool = prods[prods["category_kr"] == rcat].assign(구분="다른 상품")
                pool.loc[pool["product_id"].isin(rec["product_id"]), "구분"] = "추천"
                fig = px.scatter(pd.concat([pool, pd.DataFrame([{**rpv, "구분": "새 상품"}])]), x="price", y="weight_g",
                                 color="구분", log_x=True, log_y=True,
                                 color_discrete_map={"다른 상품": "#D3D7DD", "추천": MAIN_DEEP, "새 상품": HIGHLIGHT},
                                 labels={"price": "가격 (R$)", "weight_g": "무게 (g)"})
                fig.update_traces(selector={"name": "새 상품"}, marker={"size": 14, "symbol": "star"})
                fig.update_xaxes(dtick=1)
                fig.update_yaxes(dtick=1)
                fig.update_layout(height=290, margin={"t": 4, "b": 0},
                                  legend={"title": None, "orientation": "h", "y": -0.3, "x": 0, "xanchor": "left",
                                          "yanchor": "top"})
                st.caption("★ 새 상품 주변에 추천 상품이 모입니다")
                ui.chart(fig)

    with ui.card("rec-link"), st.expander("상품 이름 · 사진 · 링크와 연결하려면"):
        st.markdown("""
Olist는 공개 데이터를 만들 때 **상품 이름과 사진을 지웠습니다** (판매자 보호를 위한 익명화). 그래서 카드는
카테고리 · 가격 · 크기 · 판매 실적으로 만들었습니다. 실제 쇼핑몰에서는 **상품번호로 상품 정보표를 붙이면**
카드에 이름 · 사진 · 링크가 바로 뜹니다.

1. 아래 버튼으로 추천 대상 상품 2,498개의 번호 목록을 내려받습니다.
2. `name`(상품명), `image_url`(사진 주소), `link`(상품 페이지 주소) 칸을 채웁니다.
3. 파일 이름을 `product_catalog.csv`로 바꿔 `data` 폴더에 넣으면 카드가 자동으로 바뀝니다.
""")
        tmpl = prods[["product_id", "category_kr"]].assign(name="", image_url="", link="")
        st.download_button("상품 정보표 양식 내려받기 (CSV)", tmpl.to_csv(index=False).encode("utf-8-sig"),
                           "product_catalog.csv", "text/csv")

# ── 4. 모델 설명 ─────────────────────────────────────────
with tab4:
    rv = scored_reviews(models, reviews_raw)
    lab = rv[rv["review_score"] != 3]
    agree = ((lab["review_score"] <= 2) == (lab["pred"] == "부정")).mean()
    sellers = scored_sellers(models, sellers_raw)

    st.markdown("### 이 모델들은 얼마나 믿을 만한가요?")
    c1, c2 = st.columns(2, gap="medium")
    with c1, ui.card("m-sent"):
        st.markdown("#### 💬 리뷰 판정")
        st.markdown("처음 보는 리뷰 **100개 중 92개**를 맞혔습니다. "
                    "모두 '만족'이라고 찍었다면 71개였으니, 그보다 21개를 더 맞힌 셈입니다.")
        st.caption(f"최근 리뷰 {len(lab):,}건에서 별점(1·2점 = 불만, 4·5점 = 만족)과 판정이 {agree * 100:.0f}% 일치")
    with c2, ui.card("m-churn"):
        st.markdown("#### 🏪 판매 중단 예측")
        st.markdown("실제로 판매를 멈춘 판매자 **4명 중 3명**을 미리 찾아냈습니다. 대신 '멈출 것'이라고 본 판매자 중 "
                    "실제로 멈춘 건 10명 중 4명 정도라, **놓치지 않는 쪽**으로 넓게 고릅니다.")
        st.caption(f"최신 판매자 {len(sellers):,}명 중 연락 대상(50% 이상) {(sellers['churn_prob'] >= 0.5).sum():,}명")
    c3, c4 = st.columns(2, gap="medium")
    with c3, ui.card("m-seg"):
        st.markdown("#### 🏷️ 판매자 유형")
        st.markdown("정답이 정해진 문제가 아니라, 판매 기록이 **비슷한 판매자끼리 4묶음**으로 나눈 뒤 사람이 이름을 붙였습니다. "
                    "평점이 낮아 관리가 필요한 판매자는 평균 배송이 약 22일로 길고, 3건 중 1건은 출고 마감부터 넘겼습니다.")
    with c4, ui.card("m-rec"):
        st.markdown("#### 🧭 비슷한 상품 찾기")
        st.markdown("같은 카테고리에서 아무 상품이나 고를 때보다 가격 · 무게 · 크기 차이가 "
                    "**72% → 30%**로 줄었습니다.")

    with ui.card("m-caution"):
        st.markdown("### 이렇게 쓸 때 주의하세요")
        st.markdown("""
- **2016~2018년 브라질 데이터**로 배웠습니다. 지금 · 다른 나라 쇼핑몰에 쓰려면 새 데이터로 다시 학습해야 합니다.
- '판매 중단'은 **3개월 동안 주문이 0건**이라는 실습용 기준입니다. 실제 탈퇴 · 폐업과는 다를 수 있습니다.
- 배송 지연과 낮은 별점은 함께 나타나지만, **지연이 원인이라고 단정할 수는 없습니다** (상품 · 판매자 영향도 섞임).
- 상품 추천은 고객이 실제로 눌렀는지 확인하지 않았습니다. 실제로 쓰려면 A/B 테스트로 효과를 확인해야 합니다.
""")
        with st.expander("분석가용 상세 정보 (방법 · 검증 · 파일)"):
            st.markdown("""
| 모델 | 방법 | 검증 방법 | 결과 |
| --- | --- | --- | --- |
| 리뷰 판정 | TF-IDF + 로지스틱 회귀, class_weight | 중복 리뷰 제거 후 시험용 20% 먼저 분리 (7,428건) | 정확도 0.917 (기준선 0.710), 부정 F1 0.87 |
| 판매자 유형 | 로그 변환 + 표준화 + K-Means (k=4) | 엘보우 · 실루엣 점수로 k 선택, 기준일 2018-06-01 이전 정보만 | 4개 군집 (876 · 947 · 461 · 218명) |
| 판매 중단 예측 | 결측 대체 + 표준화 + 로지스틱 회귀 (balanced) | 기준일로 과거(특징) · 미래(정답) 분리, 5겹 교차검증 후 시험 데이터 1회 평가 | 재현율 0.75, 정밀도 0.44 |
| 비슷한 상품 | 로그 변환 + 표준화 + 카테고리별 KNN (k=5) | 랜덤 · 가격 규칙 추천과 같은 기준으로 비교 | 평균 특징 차이 72% → 30% |
""")
            st.markdown("**데이터 누수 방지** : 예측 시점에 알 수 없는 정보(미래 주문 · 시험 데이터)가 학습에 섞이지 않도록 "
                        "시험 데이터를 먼저 떼어 두고, 판매자 모델은 기준일 이전 기록만 썼습니다.")
            st.dataframe(serving.model_info(models), hide_index=True, width="stretch")

st.caption(SOURCE)

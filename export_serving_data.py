"""
STEP2-1. 서빙용 데이터 만들기
- 원본 CSV 9개 → 대시보드가 읽을 작은 파일 4개(data/)로 줄입니다.
- 학습할 때와 같은 판매자 특징 함수를 써서 '최신 기준일' 판매자 표를 만듭니다.
  (모델 예측은 여기서 하지 않고, 앱이 models/의 .pkl을 불러와 직접 합니다 = 서빙)

Colab : RAW_DIR = "/content/drive/MyDrive/Olist/data", OUT_DIR = "/content/drive/MyDrive/Olist/olist_serving"
로컬  : python export_serving_data.py --raw ../data --out .
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

START, END = "2017-01-01", "2018-09-01"   # 분석 기간 (2017-01 ~ 2018-08)
NOW = "2018-08-23"                         # PART 10과 같은 적용 기준일
BAD = ["canceled", "unavailable"]


# ── PART 8 · 9 · 10과 똑같은 판매자 특징 함수 (바꾸면 모델 입력이 달라지므로 그대로 둡니다) ──
def make_seller_profile(orders, items, reviews, cutoff, start="2017-01-01"):
    cutoff = pd.Timestamp(cutoff)
    o = orders[(orders["order_purchase_timestamp"] >= start)
               & (orders["order_purchase_timestamp"] < cutoff)
               & ~orders["order_status"].isin(BAD)]
    line = items.merge(o, on="order_id")
    so = line.groupby(["seller_id", "order_id"]).agg(
        purchase=("order_purchase_timestamp", "first"),
        carrier=("order_delivered_carrier_date", "first"),
        delivered=("order_delivered_customer_date", "first"),
        estimated=("order_estimated_delivery_date", "first"),
        limit=("shipping_limit_date", "max"),
        sales=("price", "sum"),
    ).reset_index()
    so["carrier"] = so["carrier"].where(so["carrier"] < cutoff)
    so["delivered"] = so["delivered"].where(so["delivered"] < cutoff)
    score = reviews[reviews["review_creation_date"] < cutoff].groupby("order_id")["review_score"].mean()
    so["score"] = so["order_id"].map(score)
    so["delivery_days"] = (so["delivered"] - so["purchase"]).dt.total_seconds() / 86400
    so["is_late"] = np.where(so["delivered"].isna(), np.nan, so["delivered"] > so["estimated"])
    so["is_ship_late"] = np.where(so["carrier"].isna(), np.nan, so["carrier"] > so["limit"])
    so["is_low_review"] = np.where(so["score"].isna(), np.nan, so["score"] <= 2)
    so["month"] = so["purchase"].dt.to_period("M")
    g = so.groupby("seller_id")
    prof = pd.DataFrame({
        "frequency": g.size(),
        "monetary": g["sales"].sum(),
        "recency_days": (cutoff - g["purchase"].max()).dt.days,
        "tenure_days": (cutoff - g["purchase"].min()).dt.days,
        "active_months": g["month"].nunique(),
        "avg_delivery_days": g["delivery_days"].mean(),
        "late_rate": g["is_late"].mean(),
        "ship_late_rate": g["is_ship_late"].mean(),
        "avg_review_score": g["score"].mean(),
        "low_review_rate": g["is_low_review"].mean(),
    })
    prof["avg_order_value"] = prof["monetary"] / prof["frequency"]
    return prof.reset_index()


def count_orders(orders, items, start, end):
    valid = orders[~orders["order_status"].isin(BAD)]
    so = items[["order_id", "seller_id"]].drop_duplicates().merge(
        valid[["order_id", "order_purchase_timestamp"]], on="order_id")
    t = so["order_purchase_timestamp"]
    return so[(t >= start) & (t < end)].groupby("seller_id").size()


def make_churn_features(orders, items, reviews, cutoff):
    cutoff = pd.Timestamp(cutoff)
    m3, m6 = cutoff - pd.DateOffset(months=3), cutoff - pd.DateOffset(months=6)
    prof = make_seller_profile(orders, items, reviews, cutoff)
    prof["orders_last_3m"] = prof["seller_id"].map(count_orders(orders, items, m3, cutoff)).fillna(0)
    prof["orders_prev_3m"] = prof["seller_id"].map(count_orders(orders, items, m6, m3)).fillna(0)
    prof["order_trend"] = prof["orders_last_3m"] - prof["orders_prev_3m"]
    for col in ["frequency", "monetary", "avg_order_value", "orders_last_3m", "orders_prev_3m"]:
        prof[f"log_{col}"] = np.log1p(prof[col])
    prof["log_order_trend"] = prof["log_orders_last_3m"] - prof["log_orders_prev_3m"]
    return prof[prof["orders_last_3m"] > 0].reset_index(drop=True)


def main(raw: Path, out: Path):
    out_data = out / "data"
    out_data.mkdir(parents=True, exist_ok=True)
    dcols = ["order_purchase_timestamp", "order_delivered_carrier_date",
             "order_delivered_customer_date", "order_estimated_delivery_date"]
    orders = pd.read_csv(raw / "olist_orders_dataset.csv", parse_dates=dcols)
    items = pd.read_csv(raw / "olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"])
    reviews = pd.read_csv(raw / "olist_order_reviews_dataset.csv", parse_dates=["review_creation_date"])
    payments = pd.read_csv(raw / "olist_order_payments_dataset.csv")
    customers = pd.read_csv(raw / "olist_customers_dataset.csv")
    products = pd.read_csv(raw / "olist_products_dataset.csv")
    trans = pd.read_csv(raw / "product_category_name_translation.csv")

    # ① orders.csv.gz : 배송 완료 주문 1건 = 1행 (EDA 탭용)
    d = orders[(orders["order_status"] == "delivered")
               & orders["order_purchase_timestamp"].between(START, END, inclusive="left")].copy()
    pay = payments.groupby("order_id")["payment_value"].sum()          # 1:N → 주문별로 먼저 합침
    main_pay = (payments.sort_values("payment_value", ascending=False)
                .drop_duplicates("order_id").set_index("order_id")["payment_type"])
    score = reviews.groupby("order_id")["review_score"].mean()
    d = d.merge(customers[["customer_id", "customer_unique_id", "customer_state"]], on="customer_id")
    o = pd.DataFrame({
        "order_id": d["order_id"],
        "customer_unique_id": d["customer_unique_id"],
        "purchase_ts": d["order_purchase_timestamp"],
        "customer_state": d["customer_state"],
        "payment_value": d["order_id"].map(pay),
        "payment_type": d["order_id"].map(main_pay),
        "delivery_days": (d["order_delivered_customer_date"] - d["order_purchase_timestamp"]).dt.total_seconds() / 86400,
        "is_late": (d["order_delivered_customer_date"] > d["order_estimated_delivery_date"]).astype(int),
        "review_score": d["order_id"].map(score),
    })
    o = o.dropna(subset=["payment_value", "delivery_days"])   # 배송 완료일이 빈 8건 제외 → 96,203건 (PPT 기준)
    o["delivery_days"] = o["delivery_days"].round(2)
    # 긴 ID → 짧은 번호 (파일 크기 절약)
    oid = {k: i for i, k in enumerate(o["order_id"])}
    cid = {k: i for i, k in enumerate(o["customer_unique_id"].unique())}
    o["order_id"] = o["order_id"].map(oid)
    o["customer_unique_id"] = o["customer_unique_id"].map(cid)
    o.to_csv(out_data / "orders.csv.gz", index=False)

    # ② items.csv.gz : 판매된 상품 1개 = 1행 (카테고리 탭용)
    it = items[items["order_id"].isin(oid)].merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
    it = it.merge(trans, on="product_category_name", how="left")
    it = pd.DataFrame({
        "order_id": it["order_id"].map(oid),
        "category": it["product_category_name_english"].fillna("unknown"),
        "price": it["price"],
    })
    it.to_csv(out_data / "items.csv.gz", index=False)

    # ③ sellers_latest.csv.gz : 최신 기준일 판매자 특징 (PART 8 · 9 모델 입력)
    latest = make_churn_features(orders, items, reviews, NOW)
    v = orders[~orders["order_status"].isin(BAD)]
    s3 = items.merge(v[["order_id", "order_purchase_timestamp"]], on="order_id")
    t = s3["order_purchase_timestamp"]
    s3 = s3[(t >= pd.Timestamp(NOW) - pd.DateOffset(months=3)) & (t < NOW)]
    latest["sales_last_3m"] = latest["seller_id"].map(s3.groupby("seller_id")["price"].sum()).fillna(0)
    latest.to_csv(out_data / "sellers_latest.csv.gz", index=False)

    # ④ reviews_recent.csv.gz : 2018-08 이후 글이 있는 리뷰 (PART 7-1 모델 입력)
    rr = reviews[(reviews["review_creation_date"] >= "2018-08-01")
                 & reviews["review_comment_message"].notna()].drop_duplicates("review_id")
    rr[["review_id", "review_score", "review_creation_date", "review_comment_message"]].to_csv(
        out_data / "reviews_recent.csv.gz", index=False)

    # ⑤ products_info.csv.gz : 상품 카드에 보여 줄 판매 실적 (상품 이름은 원본에 없음 = 익명화)
    sellers = pd.read_csv(raw / "olist_sellers_dataset.csv")
    vi = items.merge(v[["order_id", "order_purchase_timestamp"]], on="order_id")
    vi = vi[vi["order_purchase_timestamp"].between(START, END, inclusive="left")].merge(
        sellers[["seller_id", "seller_state"]], on="seller_id", how="left")
    info = vi.groupby("product_id").agg(
        sold_count=("order_id", "size"),
        seller_state=("seller_state", lambda x: x.mode().iat[0] if len(x.mode()) else ""),
        freight=("freight_value", "mean"),
    ).reset_index()
    info = info[info["sold_count"] >= 3]          # 추천 모델은 3번 이상 팔린 상품만 씀
    info["freight"] = info["freight"].round(2)
    info.to_csv(out_data / "products_info.csv.gz", index=False)

    for f in sorted(out_data.glob("*.csv.gz")):
        print(f"{f.name:24s} {f.stat().st_size / 1024:8.0f} KB")
    print(f"배송 완료 주문 {len(o):,}건 / 판매자(기준일 {NOW}) {len(latest):,}명 / 최근 리뷰 {len(rr):,}건")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../data", help="원본 CSV 9개가 있는 폴더")
    ap.add_argument("--out", default=".", help="olist_serving 폴더")
    a = ap.parse_args()
    main(Path(a.raw), Path(a.out))

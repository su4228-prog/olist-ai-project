"""
리뷰 판정 설명용 단어 통계 만들기
- 모델이 아는 단어 10,000개 각각이 '불만 리뷰(1·2점)'와 '만족 리뷰(4·5점)'에 몇 %씩 나왔는지 셉니다.
- 화면에서 "이 단어는 불만 리뷰 100건 중 31건에 나와요"처럼 근거를 보여 주는 데 씁니다.
실행 : python make_word_stats.py --raw ../data
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="../data")
a = ap.parse_args()
base = Path(__file__).parent

tfidf = joblib.load(base / "models" / "part7_1_sentiment_ml.pkl")["tfidf"]
rv = pd.read_csv(Path(a.raw) / "olist_order_reviews_dataset.csv")
rv = rv[rv["review_comment_message"].notna() & (rv["review_score"] != 3)]
rv = rv.drop_duplicates("review_comment_message")
neg = rv["review_score"] <= 2

cv = CountVectorizer(vocabulary=tfidf.vocabulary_, binary=True, lowercase=True,
                     token_pattern=tfidf.token_pattern)
X = cv.transform(rv["review_comment_message"])
words = cv.get_feature_names_out()
out = pd.DataFrame({
    "word": words,
    "neg_rate": np.asarray(X[neg.values].mean(axis=0)).ravel(),    # 불만 리뷰 중 이 단어가 나온 비율
    "pos_rate": np.asarray(X[~neg.values].mean(axis=0)).ravel(),   # 만족 리뷰 중 이 단어가 나온 비율
})
out[["neg_rate", "pos_rate"]] = out[["neg_rate", "pos_rate"]].round(4)
out.to_csv(base / "data" / "word_stats.csv.gz", index=False)
print(f"리뷰 {len(rv):,}건 (불만 {neg.sum():,} · 만족 {(~neg).sum():,}) · 단어 {len(out):,}개 저장")
print(out.set_index("word").loc[["não", "recebi", "produto", "ótimo", "atraso", "ainda"]])

"""
Olist 이커머스 분석 · AI 모델 사이트 (진입점)
실행 : streamlit run app.py
"""
import streamlit as st

import ui

# 페이지 공통 설정 · 디자인 · 사이드바 로고
st.set_page_config(page_title="Olist 이커머스 대시보드", page_icon="📦", layout="wide")
ui.setup()
st.logo("assets/logo.svg", size="large")

# 사이드바 메뉴 : 각 페이지는 views/ 폴더의 파일 하나
pages = [
    st.Page("views/home.py", title="홈", icon=":material/home:", default=True),
    st.Page("views/analysis.py", title="데이터 분석", icon=":material/bar_chart:", url_path="analysis"),
    st.Page("views/ai_models.py", title="AI 모델", icon=":material/neurology:", url_path="ai_models"),
]
st.navigation(pages).run()

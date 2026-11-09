"""화면 디자인 : 밝은 회색 바탕 + 흰 카드 + 보라 강조 (에디터형 SaaS 스타일)"""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# 이름은 그대로 두고 값만 초록 테마로 (PURPLE = 메인 진초록, LAVENDER = 라임)
PURPLE, PURPLE_SOFT, INK, MUTED, LINE = "#1F2326", "#F3F3F5", "#1F2326", "#8A9099", "#E3E4E7"
CANVAS, LAVENDER, LIME, MID = "#E9EAEC", "#FF6D2A", "#FF6D2A", "#2E70F2"
SIDEBAR, BEIGE, YELLOW, CORAL = "#1F2326", "#F3F3F5", "#FF6D2A", "#FF6D2A"
BLUE_G, ORANGE_G = "linear-gradient(120deg, #1C56E2 0%, #4F86F5 55%, #9DB9F9 100%)", "#FF6D2A"
CHART_MAIN, CHART_INK = "#2E70F2", "#1F2326"   # 그래프 색은 화면 디자인과 따로 고정 (바꾸지 않음)
RED, GREEN, GREY = "#FF6D2A", "#2E70F2", "#C9CDD3"
FONT = "Pretendard, 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif"

CSS = f"""
<style>
@import url("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500&display=swap");
/* ── 본문 : 회색 바탕 위에 흰 카드 ── */
[data-testid="stMainBlockContainer"] {{
  max-width: 1400px; padding: 3.6rem 2rem 2.5rem; margin: 0 auto;
}}
[data-testid="stHeader"] {{ background: {CANVAS}; height: 2.4rem; }}
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] {{ gap: 1rem; }}

/* ── 사이드바 : 흰색, 선택된 메뉴는 검정 알약 ── */
[data-testid="stSidebar"] {{ min-width: 228px !important; max-width: 228px !important; }}
[data-testid="stSidebarHeader"] {{ height: 4rem; padding: 1.5rem 1.4rem 0; }}
[data-testid="stSidebarHeader"] img {{ height: 26px !important; }}
[data-testid="stSidebarNavLink"] {{ border-radius: 999px; margin: 2px 12px; padding: 7px 14px; }}
[data-testid="stSidebarNavLink"] span {{ color: #4A5059 !important; }}
[data-testid="stSidebarNavLink"]:hover {{ background: #F3F3F5; }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{ background: {INK}; }}
[data-testid="stSidebarNavLink"][aria-current="page"] span {{ color: #FFFFFF !important; font-weight: 600; }}
[data-testid="stSidebarUserContent"] [data-testid="stWidgetLabel"] p {{ color: {MUTED} !important; font-size: .78rem; }}

/* ── 카드 : 큰 둥근 모서리의 흰 카드, 테두리 · 그림자 없음 ── */
[class*="st-key-card"] {{
  background: #FFFFFF; border: none !important; border-radius: 18px !important;
  padding: 20px 22px 22px !important; box-shadow: none;
}}
[class*="st-key-card"] [data-testid="stHeading"] h3,
[class*="st-key-card"] h3 {{ margin: 0 0 12px; padding: 0; font-size: 1.12rem; font-weight: 500; letter-spacing: -.01em; }}
[class*="st-key-card"] [data-testid="stVerticalBlock"] {{ gap: .7rem; }}
[class*="st-key-card"] [data-testid="stMarkdownContainer"] table {{ width: 100%; border-collapse: collapse; margin-top: 4px; border: none; }}
[class*="st-key-card"] [data-testid="stMarkdownContainer"] th {{
  background: transparent; text-align: left; color: {MUTED}; font-weight: 400; font-size: .82rem;
  border: none; border-bottom: 1px solid {LINE}; padding: 8px 10px; }}
[class*="st-key-card"] [data-testid="stMarkdownContainer"] td {{ border: none; border-bottom: 1px solid #F0F1F3; padding: 12px 10px; }}
/* 카드 안 작은 상자 (상품 카드) : 연회색 타일 */
[class*="st-key-card"] [data-testid="stVerticalBlock"] div[class][style*="border"] {{ border-radius: 14px; }}

/* 강조 카드 : 파랑 그라데이션 */
[class*="st-key-card-accent"] {{ background: {BLUE_G} !important; }}
[class*="st-key-card-accent"] * {{ color: #FFFFFF !important; }}
[class*="st-key-card-accent"] strong {{ color: #FFFFFF !important; font-weight: 700; }}

/* 같은 줄 카드 높이 맞추기 */
[class*="st-key-eq-"] [data-testid="stHorizontalBlock"] {{ align-items: stretch !important; }}
[class*="st-key-eq-"] [data-testid="stColumn"] {{ display: flex; flex-direction: column; }}
[class*="st-key-eq-"] [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {{ flex: 1 1 auto; height: 100%; }}
[class*="st-key-eq-"] [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > :last-child {{ flex: 1 1 auto; }}
[class*="st-key-eq-"] [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > :last-child [class*="st-key-card"] {{ height: 100%; }}

/* ── 지표 : 흰 카드에 가늘고 큰 숫자 ── */
[data-testid="stMetric"] {{ background: #FFFFFF; border: none; border-radius: 18px; padding: 16px 20px; box-shadow: none;
  min-height: 92px; }}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: .8rem; font-weight: 400; }}
[data-testid="stMetricValue"] div {{ white-space: nowrap; overflow: visible; color: {INK}; font-weight: 300;
  font-family: "Space Grotesk", Pretendard, sans-serif; letter-spacing: -.02em; }}

/* ── 탭 : 알약 모양 칩, 선택된 칩은 검정 ── */
[data-testid="stTabs"] [role="tablist"] {{ gap: 6px; overflow-x: auto; border-bottom: none; padding-bottom: 2px; }}
[data-testid="stTabs"] [data-baseweb="tab-highlight"], [data-testid="stTabs"] [data-baseweb="tab-border"] {{ display: none; }}
[data-testid="stTab"] {{ border-radius: 999px; background: #FFFFFF; border: 1px solid {LINE}; padding: 6px 14px !important;
  height: auto; }}
[data-testid="stTab"] p {{ font-size: .86rem; font-weight: 500; color: #3A4048; }}
[data-testid="stTab"][aria-selected="true"] {{ background: {INK}; border-color: {INK}; }}
[data-testid="stTab"][aria-selected="true"] p {{ color: #FFFFFF; }}
[data-testid="stTabPanel"] {{ padding-top: 1rem; }}

/* ── 버튼 : 주황 알약 ── */
[data-testid="stBaseButton-primary"] {{ background: {ORANGE_G}; border-color: {ORANGE_G}; font-weight: 600; }}
[data-testid="stBaseButton-primary"]:hover {{ background: #E3490A; border-color: #E3490A; }}

/* 입력칸 · 접는 칸 · 안내 상자 */
[data-testid="stDataFrame"], [data-testid="stPlotlyChart"] {{ max-width: 100%; overflow-x: auto; }}
[data-testid="stExpander"] details {{ border-radius: 14px; background: #FFFFFF; border-color: {LINE}; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {{
  background: {BEIGE} !important; border: none; border-radius: 12px; }}
[data-testid="stAlertContentInfo"], [data-testid="stAlertContentInfo"] p {{ color: {INK} !important; }}

/* ── 상단 제목 : 큰 가는 제목 + 주황 알약 ── */
.topbar {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin: 0 0 4px; }}
.crumb {{ font-size: .8rem; color: {MUTED}; }}
.crumb b {{ color: #4A5059; font-weight: 500; }}
.pill {{ display: inline-flex; align-items: center; gap: 6px; background: {ORANGE_G}; color: #FFFFFF;
  font-size: .78rem; font-weight: 600; padding: 7px 14px; border-radius: 999px; }}
.page-title {{ font-size: 2.3rem; font-weight: 300; margin: 2px 0 0; color: {INK}; line-height: 1.2; letter-spacing: -.03em; }}
.page-desc {{ color: {MUTED}; font-size: .88rem; margin: 6px 0 10px; }}
.empty {{ border: 1.5px dashed #D3D7DD; border-radius: 14px; padding: 22px 16px; margin: 2px 0 10px;
  text-align: center; color: {MUTED}; font-size: .9rem; background: #F7F7F8; }}
.result {{ border-radius: 14px; padding: 14px 18px; margin: 0 0 4px; }}
.result .t {{ font-size: 1.35rem; font-weight: 500; line-height: 1.35; }}
.result .s {{ margin-top: 2px; opacity: .85; font-size: .88rem; }}
.price {{ font-size: 1.5rem; font-weight: 300; color: {INK}; margin: 0 0 4px; font-family: "Space Grotesk", Pretendard, sans-serif; }}
.tag {{ display: inline-flex; align-items: center; font-size: .74rem; font-weight: 500; padding: 3px 10px; border-radius: 999px;
  background: #F3F3F5; color: {INK}; margin: 0 4px 4px 0; }}

/* 모델이 읽은 문장 상자 */
.readbox {{ border: none; border-radius: 14px; background: #F5F5F7; padding: 10px 16px; margin: 2px 0 6px; }}
.readbox .rb-row {{ display: grid; grid-template-columns: 88px 1fr; gap: 10px; padding: 6px 0; font-size: .88rem; line-height: 1.5; color: {INK}; }}
.readbox .rb-row + .rb-row {{ border-top: 1px solid #E6E7EA; }}
.readbox .rb-k {{ color: {MUTED}; font-size: .8rem; font-weight: 500; line-height: 1.35; }}
.readbox .rb-pt {{ font-style: italic; color: #1C56E2; }}
.readbox .rb-note {{ color: {MUTED}; font-size: .82rem; }}

/* 리뷰 판정 근거 */
.why {{ border: none; border-radius: 14px; padding: 12px 14px; margin: 0 0 8px; background: #F7F7F8; }}
.why .w {{ font-weight: 600; color: {INK}; }}
.why .m {{ color: {MUTED}; font-weight: 400; }}
.why .row {{ display: grid; grid-template-columns: 74px 1fr 42px; align-items: center; gap: 8px; font-size: .8rem; color: {MUTED}; margin-top: 4px; }}
.why .bar {{ height: 8px; background: repeating-linear-gradient(135deg, #ECEDEF 0 4px, #F7F7F8 4px 8px); border-radius: 99px; overflow: hidden; }}
.why .bar i {{ display: block; height: 100%; border-radius: 99px; }}
.why .say {{ font-size: .84rem; margin-top: 6px; color: {INK}; }}
</style>
"""


def setup():
    st.markdown(CSS, unsafe_allow_html=True)
    # Streamlit은 그래프 모듈을 처음 불러올 때 기본 디자인을 자기 것으로 바꿉니다.
    # 먼저 불러와 그 작업을 끝낸 뒤, 우리 디자인으로 다시 덮어씁니다.
    import streamlit.elements.plotly_chart  # noqa: F401
    t = go.layout.Template()
    t.layout = go.Layout(
        font={"family": FONT, "size": 12, "color": CHART_INK},
        colorway=[CHART_MAIN, "#FF6D2A", "#8FB0F7", "#FDC3A6", "#9AA0A8"],
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        margin={"l": 8, "r": 8, "t": 40, "b": 8},
        title={"font": {"size": 14, "color": CHART_INK}, "x": 0, "xanchor": "left"},
        xaxis={"gridcolor": "#EEEFF1", "linecolor": LINE, "zeroline": False, "automargin": True},
        yaxis={"gridcolor": "#EEEFF1", "linecolor": LINE, "zeroline": False, "automargin": True},
        legend={"orientation": "h", "x": 1, "xanchor": "right", "y": 1.02, "yanchor": "bottom", "title": None},
        hoverlabel={"font": {"family": FONT}},
    )
    pio.templates["olist"] = t
    pio.templates.default = "plotly_white+olist"


def header(page: str, desc: str = "", right: str = ""):
    st.markdown(f"""<div class="topbar"><div class="crumb">Olist / <b>{page}</b></div>
      {f'<span class="pill">{right}</span>' if right else ''}</div>
      <div class="page-title">{page}</div>
      {f'<p class="page-desc">{desc}</p>' if desc else ''}""", unsafe_allow_html=True)


def card(key: str, accent: bool = False):
    """폴더 모양 카드. accent=True 이면 보라색으로 채운 강조 폴더"""
    return st.container(border=True, key=f"card-{'accent-' if accent else ''}{key}")


def row(key: str):
    """안에 넣은 st.columns의 카드들이 같은 높이로 아래 끝이 맞는 줄"""
    return st.container(key=f"eq-{key}")


def empty(text: str):
    st.markdown(f'<div class="empty">{text}</div>', unsafe_allow_html=True)


def result(title: str, sub: str, bg: str, fg: str):
    st.markdown(f'<div class="result" style="background:{bg}"><div class="t" style="color:{fg}">{title}</div>'
                f'<div class="s" style="color:{fg}">{sub}</div></div>', unsafe_allow_html=True)


def tags(items):
    return "".join(f'<span class="tag">{x}</span>' for x in items)


def chart(fig, where=None, toolbar: bool = False):
    """그래프 출력 : 배경 · 글꼴을 그래프에 직접 지정해야 Streamlit이 회색으로 덮어쓰지 않습니다."""
    fig.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                      font={"family": FONT, "size": 12, "color": CHART_INK})
    if fig.layout.height is None:
        fig.update_layout(height=330)
    (where or st).plotly_chart(fig, width="stretch", theme=None,
                               config={"displayModeBar": toolbar, "displaylogo": False})

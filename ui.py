"""화면 디자인 공통 모듈 : 색상 팔레트, 전역 CSS, Plotly 테마, 카드 · 제목 등 화면 부품"""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ── 색상 팔레트 : 파스텔 라벤더 · 옐로 · 스카이블루 ──────────────────
BLUE = "#A8D6F2"           # 메인 그래프 색 (막대 · 선)
BLUE_DEEP = "#74B4E0"      # 진한 하늘색 (히트맵 최댓값)
BLUE_SOFT = "#E3F2FC"      # 선택된 메뉴 · 연한 하늘색 바탕
YELLOW = "#F8DD88"         # 포인트 노랑 (최고값 강조 · 배지)
YELLOW_SOFT = "#FCE8A8"    # 노란 지표 카드 바탕
LAVENDER = "#CCC3F6"       # 보조 보라 (긍정 · 만족)
LAVENDER_SOFT = "#E0DAFB"  # 보라 지표 카드 · 강조 카드 바탕
VIOLET = "#8D7FE6"         # 버튼 · 강조 글자
VIOLET_HOVER = "#7D6EDB"   # 버튼에 마우스를 올렸을 때
RED = "#F3AAB7"            # 경고 분홍 (부정 · 위험)
GREY = "#D9DBE5"           # 비활성 회색 (산점도의 나머지 상품)
INK = "#23253A"            # 본문 글자 (짙은 남색)
MUTED = "#8A8EA3"          # 보조 글자
LINE = "#ECEEF4"           # 테두리
CANVAS = "#F4F5F9"         # 페이지 바탕
SURFACE = "#F7F8FB"        # 카드 안 옅은 바탕 (표 머리글 · 안내 상자)
WHITE = "#FFFFFF"          # 카드 · 그래프 바탕
GRID = "#EEF0F5"           # 그래프 눈금선
ACCENT_TEXT = "#4B3BB0"    # 강조 카드 안 굵은 글자
FONT = "Pretendard, 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif"
RADIUS = "20px"            # 카드 · 지표 상자 모서리
SHADOW = "0 2px 12px rgba(35, 37, 58, .05)"
BAR_RADIUS = 6             # 그래프 막대 끝 둥글기 (px)

# 판정 결과 상자 색 : (배경, 글자) → result(title, sub, *TONE_XXX)
TONE_BLUE = ("#E3F2FC", "#1F6E9E")     # 만족 · 주력 판매자
TONE_VIOLET = ("#ECE8FD", "#5B4BC4")   # 새내기 판매자
TONE_GREY = ("#F1F2F6", "#5D6175")     # 뜸해진 판매자 · 기본값
TONE_RED = ("#FDEBEE", "#C9455C")      # 불만 · 관리 필요 판매자

# ── 전역 CSS : Streamlit 기본 요소의 모양을 덮어씀 ─────────────────
CSS = f"""
<style>
/* 본문 */
[data-testid="stMainBlockContainer"] {{
  max-width: 1400px; padding: 1.2rem 1.9rem 2.5rem; margin: 0 auto;
}}
/* Streamlit 상단 바 : 투명하게 두어 본문이 바로 위에서 시작 (오른쪽 ⋮ 메뉴만 남음) */
[data-testid="stHeader"] {{ background: transparent; height: 2.4rem; }}
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] {{ gap: 1rem; }}

/* 사이드바 : 흰 바탕, 선택된 메뉴는 하늘색 둥근 바탕 + 남색 굵은 글자 */
[data-testid="stSidebar"] {{ min-width: 220px !important; max-width: 220px !important; border-right: 1px solid {LINE}; background: {WHITE}; }}
[data-testid="stSidebarHeader"] {{ height: 3.2rem; padding: 1rem 1.25rem 0; }}
[data-testid="stSidebarHeader"] img {{ height: 26px !important; }}
[data-testid="stSidebarNavLink"] {{ border-radius: 12px; margin: 2px 12px; padding: 8px 12px; }}
[data-testid="stSidebarNavLink"] span {{ color: {MUTED}; }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{ background: {BLUE_SOFT}; }}
[data-testid="stSidebarNavLink"][aria-current="page"] span {{ color: {INK} !important; font-weight: 700; }}
/* PC 화면에서는 사이드바 접기 버튼 숨김 (모바일은 사이드바를 닫아야 하므로 유지) */
@media (min-width: 768px) {{
  [data-testid="stSidebarCollapseButton"] {{ display: none !important; }}
}}

/* 카드 : 흰 바탕 + 크게 둥근 모서리 + 옅은 그림자 (테두리 없음) */
[class*="st-key-card"] {{
  background: {WHITE}; border: 1px solid transparent !important; border-radius: {RADIUS} !important;
  padding: 18px 20px 22px !important; box-shadow: {SHADOW};
}}
[class*="st-key-card"] [data-testid="stHeading"] h3,
[class*="st-key-card"] h3 {{ margin: 0 0 10px; padding: 0; font-size: 1rem; }}
[class*="st-key-card"] [data-testid="stVerticalBlock"] {{ gap: .65rem; }}
[class*="st-key-card"] [data-testid="stMarkdownContainer"] table {{ width: 100%; border-collapse: collapse; margin-top: 4px; }}
[class*="st-key-card"] [data-testid="stMarkdownContainer"] th {{ background: {SURFACE}; text-align: left; }}

/* 강조 카드 : 라벤더 바탕 + 남색 글자 + 보라 강조 */
[class*="st-key-card-accent"] {{ background: {LAVENDER_SOFT} !important; border-color: {LAVENDER_SOFT} !important; }}
[class*="st-key-card-accent"] * {{ color: {INK} !important; }}
[class*="st-key-card-accent"] strong {{ color: {ACCENT_TEXT} !important; }}

/* 같은 줄 카드 높이 맞추기 : key가 eq- 로 시작하는 줄 안의 카드는 아래 끝을 맞춤 */
[class*="st-key-eq-"] [data-testid="stHorizontalBlock"] {{ align-items: stretch !important; }}
[class*="st-key-eq-"] [data-testid="stColumn"] {{ display: flex; flex-direction: column; }}
[class*="st-key-eq-"] [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {{ flex: 1 1 auto; height: 100%; }}
[class*="st-key-eq-"] [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > :last-child {{ flex: 1 1 auto; }}
[class*="st-key-eq-"] [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > :last-child [class*="st-key-card"] {{ height: 100%; }}

/* 지표 상자 : 보라 · 노랑 파스텔 카드가 번갈아 나옴 */
[data-testid="stMetric"] {{
  background: {LAVENDER_SOFT}; border: 0; border-radius: {RADIUS}; padding: 14px 16px;
}}
[data-testid="stColumn"]:nth-child(even) [data-testid="stMetric"] {{ background: {YELLOW_SOFT}; }}
[data-testid="stMetricLabel"] p {{ color: #4A4D63; font-size: .8rem; font-weight: 600; }}
[data-testid="stMetricValue"] div {{ white-space: nowrap; overflow: visible; color: {INK}; }}

/* 탭 */
[data-testid="stTabs"] [role="tablist"] {{ gap: 4px; overflow-x: auto; }}
[data-testid="stTab"] p {{ font-size: .9rem; font-weight: 600; }}
[data-testid="stTabPanel"] {{ padding-top: .8rem; }}

/* 버튼 : 보라 */
[data-testid="stBaseButton-primary"] {{ background: {VIOLET}; border-color: {VIOLET}; font-weight: 600; }}
[data-testid="stBaseButton-primary"]:hover {{ background: {VIOLET_HOVER}; border-color: {VIOLET_HOVER}; }}

/* 표 · 그래프가 넘치면 상자 안에서만 스크롤 */
[data-testid="stDataFrame"], [data-testid="stPlotlyChart"] {{ max-width: 100%; overflow-x: auto; }}
[data-testid="stExpander"] details {{ border-radius: 14px; background: {WHITE}; border-color: {LINE}; }}

/* 상단 제목 줄 : header() */
.topbar {{ display: flex; align-items: center; justify-content: space-between; gap: 12px;
  flex-wrap: wrap; margin: 0 0 4px; padding-right: 2.6rem; }}
.crumb {{ font-size: .85rem; color: {MUTED}; }}
.crumb b {{ color: {INK}; font-weight: 600; }}
.pill {{ display: inline-flex; align-items: center; gap: 6px; background: {YELLOW}; color: {INK};
  font-size: .78rem; font-weight: 700; padding: 6px 14px; border-radius: 999px; }}
.page-title {{ font-size: 1.5rem; font-weight: 700; margin: 0; color: {INK}; line-height: 1.3; }}
.page-desc {{ color: {MUTED}; font-size: .9rem; margin: 2px 0 6px; }}

/* 빈 결과 안내 : empty() */
.empty {{ border: 1.5px dashed #D3D6E4; border-radius: 16px; padding: 22px 16px; margin: 2px 0 10px;
  text-align: center; color: {MUTED}; font-size: .9rem; background: {SURFACE}; }}

/* 판정 결과 상자 : result() */
.result {{ border-radius: 16px; padding: 12px 16px; margin: 0 0 4px; }}
.result .t {{ font-size: 1.3rem; font-weight: 700; line-height: 1.35; }}
.result .s {{ margin-top: 2px; opacity: .85; font-size: .88rem; }}

/* 상품 카드 : 가격 · 태그 */
.price {{ font-size: 1.3rem; font-weight: 700; color: {INK}; margin: 0 0 4px; }}
.tag {{ display: inline-block; font-size: .76rem; font-weight: 600; padding: 2px 9px; border-radius: 999px;
  background: {BLUE_SOFT}; color: {INK}; margin: 0 4px 4px 0; }}

/* 모델이 읽은 문장 상자 : 한국어 입력 → 포르투갈어 번역문 */
.readbox {{ border: 1px solid {LINE}; border-radius: 14px; background: {SURFACE}; padding: 10px 14px; margin: 2px 0 6px; }}
.readbox .rb-row {{ display: grid; grid-template-columns: 88px 1fr; gap: 10px; padding: 5px 0; font-size: .88rem;
  line-height: 1.5; color: {INK}; }}
.readbox .rb-row + .rb-row {{ border-top: 1px dashed #DDE0EA; }}
.readbox .rb-k {{ color: {MUTED}; font-size: .8rem; font-weight: 600; line-height: 1.35; }}
.readbox .rb-pt {{ font-style: italic; color: {VIOLET}; }}
.readbox .rb-note {{ color: {MUTED}; font-size: .82rem; }}

/* 리뷰 판정 근거 : 단어별 등장 비율 막대 */
.why {{ border: 1px solid {LINE}; border-radius: 14px; padding: 10px 12px; margin: 0 0 8px; background: {WHITE}; }}
.why .w {{ font-weight: 700; color: {INK}; }}
.why .m {{ color: {MUTED}; font-weight: 500; }}
.why .row {{ display: grid; grid-template-columns: 74px 1fr 42px; align-items: center; gap: 8px;
  font-size: .8rem; color: {MUTED}; margin-top: 4px; }}
.why .bar {{ height: 8px; background: #EFF0F5; border-radius: 99px; overflow: hidden; }}
.why .bar i {{ display: block; height: 100%; border-radius: 99px; }}
.why .say {{ font-size: .84rem; margin-top: 6px; color: {INK}; }}
.why .say.none {{ margin: 0; color: {MUTED}; }}

/* 홈 판매자 유형 범례 : 색 네모 + 이름 · 인원 */
.legend-row {{ display: flex; justify-content: space-between; font-size: .88rem; padding: 2px 0; }}
.legend-row i {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 8px; }}
</style>
"""


def setup():
    """앱 시작 시 1번 호출 : 전역 CSS 주입 + Plotly 기본 테마를 'olist' 테마로 등록"""
    st.markdown(CSS, unsafe_allow_html=True)
    # Streamlit은 그래프 모듈을 처음 불러올 때 Plotly 기본 테마를 자기 것으로 바꿉니다.
    # 먼저 불러와 그 작업을 끝낸 뒤, 우리 테마로 다시 덮어씁니다.
    import streamlit.elements.plotly_chart  # noqa: F401

    template = go.layout.Template()
    template.layout = go.Layout(
        font={"family": FONT, "size": 12, "color": INK},
        colorway=[BLUE, YELLOW, LAVENDER, RED, "#A9ADBE"],
        paper_bgcolor=WHITE, plot_bgcolor=WHITE,
        margin={"l": 8, "r": 8, "t": 40, "b": 8},
        title={"font": {"size": 14, "color": INK}, "x": 0, "xanchor": "left"},
        xaxis={"gridcolor": GRID, "linecolor": LINE, "zeroline": False, "automargin": True},
        yaxis={"gridcolor": GRID, "linecolor": LINE, "zeroline": False, "automargin": True},
        legend={"orientation": "h", "x": 1, "xanchor": "right", "y": 1.02, "yanchor": "bottom", "title": None},
        hoverlabel={"font": {"family": FONT}},
        barcornerradius=BAR_RADIUS,
    )
    # 선 그래프 : 꺾은선 대신 살짝 부드러운 곡선 + 둥근 점
    template.data.scatter = [go.Scatter(line={"shape": "spline", "smoothing": 0.6, "width": 3},
                                        marker={"size": 7, "line": {"color": WHITE, "width": 2}})]
    pio.templates["olist"] = template
    pio.templates.default = "plotly_white+olist"


# ── 화면 부품 ────────────────────────────────────────────────
def header(page: str, desc: str = "", right: str = ""):
    """페이지 상단 : 경로(Olist / 페이지) + 오른쪽 배지 + 큰 제목 + 설명 한 줄"""
    pill = f'<span class="pill">{right}</span>' if right else ""
    description = f'<p class="page-desc">{desc}</p>' if desc else ""
    st.markdown(f"""<div class="topbar"><div class="crumb">Olist / <b>{page}</b></div>
      {pill}</div>
      <div class="page-title">{page}</div>
      {description}""", unsafe_allow_html=True)


def card(key: str, accent: bool = False):
    """흰색 카드 컨테이너. accent=True 이면 라벤더로 채운 강조 카드 (with 문으로 사용)"""
    return st.container(border=True, key=f"card-{'accent-' if accent else ''}{key}")


def row(key: str):
    """안에 넣은 st.columns의 카드들이 같은 높이로 아래 끝이 맞는 줄 (with 문으로 사용)"""
    return st.container(key=f"eq-{key}")


def empty(text: str):
    """아직 결과가 없을 때 보여 주는 점선 안내 상자"""
    st.markdown(f'<div class="empty">{text}</div>', unsafe_allow_html=True)


def result(title: str, sub: str, bg: str, fg: str):
    """모델 판정 결과 상자 : 큰 제목 + 보조 설명, 배경색 · 글자색 지정"""
    st.markdown(f'<div class="result" style="background:{bg}"><div class="t" style="color:{fg}">{title}</div>'
                f'<div class="s" style="color:{fg}">{sub}</div></div>', unsafe_allow_html=True)


def tags(items) -> str:
    """문자열 목록 → 둥근 태그 HTML (st.markdown에 넣어 사용)"""
    return "".join(f'<span class="tag">{x}</span>' for x in items)


def chart(fig, where=None, toolbar: bool = False):
    """Plotly 그래프 출력. 배경 · 글꼴을 그래프에 직접 지정해야 Streamlit이 회색으로 덮어쓰지 않습니다.
    where : 그릴 위치 (st.columns의 한 칸 등). 없으면 현재 위치"""
    fig.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE,
                      font={"family": FONT, "size": 12, "color": INK})
    if fig.layout.height is None:
        fig.update_layout(height=330)
    (where or st).plotly_chart(fig, width="stretch", theme=None,
                               config={"displayModeBar": toolbar, "displaylogo": False})

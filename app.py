import math
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. 페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="GIS 기반 열섬·오염 핫스팟 부지 선정 대시보드",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# 2. 실시간 기상 · 대기질 API (Open-Meteo, API 키 불필요)
# ---------------------------------------------------------
@st.cache_data(ttl=600)
def fetch_realtime_daejeon_env(lat=36.3650, lon=127.3450):
    """대전 유성구(충남대 일대) 실시간 기온·습도·풍속 및 PM2.5·NO2 조회"""
    result = {
        "status": "실시간 API 연동",
        "temp": None,
        "humidity": None,
        "wind": None,
        "pm25": None,
        "no2": None,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    try:
        w_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
            f"&timezone=Asia%2FSeoul"
        )
        w_res = requests.get(w_url, timeout=4).json()
        cur_w = w_res.get("current", {})
        result["temp"] = cur_w.get("temperature_2m")
        result["humidity"] = cur_w.get("relative_humidity_2m")
        result["wind"] = cur_w.get("wind_speed_10m")

        aq_url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality?"
            f"latitude={lat}&longitude={lon}"
            f"&current=pm2_5,nitrogen_dioxide"
            f"&timezone=Asia%2FSeoul"
        )
        aq_res = requests.get(aq_url, timeout=4).json()
        cur_aq = aq_res.get("current", {})
        result["pm25"] = cur_aq.get("pm2_5")
        result["no2"] = cur_aq.get("nitrogen_dioxide")
    except Exception:
        result["status"] = "오프라인 기준값 모드"
        result["temp"] = 29.5
        result["humidity"] = 62
        result["wind"] = 1.8
        result["pm25"] = 34.0
        result["no2"] = 28.0
    return result


# ---------------------------------------------------------
# 3. 충남대학교 일대 후보지 GIS 공간 데이터베이스
# ---------------------------------------------------------
CANDIDATE_SITES = [
    {
        "id": "S1",
        "name": "충남대 정문 버스정류장",
        "short_name": "충남대 정문 정류장",
        "type": "버스정류장",
        "lat": 36.3622,
        "lon": 127.3448,
        "impervious": 92,
        "heat_idx": 94,
        "poll_idx": 96,
        "ped_idx": 95,
        "label_offset": (16, -24),
        "reason": "왕복 6차선 대학로·오거리 인접, 버스 공회전 및 승객 대기 밀집 최상위",
    },
    {
        "id": "S2",
        "name": "공과대학 앞 버스정류장",
        "short_name": "공과대학 앞 정류장",
        "type": "버스정류장",
        "lat": 36.3653,
        "lon": 127.3458,
        "impervious": 84,
        "heat_idx": 86,
        "poll_idx": 88,
        "ped_idx": 89,
        "label_offset": (16, -14),
        "reason": "공대 건물군 복사열 집중 + 순환버스·통학차량 경사로 주행 배기가스",
    },
    {
        "id": "S3",
        "name": "충남대 서문 버스정류장",
        "short_name": "충남대 서문 정류장",
        "type": "버스정류장",
        "lat": 36.3666,
        "lon": 127.3414,
        "impervious": 78,
        "heat_idx": 79,
        "poll_idx": 83,
        "ped_idx": 81,
        "label_offset": (14, -14),
        "reason": "궁동 상권·서문 도로변 차량 정체 및 보행자 이동량 집중",
    },
    {
        "id": "S4",
        "name": "정문 인접 대형 주차장",
        "short_name": "정문 대형 주차장",
        "type": "주차장 (기존)",
        "lat": 36.3636,
        "lon": 127.3435,
        "impervious": 95,
        "heat_idx": 91,
        "poll_idx": 74,
        "ped_idx": 48,
        "label_offset": (16, -12),
        "reason": "아스팔트 축열로 열섬은 높으나 보행자 체류시간·직접 노출이 낮음",
    },
    {
        "id": "S5",
        "name": "농생대 앞 버스정류장",
        "short_name": "농생대 앞 정류장",
        "type": "버스정류장",
        "lat": 36.3638,
        "lon": 127.3521,
        "impervious": 72,
        "heat_idx": 74,
        "poll_idx": 80,
        "ped_idx": 72,
        "label_offset": (-145, -14),
        "reason": "동측 간선도로 차량 통행 존재, 주변 녹지 완충으로 열섬은 중간 수준",
    },
    {
        "id": "S6",
        "name": "중앙도서관 앞 광장",
        "short_name": "중앙도서관 광장",
        "type": "보행광장",
        "lat": 36.3700,
        "lon": 127.3458,
        "impervious": 68,
        "heat_idx": 72,
        "poll_idx": 42,
        "ped_idx": 78,
        "label_offset": (14, -14),
        "reason": "유동인구는 많으나 차량 통제 구역으로 도로변 배기가스 오염도 낮음",
    },
    {
        "id": "S7",
        "name": "캠퍼스 녹지대 (대조군)",
        "short_name": "캠퍼스 녹지대",
        "type": "녹지 대조군",
        "lat": 36.3672,
        "lon": 127.3446,
        "impervious": 18,
        "heat_idx": 28,
        "poll_idx": 25,
        "ped_idx": 35,
        "label_offset": (16, -14),
        "reason": "수목 그늘·증산작용으로 열섬 및 도로변 오염 영향이 가장 낮은 대조군",
    },
]


def build_heatmap_points(df: pd.DataFrame, layer_mode: str) -> list:
    """실제 도로변·후보지 주변으로 자연스럽게 퍼지는 GIS 열섬/오염 히트맵 포인트 생성"""
    rng = np.random.default_rng(42)
    pts = []
    col_map = {
        "복합 취약지수 (가중 합산)": "score",
        "열섬 핫스팟 레이어": "heat_idx",
        "예상 대기오염 레이어": "poll_idx",
    }
    target_col = col_map[layer_mode]

    for _, row in df.iterrows():
        val = float(row[target_col]) / 100.0
        pts.append([row["lat"], row["lon"], val])
        n_spread = 18 if val >= 0.75 else 10
        spread_sigma = 0.00055 if row["type"] == "버스정류장" else 0.00045
        for _ in range(n_spread):
            dlat = rng.normal(0, spread_sigma)
            dlon = rng.normal(0, spread_sigma)
            decay = math.exp(-((dlat**2 + dlon**2) / (2 * spread_sigma**2)))
            pts.append([row["lat"] + dlat, row["lon"] + dlon, max(0.08, val * decay * 0.92)])

    road_lats = np.linspace(36.3618, 36.3638, 12)
    road_lons = np.linspace(127.3420, 127.3522, 12)
    for rl, rn in zip(road_lats, road_lons):
        w = 0.72 if abs(rn - 127.3448) < 0.002 else 0.52
        pts.append([float(rl), float(rn), w])
    return pts


# ---------------------------------------------------------
# 4. 사이드바: 테마 · 가중치 · 지도 제어
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎨 화면 테마 (가독성 고정)")
    ui_theme = st.radio(
        "대시보드 컬러 테마",
        ["고대비 다크 모드 (네이비)", "고대비 라이트 모드 (PPT 맞춤)"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 🎛️ GIS 가중 중첩 분석 설정")
    st.caption("환경 요인 가중치를 조절하면 지도와 입지 순위가 실시간으로 재계산됩니다.")

    w_heat_raw = st.slider(
        "🔥 열섬 취약도 가중치",
        0, 100, 40, 5,
        help="지표 축열, 불투수 포장률, 녹지 부족 정도 반영",
    )
    w_poll_raw = st.slider(
        "🌫️ 예상 오염도 가중치",
        0, 100, 40, 5,
        help="도로변 차량 통행량, 버스 공회전, 정체 구간 배기가스 반영",
    )
    w_ped_raw = st.slider(
        "🚶 보행자 노출 가중치",
        0, 100, 20, 5,
        help="승하차 유동인구 밀집도 및 대기 체류시간 반영",
    )

    total_w = w_heat_raw + w_poll_raw + w_ped_raw
    if total_w == 0:
        w_heat, w_poll, w_ped = 0.4, 0.4, 0.2
    else:
        w_heat = w_heat_raw / total_w
        w_poll = w_poll_raw / total_w
        w_ped = w_ped_raw / total_w

    st.info(
        f"**적용 가중치 (정규화 100%)**\n"
        f"- 열섬 취약도: **{w_heat*100:.0f}%**\n"
        f"- 예상 오염도: **{w_poll*100:.0f}%**\n"
        f"- 보행자 노출: **{w_ped*100:.0f}%**"
    )

    st.markdown("---")
    st.markdown("### 🗺️ 지도 시각화 옵션")
    map_style = st.radio(
        "배경 지도 타일",
        ["실시간 위성 지도 (Esri)", "어두운 GIS 지도 (Dark)", "밝은 GIS 지도 (Light)", "표준 도로 지도 (OSM)"],
        index=0,
    )
    heat_layer = st.selectbox(
        "히트맵 오버레이 레이어",
        ["복합 취약지수 (가중 합산)", "열섬 핫스팟 레이어", "예상 대기오염 레이어"],
        index=0,
    )
    show_heatmap = st.checkbox("히트맵 오버레이 표시", value=True)
    show_labels = st.checkbox("지도 위 상위 입지 라벨 상시 표시", value=True)

    st.markdown("---")
    st.caption("💡 **캡쳐 팁**: 좌측 상단 `×` 버튼으로 사이드바를 접은 뒤 캡쳐하면 16:9 슬라이드에 더 넓고 선명하게 담깁니다.")

# ---------------------------------------------------------
# 5. 테마 팔레트 강제 지정 (OS 다크/라이트 충돌 원천 차단)
# ---------------------------------------------------------
is_dark = ui_theme.startswith("고대비 다크")

if is_dark:
    T = {
        "app_bg": "#0B1220",
        "sidebar_bg": "#111A2E",
        "sidebar_text": "#E2E8F0",
        "card_bg": "#152036",
        "card_border": "#2A3B5C",
        "card_red_bg": "#2B1822",
        "card_blue_bg": "#142444",
        "card_green_bg": "#122B27",
        "title": "#F8FAFC",
        "sub": "#CBD5E1",
        "muted": "#94A3B8",
        "chart_bg": "#152036",
        "chart_grid": "#2A3B5C",
        "chart_text": "#F1F5F9",
        "tbl_hdr_bg": "#1E3A6E",
        "tbl_hdr_txt": "#FFFFFF",
        "tbl_row_even": "#152036",
        "tbl_row_odd": "#1A2742",
        "tbl_row_top": "#261B2E",
        "tbl_txt": "#F1F5F9",
        "tbl_border": "#2A3B5C",
        "legend_bg": "rgba(15, 23, 42, 0.94)",
        "legend_border": "#334155",
        "legend_title": "#F8FAFC",
        "legend_txt": "#E2E8F0",
    }
else:
    T = {
        "app_bg": "#EEF2F6",
        "sidebar_bg": "#F8FAFC",
        "sidebar_text": "#0F172A",
        "card_bg": "#FFFFFF",
        "card_border": "#CBD5E1",
        "card_red_bg": "#FFF1F2",
        "card_blue_bg": "#EFF6FF",
        "card_green_bg": "#ECFDF5",
        "title": "#0B2559",
        "sub": "#1E293B",
        "muted": "#475569",
        "chart_bg": "#FFFFFF",
        "chart_grid": "#E2E8F0",
        "chart_text": "#0F172A",
        "tbl_hdr_bg": "#0B2559",
        "tbl_hdr_txt": "#FFFFFF",
        "tbl_row_even": "#FFFFFF",
        "tbl_row_odd": "#F8FAFC",
        "tbl_row_top": "#FEF2F2",
        "tbl_txt": "#0F172A",
        "tbl_border": "#CBD5E1",
        "legend_bg": "rgba(255, 255, 255, 0.96)",
        "legend_border": "#94A3B8",
        "legend_title": "#0B2559",
        "legend_txt": "#0F172A",
    }

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] {{
        font-family: 'Noto Sans KR', sans-serif !important;
    }}
    [data-testid="stAppViewContainer"] {{
        background-color: {T['app_bg']} !important;
        color: {T['title']} !important;
    }}
    /* 상단 기본 헤더가 본문 헤더 박스를 가리지 않도록 투명 처리 및 상단 여백 확보 */
    [data-testid="stHeader"] {{
        background: transparent !important;
        height: 2.6rem !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: {T['sidebar_bg']} !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {T['sidebar_text']} !important;
    }}
    [data-testid="stSidebar"] [data-testid="stSlider"] {{
        padding-top: 4px;
        padding-bottom: 6px;
    }}
    .block-container {{
        padding-top: 3.1rem !important;
        padding-bottom: 1.2rem !important;
        padding-left: 1.8rem !important;
        padding-right: 1.8rem !important;
        max-width: 1560px;
    }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    .header-box {{
        background: linear-gradient(135deg, #0B2559 0%, #123B8A 100%);
        padding: 18px 24px;
        border-radius: 14px;
        color: #FFFFFF !important;
        margin-bottom: 14px;
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        justify-content: space-between;
        align-items: center;
        border: 1px solid rgba(96, 165, 250, 0.35);
    }}
    .header-kicker {{
        font-size: 12px;
        font-weight: 700;
        color: #6EE7B7 !important;
        letter-spacing: 0.4px;
        margin-bottom: 4px;
        line-height: 1.4;
    }}
    .header-title {{
        font-size: 23px;
        font-weight: 800;
        margin: 0;
        color: #FFFFFF !important;
        line-height: 1.3;
    }}
    .header-sub {{
        font-size: 13px;
        color: #CBD5E1 !important;
        margin-top: 5px;
        line-height: 1.45;
    }}
    .live-badge {{
        background: rgba(15, 23, 42, 0.48);
        border: 1px solid rgba(110, 231, 183, 0.55);
        padding: 10px 16px;
        border-radius: 10px;
        text-align: right;
        min-width: 265px;
    }}
    .live-dot {{
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #34D399;
        border-radius: 50%;
        margin-right: 6px;
    }}
    .kpi-card {{
        background: {T['card_blue_bg']};
        border: 1px solid {T['card_border']};
        border-left: 5px solid #3B82F6;
        border-radius: 12px;
        padding: 13px 16px;
        margin-bottom: 10px;
        min-height: 100px;
    }}
    .kpi-card.green {{
        border-left-color: #10B981;
        background: {T['card_green_bg']};
    }}
    .kpi-card.red {{
        border-left-color: #EF4444;
        background: {T['card_red_bg']};
    }}
    .kpi-label {{
        font-size: 12px;
        font-weight: 700;
        color: {T['sub']} !important;
        margin-bottom: 4px;
        line-height: 1.35;
    }}
    .kpi-value {{
        font-size: 19px;
        font-weight: 800;
        color: {T['title']} !important;
        line-height: 1.3;
    }}
    .kpi-desc {{
        font-size: 11.8px;
        color: {T['muted']} !important;
        margin-top: 4px;
        line-height: 1.4;
    }}
    .section-title {{
        font-size: 15px;
        font-weight: 800;
        color: {T['title']} !important;
        margin: 4px 0 8px 0;
        line-height: 1.4;
    }}
    .table-wrap {{
        width: 100%;
        overflow-x: auto;
        border-radius: 10px;
        border: 1px solid {T['tbl_border']};
    }}
    table.rank-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
    }}
    table.rank-table th {{
        background: {T['tbl_hdr_bg']};
        color: {T['tbl_hdr_txt']} !important;
        font-weight: 700;
        padding: 8px 6px;
        text-align: center;
        white-space: nowrap;
        border-bottom: 1px solid {T['tbl_border']};
    }}
    table.rank-table td {{
        color: {T['tbl_txt']} !important;
        padding: 7px 6px;
        text-align: center;
        white-space: nowrap;
        border-bottom: 1px solid {T['tbl_border']};
        font-weight: 500;
    }}
    table.rank-table td.left-cell {{
        text-align: left;
        padding-left: 10px;
        font-weight: 700;
    }}
    table.rank-table tr:nth-child(even) {{ background: {T['tbl_row_even']}; }}
    table.rank-table tr:nth-child(odd) {{ background: {T['tbl_row_odd']}; }}
    table.rank-table tr.top-row {{ background: {T['tbl_row_top']} !important; }}
    .grade-badge {{
        display: inline-block;
        padding: 2px 7px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 800;
        color: #FFFFFF !important;
        white-space: nowrap;
    }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 6. 실시간 데이터 조회 및 가중치 스코어 계산
# ---------------------------------------------------------
live_env = fetch_realtime_daejeon_env()

df = pd.DataFrame(CANDIDATE_SITES)
df["열섬 기여"] = (df["heat_idx"] * w_heat).round(1)
df["오염 기여"] = (df["poll_idx"] * w_poll).round(1)
df["노출 기여"] = (df["ped_idx"] * w_ped).round(1)
df["score"] = (df["heat_idx"] * w_heat + df["poll_idx"] * w_poll + df["ped_idx"] * w_ped).round(1)
df = df.sort_values("score", ascending=False).reset_index(drop=True)
df["rank"] = df.index + 1

def assign_grade(s):
    if s >= 85:
        return "S (최우선)"
    elif s >= 78:
        return "A (우수)"
    elif s >= 65:
        return "B (보통)"
    return "C (대조군)"

df["grade"] = df["score"].apply(assign_grade)

top1 = df.iloc[0]
top2 = df.iloc[1]
parking_row = df[df["id"] == "S4"].iloc[0]

# ---------------------------------------------------------
# 7. 상단 헤더 & KPI
# ---------------------------------------------------------
temp_txt = f"{live_env['temp']:.1f}℃" if live_env["temp"] is not None else "-"
hum_txt = f"{live_env['humidity']:.0f}%" if live_env["humidity"] is not None else "-"
pm_txt = f"{live_env['pm25']:.1f} ㎍/㎥" if live_env["pm25"] is not None else "-"

st.markdown(f"""
<div class="header-box">
    <div>
        <div class="header-kicker">GIS WEIGHTED OVERLAY · 충남대학교 대덕캠퍼스 일대</div>
        <div class="header-title">열섬 핫스팟 · 예상 오염도 가중치 기반 최적 설치 부지 선정</div>
        <div class="header-sub">실시간 기상·대기질 현황과 공간 특성(불투수율·도로변 정체·보행자 노출)을 중첩해 목질계 정화·냉각 모듈 우선 부지 도출</div>
    </div>
    <div class="live-badge">
        <div style="font-size:12px; font-weight:700; color:#6EE7B7;">
            <span class="live-dot"></span>{live_env['status']} ({live_env['updated_at']})
        </div>
        <div style="font-size:13px; color:#FFFFFF; margin-top:4px; font-weight:600;">
            대전 유성구 기준 · 기온 {temp_txt} · 습도 {hum_txt} · PM2.5 {pm_txt}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

k1, k2, k3 = st.columns(3)
with k1:
    st.markdown(f"""
    <div class="kpi-card red">
        <div class="kpi-label">🥇 1순위 최우선 실증 부지 (복합 지수 {top1['score']}점 · {top1['grade']})</div>
        <div class="kpi-value">{top1['name']}</div>
        <div class="kpi-desc">열섬 {top1['heat_idx']} · 예상 오염 {top1['poll_idx']} · 보행자 노출 {top1['ped_idx']}</div>
    </div>
    """, unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">🥈 2순위 우선 실증 부지 (복합 지수 {top2['score']}점 · {top2['grade']})</div>
        <div class="kpi-value">{top2['name']}</div>
        <div class="kpi-desc">열섬 {top2['heat_idx']} · 예상 오염 {top2['poll_idx']} · 보행자 노출 {top2['ped_idx']}</div>
    </div>
    """, unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="kpi-card green">
        <div class="kpi-label">✅ 부지 확정 근거 (주차장 대비 버스정류장 우위)</div>
        <div class="kpi-value">버스정류장 핫스팟 최종 선정</div>
        <div class="kpi-desc">주차장({parking_row['score']}점)은 열섬은 높으나 오염·보행자 체류 노출에서 정류장이 우위</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 8. 메인 2단 레이아웃: 좌측 실시간 GIS 지도 / 우측 차트 + 고대비 평가표
# ---------------------------------------------------------
col_map, col_chart = st.columns([1.22, 1.08], gap="medium")

with col_map:
    st.markdown(f'<div class="section-title">🗺️ 실시간 GIS 핫스팟 분포 지도 ({heat_layer})</div>', unsafe_allow_html=True)

    if map_style == "실시간 위성 지도 (Esri)":
        m = folium.Map(location=[36.3652, 127.3462], zoom_start=16, tiles=None)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="Esri Satellite",
        ).add_to(m)
    elif map_style == "어두운 GIS 지도 (Dark)":
        m = folium.Map(location=[36.3652, 127.3462], zoom_start=16, tiles="CartoDB dark_matter")
    elif map_style == "표준 도로 지도 (OSM)":
        m = folium.Map(location=[36.3652, 127.3462], zoom_start=16, tiles="OpenStreetMap")
    else:
        m = folium.Map(location=[36.3652, 127.3462], zoom_start=16, tiles="CartoDB positron")

    if show_heatmap:
        heat_pts = build_heatmap_points(df, heat_layer)
        HeatMap(
            heat_pts,
            radius=30,
            blur=22,
            min_opacity=0.38,
            gradient={0.25: "#3B82F6", 0.50: "#10B981", 0.72: "#F59E0B", 0.90: "#EF4444", 1.0: "#991B1B"},
        ).add_to(m)

    for _, row in df.iterrows():
        if row["rank"] <= 2:
            color, fill_color, radius = "#FFFFFF", "#EF4444", 12
        elif row["type"] == "버스정류장":
            color, fill_color, radius = "#FFFFFF", "#3B82F6", 10
        elif row["id"] == "S4":
            color, fill_color, radius = "#FFFFFF", "#F59E0B", 9
        else:
            color, fill_color, radius = "#FFFFFF", "#10B981", 8

        popup_html = f"""
        <div style="font-family:sans-serif; width:235px; font-size:12px; line-height:1.45; color:#0F172A; background:#FFFFFF;">
            <div style="font-weight:800; font-size:13px; color:#0B2559; margin-bottom:4px;">
                [{row['rank']}위] {row['name']}
            </div>
            <div style="color:#475569; margin-bottom:6px;">유형: <b>{row['type']}</b> · 등급: <b>{row['grade']}</b></div>
            <table style="width:100%; border-collapse:collapse; color:#0F172A;">
                <tr><td style="color:#334155;">복합 입지점수</td><td style="text-align:right; font-weight:800; color:#DC2626;">{row['score']}점</td></tr>
                <tr><td style="color:#334155;">열섬 취약지수</td><td style="text-align:right; color:#0F172A;">{row['heat_idx']}점</td></tr>
                <tr><td style="color:#334155;">예상 오염지수</td><td style="text-align:right; color:#0F172A;">{row['poll_idx']}점</td></tr>
                <tr><td style="color:#334155;">보행자 노출지수</td><td style="text-align:right; color:#0F172A;">{row['ped_idx']}점</td></tr>
            </table>
            <div style="margin-top:6px; padding-top:5px; border-top:1px solid #E2E8F0; color:#334155;">
                {row['reason']}
            </div>
        </div>
        """

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=color,
            weight=2.5,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.92,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{row['rank']}위 {row['name']} ({row['score']}점)",
        ).add_to(m)

        if show_labels and (row["rank"] <= 2 or row["id"] in ["S4", "S7"]):
            badge_bg = "#DC2626" if row["rank"] <= 2 else ("#D97706" if row["id"] == "S4" else "#1E7A56")
            dx, dy = row["label_offset"]
            label_html = f"""
            <div style="
                position: relative; left: {dx}px; top: {dy}px; display: inline-block;
                background:{badge_bg}; color:#FFFFFF; font-weight:800; font-size:11px;
                padding:3px 8px; border-radius:6px; white-space:nowrap;
                box-shadow:0 2px 8px rgba(0,0,0,0.45); border:1.5px solid #FFFFFF;
            ">
                {row['rank']}위 {row['short_name']} ({row['score']})
            </div>
            """
            folium.Marker(
                location=[row["lat"], row["lon"]],
                icon=folium.DivIcon(icon_size=(0, 0), icon_anchor=(0, 0), html=label_html),
            ).add_to(m)

    legend_html = f"""
    <div style="
        position: fixed; bottom: 18px; left: 18px; z-index: 9999;
        background: {T['legend_bg']}; color: {T['legend_txt']};
        padding: 9px 12px; border-radius: 10px;
        border: 1.5px solid {T['legend_border']}; font-family: sans-serif; font-size: 11px;
        line-height: 1.65; box-shadow: 0 4px 12px rgba(0,0,0,0.28);
    ">
        <div style="font-weight:800; color:{T['legend_title']}; margin-bottom:3px;">GIS 부지 선정 범례</div>
        <div style="color:{T['legend_txt']};"><span style="color:#EF4444; font-weight:900;">●</span> 1~2위 최우선 선정 (버스정류장)</div>
        <div style="color:{T['legend_txt']};"><span style="color:#3B82F6; font-weight:900;">●</span> 차순위 버스정류장 후보</div>
        <div style="color:{T['legend_txt']};"><span style="color:#F59E0B; font-weight:900;">●</span> 기존 검토 주차장 (열섬 편중)</div>
        <div style="color:{T['legend_txt']};"><span style="color:#10B981; font-weight:900;">●</span> 광장 · 캠퍼스 녹지 대조군</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, width=None, height=575, returned_objects=[])

with col_chart:
    st.markdown('<div class="section-title">📊 후보지별 가중 복합 취약지수 비교 (MCDA)</div>', unsafe_allow_html=True)

    df_plot = df.iloc[::-1]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_plot["short_name"],
        x=df_plot["열섬 기여"],
        name=f"열섬 취약도 ({w_heat*100:.0f}%)",
        orientation="h",
        marker_color="#EF4444",
    ))
    fig.add_trace(go.Bar(
        y=df_plot["short_name"],
        x=df_plot["오염 기여"],
        name=f"예상 오염도 ({w_poll*100:.0f}%)",
        orientation="h",
        marker_color="#3B82F6",
    ))
    fig.add_trace(go.Bar(
        y=df_plot["short_name"],
        x=df_plot["노출 기여"],
        name=f"보행자 노출 ({w_ped*100:.0f}%)",
        orientation="h",
        marker_color="#10B981",
        text=[f" <b>{s:.1f}점</b>" for s in df_plot["score"]],
        textposition="outside",
        cliponaxis=False,
        textfont=dict(size=12, color=T["chart_text"]),
    ))

    fig.update_layout(
        barmode="stack",
        height=315,
        # 한글 Y축 라벨과 우측 점수 라벨이 절대 잘리지 않도록 좌/우/상단 여백 넉넉히 확보 + automargin 적용
        margin=dict(l=140, r=60, t=38, b=42),
        font=dict(family="Noto Sans KR, sans-serif", color=T["chart_text"], size=12),
        xaxis=dict(
            range=[0, 114],
            title=dict(text="가중 복합 점수 (100점 만점)", font=dict(size=12, color=T["chart_text"])),
            tickfont=dict(size=11.5, color=T["chart_text"]),
            gridcolor=T["chart_grid"],
            zerolinecolor=T["chart_grid"],
        ),
        yaxis=dict(
            automargin=True,
            tickfont=dict(size=12, color=T["chart_text"]),
            ticksuffix="  ",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="left",
            x=0,
            font=dict(size=11.5, color=T["chart_text"]),
        ),
        plot_bgcolor=T["chart_bg"],
        paper_bgcolor=T["chart_bg"],
    )
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

    st.markdown('<div class="section-title">📋 후보지 우선순위 평가표</div>', unsafe_allow_html=True)

    badge_colors = {
        "S (최우선)": "#DC2626",
        "A (우수)": "#2563EB",
        "B (보통)": "#D97706",
        "C (대조군)": "#059669",
    }
    rows_html = []
    for _, r in df.iterrows():
        tr_cls = ' class="top-row"' if r["rank"] <= 2 else ""
        b_col = badge_colors.get(r["grade"], "#475569")
        rows_html.append(
            f"<tr{tr_cls}>"
            f"<td><b>{r['rank']}</b></td>"
            f"<td class='left-cell'>{r['name']}</td>"
            f"<td>{r['type']}</td>"
            f"<td>{r['heat_idx']}</td>"
            f"<td>{r['poll_idx']}</td>"
            f"<td>{r['ped_idx']}</td>"
            f"<td><b>{r['score']:.1f}</b></td>"
            f"<td><span class='grade-badge' style='background:{b_col};'>{r['grade']}</span></td>"
            f"</tr>"
        )

    table_html = f"""
    <div class="table-wrap">
        <table class="rank-table">
            <thead>
                <tr>
                    <th>순위</th>
                    <th>후보지명</th>
                    <th>유형</th>
                    <th>열섬</th>
                    <th>오염</th>
                    <th>노출</th>
                    <th>총점</th>
                    <th>등급</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

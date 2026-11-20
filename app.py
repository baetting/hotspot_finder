import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
import plotly.express as px

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

# ---------------------------------------------------------
# 1. 페이지 설정 및 캡처 최적화 여백 조정
# ---------------------------------------------------------
st.set_page_config(
    page_title="유성구·서구 낮 평균온도 기반 GIS 핫스팟 선정",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 1rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.55rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌡️ 대전 유성구·서구 낮 평균온도 기반 GIS 핫스팟 선정")
st.caption("복합 환경 변수 없이 낮 평균온도(°C) 단일 기준만으로 타깃 자치구 내 열환경 우선 개선 대상지를 도출합니다.")

# ---------------------------------------------------------
# 2. 유성구·서구 전역 샘플 데이터 생성
# ---------------------------------------------------------
def load_sample_data() -> pd.DataFrame:
    data = [
        # 유성구 주요 거점
        {"자치구": "유성구", "지점명": "유성온천역 환승정류장", "lat": 36.3538, "lon": 127.3415, "day_avg_temp": 40.3},
        {"자치구": "유성구", "지점명": "충남대 정문 오거리 정류장", "lat": 36.3622, "lon": 127.3448, "day_avg_temp": 39.4},
        {"자치구": "유성구", "지점명": "봉명동 우산거리 정류장", "lat": 36.3575, "lon": 127.3489, "day_avg_temp": 39.1},
        {"자치구": "유성구", "지점명": "노은역 광장 정류장", "lat": 36.3745, "lon": 127.3182, "day_avg_temp": 38.2},
        {"자치구": "유성구", "지점명": "반석역 환승정류장", "lat": 36.3922, "lon": 127.3145, "day_avg_temp": 37.8},
        {"자치구": "유성구", "지점명": "관평동 테크노밸리 정류장", "lat": 36.4255, "lon": 127.3885, "day_avg_temp": 38.9},
        {"자치구": "유성구", "지점명": "도룡동 스마트시티 정류장", "lat": 36.3768, "lon": 127.3821, "day_avg_temp": 36.9},
        {"자치구": "유성구", "지점명": "신성동 연구단지 네거리", "lat": 36.3861, "lon": 127.3502, "day_avg_temp": 37.1},
        {"자치구": "유성구", "지점명": "학하동 복합터미널 인근", "lat": 36.3421, "lon": 127.3055, "day_avg_temp": 36.4},
        # 서구 주요 거점
        {"자치구": "서구", "지점명": "둔산동 타임월드 정류장", "lat": 36.3519, "lon": 127.3782, "day_avg_temp": 40.6},
        {"자치구": "서구", "지점명": "정부청사역 네거리 정류장", "lat": 36.3612, "lon": 127.3815, "day_avg_temp": 39.7},
        {"자치구": "서구", "지점명": "시청역 둔산대로 정류장", "lat": 36.3511, "lon": 127.3885, "day_avg_temp": 39.5},
        {"자치구": "서구", "지점명": "탄방동 로데오타운 정류장", "lat": 36.3458, "lon": 127.3852, "day_avg_temp": 38.8},
        {"자치구": "서구", "지점명": "갈마동 큰마을네거리 정류장", "lat": 36.3489, "lon": 127.3712, "day_avg_temp": 38.5},
        {"자치구": "서구", "지점명": "월평동 선사유적지 정류장", "lat": 36.3595, "lon": 127.3618, "day_avg_temp": 37.4},
        {"자치구": "서구", "지점명": "도마동 도마시장 정류장", "lat": 36.3168, "lon": 127.3765, "day_avg_temp": 39.2},
        {"자치구": "서구", "지점명": "가수원동 은아아파트 정류장", "lat": 36.3045, "lon": 127.3512, "day_avg_temp": 37.6},
        {"자치구": "서구", "지점명": "관저동 마치광장 정류장", "lat": 36.3012, "lon": 127.3365, "day_avg_temp": 38.0},
    ]
    return pd.DataFrame(data)

# ---------------------------------------------------------
# 3. 데이터 로더 및 자치구 자동 판별 함수
# ---------------------------------------------------------
def parse_uploaded_file(uploaded_file) -> pd.DataFrame:
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if filename.endswith((".geojson", ".json", ".zip")):
        if not HAS_GEOPANDAS:
            st.error("GeoJSON 또는 Shapefile을 읽으려면 `pip install geopandas` 설치가 필요합니다.")
            st.stop()
        gdf = gpd.read_file(uploaded_file)
        if gdf.crs is not None:
            gdf_proj = gdf.to_crs(epsg=5179)
            centroids = gdf_proj.geometry.centroid.to_crs(epsg=4326)
        else:
            centroids = gdf.geometry.centroid
        gdf["lon"] = centroids.x
        gdf["lat"] = centroids.y
        return pd.DataFrame(gdf.drop(columns="geometry"))

    st.error("지원하지 않는 파일 형식입니다. (.csv, .geojson, .zip 지원)")
    st.stop()

def infer_daejeon_district(lat: float, lon: float) -> str:
    """갑천 수계 및 대전 남부 행정 경계를 기준으로 유성구/서구 근사 판별"""
    if lat < 36.322 and lon >= 127.328:
        return "서구"
    if lat > 36.373 and lon < 127.405:
        return "유성구"
    if lon < 127.358:
        return "유성구"
    return "서구"

# ---------------------------------------------------------
# 4. 사이드바 컨트롤
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 타깃 권역 및 분석 설정")
    target_region = st.radio(
        "📍 타깃 자치구 선택",
        ["전체 (유성구 + 서구)", "유성구", "서구"],
        index=0
    )

    data_mode = st.radio("데이터 소스 선택", ["기본 샘플 데이터", "GIS / CSV 파일 업로드"])

    if data_mode == "GIS / CSV 파일 업로드":
        uploaded_file = st.file_uploader(
            "파일 업로드 (.csv, .geojson, .zip)",
            type=["csv", "geojson", "json", "zip"]
        )
        if uploaded_file is not None:
            raw_df = parse_uploaded_file(uploaded_file)
        else:
            st.info("파일을 업로드하기 전까지 유성구·서구 샘플 데이터를 표시합니다.")
            raw_df = load_sample_data()
    else:
        raw_df = load_sample_data()

    cols = raw_df.columns.tolist()

    default_name = cols.index("지점명") if "지점명" in cols else 0
    default_lat = cols.index("lat") if "lat" in cols else 0
    default_lon = cols.index("lon") if "lon" in cols else 0
    default_temp = cols.index("day_avg_temp") if "day_avg_temp" in cols else len(cols) - 1

    st.subheader("📌 컬럼 매핑")
    name_col = st.selectbox("지점명 컬럼", cols, index=default_name)
    lat_col = st.selectbox("위도(Latitude, ~36) 컬럼", cols, index=default_lat)
    lon_col = st.selectbox("경도(Longitude, ~127) 컬럼", cols, index=default_lon)
    temp_col = st.selectbox("낮 평균온도(°C) 컬럼", cols, index=default_temp)

    district_options = ["좌표 자동 판별 (유성구/서구)"] + cols
    default_dist_idx = district_options.index("자치구") if "자치구" in district_options else 0
    dist_col = st.selectbox("자치구 컬럼 (선택)", district_options, index=default_dist_idx)

    st.subheader("🎯 핫스팟 선정 기준")
    filter_mode = st.radio("추출 방식", ["상위 N개 선정", "기준 온도(°C) 이상"])

# ---------------------------------------------------------
# 5. 데이터 정제, 좌표 검증 및 타깃 권역 필터링
# ---------------------------------------------------------
df = raw_df[[name_col, lat_col, lon_col, temp_col]].copy()
df.columns = ["지점명", "lat", "lon", "day_avg_temp"]

if dist_col == "좌표 자동 판별 (유성구/서구)":
    df["자치구"] = None
else:
    df["자치구"] = raw_df[dist_col].astype(str).str.strip()

df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
df["day_avg_temp"] = pd.to_numeric(df["day_avg_temp"], errors="coerce")
df = df.dropna(subset=["지점명", "lat", "lon", "day_avg_temp"]).copy()

if df.empty:
    st.error("유효한 좌표 및 온도 데이터가 없습니다. 사이드바의 컬럼 매핑을 확인해 주세요.")
    st.stop()

if (df["lat"].abs() > 90).any() and (df["lon"].abs() <= 90).all():
    st.warning("⚠️ 위도(lat)와 경도(lon)가 반대로 입력되어 자동으로 교정했습니다.")
    df["lat"], df["lon"] = df["lon"].copy(), df["lat"].copy()

if (df["lat"].abs() > 90).any() or (df["lon"].abs() > 180).any():
    st.error("⚠️ 위경도(WGS84, EPSG:4326)가 아닌 평면직각좌표(TM 등)가 감지되었습니다. EPSG:4326으로 변환 후 업로드해 주세요.")
    st.stop()

if dist_col == "좌표 자동 판별 (유성구/서구)":
    df["자치구"] = [infer_daejeon_district(la, lo) for la, lo in zip(df["lat"], df["lon"])]

# 타깃 자치구 필터링 적용
if target_region in ["유성구", "서구"]:
    df = df[df["자치구"].str.contains(target_region, na=False)].copy()
    if df.empty:
        st.warning(f"선택한 '{target_region}' 권역에 해당하는 데이터가 없습니다.")
        st.stop()

df = df.sort_values(by="day_avg_temp", ascending=False).reset_index(drop=True)
df["순위"] = df.index + 1
mean_temp = df["day_avg_temp"].mean()
df["평균대비_편차(°C)"] = (df["day_avg_temp"] - mean_temp).round(2)

with st.sidebar:
    if filter_mode == "상위 N개 선정":
        top_n = st.slider("선정할 핫스팟 개수 (Top-N)", 1, len(df), min(3, len(df)))
        df["핫스팟_선정"] = df["순위"] <= top_n
    else:
        t_min_val, t_max_val = float(df["day_avg_temp"].min()), float(df["day_avg_temp"].max())
        if t_min_val == t_max_val:
            df["핫스팟_선정"] = True
        else:
            threshold = st.slider(
                "임계 낮 평균온도 (°C)",
                t_min_val,
                t_max_val,
                float(np.percentile(df["day_avg_temp"], 75)),
                step=0.1
            )
            df["핫스팟_선정"] = df["day_avg_temp"] >= threshold

    max_chart_bars = st.slider(
        "차트 표시 지점 수 (상위 N개)",
        min_value=min(5, len(df)),
        max_value=len(df),
        value=min(15, len(df))
    )

    map_theme = st.selectbox(
        "지도 배경 스타일",
        ["OpenStreetMap (도로·건물 명확)", "위성지도 (아스팔트·현장 확인용)", "CartoDB positron (심플 밝은 테마)"]
    )
    show_map_labels = st.checkbox("지도에 선정 핫스팟 이름·온도 표시", value=True)

hotspots = df[df["핫스팟_선정"]]

# ---------------------------------------------------------
# 6. 핵심 요약 지표 (KPI)
# ---------------------------------------------------------
top_site = df.iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"[{target_region}] 1순위 핫스팟", f"[{top_site['자치구']}] {top_site['지점명']}")
c2.metric("최고 낮 평균온도", f"{top_site['day_avg_temp']:.1f} °C", f"+{top_site['평균대비_편차(°C)']:.1f} °C (권역 평균 대비)")
c3.metric(f"{target_region} 낮 평균온도", f"{mean_temp:.1f} °C")
c4.metric("선정된 핫스팟 수", f"{len(hotspots)} 개 / 총 {len(df)} 개")

st.divider()

# ---------------------------------------------------------
# 7. 대형 GIS 지도 및 순위 차트 시각화 (캡처 메인 영역)
# ---------------------------------------------------------
VIS_HEIGHT = 650
col_map, col_chart = st.columns([1.35, 1])

with col_map:
    st.subheader(f"🗺️ {target_region} 낮 평균온도 분포 및 핫스팟 지도")

    center_lat, center_lon = df["lat"].mean(), df["lon"].mean()
    default_zoom = 12 if target_region == "전체 (유성구 + 서구)" else 13

    if "위성지도" in map_theme:
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=default_zoom,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery"
        )
    elif "positron" in map_theme:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=default_zoom, tiles="CartoDB positron")
    else:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=default_zoom, tiles="OpenStreetMap")

    # 광역 뷰포트에 맞춘 정규화 히트맵 레이어
    t_min, t_max = df["day_avg_temp"].min(), df["day_avg_temp"].max()
    df["heat_weight"] = (df["day_avg_temp"] - t_min) / (t_max - t_min + 1e-6) + 0.25
    heat_data = df[["lat", "lon", "heat_weight"]].values.tolist()
    heat_radius = 28 if target_region == "전체 (유성구 + 서구)" else 34
    HeatMap(heat_data, radius=heat_radius, blur=20, min_opacity=0.35, max_zoom=15).add_to(m)

    # 지점별 마커 및 캡처용 텍스트 라벨
    for _, row in df.iterrows():
        is_selected = row["핫스팟_선정"]
        color = "#FF1E1E" if row["순위"] == 1 else ("#FF8C00" if is_selected else "#2563EB")
        radius = 12 if row["순위"] == 1 else (9 if is_selected else 6)

        popup_html = f"""
        <div style="font-family: sans-serif; min-width: 170px;">
            <b>[{row['순위']}위 · {row['자치구']}] {row['지점명']}</b><br>
            낮 평균온도: <b>{row['day_avg_temp']:.1f} °C</b><br>
            권역 평균 대비: {row['평균대비_편차(°C)']:+.1f} °C<br>
            상태: {'🔥 선정 핫스팟' if is_selected else '일반 지점'}
        </div>
        """
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color="#FFFFFF",
            weight=1.8,
            fill=True,
            fill_color=color,
            fill_opacity=0.92,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{row['순위']}위 [{row['자치구']}] {row['지점명']} ({row['day_avg_temp']:.1f}°C)"
        ).add_to(m)

        if show_map_labels and is_selected:
            border_color = "#FF1E1E" if row["순위"] == 1 else "#FF8C00"
            folium.Marker(
                location=[row["lat"], row["lon"]],
                icon=folium.DivIcon(
                    html=f"""
                    <div style="
                        font-family: 'Malgun Gothic', sans-serif;
                        font-size: 11.5px;
                        font-weight: 700;
                        color: #111827;
                        background: rgba(255, 255, 255, 0.93);
                        border: 2px solid {border_color};
                        border-radius: 6px;
                        padding: 2px 7px;
                        white-space: nowrap;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.25);
                        transform: translate(11px, -11px);
                    ">
                        {row['순위']}위 [{row['자치구']}] {row['지점명']} ({row['day_avg_temp']:.1f}°C)
                    </div>
                    """
                )
            ).add_to(m)

    # 좌측 하단 범례 오버레이
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 22px;
        left: 22px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.93);
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid #CBD5E1;
        font-family: 'Malgun Gothic', sans-serif;
        font-size: 12px;
        color: #1E293B;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    ">
        <div style="font-weight: 700; margin-bottom: 6px;">대상 권역: {target_region}</div>
        <div style="margin-bottom: 4px;"><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#FF1E1E;margin-right:6px;"></span>1순위 최우선 핫스팟</div>
        <div style="margin-bottom: 4px;"><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#FF8C00;margin-right:6px;"></span>선정 핫스팟</div>
        <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#2563EB;margin-right:6px;"></span>일반 지점</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    sw = [df["lat"].min(), df["lon"].min()]
    ne = [df["lat"].max(), df["lon"].max()]
    if sw != ne:
        m.fit_bounds([sw, ne], padding=(45, 45))

    components.html(m._repr_html_(), height=VIS_HEIGHT)

with col_chart:
    st.subheader(f"📊 {target_region} 낮 평균온도 상위 순위")
    chart_df = df.head(max_chart_bars).copy()
    chart_df["구분"] = chart_df["핫스팟_선정"].map({True: "선정 핫스팟", False: "일반 지점"})
    chart_df["표시라벨"] = "[" + chart_df["자치구"] + "] " + chart_df["지점명"]

    fig = px.bar(
        chart_df.sort_values("day_avg_temp", ascending=True),
        x="day_avg_temp",
        y="표시라벨",
        orientation="h",
        color="구분",
        color_discrete_map={"선정 핫스팟": "#EF4444", "일반 지점": "#94A3B8"},
        text="day_avg_temp",
        labels={"day_avg_temp": "낮 평균온도 (°C)", "표시라벨": ""}
    )
    fig.add_vline(
        x=mean_temp,
        line_dash="dash",
        line_color="#2563EB",
        annotation_text=f"권역 평균 ({mean_temp:.1f}°C)",
        annotation_position="top right"
    )
    fig.update_traces(texttemplate="%{text:.1f}°C", textposition="outside")
    fig.update_layout(
        height=VIS_HEIGHT,
        margin=dict(l=10, r=45, t=35, b=20),
        xaxis_range=[max(0, float(chart_df["day_avg_temp"].min()) - 2.5), t_max + 2.2],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=12.5)
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 8. 결과 CSV 내보내기 (사이드바 하단 배치)
# ---------------------------------------------------------
with st.sidebar:
    st.divider()
    export_cols = ["순위", "자치구", "지점명", "day_avg_temp", "평균대비_편차(°C)", "lat", "lon", "핫스팟_선정"]
    csv_bytes = df[export_cols].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 핫스팟 선정 결과 CSV 다운로드",
        data=csv_bytes,
        file_name=f"hotspot_{target_region}_daytime_temp.csv",
        mime="text/csv",
        use_container_width=True
    )

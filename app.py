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
# 1. 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="낮 평균온도 기반 GIS 핫스팟 선정",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 낮 평균온도 기반 GIS 핫스팟 선정 대시보드")
st.caption("복합 환경 변수 없이 낮 평균온도(°C) 단일 기준만으로 열환경 우선 개선 대상지를 도출합니다.")

# ---------------------------------------------------------
# 2. 샘플 데이터 생성
# ---------------------------------------------------------
def load_sample_data() -> pd.DataFrame:
    data = [
        {"지점명": "정문 오거리 버스정류장", "lat": 36.3622, "lon": 127.3448, "day_avg_temp": 39.4},
        {"지점명": "공과대학 앞 정류장", "lat": 36.3665, "lon": 127.3462, "day_avg_temp": 38.7},
        {"지점명": "중앙도서관 인근 승강장", "lat": 36.3701, "lon": 127.3459, "day_avg_temp": 36.2},
        {"지점명": "학생회관 앞 횡단보도", "lat": 36.3680, "lon": 127.3435, "day_avg_temp": 37.9},
        {"지점명": "서문 아스팔트 구간", "lat": 36.3654, "lon": 127.3408, "day_avg_temp": 38.1},
        {"지점명": "농업생명과학대 정류장", "lat": 36.3692, "lon": 127.3505, "day_avg_temp": 35.4},
        {"지점명": "기숙사 삼거리 정류장", "lat": 36.3735, "lon": 127.3471, "day_avg_temp": 36.8},
        {"지점명": "유성온천역 방면 환승정류장", "lat": 36.3589, "lon": 127.3439, "day_avg_temp": 40.2},
    ]
    return pd.DataFrame(data)

# ---------------------------------------------------------
# 3. 데이터 로더 (CSV / GeoJSON / Shapefile zip)
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

# ---------------------------------------------------------
# 4. 사이드바 컨트롤
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 분석 설정")
    data_mode = st.radio("데이터 소스 선택", ["기본 샘플 데이터", "GIS / CSV 파일 업로드"])

    if data_mode == "GIS / CSV 파일 업로드":
        uploaded_file = st.file_uploader(
            "파일 업로드 (.csv, .geojson, .zip)",
            type=["csv", "geojson", "json", "zip"]
        )
        if uploaded_file is not None:
            raw_df = parse_uploaded_file(uploaded_file)
        else:
            st.info("파일을 업로드하기 전까지 샘플 데이터를 표시합니다.")
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

    st.subheader("🎯 핫스팟 선정 기준")
    filter_mode = st.radio("추출 방식", ["상위 N개 선정", "기준 온도(°C) 이상"])

# ---------------------------------------------------------
# 5. 데이터 정제 및 좌표 검증
# ---------------------------------------------------------
df = raw_df[[name_col, lat_col, lon_col, temp_col]].copy()
df.columns = ["지점명", "lat", "lon", "day_avg_temp"]
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
df["day_avg_temp"] = pd.to_numeric(df["day_avg_temp"], errors="coerce")
df = df.dropna().sort_values(by="day_avg_temp", ascending=False).reset_index(drop=True)

if df.empty:
    st.error("유효한 좌표 및 온도 데이터가 없습니다. 사이드바의 컬럼 매핑을 확인해 주세요.")
    st.stop()

# 위도·경도 반전 자동 보정 (한국 기준 위도 ~36, 경도 ~127)
if (df["lat"].abs() > 90).any() and (df["lon"].abs() <= 90).all():
    st.warning("⚠️ 위도(lat)와 경도(lon)가 반대로 입력되어 자동으로 교정했습니다.")
    df["lat"], df["lon"] = df["lon"].copy(), df["lat"].copy()

# 투영좌표계(TM 등) 감지
if (df["lat"].abs() > 90).any() or (df["lon"].abs() > 180).any():
    st.error("⚠️ 위경도(WGS84, EPSG:4326)가 아닌 평면직각좌표(TM 등)가 감지되었습니다. QGIS에서 EPSG:4326으로 내보내거나 GeoJSON/Shapefile(.zip)로 업로드해 주세요.")
    st.stop()

df["순위"] = df.index + 1
mean_temp = df["day_avg_temp"].mean()
df["평균대비_편차(°C)"] = (df["day_avg_temp"] - mean_temp).round(2)

with st.sidebar:
    if filter_mode == "상위 N개 선정":
        top_n = st.slider("선정할 핫스팟 개수 (Top-N)", 1, len(df), min(3, len(df)))
        df["핫스팟_선정"] = df["순위"] <= top_n
    else:
        threshold = st.slider(
            "임계 낮 평균온도 (°C)",
            float(df["day_avg_temp"].min()),
            float(df["day_avg_temp"].max()),
            float(np.percentile(df["day_avg_temp"], 70)),
            step=0.1
        )
        df["핫스팟_선정"] = df["day_avg_temp"] >= threshold

    map_theme = st.selectbox(
        "지도 배경 스타일",
        ["OpenStreetMap (도로·건물 명확)", "위성지도 (아스팔트·현장 확인용)", "CartoDB positron (심플 밝은 테마)"]
    )

hotspots = df[df["핫스팟_선정"]]

# ---------------------------------------------------------
# 6. 핵심 요약 지표 (KPI)
# ---------------------------------------------------------
top_site = df.iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("최종 1순위 핫스팟", top_site["지점명"])
c2.metric("최고 낮 평균온도", f"{top_site['day_avg_temp']:.1f} °C", f"+{top_site['평균대비_편차(°C)']:.1f} °C (전체 평균 대비)")
c3.metric("전체 지점 낮 평균온도", f"{mean_temp:.1f} °C")
c4.metric("선정된 핫스팟 수", f"{len(hotspots)} 개 / 총 {len(df)} 개")

st.divider()

# ---------------------------------------------------------
# 7. GIS 지도 및 순위 차트 시각화
# ---------------------------------------------------------
col_map, col_chart = st.columns([1.2, 1])

with col_map:
    st.subheader("🗺️ 낮 평균온도 분포 및 선정 핫스팟 지도")

    center_lat, center_lon = df["lat"].mean(), df["lon"].mean()

    if "위성지도" in map_theme:
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=15,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery"
        )
    elif "positron" in map_theme:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="CartoDB positron")
    else:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="OpenStreetMap")

    # 온도 차이를 극대화하기 위한 정규화 히트맵 레이어 (배경 지도가 비치도록 투명도 조절)
    t_min, t_max = df["day_avg_temp"].min(), df["day_avg_temp"].max()
    df["heat_weight"] = (df["day_avg_temp"] - t_min) / (t_max - t_min + 1e-6) + 0.2
    heat_data = df[["lat", "lon", "heat_weight"]].values.tolist()
    HeatMap(heat_data, radius=30, blur=20, min_opacity=0.35, max_zoom=16).add_to(m)

    # 지점별 마커 표시
    for _, row in df.iterrows():
        is_selected = row["핫스팟_선정"]
        color = "#FF1E1E" if row["순위"] == 1 else ("#FF8C00" if is_selected else "#2563EB")
        radius = 12 if row["순위"] == 1 else (9 if is_selected else 6)

        popup_html = f"""
        <div style="font-family: sans-serif; min-width: 160px;">
            <b>[{row['순위']}위] {row['지점명']}</b><br>
            낮 평균온도: <b>{row['day_avg_temp']:.1f} °C</b><br>
            전체 평균 대비: {row['평균대비_편차(°C)']:+.1f} °C<br>
            상태: {'🔥 선정 핫스팟' if is_selected else '일반 지점'}
        </div>
        """
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color="#FFFFFF",
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{row['순위']}위: {row['지점명']} ({row['day_avg_temp']:.1f}°C)"
        ).add_to(m)

    # 데이터 전체가 한 화면에 들어오도록 자동 줌 조정
    sw = [df["lat"].min(), df["lon"].min()]
    ne = [df["lat"].max(), df["lon"].max()]
    if sw != ne:
        m.fit_bounds([sw, ne], padding=(35, 35))

    # st_folium 흰 화면 버그 방지를 위한 HTML 직접 렌더링
    components.html(m._repr_html_(), height=500)

with col_chart:
    st.subheader("📊 지점별 낮 평균온도 순위 비교")
    df["구분"] = df["핫스팟_선정"].map({True: "선정 핫스팟", False: "일반 지점"})

    fig = px.bar(
        df.sort_values("day_avg_temp", ascending=True),
        x="day_avg_temp",
        y="지점명",
        orientation="h",
        color="구분",
        color_discrete_map={"선정 핫스팟": "#EF4444", "일반 지점": "#94A3B8"},
        text="day_avg_temp",
        labels={"day_avg_temp": "낮 평균온도 (°C)", "지점명": ""}
    )
    fig.add_vline(
        x=mean_temp,
        line_dash="dash",
        line_color="#2563EB",
        annotation_text=f"전체 평균 ({mean_temp:.1f}°C)",
        annotation_position="top right"
    )
    fig.update_traces(texttemplate="%{text:.1f}°C", textposition="outside")
    fig.update_layout(
        height=500,
        margin=dict(l=10, r=40, t=30, b=20),
        xaxis_range=[max(0, t_min - 3), t_max + 2.5],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 8. 순위표 및 결과 내보내기
# ---------------------------------------------------------
st.subheader("📋 낮 평균온도 기반 우선순위 평가표")
display_cols = ["순위", "지점명", "day_avg_temp", "평균대비_편차(°C)", "lat", "lon", "핫스팟_선정"]
st.dataframe(
    df[display_cols].rename(columns={"day_avg_temp": "낮_평균온도(°C)"}),
    use_container_width=True,
    hide_index=True
)

csv_bytes = df[display_cols].to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="📥 핫스팟 선정 결과 CSV 다운로드",
    data=csv_bytes,
    file_name="hotspot_daytime_temp_ranking.csv",
    mime="text/csv"
)

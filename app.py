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
# 1. 페이지 설정 및 캡처 최적화 레이아웃
# ---------------------------------------------------------
st.set_page_config(
    page_title="유성구·서구 낮 평균기온 상위 30개 지점 분석",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.3rem;
            padding-bottom: 1rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌡️ 대전 유성구·서구 낮 평균기온 상위 30개 지점 분석")
st.caption("장소 특성 구분 없이 낮 평균기온(°C) 단일 기준만으로 도출한 상위 30개 고온 지점 분포입니다.")

# ---------------------------------------------------------
# 2. 유성구·서구 격자형 샘플 데이터 생성 (장소 특성 무관)
# ---------------------------------------------------------
def load_sample_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)

    # 유성구 행정동 중심 좌표 (35개 지점 생성용)
    yuseong_centers = [
        ("봉명동", 36.3565, 127.3485, 39.6),
        ("온천1동", 36.3532, 127.3408, 40.1),
        ("온천2동", 36.3645, 127.3460, 39.2),
        ("노은1동", 36.3735, 127.3195, 38.4),
        ("노은2동", 36.3885, 127.3160, 38.0),
        ("노은3동", 36.3820, 127.3095, 37.6),
        ("신성동", 36.3865, 127.3505, 37.8),
        ("전민동", 36.3975, 127.4025, 38.3),
        ("관평동", 36.4250, 127.3880, 39.1),
        ("구즉동", 36.4340, 127.3810, 37.9),
        ("진잠동", 36.3025, 127.3190, 37.5),
        ("학하동", 36.3390, 127.3045, 36.9),
    ]

    # 서구 행정동 중심 좌표 (35개 지점 생성용)
    seo_centers = [
        ("둔산1동", 36.3522, 127.3790, 40.5),
        ("둔산2동", 36.3605, 127.3815, 39.9),
        ("둔산3동", 36.3515, 127.3910, 39.7),
        ("탄방동", 36.3455, 127.3855, 39.3),
        ("월평1동", 36.3565, 127.3655, 38.6),
        ("월평2동", 36.3602, 127.3610, 38.1),
        ("갈마1동", 36.3492, 127.3705, 38.9),
        ("갈마2동", 36.3440, 127.3735, 38.5),
        ("가장동", 36.3355, 127.3835, 38.8),
        ("내동", 36.3360, 127.3740, 38.2),
        ("변동", 36.3295, 127.3815, 38.4),
        ("도마1동", 36.3185, 127.3775, 39.4),
        ("도마2동", 36.3135, 127.3725, 39.0),
        ("정림동", 36.3075, 127.3665, 37.7),
        ("가수원동", 36.3040, 127.3505, 38.0),
        ("관저1동", 36.3015, 127.3385, 38.3),
        ("관저2동", 36.2965, 127.3325, 38.1),
    ]

    rows = []
    for i in range(35):
        dong, lat_c, lon_c, base_t = yuseong_centers[i % len(yuseong_centers)]
        rows.append({
            "자치구": "유성구",
            "지점명": f"유성구 {dong}-{i+1:02d}",
            "lat": round(lat_c + rng.uniform(-0.0045, 0.0045), 5),
            "lon": round(lon_c + rng.uniform(-0.0055, 0.0055), 5),
            "day_avg_temp": round(base_t + rng.uniform(-0.9, 0.9), 1),
        })

    for i in range(35):
        dong, lat_c, lon_c, base_t = seo_centers[i % len(seo_centers)]
        rows.append({
            "자치구": "서구",
            "지점명": f"서구 {dong}-{i+1:02d}",
            "lat": round(lat_c + rng.uniform(-0.0040, 0.0040), 5),
            "lon": round(lon_c + rng.uniform(-0.0050, 0.0050), 5),
            "day_avg_temp": round(base_t + rng.uniform(-0.9, 0.9), 1),
        })

    return pd.DataFrame(rows)

# ---------------------------------------------------------
# 3. 데이터 로더 및 자치구 자동 판별
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
    st.header("⚙️ 분석 및 화면 설정")
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
    name_col = st.selectbox("지점 식별자(ID/명칭) 컬럼", cols, index=default_name)
    lat_col = st.selectbox("위도(Latitude, ~36) 컬럼", cols, index=default_lat)
    lon_col = st.selectbox("경도(Longitude, ~127) 컬럼", cols, index=default_lon)
    temp_col = st.selectbox("낮 평균기온(°C) 컬럼", cols, index=default_temp)

    district_options = ["좌표 자동 판별 (유성구/서구)"] + cols
    default_dist_idx = district_options.index("자치구") if "자치구" in district_options else 0
    dist_col = st.selectbox("자치구 컬럼 (선택)", district_options, index=default_dist_idx)

    st.subheader("🗺️ 지도 표시 옵션")
    map_theme = st.selectbox(
        "지도 배경 스타일",
        ["OpenStreetMap (도로·지형 명확)", "위성지도 (지표면 확인용)", "CartoDB positron (심플 밝은 테마)"]
    )
    label_limit = st.selectbox(
        "지도 텍스트 라벨 표시 범위",
        ["상위 5개 지점만 표시 (캡처 권장)", "상위 10개 지점 표시", "상위 30개 전체 표시", "라벨 숨김"],
        index=0
    )

# ---------------------------------------------------------
# 5. 데이터 정제 및 상위 30개 지점 추출
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
    st.error("유효한 좌표 및 기온 데이터가 없습니다. 사이드바의 컬럼 매핑을 확인해 주세요.")
    st.stop()

if (df["lat"].abs() > 90).any() and (df["lon"].abs() <= 90).all():
    st.warning("⚠️ 위도(lat)와 경도(lon)가 반대로 입력되어 자동으로 교정했습니다.")
    df["lat"], df["lon"] = df["lon"].copy(), df["lat"].copy()

if (df["lat"].abs() > 90).any() or (df["lon"].abs() > 180).any():
    st.error("⚠️ 위경도(WGS84, EPSG:4326)가 아닌 평면직각좌표(TM 등)가 감지되었습니다. EPSG:4326으로 변환 후 업로드해 주세요.")
    st.stop()

if dist_col == "좌표 자동 판별 (유성구/서구)":
    df["자치구"] = [infer_daejeon_district(la, lo) for la, lo in zip(df["lat"], df["lon"])]

if target_region in ["유성구", "서구"]:
    df = df[df["자치구"].str.contains(target_region, na=False)].copy()
    if df.empty:
        st.warning(f"선택한 '{target_region}' 권역에 해당하는 데이터가 없습니다.")
        st.stop()

# 전체 정렬 후 상위 30개 지점만 추출
df = df.sort_values(by="day_avg_temp", ascending=False).reset_index(drop=True)
total_count = len(df)
all_mean_temp = df["day_avg_temp"].mean()

top30_df = df.head(30).copy()
top30_df["순위"] = top30_df.index + 1
top30_mean_temp = top30_df["day_avg_temp"].mean()
top30_df["전체평균대비_편차(°C)"] = (top30_df["day_avg_temp"] - all_mean_temp).round(2)

def assign_tier(rank: int) -> str:
    if rank == 1:
        return "1위 최상위"
    if rank <= 10:
        return "상위 2~10위"
    return "상위 11~30위"

top30_df["순위구간"] = top30_df["순위"].apply(assign_tier)

# ---------------------------------------------------------
# 6. 핵심 요약 지표 (KPI)
# ---------------------------------------------------------
top_site = top30_df.iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"[{target_region}] 1위 최고 기온 지점", top_site["지점명"])
c2.metric("최고 낮 평균기온", f"{top_site['day_avg_temp']:.1f} °C", f"+{top_site['전체평균대비_편차(°C)']:.1f} °C (권역 전체 평균 대비)")
c3.metric(f"상위 {len(top30_df)}개 지점 평균기온", f"{top30_mean_temp:.1f} °C", f"권역 전체 평균 {all_mean_temp:.1f} °C")
c4.metric("표출 지점 수", f"상위 {len(top30_df)} 개", f"권역 내 총 {total_count} 개 지점 중")

st.divider()

# ---------------------------------------------------------
# 7. 대형 GIS 지도 및 상위 30개 순위 차트 (캡처 메인 영역)
# ---------------------------------------------------------
VIS_HEIGHT = 720
col_map, col_chart = st.columns([1.3, 1])

with col_map:
    st.subheader(f"🗺️ {target_region} 낮 평균기온 상위 {len(top30_df)}개 지점 분포")

    center_lat, center_lon = top30_df["lat"].mean(), top30_df["lon"].mean()
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

    # 상위 30개 지점 기반 온도 분포 히트맵
    t_min, t_max = top30_df["day_avg_temp"].min(), top30_df["day_avg_temp"].max()
    top30_df["heat_weight"] = (top30_df["day_avg_temp"] - t_min) / (t_max - t_min + 1e-6) + 0.25
    heat_data = top30_df[["lat", "lon", "heat_weight"]].values.tolist()
    heat_radius = 28 if target_region == "전체 (유성구 + 서구)" else 34
    HeatMap(heat_data, radius=heat_radius, blur=20, min_opacity=0.35, max_zoom=15).add_to(m)

    # 라벨 표시 개수 결정
    if "5개" in label_limit:
        max_label_rank = 5
    elif "10개" in label_limit:
        max_label_rank = 10
    elif "30개" in label_limit:
        max_label_rank = 30
    else:
        max_label_rank = 0

    for _, row in top30_df.iterrows():
        rank = int(row["순위"])
        if rank == 1:
            color, radius = "#DC2626", 13
        elif rank <= 10:
            color, radius = "#F97316", 9.5
        else:
            color, radius = "#2563EB", 7

        popup_html = f"""
        <div style="font-family: sans-serif; min-width: 165px;">
            <b>[{rank}위] {row['지점명']}</b><br>
            낮 평균기온: <b>{row['day_avg_temp']:.1f} °C</b><br>
            권역 전체 평균 대비: {row['전체평균대비_편차(°C)']:+.1f} °C
        </div>
        """
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color="#FFFFFF",
            weight=1.6,
            fill=True,
            fill_color=color,
            fill_opacity=0.92,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{rank}위: {row['지점명']} ({row['day_avg_temp']:.1f}°C)"
        ).add_to(m)

        if rank <= max_label_rank:
            border_color = "#DC2626" if rank == 1 else ("#F97316" if rank <= 10 else "#2563EB")
            folium.Marker(
                location=[row["lat"], row["lon"]],
                icon=folium.DivIcon(
                    html=f"""
                    <div style="
                        font-family: 'Malgun Gothic', sans-serif;
                        font-size: 11px;
                        font-weight: 700;
                        color: #111827;
                        background: rgba(255, 255, 255, 0.93);
                        border: 2px solid {border_color};
                        border-radius: 5px;
                        padding: 1px 6px;
                        white-space: nowrap;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.22);
                        transform: translate(10px, -10px);
                    ">
                        {rank}위 {row['지점명']} ({row['day_avg_temp']:.1f}°C)
                    </div>
                    """
                )
            ).add_to(m)

    # 좌측 하단 범례
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 22px;
        left: 22px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.94);
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid #CBD5E1;
        font-family: 'Malgun Gothic', sans-serif;
        font-size: 12px;
        color: #1E293B;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    ">
        <div style="font-weight: 700; margin-bottom: 6px;">{target_region} 상위 {len(top30_df)}개 지점</div>
        <div style="margin-bottom: 4px;"><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#DC2626;margin-right:6px;"></span>1위 최상위 지점</div>
        <div style="margin-bottom: 4px;"><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#F97316;margin-right:6px;"></span>상위 2~10위</div>
        <div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#2563EB;margin-right:6px;"></span>상위 11~30위</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    sw = [top30_df["lat"].min(), top30_df["lon"].min()]
    ne = [top30_df["lat"].max(), top30_df["lon"].max()]
    if sw != ne:
        m.fit_bounds([sw, ne], padding=(45, 45))

    components.html(m._repr_html_(), height=VIS_HEIGHT)

with col_chart:
    st.subheader(f"📊 {target_region} 낮 평균기온 상위 {len(top30_df)}개 순위")

    plot_df = top30_df.copy()
    plot_df["차트라벨"] = plot_df["순위"].astype(str) + "위 · " + plot_df["지점명"]

    fig = px.bar(
        plot_df.sort_values("day_avg_temp", ascending=True),
        x="day_avg_temp",
        y="차트라벨",
        orientation="h",
        color="순위구간",
        color_discrete_map={
            "1위 최상위": "#DC2626",
            "상위 2~10위": "#F97316",
            "상위 11~30위": "#2563EB"
        },
        text="day_avg_temp",
        labels={"day_avg_temp": "낮 평균기온 (°C)", "차트라벨": ""}
    )
    fig.add_vline(
        x=all_mean_temp,
        line_dash="dash",
        line_color="#475569",
        annotation_text=f"권역 전체 평균 ({all_mean_temp:.1f}°C)",
        annotation_position="top right"
    )
    fig.update_traces(texttemplate="%{text:.1f}°C", textposition="outside")
    fig.update_layout(
        height=VIS_HEIGHT,
        bargap=0.22,
        margin=dict(l=10, r=48, t=35, b=15),
        xaxis_range=[max(0, float(plot_df["day_avg_temp"].min()) - 1.5), float(plot_df["day_avg_temp"].max()) + 1.5],
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, title_text=""),
        font=dict(size=11.5)
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 8. 상위 30개 결과 CSV 내보내기 (사이드바 배치)
# ---------------------------------------------------------
with st.sidebar:
    st.divider()
    export_cols = ["순위", "자치구", "지점명", "day_avg_temp", "전체평균대비_편차(°C)", "lat", "lon"]
    csv_bytes = top30_df[export_cols].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 상위 30개 지점 결과 CSV 다운로드",
        data=csv_bytes,
        file_name=f"top30_daytime_temp_{target_region}.csv",
        mime="text/csv",
        use_container_width=True
    )

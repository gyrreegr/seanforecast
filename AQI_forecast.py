"""
AQI 空氣品質預報面量圖產生器 + 自動疊圖
自動下載 CSV 資料，結合縣市 SHP 底圖，產出 3 天預報並合成至固定底圖
"""

import requests
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import os
import urllib3
from PIL import Image  # 新增：用於影像合成

# 關閉不安全的 SSL 憑證警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================================
# === 使用者設定區 (請依需求修改) ============================
# ============================================================

# 輸出目錄
WORK_DIR = "./outputs"
OUTPUT_DIR = os.path.join(WORK_DIR, "Output")

# 縣市界線 SHP 路徑
SHP_PATH = "./COUNTY_MOI_1130718.shp"

# 【新增】底圖路徑與最終輸出檔名
BASE_IMAGE_PATH = "./AQI_background.png"  # 👈 請修改為您那張大底圖的路徑
FINAL_OUTPUT_NAME = "AQI_Forecast_Composite_3Days.png"

# CSV 下載連結
CSV_URL = (
    "https://data.moenv.gov.tw/api/v2/aqf_p_01"
    "?api_key=b7df779e-71a6-4148-8379-5afbd441d803"
    "&limit=1000&sort=publishtime%20desc&format=CSV"
)

# SHP 中縣市名稱欄位
SHP_NAME_COL = "COUNTYNAME"

# 【新增】版面配置 (依據您提供的像素，四捨五入至整數)
LAYOUT_CONFIG = [
    # 第一天 (明日)
    {'w': 1114, 'h': 1745, 'x': 190,  'y': 497},
    # 第二天 (後日)
    {'w': 1114, 'h': 1745, 'x': 1711, 'y': 497},
    # 第三天 (大後日)
    {'w': 1114, 'h': 1745, 'x': 3232, 'y': 497}
]

# ============================================================
# === AQI 設定 ================================================
# ============================================================

AQI_BINS   = [0, 50, 100, 150, 200, 300, 500]
AQI_COLORS = ["#7ed957", "#fffb26", "#ff9734", "#ca0034", "#670099", "#7e0123"]
AQI_LABELS = [
    "良好 (0–50)",
    "普通 (51–100)",
    "對敏感族群不健康 (101–150)",
    "對所有族群不健康 (151–200)",
    "非常不健康 (201–300)",
    "危害 (301–500)",
]

# area → 縣市對應表
AREA_TO_COUNTIES = {
    "北部":  ["新北市", "臺北市", "桃園市", "基隆市"],
    "竹苗":  ["新竹市", "新竹縣", "苗栗縣"],
    "宜蘭":  ["宜蘭縣"],
    "中部":  ["臺中市", "彰化縣", "南投縣"],
    "雲嘉南": ["雲林縣", "嘉義市", "嘉義縣", "臺南市"],
    "高屏":  ["高雄市", "屏東縣"],
    "花東":  ["花蓮縣", "臺東縣"],
    "澎湖":  ["澎湖縣"],
    "金門":  ["金門縣"],
    "馬祖":  ["連江縣"],
}

# ============================================================

def classify_aqi(val):
    """回傳 AQI 對應的顏色"""
    if pd.isna(val):
        return "#cccccc"
    for i in range(len(AQI_BINS) - 1):
        if AQI_BINS[i] <= val <= AQI_BINS[i + 1]:
            return AQI_COLORS[i]
    return "#cccccc"

def download_csv(url):
    print("正在下載 AQI 預報資料...")
    resp = requests.get(url, timeout=30, verify=False)
    resp.raise_for_status()
    content = resp.content.decode("utf-8-sig")
    from io import StringIO
    df = pd.read_csv(StringIO(content))
    print(f"  下載完成，共 {len(df)} 筆資料")
    return df

def build_county_aqi(df_day, gdf):
    """將當日 AQI 依 area 對應到各縣市"""
    county_aqi = {}
    for _, row in df_day.iterrows():
        area = str(row["area"]).strip()
        aqi  = row["aqi"]
        counties = AREA_TO_COUNTIES.get(area, [area])
        for c in counties:
            county_aqi[c] = aqi

    gdf = gdf.copy()
    gdf["aqi_value"] = gdf[SHP_NAME_COL].map(county_aqi)
    gdf["color"]     = gdf["aqi_value"].apply(classify_aqi)
    return gdf

def draw_transparent_map(gdf_day, output_path):
    """繪製滿版、無邊框、透明背景的面量圖"""
    # 設定畫布比例，盡量接近您的目標長寬比 (1114/1745 ~ 0.638)
    fig = plt.figure(figsize=(6.38, 10), dpi=200)
    
    # 【關鍵】使用 add_axes 強制讓地圖填滿整個畫布，去除所有白邊與 padding
    ax = fig.add_axes([0, 0, 1, 1], projection=None)

    # 畫底圖
    for color, group in gdf_day.groupby("color"):
        group.plot(ax=ax, color=color, edgecolor="black", linewidth=0.5)

    # 縣市邊界再疊一層
    gdf_day.boundary.plot(ax=ax, color="#555555", linewidth=1)

    # 設定經緯度範圍 (鎖定台灣本島範圍，確保每次縮放比例一致)
    lat_min, lat_max = 21.8, 25.4
    lon_min, lon_max = 119.8, 122.2
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    
    # 關閉坐標軸
    ax.set_axis_off()

    # 儲存為透明背景
    plt.savefig(output_path, dpi=200, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. 載入底圖 ──
    if not os.path.exists(BASE_IMAGE_PATH):
        print(f"嚴重錯誤: 找不到底圖 {BASE_IMAGE_PATH}")
        return
    base_img = Image.open(BASE_IMAGE_PATH).convert("RGBA")

    # ── 2. 下載並讀取 CSV ──
    df = download_csv(CSV_URL)
    df.columns = [c.strip().lower() for c in df.columns]

    df["forecastdate"] = pd.to_datetime(df["forecastdate"]).dt.date
    df["aqi"]          = pd.to_numeric(df["aqi"], errors="coerce")

    # ── 3. 決定要繪製的 3 天（今日 + 1、+ 2、+ 3） ──
    today = datetime.now().date()
    target_dates = [today + timedelta(days=d) for d in range(1, 4)]
    print(f"\n執行日期：{today}，將繪製：{[str(d) for d in target_dates]}\n")

    # ── 4. 讀取 SHP ──
    print("正在載入 SHP 檔案...")
    gdf = gpd.read_file(SHP_PATH, encoding="utf-8")
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # ── 5. 逐日繪圖與疊圖 ──
    for i, target_date in enumerate(target_dates):
        df_day = df[df["forecastdate"] == target_date]
        if df_day.empty:
            print(f"警告：{target_date} 無預報資料，略過該日。")
            continue

        if "publishtime" in df_day.columns:
            df_day = (
                df_day.sort_values("publishtime", ascending=False)
                .drop_duplicates(subset=["area"])
            )

        gdf_day = build_county_aqi(df_day, gdf)
        
        # 產生暫存的透明地圖
        temp_png = os.path.join(OUTPUT_DIR, f"temp_aqi_{i}.png")
        print(f"正在產生 {target_date} 面量圖...")
        draw_transparent_map(gdf_day, temp_png)

        # 讀取暫存圖並疊加至大底圖
        if os.path.exists(temp_png):
            overlay_img = Image.open(temp_png).convert("RGBA")
            
            # 讀取您的尺寸與座標設定
            cfg = LAYOUT_CONFIG[i]
            target_size = (cfg['w'], cfg['h'])
            paste_pos = (cfg['x'], cfg['y'])
            
            # 縮放影像 (LANCZOS 演算法畫質最佳)
            overlay_resized = overlay_img.resize(target_size, Image.Resampling.LANCZOS)
            
            # 將縮放後的地圖貼到底圖上
            base_img.paste(overlay_resized, paste_pos, overlay_resized)
            print(f"  ✓ {target_date} 已合成至底圖。")
            
            # 刪除暫存檔 (可依需求決定是否保留)
            os.remove(temp_png)

    # ── 6. 儲存最終合成圖 ──
    final_path = os.path.join(OUTPUT_DIR, FINAL_OUTPUT_NAME)
    base_img.save(final_path)
    print(f"\n🎉 全部完成！最終合成圖已儲存至：{final_path}")

if __name__ == "__main__":
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    main()
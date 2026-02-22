import os
import requests
import urllib3
import io
import numpy as np
from PIL import Image, ImageDraw

# 關閉不安全的 SSL 憑證警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# ⚙️ 設定區：檔案路徑與目錄
# ==========================================
WORK_DIR = "./outputs"
OUTPUT_DIR = os.path.join(WORK_DIR, "Output")

# 底圖設定 (請換成您實際的底圖路徑)
BASE_MAP_1 = "./7daysforecast_background_1.png" # 供第 1~4 天使用
BASE_MAP_2 = "./7daysforecast_background_2.png" # 供第 5~7 天使用

# 輸出檔名設定
OUTPUT_NAME_1 = "ECMWF_Forecast_Days_1_to_4.png"
OUTPUT_NAME_2 = "ECMWF_Forecast_Days_5_to_7.png"

# 建立輸出資料夾
os.makedirs(OUTPUT_DIR, exist_ok=True)

# URL 與 API 設定
CSV_URL = "https://watch.ncdr.nat.gov.tw/php/list_realtime_date_csv.php?v=CHART_ECMWF_WRFDS"
IMG_TEMPLATE = "https://watch.ncdr.nat.gov.tw/00_Wxmap/2F7_ECMWF_0.25deg/{YYYYMM}/{YYYYMMDDHH}/ecwrf_rain_{YYYYMMDDHH}_f{XX}.png"

# ==========================================
# 🛠 版面配置與遮罩設定 (自動四捨五入)
# ==========================================
# 定義 7 天各自的座標與要去除的區域 (遮罩)
LAYOUT_CONFIGS = {
    1: { # 第 1 天
        'base': 1,
        'layout': {'w': 946.2, 'h': 1628.3, 'x': 171.8, 'y': 574.2},
        'masks': [
            {'w': 473.1, 'h': 123.2, 'x': 171.8, 'y': 584.9},
            {'w': 212.2, 'h': 161.0, 'x': 169.7, 'y': 2033.4},
            {'w': 114.8, 'h': 814.1, 'x': 1003.1, 'y': 1388.3}
        ]
    },
    2: { # 第 2 天
        'base': 1,
        'layout': {'w': 946.2, 'h': 1628.3, 'x': 1191.3, 'y': 574.2},
        'masks': [
            {'w': 473.1, 'h': 123.2, 'x': 1191.3, 'y': 584.9},
            {'w': 212.2, 'h': 161.0, 'x': 1191.3, 'y': 2033.4},
            {'w': 114.8, 'h': 814.1, 'x': 2022.6, 'y': 1388.3}
        ]
    },
    3: { # 第 3 天
        'base': 1,
        'layout': {'w': 946.2, 'h': 1628.3, 'x': 2349.5, 'y': 574.2},
        'masks': [
            {'w': 473.1, 'h': 123.2, 'x': 2349.5, 'y': 584.9},
            {'w': 212.2, 'h': 161.0, 'x': 2349.5, 'y': 2033.4},
            {'w': 114.8, 'h': 814.1, 'x': 3178.6, 'y': 1388.3}
        ]
    },
    4: { # 第 4 天
        'base': 1,
        'layout': {'w': 946.2, 'h': 1628.3, 'x': 3431.0, 'y': 574.2},
        'masks': [
            {'w': 473.1, 'h': 123.2, 'x': 3431.0, 'y': 584.9},
            {'w': 212.2, 'h': 161.0, 'x': 3431.0, 'y': 2033.4},
            {'w': 114.8, 'h': 814.1, 'x': 4262.3, 'y': 1388.3}
        ]
    },
    5: { # 第 5 天
        'base': 2,
        'layout': {'w': 946.2, 'h': 1628.3, 'x': 171.8, 'y': 574.2},
        'masks': [
            {'w': 473.1, 'h': 123.2, 'x': 171.8, 'y': 584.9},
            {'w': 212.2, 'h': 161.0, 'x': 169.7, 'y': 2033.4},
            {'w': 114.8, 'h': 814.1, 'x': 1003.1, 'y': 1388.3}
        ]
    },
    6: { # 第 6 天
        'base': 2,
        'layout': {'w': 946.2, 'h': 1628.3, 'x': 1191.3, 'y': 574.2},
        'masks': [
            {'w': 473.1, 'h': 123.2, 'x': 1191.3, 'y': 584.9},
            {'w': 212.2, 'h': 161.0, 'x': 1191.3, 'y': 2033.4},
            {'w': 114.8, 'h': 814.1, 'x': 2022.6, 'y': 1388.3}
        ]
    },
    7: { # 第 7 天
        'base': 2,
        'layout': {'w': 946.2, 'h': 1628.3, 'x': 2349.5, 'y': 574.2},
        'masks': [
            {'w': 473.1, 'h': 123.2, 'x': 2349.5, 'y': 584.9},
            {'w': 212.2, 'h': 161.0, 'x': 2349.5, 'y': 2033.4},
            {'w': 114.8, 'h': 814.1, 'x': 3178.6, 'y': 1388.3}
        ]
    }
}

# ==========================================
# 🧠 核心處理邏輯
# ==========================================

def get_init_time(csv_url):
    """取得資料最新初始時間 (YYYYMMDDHHMM)"""
    try:
        r = requests.get(csv_url, verify=False, timeout=10)
        r.raise_for_status()
        content = r.text.strip()
        if ',' in content:
            return content.split(',')[1].strip()
        return None
    except Exception as e:
        print(f"取得初始時間失敗: {e}")
        return None

def download_image(url):
    """下載影像並回傳 PIL Image 物件"""
    try:
        r = requests.get(url, verify=False, timeout=15)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGBA")
    except Exception as e:
        print(f" 下載失敗: {url}\n ({e})")
        return None

def make_white_transparent(img, threshold=220):
    """將白色背景轉為透明"""
    img = img.convert("RGBA")
    data = np.array(img)
    
    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    # 判斷接近白色的像素
    white_mask = (r > threshold) & (g > threshold) & (b > threshold)
    
    # 將符合條件的像素 Alpha 設為 0 (透明)
    data[..., 3][white_mask] = 0
    return Image.fromarray(data)

def process_day(day_idx, init_time_str, canvases):
    """處理單日資料並貼到對應底圖上"""
    config = LAYOUT_CONFIGS[day_idx]
    base_idx = config['base']
    canvas = canvases[base_idx]
    
    # 1. 組合 URL (f01 ~ f07)
    yyyy_mm = init_time_str[:6]
    yyyy_mm_dd_hh = init_time_str[:10]
    fxx = f"{day_idx:02d}"
    
    url = IMG_TEMPLATE.format(
        YYYYMM=yyyy_mm, 
        YYYYMMDDHH=yyyy_mm_dd_hh, 
        XX=fxx
    )
    
    print(f"[Day {day_idx}] 下載與處理: {url}")
    img = download_image(url)
    if not img: return

    # 2. 去除白底
    img = make_white_transparent(img)

    # 3. 建立與底圖大小相同的透明中繼圖層
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    
    # 4. 縮放與貼上
    cfg_L = config['layout']
    target_size = (int(round(cfg_L['w'])), int(round(cfg_L['h'])))
    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
    
    paste_pos = (int(round(cfg_L['x'])), int(round(cfg_L['y'])))
    layer.paste(img_resized, paste_pos, img_resized)

    # 5. 利用 Alpha 遮罩將「不要的區域」透明化
    alpha_mask = layer.split()[3]
    draw = ImageDraw.Draw(alpha_mask)

    for mask in config['masks']:
        mx, my = int(round(mask['x'])), int(round(mask['y']))
        mw, mh = int(round(mask['w'])), int(round(mask['h']))
        # 畫上全透明方塊 (fill=0)
        draw.rectangle([mx, my, mx + mw, my + mh], fill=0)

    # 套用遮罩
    layer.putalpha(alpha_mask)
    
    # 6. 合成至最終畫布
    canvas.alpha_composite(layer)
    print(f" ✓ Day {day_idx} 已成功合成至底圖 {base_idx}")

# ==========================================
# 🚀 主程式執行
# ==========================================
def main():
    print("="*50)
    print(" ECMWF WRF 7天預報自動下載與合成程式")
    print("="*50)

    # 確認底圖存在
    if not os.path.exists(BASE_MAP_1) or not os.path.exists(BASE_MAP_2):
        print(f"嚴重錯誤: 找不到底圖檔案，請確認路徑設定正確。")
        return

    # 載入底圖
    canvases = {
        1: Image.open(BASE_MAP_1).convert("RGBA"),
        2: Image.open(BASE_MAP_2).convert("RGBA")
    }

    # 取得最新初始時間
    print("\n獲取最新初始時間...")
    init_time_str = get_init_time(CSV_URL)
    if not init_time_str:
        print("終止作業：無法取得初始時間")
        return
    print(f"初始時間為: {init_time_str}")

    # 依序處理 1~7 天
    for day_idx in range(1, 8):
        process_day(day_idx, init_time_str, canvases)

    # 存檔輸出
    out_path_1 = os.path.join(OUTPUT_DIR, OUTPUT_NAME_1)
    out_path_2 = os.path.join(OUTPUT_DIR, OUTPUT_NAME_2)
    
    canvases[1].save(out_path_1, format="PNG")
    canvases[2].save(out_path_2, format="PNG")
    
    print("\n🎉 作業完成！")
    print(f"輸出圖 1 (Day 1-4): {out_path_1}")
    print(f"輸出圖 2 (Day 5-7): {out_path_2}")

if __name__ == "__main__":
    main()
import os
import requests
import urllib3
from PIL import Image, ImageDraw
import io
import numpy as np

# 關閉不安全的 SSL 憑證警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# ⚙️ 設定區：檔案路徑與目錄
# ==========================================
WORK_DIR = "./outputs"
OUTPUT_DIR = os.path.join(WORK_DIR, "Output")

# 底圖設定 (請換成您實際的底圖路徑)
BASE_MAP_TOMORROW = "./twodays_background_1.png"
BASE_MAP_DAYAFTER = "./twodays_background_2.png"

# 輸出檔名設定
OUTPUT_NAME_TOMORROW = "Model_Forecast_Tomorrow.png"
OUTPUT_NAME_DAYAFTER = "Model_Forecast_DayAfter.png"

# 建立輸出資料夾
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 🛠 預報模型設定與參數
# ==========================================

def get_cwa_qpf_fxx(init_time_str, day_offset):
    """CWA QPF 的 fxx 判定邏輯"""
    hh = int(init_time_str[8:10]) # 取出 HH
    if hh in (3,21):
        if day_offset == 1: return "39"
        else: return None  # 後天不產出
    elif hh in (9,15):
        if day_offset == 1: return "15"
        elif day_offset == 2: return "51"
        else: return None
    return None

def get_standard_fxx(init_time_str, day_offset):
    """ECMWF, GFS, GSM 的通用 fxx 判定邏輯 (01, 02)"""
    return f"{day_offset:02d}"

MODELS = {
    'cwa_qpf': {
        'csv_url': 'https://watch.ncdr.nat.gov.tw/php/list_realtime_date_csv.php?v=CWB_QPF_OFFICIAL',
        'img_template': 'https://watch.ncdr.nat.gov.tw/00_Wxmap/5F11_CWB_QPF_OFFICIAL/{YYYYMM}/O01_{YYYYMMDDHH}_f{XX}_d12s.gif',
        'layout': {'w': 904.1, 'h': 1629, 'x': 190.5, 'y': 572.6},
        'masks': [
            {'w': 415.4, 'h': 189.4, 'x': 189.9, 'y': 572.6},
            {'w': 119.5, 'h': 785.8, 'x': 938.5, 'y': 1095.3},
            {'w': 415.4, 'h': 293.9, 'x': 642.6, 'y': 1881}
        ],
        'keep_box': None,
        'get_fxx': get_cwa_qpf_fxx
    },
    'ecmwf_wrf': {
        'csv_url': 'https://watch.ncdr.nat.gov.tw/php/list_realtime_date_csv.php?v=CHART_ECMWF_WRFDS',
        'img_template': 'https://watch.ncdr.nat.gov.tw/00_Wxmap/2F7_ECMWF_0.25deg/{YYYYMM}/{YYYYMMDDHH}/ecwrf_rain_{YYYYMMDDHH}_f{XX}.png',
        'layout': {'w': 952.6, 'h': 1639.3, 'x': 1192.5, 'y': 566.2},
        'masks': [
            {'w': 415.4, 'h': 139.5, 'x': 1195.8, 'y': 566.2},
            {'w': 230.2, 'h': 150.2, 'x': 1195.8, 'y': 2055.2},
            {'w': 98.1, 'h': 834.2, 'x': 2046.9, 'y': 1371.2}
        ],
        'keep_box': None,
        'get_fxx': get_standard_fxx
    },
    'gfs_fnv3': {
        'csv_url': 'https://watch.ncdr.nat.gov.tw/php/list_realtime_date_csv.php?v=WRF2WEEKS_RAIN',
        # 注意此 URL 使用的是 {YYYYMMDDHHmm} 12碼
        'img_template': 'https://watch.ncdr.nat.gov.tw/00_Wxmap/5F24_NCDR_WRF_2WEEKS/{YYYYMM}/{YYYYMMDDHHmm}/rain_{YYYYMMDDHHmm}_f{XX}.gif',
        'layout': {'w': 1318.2, 'h': 1721.6, 'x': 1990.8, 'y': 472.8},
        'keep_box': {'w': 1024, 'h': 1664.5, 'x': 2285, 'y': 529.9}, # 裁減(要的區域)
        'masks': [
            {'w': 309.5, 'h': 57.1, 'x': 2285, 'y': 544.1},
            {'w': 236.1, 'h': 196.7, 'x': 2285, 'y': 1997.7},
            {'w': 143.5, 'h': 1057.5, 'x': 3165.4, 'y': 1136.9}
        ],
        'get_fxx': get_standard_fxx
    },
    'gsm_ai': {
        'csv_url': 'https://watch.ncdr.nat.gov.tw/php/list_realtime_date_csv.php?v=WRF2WEEKS_RAIN',
        'img_template': 'https://watch.ncdr.nat.gov.tw/00_Wxmap/2F8_JMAGSM_0.5deg/{YYYYMM}/{YYYYMMDDHH}/jmamsrn_{YYYYMMDDHH}_{XX}.png', # 注意 gsm 是直接接 _{XX}
        'layout': {'w': 1138.7, 'h': 1699.5, 'x': 3354.3, 'y': 506},
        'masks': [
            {'w': 453.2, 'h': 134.4, 'x': 3381.9, 'y': 568.7},
            {'w': 316.1, 'h': 177.5, 'x': 3407, 'y': 2028},
            {'w': 205.1, 'h': 1036.5, 'x': 4287.8, 'y': 1169}
        ],
        'keep_box': None,
        'get_fxx': get_standard_fxx
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
        # 內容格式通常為 "KEY_date,202602211200"
        if ',' in content:
            return content.split(',')[1].strip()
        return None
    except Exception as e:
        print(f"取得初始時間失敗 ({csv_url}): {e}")
        return None

def download_image(url):
    """下載影像並回傳 PIL Image 物件 (轉為 RGBA)"""
    try:
        r = requests.get(url, verify=False, timeout=15)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGBA")
        return img
    except Exception as e:
        print(f" 下載失敗: {url}\n ({e})")
        return None

# ==========================================
# 新增：去白底函式
# ==========================================
def make_white_transparent(img, threshold=200):
    """將白色背景轉為透明"""
    img = img.convert("RGBA")
    data = np.array(img)
    
    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    # 判斷白色 (R,G,B 都大於閥值)
    white_mask = (r > threshold) & (g > threshold) & (b > threshold)
    
    # 將符合條件的像素 Alpha 設為 0 (透明)
    data[..., 3][white_mask] = 0
    
    return Image.fromarray(data)

# ==========================================
# 替換：處理與合成邏輯 (加入去白底步驟)
# ==========================================
# ==========================================
# 替換：處理與合成邏輯 (修正 keep_box 破壞去背的問題)
# ==========================================
def process_and_composite(canvas, model_name, model_config, day_offset):
    """處理單一預報模型並合成至畫布"""
    print(f"\n[{model_name}] 準備處理 Day {day_offset}...")
    
    # 1. 取得初始時間
    init_time_str = get_init_time(model_config['csv_url'])
    if not init_time_str:
        print(f" 錯誤: 無法取得 {model_name} 的初始時間")
        return

    # 2. 判斷 fXX
    fxx = model_config['get_fxx'](init_time_str, day_offset)
    if not fxx:
        print(f" 提示: 依據規則，{model_name} 在此日期 (Day {day_offset}) 不產出圖片。跳過。")
        return

    # 3. 組合 URL
    yyyy_mm = init_time_str[:6]
    yyyy_mm_dd_hh = init_time_str[:10]
    
    url = model_config['img_template'].format(
        YYYYMM=yyyy_mm,
        YYYYMMDDHH=yyyy_mm_dd_hh,
        YYYYMMDDHHmm=init_time_str,
        XX=fxx
    )
    
    print(f" 正在下載: {url}")
    img = download_image(url)
    if not img: return

    # 將下載的圖片白色背景轉為透明
    img = make_white_transparent(img)

    # 4. 建立透明圖層以進行精準裁切與合成
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    
    # 縮放下載的影像
    cfg_L = model_config['layout']
    target_size = (int(round(cfg_L['w'])), int(round(cfg_L['h'])))
    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # 貼上透明圖層的指定座標 (使用自身作為遮罩保留透明度)
    paste_pos = (int(round(cfg_L['x'])), int(round(cfg_L['y'])))
    layer.paste(img_resized, paste_pos, img_resized)

    # 5. 製作遮罩 (Alpha Mask)
    # 取出目前的 Alpha 通道，保留剛剛去白底的效果
    alpha_mask = layer.split()[3]
    draw = ImageDraw.Draw(alpha_mask)

    # 若有「裁減尺寸(要的區域)」
    # 修正：不覆蓋內部 Alpha，而是將「要的區域以外」的四周填滿透明(0)
    keep = model_config.get('keep_box')
    if keep:
        kx, ky = int(round(keep['x'])), int(round(keep['y']))
        kw, kh = int(round(keep['w'])), int(round(keep['h']))
        
        # 畫四個透明矩形(fill=0)，把保留區塊外的地方遮掉
        draw.rectangle([0, 0, canvas.width, ky], fill=0)                     # 上方區域
        draw.rectangle([0, ky + kh, canvas.width, canvas.height], fill=0)    # 下方區域
        draw.rectangle([0, ky, kx, ky + kh], fill=0)                         # 左側區域
        draw.rectangle([kx + kw, ky, canvas.width, ky + kh], fill=0)         # 右側區域

    # 執行「不要的區域」裁切 (塗黑=透明)
    for mask in model_config['masks']:
        mx, my = int(round(mask['x'])), int(round(mask['y']))
        mw, mh = int(round(mask['w'])), int(round(mask['h']))
        draw.rectangle([mx, my, mx + mw, my + mh], fill=0)

    # 6. 套用遮罩並合成至最終畫布
    layer.putalpha(alpha_mask)
    canvas.alpha_composite(layer)
    print(f" ✓ {model_name} 去白底並合成成功！")

# ==========================================
# 🚀 主程式執行
# ==========================================
def create_forecast_card(base_map_path, output_filename, day_offset):
    print(f"\n{'='*50}")
    print(f"開始產生 Day {day_offset} 預報圖...")
    print(f"{'='*50}")
    
    if not os.path.exists(base_map_path):
        print(f"嚴重錯誤: 找不到底圖 {base_map_path}")
        return

    # 載入底圖
    canvas = Image.open(base_map_path).convert("RGBA")

    # 依序處理 4 個模型
    for model_name, config in MODELS.items():
        process_and_composite(canvas, model_name, config, day_offset)

    # 儲存
    out_path = os.path.join(OUTPUT_DIR, output_filename)
    canvas.save(out_path, format="PNG")
    print(f"\n🎉 圖片儲存成功: {out_path}\n")

def main():
    # Day 1: 明天
    create_forecast_card(BASE_MAP_TOMORROW, OUTPUT_NAME_TOMORROW, day_offset=1)
    
    # Day 2: 後天
    create_forecast_card(BASE_MAP_DAYAFTER, OUTPUT_NAME_DAYAFTER, day_offset=2)
    
    print("所有作業處理完畢！")

if __name__ == "__main__":

    main()

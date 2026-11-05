%%writefile app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests
from bs4 import BeautifulSoup
import datetime
import re
import itertools

# ページの基本設定
st.set_page_config(page_title="ボートレースAI【決まり手完全連動・進化版】", layout="centered")

# 1. AIと各種辞書の読み込み（Googleドライブのパス、または直下のパスを指定）
@st.cache_resource
def load_brain():
    # ファイル名が 'boat_ai_brain_v9.pkl' の場合（もしドライブのパスならここを '/content/drive/MyDrive/boat_ai_brain_v9.pkl' に変更してな）
    with open('boat_ai_brain_v9.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    return (
        saved_data['model_1st'],
        saved_data['model_2nd'],
        saved_data['model_3rd'],
        saved_data['racer_stats'],
        saved_data['motor_stats'],
        saved_data['features']
    )

model_1st, model_2nd, model_3rd, racer_stat_dict, motor_stat_dict, train_features = load_brain()

VENUE_MAP = {
    "桐生": "01", "戸田": "02", "江戸川": "03", "平和島": "04", "多摩川": "05", "浜名湖": "06",
    "蒲郡": "07", "常滑": "08", "津": "09", "三国": "10", "びわこ": "11", "住之江": "12",
    "尼崎": "13", "鳴門": "14", "丸亀": "15", "児島": "16", "宮島": "17", "徳山": "18",
    "下関": "19", "若松": "20", "芦屋": "21", "福岡": "22", "唐津": "23", "大村": "24"
}

# 2. スクレイピング関数（今節の調子・ST・展示を網羅）
def scrape_boatrace_data_v9(target_date, venue, race_num):
    clean_date = str(target_date).replace("-", "").replace("/", "")
    jcd = VENUE_MAP[venue]
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    base_url = "https://www.boatrace.jp/owpc/pc/race"
    params = f"?rno={int(race_num)}&jcd={jcd}&hd={clean_date}"
    url_list = f"{base_url}/racelist{params}"
    url_before = f"{base_url}/beforeinfo{params}"
    
    boat_data = {}
    env_data = {'風速': 0.0, '波高': 0.0, '会場': int(jcd)}
    
    try:
        res_list = requests.get(url_list, headers=headers)
        res_list.encoding = 'utf-8'
        soup_list = BeautifulSoup(res_list.text, 'html.parser')
        
        tbodies_list = soup_list.find_all('tbody', class_='is-fs12')
        if len(tbodies_list) < 6:
            return None, None, False, f"出走表が見つかりません。URL: {url_list}"
            
        for waku, tbody in enumerate(tbodies_list[:6], 1):
            text_data = tbody.text.replace('\n', ' ').replace('\r', ' ')
            toban_match = re.search(r'(\d{4})\s*/\s*([A-B][1-2])', text_data)
            
            toban = int(toban_match.group(1)) if toban_match else 0
            rank = toban_match.group(2) if toban_match else "B2"
            
            if toban == 0:
                t_m2 = re.search(r'(\d{4})', text_data)
                toban = int(t_m2.group(1)) if t_m2 else 0
            
            recent_results = re.findall(r'\b([1-6])\b', tbody.text)
            numeric_results = [int(x) for x in recent_results if x in ['1','2','3','4','5','6']]
            if numeric_results:
                avg_finish = sum(numeric_results[-3:]) / len(numeric_results[-3:])
            else:
                avg_finish = 3.5
            
            tds = tbody.find_all('td')
            try:
                motor_text = tds[6].text.strip()
                motor = int(re.search(r'^(\d{1,3})', motor_text).group(1))
            except: motor = 0
            
            boat_data[waku] = {
                '登番': toban, 
                '級別': rank, 
                'モーター': motor, 
                '展示': 0.0,
                '今節調子スコア': max(0.5, (6.0 - avg_finish) / 2.0)
            }
            
        res_before = requests.get(url_before, headers=headers)
        res_before.encoding = 'utf-8'
        soup_before = BeautifulSoup(res_before.text, 'html.parser')
        
        before_text = soup_before.text.replace('\n', ' ')
        wind_match = re.search(r'風速.*?(\d+)m', before_text)
        wave_match = re.search(r'波高.*?(\d+)cm', before_text)
        if wind_match: env_data['風速'] = float(wind_match.group(1))
        if wave_match: env_data['波高'] = float(wave_match.group(1))
            
        tbodies_before = soup_before.find_all('tbody', class_='is-fs12')
        if len(tbodies_before) < 6:
            return None, None, False, f"直前情報が未公開です。"
            
        for waku, tbody in enumerate(tbodies_before[:6], 1):
            tenji_match = re.findall(r'[5-7]\.\d{2}', tbody.text)
            if tenji_match:
                boat_data[waku]['展示'] = float(tenji_match[0])
                
        return boat_data, env_data, True, url_before
    except Exception as e:
        return None, None, False, f"読み込み失敗: {str(e)}"

# 3. 予想実行関数
def predict_race_auto_v9(target_date, venue, race_num):
    boat_data, env_data, success, msg = scrape_boatrace_data_v9(target_date, venue, race_num)
    if not success: return msg
    
    input_dict = {'会場': env_data['会場'], '風速': env_data['風速'], '波高': env_data['波高']}
    
    for w in range(1, 7):
        toban = boat_data[w]['登番']
        motor = boat_data[w]['モーター']
        input_dict[f'{w}枠登番'] = toban
        input_dict[f'{w}枠展示'] = boat_data[w]['展示']
        
        c_stats = racer_stat_dict.get((toban, w), {})
        input_dict[f'{w}枠_1着率'] = c_stats.get('rate_1st', 0.0)
        input_dict[f'{w}枠_2連対率'] = c_stats.get('rate_2ren', 0.0)
        input_dict[f'{w}枠_3連対率'] = c_stats.get('rate_3ren', 0.0)
        input_dict[f'{w}枠_平均ST'] = c_stats.get('avg_st', 0.17)
        
        m_stats = motor_stat_dict.get((env_data['会場'], motor), {})
        input_dict[f'{w}枠_M2連対'] = m_stats.get('m_rate_2ren', 0.0)
        input_dict[f'{w}枠_M3連対'] = m_stats.get('m_rate_3ren', 0.0)

    X_pred = pd.DataFrame([input_dict], columns=train_features)
    X_pred['会場'] = X_pred['会場'].astype('category')
    for w in range(1, 7):
        X_pred[f'{w}枠登番'] = X_pred[f'{w}枠登番'].astype('category')
        
    preds_1st = model_1st.predict_proba(X_pred)[0]
    preds_2nd = model_2nd.predict_proba(X_pred)[0]
    preds_3rd = model_3rd.predict_proba(X_pred)[0]
    
    adjusted_1st = np.zeros(6)
    for i in range(6):
        waku = i + 1
        rank = boat_data[waku]['級別']
        rhythm = boat_data[waku]['今節調子スコア']
        c_1st_rate = input_dict[f'{waku}枠_1着率']
        
        r_weight = 1.3 if rank == "A1" else (1.1 if rank == "A2" else (1.0 if rank == "B1" else 0.7))
        technique_bonus = 1.0 + (c_1st_rate * 2.5)
        waku_base = 1.4 if waku == 1 else (1.1 if waku == 2 else (1.0 if waku == 3 else (0.9 if waku == 4 else 0.7)))
        
        adjusted_1st[i] = preds_1st[i] * r_weight * waku_base * technique_bonus * rhythm
        
    preds_1st = adjusted_1st / np.sum(adjusted_1st)

    combos = list(itertools.permutations([0, 1, 2, 3, 4, 5], 3))
    raw_probs = []
    for (r1, r2, r3) in combos:
        if r1 == r2 or r2 == r3 or r1 == r3: continue
        p = preds_1st[r1] * preds_2nd[r2] * preds_3rd[r3]
        raw_probs.append((f"{r1+1}-{r2+1}-{r3+1}", p))
        
    total_p = sum(x[1] for x in raw_probs)
    final_probs = [(c, (p / total_p) * 100) for c, p in raw_probs]
    
    result_text = f"🌐 【決まり手完全連動・進化版】解析完了！\n"
    result_text += f"🌪️ 水面環境: 風速 {env_data['風速']}m / 波高 {env_data['波高']}cm\n\n"
    
    for i in range(1, 7):
        r = boat_data[i]['級別']
        st = input_dict[f'{i}枠_平均ST']
        c_1st = input_dict[f'{i}枠_1着率'] * 100
        result_text += f"{i}枠({r}): 当該コース1着実績(得意技) {c_1st:.1f}% / ST:{st:.2f}\n"
        
    prob_sorted = sorted(final_probs, key=lambda x: x[1], reverse=True)
    
    result_text += f"\n🎯 得意技・展開考慮の推奨買い目（上位6点）\n"
    result_text += "-" * 50 + "\n"
    for i, item in enumerate(prob_sorted[:6]):
        result_text += f"推奨 {i+1}点目: 【{item[0]}】 予想確率: {item[1]:>4.1f}%\n"

    result_text += "-" * 50 + "\n"
    result_text += f"・本命軸: 【{prob_sorted[0][0]}】 (確率 {prob_sorted[0][1]:.1f}%)\n"

    return result_text

# Streamlitの画面デザイン
st.title("🚤 ボートレースAI【決まり手完全連動・進化版】")

today_str = datetime.datetime.now().strftime('%Y%m%d')

col1, col2, col3 = st.columns(3)
with col1:
    target_date = st.text_input("日付 (例: 20261103)", value=today_str)
with col2:
    venue = st.selectbox("会場", options=list(VENUE_MAP.keys()), index=9)
with col3:
    race_num = st.number_input("レース番号", min_value=1, max_value=12, value=5, step=1)

if st.button("⚡ 得意技を反映して予想！", type="primary"):
    with st.spinner("選手の決まり手実績・コース適性を分析中..."):
        res = predict_race_auto_v9(target_date, venue, race_num)
        st.text_area("AIの進化解析結果", value=res, height=450)

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
st.set_page_config(page_title="ボートレースAI【外枠リスク評価・完全版】", layout="centered")

# 1. AIと各種辞書の読み込み
@st.cache_resource
def load_brain():
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

# 2. スクレイピング関数
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
                '今節調子スコア': max(0.5, (6.0 - avg_finish) / 2.5)
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

# 3. 予想実行関数（★外枠での大敗傾向＝3連対率が低い選手の評価を厳しく下げるロジックを搭載）
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
    
    # 【1着の適正化：外枠リスク評価の連動】
    adjusted_1st = np.zeros(6)
    for i in range(6):
        waku = i + 1
        rank = boat_data[waku]['級別']
        rhythm = boat_data[waku]['今節調子スコア']
        t_rate_3ren = input_dict[f'{waku}枠_3連対率'] # そのコースでの3連対率
        
        r_weight = 1.7 if rank == "A1" else (1.2 if rank == "A2" else (0.9 if rank == "B1" else 0.4))
        
        # 4〜6コース（外枠）で、もともと3連対率が低い（＝4〜6着が多い）選手は評価をさらに大きく減点する
        if waku >= 4:
            # 3連対率が20%未満なら容赦なく評価を下げる
            outer_risk_penalty = max(0.3, t_rate_3ren * 3.0) 
        else:
            outer_risk_penalty = 1.0
            
        c_weight = 2

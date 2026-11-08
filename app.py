import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import datetime
import re
import itertools
import time
import plotly.express as px
import concurrent.futures 
import xgboost as xgb

st.set_page_config(page_title="ボートレースAI【v12 展開読み＋アンサンブル完全体】", layout="wide")

# 1. 究極のAI（v12 アンサンブル脳）の読み込み
@st.cache_resource
def load_brain():
    with open('boat_ai_brain_v12_ensemble.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    return (
        saved_data['model_lgb'],
        saved_data['model_xgb'],
        saved_data['features'],
        saved_data['racer_stats']
    )

try:
    model_lgb, model_xgb, train_features, racer_stats = load_brain()
except FileNotFoundError:
    st.error("⚠️ `boat_ai_brain_v12_ensemble.pkl` が見つかりません。学習スクリプトを実行してください。")
    st.stop()

VENUE_MAP = {
    "桐生": "01", "戸田": "02", "江戸川": "03", "平和島": "04", "多摩川": "05", "浜名湖": "06",
    "蒲郡": "07", "常滑": "08", "津": "09", "三国": "10", "びわこ": "11", "住之江": "12",
    "尼崎": "13", "鳴門": "14", "丸亀": "15", "児島": "16", "宮島": "17", "徳山": "18",
    "下関": "19", "若松": "20", "芦屋": "21", "福岡": "22", "唐津": "23", "大村": "24"
}

session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[403, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20))
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
})

def scrape_boatrace_data_v12(target_date, venue, race_num):
    clean_date = str(target_date).replace("-", "").replace("/", "")
    jcd = VENUE_MAP[venue]
    
    url_list = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={int(race_num)}&jcd={jcd}&hd={clean_date}"
    url_before = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={int(race_num)}&jcd={jcd}&hd={clean_date}"
    
    boat_data = {}
    env_data = {'風速': 0.0, '波高': 0.0}
    
    try:
        res_list = session.get(url_list, timeout=15)
        res_list.encoding = 'utf-8'
        if res_list.status_code != 200: return None, None, False, "通信ブロック"

        soup_list = BeautifulSoup(res_list.text, 'html.parser')
        tbodies_list = soup_list.find_all('tbody', class_='is-fs12')
        if len(tbodies_list) < 6: return None, None, False, "データ未公開"
            
        for waku, tbody in enumerate(tbodies_list[:6], 1):
            text_data = tbody.text.replace('\n', ' ').replace('\r', ' ')
            toban_match = re.search(r'(\d{4})\s*/\s*([A-B][1-2])', text_data)
            toban = int(toban_match.group(1)) if toban_match else 0
            try: motor = int(re.findall(r'\b(\d{1,3})\b', tbody.text)[-6])
            except: motor = 0
            boat_data[waku] = {'登番': toban, 'モーター': motor, '展示': 0.0, '進入コース': waku, '平均ST': 0.17}
            
        time.sleep(0.5) 
        res_before = session.get(url_before, timeout=15)
        res_before.encoding = 'utf-8'
        soup_before = BeautifulSoup(res_before.text, 'html.parser')
        before_text = soup_before.text.replace('\n', ' ')
        
        wind_match = re.search(r'風速.*?(\d+)m', before_text)
        wave_match = re.search(r'波高.*?(\d+)cm', before_text)
        if wind_match: env_data['風速'] = float(wind_match.group(1))
        if wave_match: env_data['波高'] = float(wave_match.group(1))
            
        tbodies_before = soup_before.find_all('tbody', class_='is-fs12')
        for waku, tbody in enumerate(tbodies_before[:6], 1):
            t_text = tbody.text.replace('\n', ' ')
            floats = re.findall(r'[-]?\d\.\d{1,2}', tbody.text)
            if len(floats) >= 4:
                boat_data[waku]['展示'] = float(floats[0]) if float(floats[0]) > 5.0 else 0.0
            st_match = re.findall(r'0\.\d{2}', t_text)
            if st_match:
                boat_data[waku]['平均ST'] = float(st_match[0]) 

        course_match = re.findall(r'(\d)コース', before_text)
        if len(course_match) >= 6:
            for i, c in enumerate(course_match[:6]):
                actual_waku = int(c)
                if 1 <= actual_waku <= 6: boat_data[actual_waku]['進入コース'] = i + 1
                    
        return boat_data, env_data, True, ""
    except Exception as e:
        return None, None, False, f"エラー: {str(e)}"

def fetch_actual_result(target_date, jcd, race_num):
    clean_date = str(target_date).replace("-", "").replace("/", "")
    url = f"https://www.boatrace.jp/owpc/pc/race/raceresult?rno={int(race_num)}&jcd={jcd}&hd={clean_date}"
    try:
        res = session.get(url, timeout=15)
        res.encoding = 'utf-8'
        if res.status_code != 200: return "通信ブロック", 0
        soup = BeautifulSoup(res.text, 'html.parser')
        trs = soup.find_all('tr')
        for tr in trs:
            if '3連単' in tr.text:
                tds = tr.find_all('td')
                if len(tds) >= 3:
                    raw_text = tds[1].text.strip()
                    digits = re.findall(r'[1-6]', raw_text)
                    result_str = f"{digits[0]}-{digits[1]}-{digits[2]}" if len(digits) >= 3 else raw_text
                    payout_val = max([int(n) for n in re.findall(r'\d+', tds[2].text.replace(',', '')) if int(n) >= 100], default=0)
                    return result_str, payout_val
        return "データなし(未発走)", 0
    except:
        return "通信エラー", 0

def predict_race_auto_v12(target_date, venue, race_num, is_backtest=False):
    boat_data, env_data, success, msg = scrape_boatrace_data_v12(target_date, venue, race_num)
    if not success: return {"error": msg}
    
    # 🔥 v12専用: 相対特徴量（展開読み）とマスタデータの計算
    st_vals = [boat_data[w]['平均ST'] for w in range(1, 7)]
    tenji_vals = [boat_data[w]['展示'] for w in range(1, 7) if boat_data[w]['展示'] > 0]
    tenji_mean = np.mean(tenji_vals) if tenji_vals else 0.0

    input_list = []
    for w in range(1, 7):
        toban = boat_data[w]['登番']
        win_rate = racer_stats['wins'].get(toban, 0) / max(1, racer_stats['races'].get(toban, 1))
        inside_st = st_vals[w-2] if w > 1 else st_vals[w-1]
        st_diff_inside = boat_data[w]['平均ST'] - inside_st
        tenji_diff_avg = boat_data[w]['展示'] - tenji_mean if boat_data[w]['展示'] > 0 else 0.0
        
        input_dict = {
            '枠番': w, '登番': toban, 'モーター': boat_data[w]['モーター'],
            '展示': boat_data[w]['展示'], '進入コース': boat_data[w]['進入コース'],
            '平均ST': boat_data[w]['平均ST'], '風速': env_data['風速'], '波高': env_data['波高'],
            '1着率': win_rate, '内側ST差': st_diff_inside, '展示平均差': tenji_diff_avg
        }
        input_list.append({col: input_dict.get(col, 0.0) for col in train_features})

    X_pred = pd.DataFrame(input_list)[train_features]
    
    for col in X_pred.columns:
        if str(X_pred[col].dtype) == 'object': X_pred[col] = pd.to_numeric(X_pred[col], errors='coerce').fillna(0.0)

    # 🔥 v12専用: LightGBMとXGBoostのアンサンブル（多数決）処理
    # 1. LightGBMの確率算出
    lgb_scores = model_lgb.predict(X_pred)
    z_lgb = (lgb_scores - np.mean(lgb_scores)) / (np.std(lgb_scores) + 1e-9)
    prob_lgb = np.exp(z_lgb * 1.5) / np.sum(np.exp(z_lgb * 1.5))
    
    # 2. XGBoostの確率算出
    xgb_scores = model_xgb.predict(X_pred)
    z_xgb = (xgb_scores - np.mean(xgb_scores)) / (np.std(xgb_scores) + 1e-9)
    prob_xgb = np.exp(z_xgb * 1.5) / np.sum(np.exp(z_xgb * 1.5))
    
    # 3. 2つのAIの意見を合体（平均化）
    win_probs = (prob_lgb + prob_xgb) / 2.0
    
    # XGBとLGBの意見が割れている（分散が大きい）場合は確信度を下げるペナルティ
    disagreement_penalty = np.std([prob_lgb[0], prob_xgb[0]]) * 0.5 
    
    top_indices = np.argsort(win_probs)[::-1][:4]
    t1, t2, t3, t4 = [i + 1 for i in top_indices]
    top1_win_prob = (win_probs[top_indices[0]] - disagreement_penalty) * 100.0

    combos = list(itertools.permutations([0, 1, 2, 3, 4, 5], 3))
    raw_probs = []
    for (r1, r2, r3) in combos:
        p1 = win_probs[r1]
        p2 = win_probs[r2] / max(0.0001, 1.0 - win_probs[r1])
        p3 = win_probs[r3] / max(0.0001, 1.0 - win_probs[r1] - win_probs[r2])
        raw_probs.append((f"{r1+1}-{r2+1}-{r3+1}", p1 * p2 * p3 * 100))
        
    prob_sorted = sorted(raw_probs, key=lambda x: x[1], reverse=True)
    
    bets = []
    if top1_win_prob >= 40.0: 
        bet_type_name = "💎 究極・鉄板（激絞り4点）"
        bets = [f"{t1}-{t2}-{t3}", f"{t1}-{t2}-{t4}", f"{t1}-{t3}-{t2}", f"{t1}-{t3}-{t4}"]
    else: 
        bet_type_name = "🎯 本命＆逆転（フォーメーション8点）"
        for second in [t2, t3, t4]:
            for third in [t2, t3, t4]:
                if second != third: bets.append(f"{t1}-{second}-{third}")
        for third in [t3, t4]:
            bets.append(f"{t2}-{t1}-{third}")

    is_ken = (top1_win_prob < 25.0) or (env_data['風速'] >= 6.0)
    if is_ken:
        bet_type_name = f"⚠️ 見送り推奨 (AI意見割れ・条件悪化)"

    is_easy_hit = (top1_win_prob >= 40.0) and not is_ken

    if is_backtest:
        return {
            "race_num": f"{race_num}R", "本命": prob_sorted[0][0], "total_bets": bets,
            "bet_type": bet_type_name, "is_ken": is_ken, "is_easy_hit": is_easy_hit, "top1_win_prob": top1_win_prob
        }
    
    result_text = f"🌐 【v12 完全体アンサンブルAI】解析完了！\n"
    result_text += f"🌪️ 水面環境: {venue} / 風速 {env_data['風速']}m\n"
    result_text += f"🎯 アンサンブル1位抜け確率: {top1_win_prob

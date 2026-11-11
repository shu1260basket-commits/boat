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
import time
import concurrent.futures 

st.set_page_config(page_title="ボートレースAI【v22 アルティメット・スナイパー】", layout="wide")

@st.cache_resource
def load_brain():
    with open('boat_ai_brain_v22.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    return saved_data

try:
    brain = load_brain()
    model_lgb = brain['model_lgb']
    train_features = brain['features']
    racer_course_stats = brain['racer_course_stats']
    racer_venue_stats = brain['racer_venue_stats']
    motor_stats = brain['motor_stats']
    racer_names = brain.get('racer_names', {}) 
    COURSE_AVG_1ST = brain['course_avg_1st']
    id_to_combo = brain['id_to_combo']
except Exception as e:
    st.error(f"⚠️ `boat_ai_brain_v22.pkl` の読み込みエラー: {e}")
    st.stop()

VENUE_MAP = {
    "桐生": 1, "戸田": 2, "江戸川": 3, "平和島": 4, "多摩川": 5, "浜名湖": 6,
    "蒲郡": 7, "常滑": 8, "津": 9, "三国": 10, "びわこ": 11, "住之江": 12,
    "尼崎": 13, "鳴門": 14, "丸亀": 15, "児島": 16, "宮島": 17, "徳山": 18,
    "下関": 19, "若松": 20, "芦屋": 21, "福岡": 22, "唐津": 23, "大村": 24
}

session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[403, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20))
session.headers.update({'User-Agent': 'Mozilla/5.0'})

def scrape_boatrace_data(target_date, venue, race_num):
    clean_date = str(target_date).replace("-", "").replace("/", "")
    jcd_str = str(VENUE_MAP[venue]).zfill(2)
    url_list = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={int(race_num)}&jcd={jcd_str}&hd={clean_date}"
    url_before = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={int(race_num)}&jcd={jcd_str}&hd={clean_date}"
    
    boat_data = {}
    env_data = {'風速': 0.0, '波高': 0.0}
    
    try:
        res_list = session.get(url_list, timeout=15)
        res_list.encoding = 'utf-8'
        if res_list.status_code != 200: return None, None, False, f"通信ブロック ({res_list.status_code})"

        soup_list = BeautifulSoup(res_list.text, 'html.parser')
        tbodies_list = soup_list.find_all('tbody', class_='is-fs12')
        if len(tbodies_list) < 6: return None, None, False, "休催日 または データ未公開"
            
        for waku, tbody in enumerate(tbodies_list[:6], 1):
            text_data = tbody.text.replace('\n', ' ')
            toban = int(re.search(r'(\d{4})', text_data).group(1)) if re.search(r'(\d{4})', text_data) else 0
            try: motor = int(re.findall(r'\b(\d{1,3})\b', tbody.text)[-6])
            except: motor = 0
            boat_data[waku] = {'登番': toban, 'モーター': motor, '展示': 0.0, '進入コース': waku, '平均ST': 0.17}
            
        time.sleep(0.5) 
        res_before = session.get(url_before, timeout=15)
        res_before.encoding = 'utf-8'
        soup_before = BeautifulSoup(res_before.text, 'html.parser')
        before_text = soup_before.text.replace('\n', ' ')
        
        wind_match = re.search(r'風.*?(\d+)m', before_text)
        wave_match = re.search(r'波.*?(\d+)cm', before_text)
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
        return None, None, False, f"通信/解析エラー: {str(e)}"

def predict_race_auto_v22(target_date, venue, race_num):
    boat_data, env_data, success, msg = scrape_boatrace_data(target_date, venue, race_num)
    if not success: return {"error": msg}
    
    jcd = int(VENUE_MAP[venue])
    
    st_array = [boat_data[w]['平均ST'] for w in range(1, 7)]
    tenji_array = [boat_data[w]['展示'] for w in range(1, 7)]
    tenji_mean = np.mean(tenji_array)

    # 1レース分の特徴量を横持ちで生成
    race_feat = {
        'jcd': jcd,
        'wind': env_data['風速'],
        'wave': env_data['波高']
    }
    
    for i in range(1, 7):
        toban = boat_data[i]['登番']
        course = boat_data[i]['進入コース']
        
        total_runs, total_wins, total_value_added = 0, 0, 0.0
        for c in range(1, 7):
            c_st = racer_course_stats.get((toban, c), {'runs':0, '1st':0, '2nd':0, '3rd':0, '4th6th':0})
            total_runs += c_st['runs']
            total_value_added += c_st['1st'] - (c_st['runs'] * COURSE_AVG_1ST.get(c, 0.0))

        true_skill = total_value_added / max(1, total_runs)

        c_stats = racer_course_stats.get((toban, course), {'runs':1, '1st':0, '2nd':0, '3rd':0})
        m_stats = motor_stats.get((jcd, boat_data[i]['モーター']), {'runs':1, '1st':0, '2nd':0})
        
        inside_st = st_array[i-2] if i > 1 else st_array[i-1]
        runs_c = max(1, c_stats['runs'])
        
        race_feat[f'w{i}_course'] = course
        race_feat[f'w{i}_win_rate'] = c_stats['1st'] / runs_c
        race_feat[f'w{i}_sanren_rate'] = (c_stats['1st'] + c_stats['2nd'] + c_stats['3rd']) / runs_c
        race_feat[f'w{i}_skill'] = true_skill
        race_feat[f'w{i}_motor'] = (m_stats['1st'] + m_stats['2nd']) / max(1, m_stats['runs'])
        race_feat[f'w{i}_st_dev'] = boat_data[i]['平均ST'] - inside_st
        race_feat[f'w{i}_tenji_dev'] = boat_data[i]['展示'] - tenji_mean
        
        # プロファイル表示用
        boat_data[i]['true_skill'] = true_skill
        boat_data[i]['sanren_rate'] = (c_stats['1st'] + c_stats['2nd'] + c_stats['3rd']) / runs_c * 100.0

    ordered_dict = {col: race_feat.get(col, 0.0) for col in train_features}
    X_pred = pd.DataFrame([ordered_dict])
    
    # 🔥 AIに「120通りの買い目」の確率を一撃で出力させる！！
    probs = model_lgb.predict_proba(X_pred)[0]
    
    raw_probs = []
    for cls_id, p in enumerate(probs):
        combo = id_to_combo.get(cls_id, "")
        if combo:
            raw_probs.append((combo, p))
            
    prob_sorted = sorted([(c, p * 100) for c, p in raw_probs], key=lambda x: x[1], reverse=True)
    
    top1_3t_prob = prob_sorted[0][1]
    bets = [x[0] for x in prob_sorted[:4]]
    
    is_ken = False
    ken_reason = ""
    if top1_3t_prob < 10.0:
        is_ken = True
        ken_reason = "展開が混戦すぎる（オッズ割れ・トリガミの危険）"
        bet_type_name = "🎯 展開狙い（フォーメーション8点）"
        bets = [x[0] for x in prob_sorted[:8]]
    else:
        bet_type_name = "💎 超鉄板（激絞り4点）"

    if is_ken:
        bet_type_name = f"⚠️ 見送り推奨 (仮の買い目: {bet_type_name})"
    
    result_text = f"🌐 【v22 アルティメット・スナイパー】解析完了！\n"
    result_text += f"🎯 AI推奨 3連単1番人気 信頼度: {top1_3t_prob:.1f}%\n\n"
    
    if is_ken:
        result_text += f"⚠️ 【AI判定: ケン（見送り）】\n⛔ 理由: {ken_reason}\n\n"
    else:
        result_text += f"✨ 【AI投資判定】 買い目完全にロックオン！\n\n"
    
    result_text += "📊 各艇の実力・相性データ\n"
    for i in range(1, 7):
        t_name = racer_names.get(boat_data[i]['登番'], "")
        c_actual = boat_data[i]['進入コース']
        result_text += f"{i}枠 {t_name}: 【真の実力: {boat_data[i]['true_skill']:+.3f} / {c_actual}ｺｰｽ3連対率: {boat_data[i]['sanren_rate']:.1f}%】\n"
        
    result_text += f"\n🎯 【AI直結・120通り解析】3連単 上位8点予想\n"
    result_text += "-" * 50 + "\n"
    for i in range(min(8, len(prob_sorted))):
        result_text += f"第{i+1}推奨: 【{prob_sorted[i][0]}】 (確率 {prob_sorted[i][1]:.1f}%)\n"
        
    result_text += f"\n🎯 激絞り買い目指示\n"
    result_text += "-" * 50 + "\n"
    result_text += f"👉 展開予想買い目（{len(bets)}点）: {', '.join(bets)}\n"
    result_text += "-" * 50 + "\n"

    return {"text": result_text, "boat_data": boat_data}

st.title("🚤 ボートレースAI【v22 アルティメット・スナイパー】")

today_str = datetime.datetime.now().strftime('%Y%m%d')

st.write("AIが『120通りの買い目』を直接予測！スジの矛盾や計算バグを完全に排除した最終形態です。")
col1, col2, col3 = st.columns(3)
with col1: target_date = st.text_input("日付 (例: 20261105)", value=today_str, key="date1")
with col2: venue = st.selectbox("会場", options=list(VENUE_MAP.keys()), index=9, key="venue1")
with col3: race_num = st.number_input("レース番号", min_value=1, max_value=12, value=5, step=1, key="race1")

if st.button("⚡ v22 アルティメットAI予想！", type="primary"):
    with st.spinner("120通りの展開を並列計算中..."):
        res_dict = predict_race_auto_v22(target_date, venue, race_num)
        if "error" in res_dict:
            st.error(f"【予想失敗】 {res_dict['error']}")
        else:
            st.text_area("AIのスコア解析結果", value=res_dict["text"], height=550)

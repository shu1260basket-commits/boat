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

st.set_page_config(page_title="ボートレースAI【v23 フル覚醒スナイパー】", layout="wide")

@st.cache_resource
def load_brain():
    with open('boat_ai_brain_v23.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    return saved_data

try:
    brain = load_brain()
    model_lgb = brain['model_lgb']
    model_xgb = brain['model_xgb']
    train_features = brain['features']
    racer_course_stats = brain['racer_course_stats']
    racer_venue_stats = brain['racer_venue_stats']
    motor_stats = brain['motor_stats']
    racer_names = brain.get('racer_names', {}) 
    suji_stats = brain['suji_stats']
    COURSE_AVG_1ST = brain['course_avg_1st']
except Exception as e:
    st.error(f"⚠️ `boat_ai_brain_v23.pkl` の読み込みエラー: {e}")
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
    env_data = {'風速': 0.0, '波高': 0.0, '風向': 0}
    
    try:
        # 出走表（体重、F数）
        res_list = session.get(url_list, timeout=15)
        res_list.encoding = 'utf-8'
        soup_list = BeautifulSoup(res_list.text, 'html.parser')
        tbodies_list = soup_list.find_all('tbody', class_='is-fs12')
            
        for waku, tbody in enumerate(tbodies_list[:6], 1):
            text_data = tbody.text.replace('\n', ' ')
            toban = int(re.search(r'(\d{4})', text_data).group(1)) if re.search(r'(\d{4})', text_data) else 0
            
            # 体重とF数
            weight = float(re.search(r'(\d{2}\.\d)kg', text_data).group(1)) if re.search(r'(\d{2}\.\d)kg', text_data) else 51.0
            f_count = int(re.search(r'F(\d)', text_data).group(1)) if re.search(r'F(\d)', text_data) else 0
            
            try: motor = int(re.findall(r'\b(\d{1,3})\b', tbody.text)[-6])
            except: motor = 0
            boat_data[waku] = {'登番': toban, 'モーター': motor, '体重': weight, 'F持ち': f_count, '展示': 0.0, '進入コース': waku, '平均ST': 0.17, '展示ST': 0.17}
            
        time.sleep(0.5) 
        # 直前情報（展示タイム、展示ST、風向）
        res_before = session.get(url_before, timeout=15)
        res_before.encoding = 'utf-8'
        soup_before = BeautifulSoup(res_before.text, 'html.parser')
        before_text = soup_before.text.replace('\n', ' ')
        
        wind_match = re.search(r'風.*?(\d+)m', before_text)
        wave_match = re.search(r'波.*?(\d+)cm', before_text)
        if wind_match: env_data['風速'] = float(wind_match.group(1))
        if wave_match: env_data['波高'] = float(wave_match.group(1))
        
        # 簡易風向（向かい風=1, 追い風=2, 横風=3, 無風=0）画像クラス等から取得がベストだがテキストベースでフォールバック
        if '向かい風' in before_text: env_data['風向'] = 1
        elif '追い風' in before_text: env_data['風向'] = 2
        else: env_data['風向'] = 0
            
        tbodies_before = soup_before.find_all('tbody', class_='is-fs12')
        for waku, tbody in enumerate(tbodies_before[:6], 1):
            t_text = tbody.text.replace('\n', ' ')
            floats = re.findall(r'[-]?\d\.\d{1,2}', tbody.text)
            if len(floats) >= 4:
                boat_data[waku]['展示'] = float(floats[0]) if float(floats[0]) > 5.0 else 0.0
            
            # 展示STの取得（beforeinfoのテキスト構造に依存）
            st_matches = re.findall(r'(F?0\.\d{2})', t_text)
            if len(st_matches) >= 2:
                # 平均STと展示STが混在するため、後者を展示STとする
                try: boat_data[waku]['展示ST'] = float(st_matches[-1].replace('F', '-'))
                except: pass

        # スタート展示の進入コースを本番コースとする
        course_match = re.findall(r'(\d)コース', before_text)
        if len(course_match) >= 6:
            for i, c in enumerate(course_match[:6]):
                actual_waku = int(c)
                if 1 <= actual_waku <= 6: boat_data[actual_waku]['進入コース'] = i + 1
                    
        return boat_data, env_data, True, ""
    except Exception as e:
        return None, None, False, f"通信/解析エラー: {str(e)}"

def predict_race_auto_v23(target_date, venue, race_num):
    boat_data, env_data, success, msg = scrape_boatrace_data(target_date, venue, race_num)
    if not success: return {"error": msg}
    
    jcd = int(VENUE_MAP[venue])
    st_array = [boat_data[w]['平均ST'] for w in range(1, 7)]
    tenji_array = [boat_data[w]['展示'] for w in range(1, 7)]
    tenji_mean = np.mean(tenji_array)
    course_map = {w-1: boat_data[w]['進入コース'] for w in range(1, 7)}

    input_list = []
    for i in range(6):
        waku = i + 1
        toban = boat_data[waku]['登番']
        motor = boat_data[waku]['モーター']
        course = boat_data[waku]['進入コース']
        
        total_runs_all, total_value = 0, 0.0
        for c in range(1, 7):
            c_st = racer_course_stats.get((toban, c), {'runs':0, '1st':0, '2nd':0, '3rd':0})
            total_runs_all += c_st['runs']
            total_value += c_st['1st'] - (c_st['runs'] * COURSE_AVG_1ST.get(c, 0.0))

        true_skill = total_value / max(1, total_runs_all)
        c_stats = racer_course_stats.get((toban, course), {'runs':1, '1st':0, '2nd':0, '3rd':0})
        v_stats = racer_venue_stats.get((toban, jcd), {'runs':1, '1st':0, '2nd':0})
        m_stats = motor_stats.get((jcd, motor), {'runs':1, '1st':0, '2nd':0})
        
        inside_st = st_array[i-1] if i > 0 else st_array[i]
        runs_c = max(1, c_stats['runs'])
        win_rate = c_stats['1st'] / runs_c
        sanren_rate = (c_stats['1st'] + c_stats['2nd'] + c_stats['3rd']) / runs_c

        input_dict = {
            '会場ID': jcd,
            'レース番号': race_num,
            '枠番': waku, '進入コース': course, 
            '風速': env_data['風速'], '風向': env_data['風向'], '波高': env_data['波高'],
            'コース1着率': win_rate,
            'コース3連対率': sanren_rate,
            '当地1着率': v_stats['1st'] / max(1, v_stats['runs']),
            'モーター2連対率': (m_stats['1st'] + m_stats['2nd']) / max(1, m_stats['runs']),
            '真の実力値': true_skill,
            '内側ST差': boat_data[waku]['平均ST'] - inside_st, 
            '展示平均差': boat_data[waku]['展示'] - tenji_mean,
            'F持ち': boat_data[waku]['F持ち'],
            '体重': boat_data[waku]['体重'],
            '展示ST': boat_data[waku]['展示ST']
        }
        ordered_dict = {col: input_dict.get(col, 0.0) for col in train_features}
        input_list.append(ordered_dict)
        
        boat_data[waku]['win_rate'] = win_rate
        boat_data[waku]['true_skill'] = true_skill

    X_pred = pd.DataFrame(input_list)[train_features]
    if '会場ID' in X_pred.columns: X_pred['会場ID'] = X_pred['会場ID'].astype(int)
    if '風向' in X_pred.columns: X_pred['風向'] = X_pred['風向'].astype(int)
    
    lgb_probs = model_lgb.predict_proba(X_pred)[:, 1]
    xgb_probs = model_xgb.predict_proba(X_pred)[:, 1]
    ensemble_scores = (lgb_probs + xgb_probs) / 2.0
    
    score_details = {}
    for i in range(6):
        waku = i + 1
        t_name = racer_names.get(boat_data[waku]['登番'], "")
        f_str = f" [F{boat_data[waku]['F持ち']}]" if boat_data[waku]['F持ち'] > 0 else ""
        c_actual = boat_data[waku]['進入コース']
        score_details[waku] = f"舟券内AI確率: {ensemble_scores[i]*100:.1f}% (進入{c_actual} / 展ST {boat_data[waku]['展示ST']}){f_str}"
    
    p1_weights = []
    for i in range(1, 7):
        base_win = max(0.001, boat_data[i]['win_rate'])
        skill = boat_data[i]['true_skill']
        p1_weights.append(base_win * (1.0 + max(0, skill * 1.5)))
    p1_probs = np.array(p1_weights) / sum(p1_weights)

    potentials = []
    for i in range(1, 7):
        ai_prob = ensemble_scores[i-1] 
        skill = boat_data[i]['true_skill']
        potentials.append(ai_prob * (1.0 + max(0, skill * 1.5)))
        
    combos = list(itertools.permutations([0, 1, 2, 3, 4, 5], 3))
    raw_probs = []
    
    for (r1, r2, r3) in combos:
        c1, c2, c3 = course_map[r1], course_map[r2], course_map[r3]
        P1 = p1_probs[r1]
        
        w2_list = []
        for cand2 in range(6):
            if cand2 == r1: w2_list.append(0)
            else:
                c_cand2 = course_map[cand2]
                suji2 = suji_stats['12_matrix'].get((c1, c_cand2), 0) / max(1, suji_stats['1st_count'].get(c1, 1))
                w2_list.append(potentials[cand2] * (1.0 + suji2 * 2.0))
        P2 = w2_list[r2] / max(0.0001, sum(w2_list))
        
        w3_list = []
        for cand3 in range(6):
            if cand3 in (r1, r2): w3_list.append(0)
            else:
                c_cand3 = course_map[cand3]
                suji3 = suji_stats['123_matrix'].get((c1, c2, c_cand3), 0) / max(1, suji_stats['12_matrix'].get((c1, c2), 1))
                w3_list.append(potentials[cand3] * (1.0 + suji3 * 2.0))
        P3 = w3_list[r3] / max(0.0001, sum(w3_list))
        
        score = P1 * P2 * P3
        raw_probs.append((f"{r1+1}-{r2+1}-{r3+1}", score))
        
    prob_sorted = sorted([(c, s * 100) for c, s in raw_probs], key=lambda x: x[1], reverse=True)
    
    top1_3t_prob = prob_sorted[0][1]
    bets = [x[0] for x in prob_sorted[:4]]
    
    result_text = f"🌐 【v23 展開完全認知スナイパー】解析完了！\n"
    wind_txt = "向かい風" if env_data['風向'] == 1 else "追い風" if env_data['風向'] == 2 else "横風/無風"
    result_text += f"🌪️ 水面環境: {venue}{race_num}R / {wind_txt} {env_data['風速']}m\n"
    
    if top1_3t_prob < 15.0:
        result_text += f"⚠️ 【AI判定: ケン（見送り）】\n⛔ 理由: 展開が混戦すぎる（オッズ割れ・トリガミの危険）\n\n"
    else: result_text += f"✨ 【AI投資判定】 最強のスイートスポット！勝負可能！\n\n"
    
    result_text += "📊 各艇のAIスコア詳細（展開加味）\n"
    for i in range(1, 7):
        detail = score_details[i]
        t_name = racer_names.get(boat_data[i]['登番'], "")
        result_text += f"{i}枠 {t_name}: 【{detail}】\n"
        
    result_text += f"\n🎯 3連単 上位8点予想\n"
    result_text += "-" * 50 + "\n"
    for i in range(min(8, len(prob_sorted))):
        result_text += f"第{i+1}推奨: 【{prob_sorted[i][0]}】 (確率 {prob_sorted[i][1]:.1f}%)\n"

    return {"text": result_text, "boat_data": boat_data}

st.title("🚤 ボートレースAI【v23 フル覚醒スナイパー】")
today_str = datetime.datetime.now().strftime('%Y%m%d')

st.write("風向き、体重、F持ち、展示ST、進入予想をすべて組み込んだ完全体AIです。")
col1, col2, col3 = st.columns(3)
with col1: target_date = st.text_input("日付 (例: 20261105)", value=today_str, key="date1")
with col2: venue = st.selectbox("会場", options=list(VENUE_MAP.keys()), index=9, key="venue1")
with col3: race_num = st.number_input("レース番号", min_value=1, max_value=12, value=5, step=1, key="race1")

if st.button("⚡ v23 フル覚醒AI予想！", type="primary"):
    with st.spinner("展開データ（風向/F持ち/展示ST/進入）をスクレイピング・解析中..."):
        res_dict = predict_race_auto_v23(target_date, venue, race_num)
        if "error" in res_dict:
            st.error(f"【予想失敗】 {res_dict['error']}")
        else:
            st.text_area("AIのスコア解析結果", value=res_dict["text"], height=550)

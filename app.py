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

# ページの基本設定
st.set_page_config(page_title="ボートレースAI【プロ舟券師ロジック・v10究極完全版】", layout="wide")

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

VENUE_BRANCH_MAP = {
    "桐生": "群馬", "戸田": "埼玉", "江戸川": "東京", "平和島": "東京", "多摩川": "東京",
    "浜名湖": "静岡", "蒲郡": "愛知", "常滑": "愛知", "津": "三重",
    "三国": "福井", "びわこ": "滋賀", "住之江": "大阪", "尼崎": "兵庫",
    "鳴門": "徳島", "丸亀": "香川", "児島": "岡山", "宮島": "広島", "徳山": "山口", "下関": "山口",
    "若松": "福岡", "芦屋": "福岡", "福岡": "福岡", "唐津": "佐賀", "大村": "長崎"
}

VENUE_1ST_BONUS = {
    "大村": 15, "芦屋": 10, "徳山": 10, "下関": 5, "住之江": 5,
    "戸田": -15, "江戸川": -10, "平和島": -10, "鳴門": -5
}

session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
session.mount('https://', adapter)
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

# 2. 全部入りスクレイピング＆相対評価解析エンジン
def scrape_boatrace_data_v10(target_date, venue, race_num):
    clean_date = str(target_date).replace("-", "").replace("/", "")
    jcd = VENUE_MAP[venue]
    
    base_url = "https://www.boatrace.jp/owpc/pc/race"
    params = f"?rno={int(race_num)}&jcd={jcd}&hd={clean_date}"
    url_list = f"{base_url}/racelist{params}"
    url_before = f"{base_url}/beforeinfo{params}"
    
    boat_data = {}
    env_data = {'風速': 0.0, '波高': 0.0, '会場': int(jcd), '潮': '普通'}
    
    try:
        res_list = session.get(url_list, timeout=15)
        res_list.encoding = 'utf-8'
        if res_list.status_code != 200:
            return None, None, False, f"通信ブロック ({res_list.status_code})"

        soup_list = BeautifulSoup(res_list.text, 'html.parser')
        tbodies_list = soup_list.find_all('tbody', class_='is-fs12')
        if len(tbodies_list) < 6:
            return None, None, False, "休催日 または データ未公開"
            
        for waku, tbody in enumerate(tbodies_list[:6], 1):
            text_data = tbody.text.replace('\n', ' ').replace('\r', ' ')
            toban_match = re.search(r'(\d{4})\s*/\s*([A-B][1-2])', text_data)
            toban = int(toban_match.group(1)) if toban_match else 0
            rank = toban_match.group(2) if toban_match else "B2"
            
            f_match = re.search(r'F(\d)', text_data)
            f_count = int(f_match.group(1)) if f_match else 0
            
            branch_match = re.search(r'(群馬|埼玉|東京|静岡|愛知|三重|福井|滋賀|大阪|兵庫|徳島|香川|岡山|広島|山口|福岡|佐賀|長崎)', text_data)
            branch = branch_match.group(1) if branch_match else "不明"
            
            recent_results = re.findall(r'\b([1-6])\b', tbody.text)
            numeric_results = [int(x) for x in recent_results if x in ['1','2','3','4','5','6']]
            avg_finish = sum(numeric_results[-3:]) / len(numeric_results[-3:]) if numeric_results else 3.5
            
            # ③ 得意戦法（まくり・差しの傾向）をテキストから簡易検出
            tactics = "差し"
            if "まくり" in text_data: tactics = "まくり"
            elif "逃げ" in text_data: tactics = "逃げ"
            
            try:
                motor_match = re.findall(r'\b(\d{1,3})\b', tbody.text)
                motor = int(motor_match[-6]) if len(motor_match) > 6 else 0
            except: motor = 0
            
            boat_data[waku] = {
                '登番': toban, '級別': rank, 'モーター': motor, 
                '展示': 0.0, '一周': 0.0, '回り足': 0.0, 'チルト': 0.0, 
                'F': f_count, '支部': branch, '得意戦法': tactics, '部品交換': False,
                '進入コース': waku, '直近STブレ': 0.0,
                '今節調子スコア': max(0.5, (6.0 - avg_finish) / 2.0)
            }
            
        time.sleep(0.5) 
        
        res_before = session.get(url_before, timeout=15)
        res_before.encoding = 'utf-8'
        soup_before = BeautifulSoup(res_before.text, 'html.parser')
        
        before_text = soup_before.text.replace('\n', ' ')
        wind_match = re.search(r'風速.*?(\d+)m', before_text)
        wave_match = re.search(r'波高.*?(\d+)cm', before_text)
        if wind_match: env_data['風速'] = float(wind_match.group(1))
        if wave_match: env_data['波高'] = float(wave_match.group(1))
        
        if "満潮" in before_text: env_data['潮'] = '満潮'
        elif "干潮" in before_text: env_data['潮'] = '干潮'
            
        tbodies_before = soup_before.find_all('tbody', class_='is-fs12')
        for waku, tbody in enumerate(tbodies_before[:6], 1):
            t_text = tbody.text.replace('\n', ' ')
            if "リング" in t_text or "ピストン" in t_text or "シリンダ" in t_text:
                boat_data[waku]['部品交換'] = True

            floats = re.findall(r'[-]?\d\.\d{1,2}', tbody.text)
            if len(floats) >= 4:
                boat_data[waku]['展示'] = float(floats[0]) if float(floats[0]) > 5.0 else 0.0
                boat_data[waku]['チルト'] = float(floats[1]) if -0.5 <= float(floats[1]) <= 3.0 else 0.0
                boat_data[waku]['一周'] = float(floats[2]) if float(floats[2]) > 30.0 else 0.0
                boat_data[waku]['回り足'] = float(floats[3]) if float(floats[3]) > 3.0 else 0.0

            # ④ 直近STブレ（今節の平均STのバラつき・ビビり度を数値化）
            st_match = re.findall(r'0\.\d{2}', t_text)
            if st_match:
                st_vals = [float(s) for s in st_match if float(s) < 0.4]
                if st_vals:
                    boat_data[waku]['直近STブレ'] = np.std(st_vals) # ばらつきが大きいほどビビり・ムラがある

        # ② ピット離れ・進入コースの変動を確実に取得
        course_match = re.findall(r'(\d)コース', before_text)
        if len(course_match) >= 6:
            for i, c in enumerate(course_match[:6]):
                actual_waku = int(c)
                if 1 <= actual_waku <= 6:
                    boat_data[actual_waku]['進入コース'] = i + 1
                    
        return boat_data, env_data, True, url_before
    except requests.exceptions.Timeout:
        return None, None, False, "タイムアウト(リトライ上限)"
    except Exception as e:
        return None, None, False, f"通信エラー"

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
                    
                    payout_cell_text = tds[2].text.replace(',', '')
                    numbers = re.findall(r'\d+', payout_cell_text)
                    payout_val = 0
                    if numbers:
                        payout_candidates = [int(n) for n in numbers if int(n) >= 100]
                        payout_val = max(payout_candidates) if payout_candidates else int(numbers[0])
                    return result_str, payout_val
        return "データなし(休催等)", 0
    except:
        return "通信エラー", 0

def predict_race_auto_v10(target_date, venue, race_num, is_backtest=False):
    boat_data, env_data, success, msg = scrape_boatrace_data_v10(target_date, venue, race_num)
    if not success: return {"error": msg}
    
    input_dict = {'会場': env_data['会場'], '風速': env_data['風速'], '波高': env_data['波高']}
    
    for w in range(1, 7):
        toban = boat_data[w]['登番']
        motor = boat_data[w]['モーター']
        input_dict[f'{w}枠登番'] = toban
        input_dict[f'{w}枠展示'] = boat_data[w]['展示']
        
        c_stats = racer_stat_dict.get(int(toban), racer_stat_dict.get(str(toban), {}))
        rate_1st = float(c_stats.get('1着率', 0.0))
        rate_2ren = float(c_stats.get('2連対率', 0.0))
        rate_3ren = float(c_stats.get('3連対率', 0.0))
        avg_st = float(c_stats.get('平均ST', 0.17))

        if rate_1st > 1.0: rate_1st /= 100.0
        if rate_2ren > 1.0: rate_2ren /= 100.0
        if rate_3ren > 1.0: rate_3ren /= 100.0

        input_dict[f'{w}枠_1着率'] = rate_1st
        input_dict[f'{w}枠_2連対率'] = rate_2ren
        input_dict[f'{w}枠_3連対率'] = rate_3ren
        input_dict[f'{w}枠_平均ST'] = avg_st
        
        m_stats = motor_stat_dict.get((int(env_data['会場']), int(motor)), {})
        if not m_stats: m_stats = motor_stat_dict.get(int(motor), motor_stat_dict.get(str(motor), {}))
        m_2ren, m_3ren = 0.0, 0.0
        for k, v in m_stats.items():
            if '2連' in str(k) or '2ren' in str(k): m_2ren = float(v)
            if '3連' in str(k) or '3ren' in str(k): m_3ren = float(v)
        input_dict[f'{w}枠_M2連対'] = m_2ren / 100.0 if m_2ren > 1.0 else m_2ren
        input_dict[f'{w}枠_M3連対'] = m_3ren / 100.0 if m_3ren > 1.0 else m_3ren

    X_pred = pd.DataFrame([input_dict], columns=train_features)
    booster = getattr(model_1st, 'booster_', getattr(model_1st, '_Booster', None))
    if booster and hasattr(booster, 'pandas_categorical') and booster.pandas_categorical:
        for item in booster.pandas_categorical:
            col_name = item[0] if isinstance(item, (list, tuple)) else item.get('col_name')
            cats = item[2] if isinstance(item, (list, tuple)) else item.get('categories')
            if col_name and col_name in X_pred.columns:
                X_pred[col_name] = pd.Categorical(X_pred[col_name], categories=cats)

    for col in X_pred.columns:
        if str(X_pred[col].dtype) == 'object':
            X_pred[col] = pd.to_numeric(X_pred[col], errors='coerce').fillna(0.0)

    preds_1st = model_1st.predict_proba(X_pred)[0]
    preds_2nd = model_2nd.predict_proba(X_pred)[0]
    preds_3rd = model_3rd.predict_proba(X_pred)[0]
    
    course_multipliers = {1: 2.5, 2: 1.5, 3: 1.0, 4: 0.8, 5: 0.4, 6: 0.1}
    adjusted_scores_1st = np.zeros(6)
    adjusted_scores_2nd = np.zeros(6)
    adjusted_scores_3rd = np.zeros(6)
    score_details = {}
    graph_data = [] 
    
    rank_score_map = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
    course_to_waku = {boat_data[w]['進入コース']: w for w in range(1, 7)}
    in_waku = course_to_waku.get(1, 1)
    in_rank_val = rank_score_map.get(boat_data[in_waku]['級別'], 2)
    course_3_waku = course_to_waku.get(3, 3)
    course_3_rank_val = rank_score_map.get(boat_data[course_3_waku]['級別'], 2)
    wind = env_data['風速']
    tide = env_data['潮']
    v_bonus = VENUE_1ST_BONUS.get(venue, 0)
    
    # ① 相対的な展示気配（全6艇の中で誰が一番タイムが良いか）の計算
    tenji_list = [boat_data[w]['展示'] for w in range(1, 7) if boat_data[w]['展示'] > 0]
    best_tenji = min(tenji_list) if tenji_list else 0
    mawari_list = [boat_data[w]['回り足'] for w in range(1, 7) if boat_data[w]['回り足'] > 0]
    best_mawari = max(mawari_list) if mawari_list else 0
    
    for i in range(6):
        waku = i + 1
        rank = boat_data[waku]['級別']
        rank_val = rank_score_map.get(rank, 2)
        tenji = boat_data[waku]['展示']
        mawari = boat_data[waku]['回り足']
        f_count = boat_data[waku]['F']
        tilt = boat_data[waku]['チルト']
        tactics = boat_data[waku]['得意戦法']
        st_std = boat_data[waku]['直近STブレ']
        actual_course = boat_data[waku]['進入コース']
        
        rank_pt = {"A1": 25, "A2": 15, "B1": 5, "B2": 0}.get(rank, 5)
        motor_pt = input_dict[f'{waku}枠_M2連対'] * 100 * 0.4
        skill_pt = input_dict[f'{waku}枠_1着率'] * course_multipliers[actual_course] * 100.0
        course_pt = {1: 20, 2: 10, 3: 5, 4: 5, 5: 0, 6: -5}[actual_course]
        if actual_course == 1: course_pt += v_bonus 
            
        scenario_pt = 0
        if actual_course == 1 and in_rank_val <= 2: scenario_pt -= 25 
        if actual_course > 1 and rank_val > in_rank_val:
            scenario_pt += 25 
            if actual_course == 2: scenario_pt += 10
        if actual_course == 4 and rank_val > course_3_rank_val: scenario_pt += 15 
            
        # ③ 得意戦法ボーナス（4カドで「まくり」巧者なら大幅プラス！）
        if actual_course == 4 and tactics == "まくり": scenario_pt += 25
        elif actual_course == 1 and tactics == "逃げ": scenario_pt += 15
            
        # ④ 直近STブレ（ムラがある・ビビっている選手はマイナス）
        if st_std > 0.05: scenario_pt -= 15
            
        # ① 相対評価（トップクラスの展示・回り足ならボーナス加算）
        kehai_pt = 0
        if mawari > 0 and mawari == best_mawari: kehai_pt += 20 # 節一の回り足
        if tenji > 0 and tenji == best_tenji: kehai_pt += 20     # 最速の直線足
        if tilt >= 0.5 and actual_course >= 4: kehai_pt += 25    # アウト伸び型
        
        base_bonus = rank_pt + motor_pt + skill_pt + course_pt + scenario_pt + kehai_pt
        
        adjusted_scores_1st[i] = max(1.0, (preds_1st[i] * 50.0) + base_bonus)
        adjusted_scores_2nd[i] = max(1.0, (preds_2nd[i] * 50.0) + base_bonus)
        adjusted_scores_3rd[i] = max(1.0, (preds_3rd[i] * 50.0) + base_bonus)
        
        score_details[waku] = f"1着力 {adjusted_scores_1st[i]:>4.1f} | 2着力 {adjusted_scores_2nd[i]:>4.1f} | 3着力 {adjusted_scores_3rd[i]:>4.1f} (戦法:{tactics})"
        
        graph_data.extend([
            {"枠": f"{waku}枠", "項目": "① AI・実力", "スコア": round(max(0, (preds_1st[i]*50) + rank_pt), 1)},
            {"枠": f"{waku}枠", "項目": "② モーター", "スコア": round(max(0, motor_pt * 2.5), 1)},
            {"枠": f"{waku}枠", "項目": "③ コース技", "スコア": round(max(0, skill_pt + course_pt + 15), 1)},
            {"枠": f"{waku}枠", "項目": "④ 展開有利", "スコア": round(max(0, scenario_pt + 30), 1)},
            {"枠": f"{waku}枠", "項目": "⑤ 直前気配", "スコア": round(max(0, kehai_pt + 20), 1)}
        ])
        
    sum_1st = np.sum(adjusted_scores_1st)
    combos = list(itertools.permutations([0, 1, 2, 3, 4, 5], 3))
    raw_probs = []
    
    for (r1, r2, r3) in combos:
        s1 = adjusted_scores_1st[r1]
        s2 = adjusted_scores_2nd[r2]
        s3 = adjusted_scores_3rd[r3]
        p1 = s1 / sum_1st
        p2 = s2 / max(1.0, np.sum(adjusted_scores_2nd) - adjusted_scores_2nd[r1])
        p3 = s3 / max(1.0, np.sum(adjusted_scores_3rd) - adjusted_scores_3rd[r1] - adjusted_scores_3rd[r2])
        raw_probs.append((f"{r1+1}-{r2+1}-{r3+1}", p1 * p2 * p3))
        
    total_p = sum(x[1] for x in raw_probs)
    final_probs = [(c, (p / total_p) * 100) for c, p in raw_probs]
    prob_sorted = sorted(final_probs, key=lambda x: x[1], reverse=True)
    
    ai_top8 = [item[0] for item in prob_sorted[:8]]
    ai_top8_probs = [item[1] for item in prob_sorted[:8]]
    
    combined_scores = adjusted_scores_1st + adjusted_scores_2nd + adjusted_scores_3rd
    top_indices = np.argsort(combined_scores)[::-1][:4]
    t1, t2, t3, t4 = [i + 1 for i in top_indices]
    
    form_bets = []
    for second in [t2, t3, t4]:
        for third in [t2, t3, t4]:
            if second != third: form_bets.append(f"{t1}-{second}-{third}")
    for third in [t3, t4]: form_bets.append(f"{t2}-{t1}-{third}")
        
    total_16_bets = list(dict.fromkeys(ai_top8 + form_bets))
    
    top1_score = np.max(adjusted_scores_1st)
    top3_score = np.sort(adjusted_scores_1st)[-3]
    is_ken = (top1_score - top3_score) < 15.0 or env_data['風速'] >= 8.0 
    
    if is_backtest:
        return {
            "race_num": f"{race_num}R",
            "本命": ai_top8[0],
            "total_16_bets": total_16_bets,
            "ai_top8_probs": ai_top8_probs,
            "is_ken": is_ken
        }
    
    result_text = f"🌐 【プロ舟券師ロジック・v10究極完全版】解析完了！\n"
    result_text += f"🌪️ 水面環境: {venue} / 風速 {env_data['風速']}m / 潮: {env_data['潮']}\n\n"
    
    if is_ken:
        result_text += "⚠️ 【判定：見送り推奨（ケン）】\n※実力が拮抗していますが、v10解析データを以下に出力します。\n\n"
    
    result_text += "📊 各艇の評価スコア内訳（相対気配・戦法・STブレ考慮済）\n"
    for i in range(1, 7):
        r = boat_data[i]['級別']
        f_info = f"[F{boat_data[i]['F']}]" if boat_data[i]['F'] > 0 else ""
        parts = "[換]" if boat_data[i]['部品交換'] else ""
        detail = score_details[i]
        result_text += f"{i}枠({r}){f_info}{parts}: 【{detail}】\n"
        
    result_text += f"\n🎯 AI自信のピンポイント予想（上位8点）\n"
    result_text += "-" * 50 + "\n"
    for i in range(min(8, len(prob_sorted))):
        result_text += f"第{i+1}推奨: 【{prob_sorted[i][0]}】 (確率 {prob_sorted[i][1]:.1f}%)\n"
        
    result_text += f"\n🎯 黄金フォーメーション（8点）\n"
    result_text += "-" * 50 + "\n"
    result_text += f"【本命軸】 1着: {t1} - 2,3着: {t2},{t3},{t4} (6点)\n"
    result_text += f"【逆転軸】 1着: {t2} - 2着: {t1} - 3着: {t3},{t4} (2点)\n"
    result_text += f"👉 合わせた網羅買い目（重複除外 {len(total_16_bets)}点）: {', '.join(total_16_bets)}\n"
    result_text += "-" * 50 + "\n"

    return {"text": result_text, "graph_data": pd.DataFrame(graph_data)}

# Streamlitの画面デザイン
st.title("🚤 ボートレースAI【プロ舟券師ロジック・v10究極完全版】")

tab1, tab2, tab3 = st.tabs(["🚤 通常予想モード", "📊 1日丸ごとバックテスト", "🎯 全国スナイパー"])

today_str = datetime.datetime.now().strftime('%Y%m%d')

with tab1:
    st.write("相対展示気配・進入変動・得意戦法・直近STブレまで完全統合した最高峰の予想モードです。")
    col1, col2, col3 = st.columns(3)
    with col1: target_date = st.text_input("日付 (例: 20261105)", value=today_str, key="date1")
    with col2: venue = st.selectbox("会場", options=list(VENUE_MAP.keys()), index=9, key="venue1")
    with col3: race_num = st.number_input("レース番号", min_value=1, max_value=12, value=5, step=1, key="race1")

    if st.button("⚡ v10エンジンで究極予想！", type="primary"):
        with st.spinner("全4大強化要素を解析し、チャートを生成中..."):
            res_dict = predict_race_auto_v10(target_date, venue, race_num)
            if "error" in res_dict:
                st.error(res_dict["error"])
            else:
                st.text_area("AIのスコア解析結果", value=res_dict["text"], height=450)
                df_graph = res_dict["graph_data"]
                fig = px.bar(
                    df_graph, x='項目', y='スコア', color='枠', 
                    barmode='group', template="plotly_dark",
                    text_auto='.0f', 
                    color_discrete_sequence=['#ffffff', '#aaaaaa', '#ff4444', '#44aaff', '#ffee00', '#44ff44']
                )
                fig.update_layout(title="📊 全艇パラメーター比較（項目別）", xaxis_title=None, yaxis_title="評価スコア", legend_title="枠番")
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.write("AI予想8点＋黄金フォーメーション8点（計最大16点）を網羅し、予算に合わせた自動資金配分で回収率をシミュレーションします。")
    col1_bt, col2_bt, col3_bt = st.columns(3)
    with col1_bt: bt_date = st.text_input("日付 (例: 20261105)", value=today_str, key="date2")
    with col2_bt: bt_venue = st.selectbox("会場", options=list(VENUE_MAP.keys()), index=9, key="venue2")
    with col3_bt: bt_budget = st.number_input("1レースあたりの予算（円）", min_value=800, max_value=50000, value=3000, step=500, key="budget2")
        
    if st.button("🔄 1R〜12R v10バックテスト＆資金配分開始", type="primary"):
        progress_text = "v10バックテスト実行中..."
        my_bar = st.progress(0, text=progress_text)
        results = []
        jcd = VENUE_MAP[bt_venue]
        scan_count = 0
        total_scans = 12
        total_invest = 0
        total_return = 0
        
        def run_backtest_single(r_num):
            res_dict = predict_race_auto_v10(bt_date, bt_venue, r_num, is_backtest=True)
            actual_result, actual_payout_val = fetch_actual_result(bt_date, jcd, r_num)
            time.sleep(0.3) 
            
            if res_dict and "error" not in res_dict:
                is_ken = res_dict["is_ken"]
                bets = res_dict["total_16_bets"]
                pinpoint = res_dict["本命"]
                
                status = "-"
                profit = 0
                invest = 0
                return_val = 0
                
                num_bets = len(bets)
                if num_bets > 0:
                    base_amt = max(100, (bt_budget // num_bets // 100) * 100)
                    invest = base_amt * num_bets
                    if actual_result in bets:
                        status = "🎯 的中！" if not is_ken else "⚠️ ケン的中！"
                        return_val = int(actual_payout_val * (base_amt / 100))
                        profit = return_val - invest
                    else:
                        status = "❌ ハズレ" if not is_ken else "⚠️ ケン外れ"
                        profit = -invest
                
                return {
                    "R": f"{r_num}R", "AI本命(1点)": pinpoint, "買い目(最大16点)": f"{len(bets)}点買い",
                    "実際の結果": actual_result, "払戻金(100円)": f"¥{actual_payout_val:,}" if actual_payout_val > 0 else "-",
                    "判定": status, "収支": profit, "投資": invest, "回収": return_val
                }
            else:
                err_msg = res_dict["error"] if res_dict else "データ未公開"
                return {"R": f"{r_num}R", "AI本命(1点)": "-", "買い目(最大16点)": f"【失敗】 {err_msg}", "実際の結果": actual_result, "払戻金(100円)": "-", "判定": "-", "収支": 0, "投資": 0, "回収": 0}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(run_backtest_single, r): r for r in range(1, 13)}
            for future in concurrent.futures.as_completed(futures):
                scan_count += 1
                my_bar.progress(scan_count / total_scans, text=f"v10スキャン中... 完了 {scan_count}/12 レース")
                res = future.result()
                results.append(res)
                total_invest += res["投資"]
                total_return += res["回収"]
                
        my_bar.empty()
        net_profit = total_return - total_invest
        roi = (total_return / total_invest * 100) if total_invest > 0 else 0
        
        st.success(f"{bt_venue}全12レースのv10バックテストが完了しました！")
        df_results = pd.DataFrame(results)
        df_results['R_num'] = df_results['R'].str.replace('R', '').astype(int)
        df_results = df_results.sort_values(by='R_num').drop(columns=['R_num']).set_index("R")
        
        df_display = df_results[['AI本命(1点)', '買い目(最大16点)', '実際の結果', '払戻金(100円)', '判定', '収支']].copy()
        df_display['収支'] = df_display['収支'].apply(lambda x: f"+{x:,}円" if x > 0 else f"{x:,}円" if x < 0 else "-")
        st.dataframe(df_display, use_container_width=True)
        
        st.markdown("### 📊 v10トータル収支（16点網羅買い・資金配分シミュレーション）")
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        col_sum1.metric("総投資額", f"¥{total_invest:,}")
        col_sum2.metric("総払戻額", f"¥{total_return:,}")
        col_sum3.metric("回収率", f"{roi:.1f}%", f"{net_profit:,}円")

with tab3:
    st.header("🎯 全国スナイパー（v10 爆速検知モード）")
    st.write("相対気配やSTブレ、戦法相性を加味した高精度な波乱・鉄板検知を行います。")
    col1_sn, col2_sn = st.columns([1, 3])
    with col1_sn: snipe_date = st.text_input("スキャン日付", value=today_str, key="date4")
    with col2_sn: target_venues = st.multiselect("スキャン対象の会場", options=list(VENUE_MAP.keys()), default=["平和島", "戸田", "江戸川", "鳴門", "福岡"])
    
    if st.button("📡 v10スナイパー探知を開始する", type="primary"):
        if not target_venues: st.error("会場を1つ以上選択してください。")
        else:
            hot_races = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            race_targets = [(venue, r_num) for venue in target_venues for r_num in range(1, 13)]
            total_scans = len(race_targets)
            scan_count = 0
            
            def evaluate_race_for_sniper(target):
                venue, r_num = target
                boat_data, env_data, success, _ = scrape_boatrace_data_v10(snipe_date, venue, r_num)
                time.sleep(0.3) 
                if not success: return None
                
                haran_reasons, teppan_reasons = [], []
                if boat_data[1]['級別'] in ['B1', 'B2'] and any(boat_data[w]['級別'] in ['A1', 'A2'] for w in [2, 3, 4]):
                    haran_reasons.append("🚨 イン激弱（1枠B級 / 外にA級）")
                if boat_data[3]['級別'] in ['B1', 'B2'] and boat_data[4]['級別'] in ['A1', 'A2'] and boat_data[4]['得意戦法'] == 'まくり':
                    haran_reasons.append("🔥 4カドまくり巧者の特大チャンス")
                if env_data['風速'] >= 6.0: haran_reasons.append(f"🌪️ 荒れ水面（風速{env_data['風速']}m）")
                
                if boat_data[1]['級別'] == 'A1' and boat_data[1]['F'] == 0 and env_data['風速'] <= 3.0:
                    if boat_data[1]['直近STブレ'] < 0.03: # スタートが安定している
                        teppan_reasons.append("💎 イン超鉄板（A1・F無・ST安定・無風）")

                if haran_reasons:
                    return {"会場": venue, "R": f"{r_num}R", "狙い目": "🔥 波乱(穴)", "探知理由": " ＋ ".join(haran_reasons), "風速": f"{env_data['風速']}m", "1枠": boat_data[1]['級別'], "4枠": boat_data[4]['級別']}
                elif teppan_reasons:
                    return {"会場": venue, "R": f"{r_num}R", "狙い目": "💎 鉄板(本命)", "探知理由": " ＋ ".join(teppan_reasons), "風速": f"{env_data['風速']}m", "1枠": boat_data[1]['級別'], "4枠": boat_data[4]['級別']}
                return None

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(evaluate_race_for_sniper, t): t for t in race_targets}
                for future in concurrent.futures.as_completed(futures):
                    scan_count += 1
                    status_text.text(f"📡 v10スキャン中... 完了 {scan_count}/{total_scans} レース")
                    progress_bar.progress(scan_count / total_scans)
                    res = future.result()
                    if res: hot_races.append(res)
                        
            status_text.empty()
            progress_bar.empty()
            if hot_races:
                st.success(f"🎯 {len(hot_races)}件の狙い目レースを探知しました！")
                df_hot = pd.DataFrame(hot_races)
                df_hot['R_num'] = df_hot['R'].str.replace('R', '').astype(int)
                df_hot = df_hot.sort_values(by=['会場', 'R_num']).drop(columns=['R_num']).reset_index(drop=True)
                st.dataframe(df_hot, use_container_width=True)
            else:
                st.info("現在指定された会場には、v10条件を満たすレースはありませんでした。")

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests
from bs4 import BeautifulSoup
import datetime
import re
import itertools
import time

# ページの基本設定
st.set_page_config(page_title="ボートレースAI【プロ舟券師ロジック・完全版】", layout="wide")

# 1. AIと各種辞書の読み込み
@st.cache_resource
def load_brain():
    with open('boat_ai_brain_v9 .pkl', 'rb') as f:
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

# 2. スクレイピング関数（F持ち・支部・チルト対応版）
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
            
            # F（フライング）の取得
            f_match = re.search(r'F(\d)', text_data)
            f_count = int(f_match.group(1)) if f_match else 0
            
            # 支部の取得
            branch_match = re.search(r'(群馬|埼玉|東京|静岡|愛知|三重|福井|滋賀|大阪|兵庫|徳島|香川|岡山|広島|山口|福岡|佐賀|長崎)', text_data)
            branch = branch_match.group(1) if branch_match else "不明"
            
            recent_results = re.findall(r'\b([1-6])\b', tbody.text)
            numeric_results = [int(x) for x in recent_results if x in ['1','2','3','4','5','6']]
            avg_finish = sum(numeric_results[-3:]) / len(numeric_results[-3:]) if numeric_results else 3.5
            
            try:
                motor_match = re.findall(r'\b(\d{1,3})\b', tbody.text)
                motor = int(motor_match[-6]) if len(motor_match) > 6 else 0
            except: motor = 0
            
            boat_data[waku] = {
                '登番': toban, '級別': rank, 'モーター': motor, 
                '展示': 0.0, 'チルト': 0.0, 'F': f_count, '支部': branch,
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
        for waku, tbody in enumerate(tbodies_before[:6], 1):
            # 展示タイムとチルトの取得
            floats = re.findall(r'[-]?\d\.\d{1,2}', tbody.text)
            if floats:
                # 一般的に最初の少数が展示タイム、次がチルト
                boat_data[waku]['展示'] = float(floats[0]) if float(floats[0]) > 5.0 else 0.0
                for val in floats:
                    if -0.5 <= float(val) <= 3.0:
                        boat_data[waku]['チルト'] = float(val)
                        
        return boat_data, env_data, True, url_before
    except Exception as e:
        return None, None, False, f"読み込み失敗: {str(e)}"

# 3. 予想実行関数
def predict_race_auto_v9(target_date, venue, race_num, is_backtest=False):
    boat_data, env_data, success, msg = scrape_boatrace_data_v9(target_date, venue, race_num)
    if not success: return None if is_backtest else msg
    
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
        if not m_stats:
            m_stats = motor_stat_dict.get(int(motor), motor_stat_dict.get(str(motor), {}))
            
        m_2ren, m_3ren = 0.0, 0.0
        for k, v in m_stats.items():
            ks = str(k)
            if '2連' in ks or '2ren' in ks: m_2ren = float(v)
            if '3連' in ks or '3ren' in ks: m_3ren = float(v)
            
        if m_2ren > 1.0: m_2ren /= 100.0
        if m_3ren > 1.0: m_3ren /= 100.0
        
        input_dict[f'{w}枠_M2連対'] = m_2ren
        input_dict[f'{w}枠_M3連対'] = m_3ren

    X_pred = pd.DataFrame([input_dict], columns=train_features)
    
    booster = getattr(model_1st, 'booster_', getattr(model_1st, '_Booster', None))
    if booster and hasattr(booster, 'pandas_categorical') and booster.pandas_categorical:
        for item in booster.pandas_categorical:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                col_name, cats = item[0], item[2]
            elif isinstance(item, dict):
                col_name = item.get('col_name') or item.get('name')
                cats = item.get('categories') or item.get('cats')
            else: continue
            if col_name and col_name in X_pred.columns:
                if len(cats) > 0:
                    if isinstance(cats[0], int): X_pred[col_name] = pd.to_numeric(X_pred[col_name], errors='coerce')
                    elif isinstance(cats[0], str): X_pred[col_name] = X_pred[col_name].astype(str)
                X_pred[col_name] = pd.Categorical(X_pred[col_name], categories=cats)

    for col in X_pred.columns:
        if str(X_pred[col].dtype) == 'object':
            X_pred[col] = pd.to_numeric(X_pred[col], errors='coerce').fillna(0.0)

    preds_1st = model_1st.predict_proba(X_pred)[0]
    
    course_multipliers = {1: 2.5, 2: 1.5, 3: 1.0, 4: 0.8, 5: 0.4, 6: 0.1}
    adjusted_scores = np.zeros(6)
    score_details = {}
    
    in_rank = boat_data[1]['級別']
    rank_score_map = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
    in_rank_val = rank_score_map.get(in_rank, 2)
    course_3_rank_val = rank_score_map.get(boat_data[3]['級別'], 2)
    wind = env_data['風速']
    v_bonus = VENUE_1ST_BONUS.get(venue, 0)
    
    st_123_avg = (input_dict['1枠_平均ST'] + input_dict['2枠_平均ST'] + input_dict['3枠_平均ST']) / 3.0
    st_4 = input_dict['4枠_平均ST']
    
    tenji_times = [boat_data[w]['展示'] for w in range(1, 7) if boat_data[w]['展示'] > 0]
    avg_tenji = sum(tenji_times) / len(tenji_times) if tenji_times else 0
    
    local_branch = VENUE_BRANCH_MAP.get(venue, "")
    
    for i in range(6):
        waku = i + 1
        rank = boat_data[waku]['級別']
        rank_val = rank_score_map.get(rank, 2)
        tenji = boat_data[waku]['展示']
        f_count = boat_data[waku]['F']
        tilt = boat_data[waku]['チルト']
        branch = boat_data[waku]['支部']
        
        ai_pt = preds_1st[i] * 50.0
        rank_pt = {"A1": 25, "A2": 15, "B1": 5, "B2": 0}.get(rank, 5)
        motor_pt = input_dict[f'{waku}枠_M2連対'] * 100 * 0.4
        skill_pt = input_dict[f'{waku}枠_1着率'] * course_multipliers[waku] * 100.0
        
        course_pt = {1: 20, 2: 10, 3: 5, 4: 5, 5: 0, 6: -5}[waku]
        if waku == 1: course_pt += v_bonus 
            
        scenario_pt = 0
        if waku == 1 and in_rank_val <= 2: scenario_pt -= 25 
        if waku > 1 and rank_val > in_rank_val:
            scenario_pt += 25 
            if waku == 2: scenario_pt += 10
            
        if waku == 4 and rank_val > course_3_rank_val: scenario_pt += 15 
            
        if waku == 4 and st_4 < (st_123_avg - 0.02): scenario_pt += 20 
        elif waku == 1 and input_dict['1枠_平均ST'] > 0.18: scenario_pt -= 15 
            
        if wind >= 4.0:
            if waku in [1, 2]: scenario_pt -= 10
            elif waku == 4: scenario_pt += 15
            elif waku in [3, 5, 6]: scenario_pt += 5
            
        # 🔥【心理・環境・チルト追加ロジック】
        if f_count == 1: scenario_pt -= 15
        elif f_count >= 2: scenario_pt -= 30
        
        if branch == local_branch: scenario_pt += 10 # 地元ボーナス
        
        if tilt >= 0.5 and waku >= 4:
            scenario_pt += 20 # 伸び型チルトのまくり警戒
            
        kehai_pt = 0
        if tenji > 0 and avg_tenji > 0: kehai_pt += (avg_tenji - tenji) * 150 
        rhythm_score = boat_data[waku]['今節調子スコア']
        kehai_pt += (rhythm_score - 1.25) * 10 
            
        total_score = max(1.0, ai_pt + rank_pt + motor_pt + skill_pt + course_pt + scenario_pt + kehai_pt)
        adjusted_scores[i] = total_score
        
        score_details[waku] = f"総点 {total_score:>4.1f} (AI:{ai_pt:.1f}|級:{rank_pt}|機:{motor_pt:.1f}|枠地:{course_pt}|展開:{scenario_pt}|気配:{kehai_pt:.1f})"
        
    sum_scores = np.sum(adjusted_scores)
    
    combos = list(itertools.permutations([0, 1, 2, 3, 4, 5], 3))
    raw_probs = []
    
    for (r1, r2, r3) in combos:
        score1 = adjusted_scores[r1]
        score2 = adjusted_scores[r2]
        score3 = adjusted_scores[r3]
        
        p1 = score1 / sum_scores
        p2 = score2 / (sum_scores - score1)
        p3 = score3 / (sum_scores - score1 - score2)
        
        p = p1 * p2 * p3
        raw_probs.append((f"{r1+1}-{r2+1}-{r3+1}", p))
        
    total_p = sum(x[1] for x in raw_probs)
    final_probs = [(c, (p / total_p) * 100) for c, p in raw_probs]
    prob_sorted = sorted(final_probs, key=lambda x: x[1], reverse=True)
    
    if is_backtest:
        return {
            "race_num": race_num,
            "本命": prob_sorted[0][0],
            "対抗": prob_sorted[1][0],
            "穴": prob_sorted[2][0]
        }
    
    result_text = f"🌐 【プロ舟券師ロジック完全搭載版】解析完了！\n"
    result_text += f"🌪️ 水面環境: {venue} / 風速 {env_data['風速']}m / 波高 {env_data['波高']}cm\n\n"
    
    result_text += "📊 各艇の評価スコア内訳（F持ち・地元・チルト考慮済）\n"
    for i in range(1, 7):
        r = boat_data[i]['級別']
        f_info = f"[F{boat_data[i]['F']}]" if boat_data[i]['F'] > 0 else ""
        tilt = boat_data[i]['チルト']
        detail = score_details[i]
        result_text += f"{i}枠({r}){f_info} チルト{tilt}: 【{detail}】\n"
        
    result_text += f"\n🎯 スコア・展開考慮の推奨買い目（上位6点）\n"
    result_text += "-" * 50 + "\n"
    for i, item in enumerate(prob_sorted[:6]):
        result_text += f"推奨 {i+1}点目: 【{item[0]}】 予想確率: {item[1]:>4.1f}%\n"

    result_text += "-" * 50 + "\n"
    result_text += f"・本命軸: 【{prob_sorted[0][0]}】 (確率 {prob_sorted[0][1]:.1f}%)\n"

    return result_text

# Streamlitの画面デザイン
st.title("🚤 ボートレースAI【プロ舟券師ロジック完全版】")

tab1, tab2 = st.tabs(["🚤 通常予想モード", "📊 1日丸ごとバックテスト"])

today_str = datetime.datetime.now().strftime('%Y%m%d')

with tab1:
    st.write("最新のF持ち・地元アドバンテージ・チルト角のデータを自動取得し、プロの思考で予想します。")
    col1, col2, col3 = st.columns(3)
    with col1:
        target_date = st.text_input("日付 (例: 20261105)", value=today_str, key="date1")
    with col2:
        venue = st.selectbox("会場", options=list(VENUE_MAP.keys()), index=9, key="venue1")
    with col3:
        race_num = st.number_input("レース番号", min_value=1, max_value=12, value=5, step=1, key="race1")

    if st.button("⚡ プロの思考で予想！", type="primary"):
        with st.spinner("選手心理・モーター気配をスコア化して分析中..."):
            res = predict_race_auto_v9(target_date, venue, race_num)
            st.text_area("AIのスコア解析結果", value=res, height=550)

with tab2:
    st.write("指定した会場の1R〜12Rを一気にシミュレーションし、本命・対抗・穴の買い目を一覧表示します。（※過去データ検証用）")
    col1_bt, col2_bt = st.columns(2)
    with col1_bt:
        bt_date = st.text_input("日付 (例: 20261105)", value=today_str, key="date2")
    with col2_bt:
        bt_venue = st.selectbox("会場", options=list(VENUE_MAP.keys()), index=9, key="venue2")
        
    if st.button("🔄 1R〜12R 一括バックテスト開始", type="primary"):
        progress_text = "バックテスト実行中..."
        my_bar = st.progress(0, text=progress_text)
        
        results = []
        for r_num in range(1, 13):
            my_bar.progress(r_num / 12.0, text=f"{r_num}Rを解析中...")
            res_dict = predict_race_auto_v9(bt_date, bt_venue, r_num, is_backtest=True)
            if res_dict:
                results.append(res_dict)
            else:
                results.append({"race_num": r_num, "本命": "データ未公開", "対抗": "-", "穴": "-"})
            time.sleep(0.5) # サーバー負荷軽減のためのウェイト
            
        my_bar.empty()
        st.success(f"{bt_venue}全12レースのバックテストが完了しました！")
        
        df_results = pd.DataFrame(results)
        df_results.set_index("race_num", inplace=True)
        st.dataframe(df_results, use_container_width=True)

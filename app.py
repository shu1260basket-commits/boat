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
st.set_page_config(page_title="ボートレースAI【プロ舟券師ロジック・最終進化版】", layout="wide")

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

# 🔥 通信安定化（リトライ機構）のセットアップ
session = requests.Session()
# タイムアウト等で失敗した場合、最大3回（1秒間隔で）自動リトライするプロ仕様の設定
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

# 2. スクレイピング関数
def scrape_boatrace_data_v9(target_date, venue, race_num):
    clean_date = str(target_date).replace("-", "").replace("/", "")
    jcd = VENUE_MAP[venue]
    
    base_url = "https://www.boatrace.jp/owpc/pc/race"
    params = f"?rno={int(race_num)}&jcd={jcd}&hd={clean_date}"
    url_list = f"{base_url}/racelist{params}"
    url_before = f"{base_url}/beforeinfo{params}"
    
    boat_data = {}
    env_data = {'風速': 0.0, '波高': 0.0, '会場': int(jcd), '潮': '普通'}
    
    try:
        # 🔥 timeoutを15秒に延長し、sessionを使ってリトライさせる
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
            
            try:
                motor_match = re.findall(r'\b(\d{1,3})\b', tbody.text)
                motor = int(motor_match[-6]) if len(motor_match) > 6 else 0
            except: motor = 0
            
            boat_data[waku] = {
                '登番': toban, '級別': rank, 'モーター': motor, 
                '展示': 0.0, '一周': 0.0, '回り足': 0.0, 'チルト': 0.0, 
                'F': f_count, '支部': branch, '部品交換': False,
                '進入コース': waku, 
                '今節調子スコア': max(0.5, (6.0 - avg_finish) / 2.0)
            }
            
        time.sleep(0.5) # 🔥 サーバーBAN対策の深呼吸（超重要）
        
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
        if res.status_code != 200:
            return "通信ブロック", "-"
            
        soup = BeautifulSoup(res.text, 'html.parser')
        
        trs = soup.find_all('tr')
        for tr in trs:
            if '3連単' in tr.text:
                tds = tr.find_all('td')
                if len(tds) >= 3:
                    result_str = tds[1].text.strip().replace('\r', '').replace('\n', '').replace(' ', '')
                    payout_str = tds[2].text.strip().replace('\r', '').replace('\n', '').replace(' ', '')
                    if payout_str and payout_str != "円": payout_str += "円"
                    return result_str, payout_str
        return "データなし(休催等)", "-"
    except requests.exceptions.Timeout:
        return "タイムアウト", "-"
    except:
        return "通信エラー", "-"

def predict_race_auto_v9(target_date, venue, race_num, is_backtest=False):
    boat_data, env_data, success, msg = scrape_boatrace_data_v9(target_date, venue, race_num)
    if not success: 
        return {"error": msg}
    
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
    
    st_123_avg = (input_dict[f'{course_to_waku.get(1,1)}枠_平均ST'] + input_dict[f'{course_to_waku.get(2,2)}枠_平均ST'] + input_dict[f'{course_to_waku.get(3,3)}枠_平均ST']) / 3.0
    st_4_waku = course_to_waku.get(4, 4)
    st_4 = input_dict[f'{st_4_waku}枠_平均ST']
    
    tenji_times = [boat_data[w]['展示'] for w in range(1, 7) if boat_data[w]['展示'] > 0]
    avg_tenji = sum(tenji_times) / len(tenji_times) if tenji_times else 0
    mawari_times = [boat_data[w]['回り足'] for w in range(1, 7) if boat_data[w]['回り足'] > 0]
    avg_mawari = sum(mawari_times) / len(mawari_times) if mawari_times else 0
    local_branch = VENUE_BRANCH_MAP.get(venue, "")
    
    for i in range(6):
        waku = i + 1
        rank = boat_data[waku]['級別']
        rank_val = rank_score_map.get(rank, 2)
        tenji = boat_data[waku]['展示']
        mawari = boat_data[waku]['回り足']
        f_count = boat_data[waku]['F']
        tilt = boat_data[waku]['チルト']
        branch = boat_data[waku]['支部']
        parts = boat_data[waku]['部品交換']
        actual_course = boat_data[waku]['進入コース']
        
        ai_pt = preds_1st[i] * 50.0
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
        if actual_course == 4 and st_4 < (st_123_avg - 0.02): scenario_pt += 20 
        elif actual_course == 1 and input_dict[f'{waku}枠_平均ST'] > 0.18: scenario_pt -= 15 
            
        if wind >= 4.0:
            if actual_course in [1, 2]: scenario_pt -= 10
            elif actual_course == 4: scenario_pt += 15
            elif actual_course in [3, 5, 6]: scenario_pt += 5
            
        if tide == '満潮':
            if actual_course in [1, 2]: scenario_pt += 10
            elif actual_course in [3, 4, 5, 6]: scenario_pt -= 5
        elif tide == '干潮':
            if actual_course in [1, 2]: scenario_pt -= 5
            elif actual_course in [3, 4, 5, 6]: scenario_pt += 10
            
        if actual_course == 1 and waku >= 4: scenario_pt -= 20 
        if f_count == 1: scenario_pt -= 15
        elif f_count >= 2: scenario_pt -= 30
        if branch == local_branch: scenario_pt += 10 
        if tilt >= 0.5 and actual_course >= 4: scenario_pt += 20 
        if parts: scenario_pt += 5 
            
        kehai_pt = 0
        if actual_course in [1, 2] and mawari > 0 and avg_mawari > 0:
            kehai_pt += (avg_mawari - mawari) * 150
        elif actual_course >= 3 and tenji > 0 and avg_tenji > 0:
            kehai_pt += (avg_tenji - tenji) * 150 
            
        rhythm_score = boat_data[waku]['今節調子スコア']
        kehai_pt += (rhythm_score - 1.25) * 10 
            
        total_score = max(1.0, ai_pt + rank_pt + motor_pt + skill_pt + course_pt + scenario_pt + kehai_pt)
        adjusted_scores[i] = total_score
        
        score_details[waku] = f"得点 {total_score:>4.1f} (入:{actual_course}C|AI:{ai_pt:.1f}|級:{rank_pt}|機:{motor_pt:.1f}|技:{skill_pt:.1f}|展開:{scenario_pt}|気:{kehai_pt:.1f})"
        
        graph_data.extend([
            {"枠": f"{waku}枠", "項目": "① AI・実力", "スコア": round(max(0, ai_pt + rank_pt), 1)},
            {"枠": f"{waku}枠", "項目": "② モーター", "スコア": round(max(0, motor_pt * 2.5), 1)},
            {"枠": f"{waku}枠", "項目": "③ コース技", "スコア": round(max(0, skill_pt + course_pt + 15), 1)},
            {"枠": f"{waku}枠", "項目": "④ 展開有利", "スコア": round(max(0, scenario_pt + 30), 1)},
            {"枠": f"{waku}枠", "項目": "⑤ 直前気配", "スコア": round(max(0, kehai_pt + 20), 1)}
        ])
        
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
    
    top1_score = np.max(adjusted_scores)
    top3_score = np.sort(adjusted_scores)[-3]
    is_ken = (top1_score - top3_score) < 15.0 or env_data['風速'] >= 8.0 
    
    if is_backtest:
        return {
            "race_num": f"{race_num}R",
            "本命": prob_sorted[0][0] if not is_ken else "見送り推奨",
            "対抗": prob_sorted[1][0] if not is_ken else "-",
            "穴": prob_sorted[2][0] if not is_ken else "-"
        }
    
    result_text = f"🌐 【プロ舟券師ロジック・最終形態】解析完了！\n"
    result_text += f"🌪️ 水面環境: {venue} / 風速 {env_data['風速']}m / 潮: {env_data['潮']}\n\n"
    
    if is_ken:
        result_text += "⚠️ 【見送り推奨（ケン）】\n上位の実力が拮抗しているか、水面が荒れすぎて難解なレースです。\n\n"
    
    result_text += "📊 各艇の評価スコア内訳（進入コース・部品交換・潮回り考慮済）\n"
    for i in range(1, 7):
        r = boat_data[i]['級別']
        f_info = f"[F{boat_data[i]['F']}]" if boat_data[i]['F'] > 0 else ""
        parts = "[換]" if boat_data[i]['部品交換'] else ""
        detail = score_details[i]
        result_text += f"{i}枠({r}){f_info}{parts}: 【{detail}】\n"
        
    result_text += f"\n🎯 スコア・展開考慮の推奨買い目（上位6点）\n"
    result_text += "-" * 50 + "\n"
    for i, item in enumerate(prob_sorted[:6]):
        result_text += f"推奨 {i+1}点目: 【{item[0]}】 予想確率: {item[1]:>4.1f}%\n"

    result_text += "-" * 50 + "\n"
    result_text += f"・本命軸: 【{prob_sorted[0][0]}】 (確率 {prob_sorted[0][1]:.1f}%)\n"

    return {"text": result_text, "graph_data": pd.DataFrame(graph_data)}

# Streamlitの画面デザイン
st.title("🚤 ボートレースAI【プロ舟券師ロジック・完全版】")

tab1, tab2, tab3 = st.tabs(["🚤 通常予想モード", "📊 1日丸ごとバックテスト", "🎯 全国スナイパー"])

today_str = datetime.datetime.now().strftime('%Y%m%d')

with tab1:
    st.write("潮の満ち引き・進入コースの変化・部品交換まで完全に読み切り、項目別の棒グラフでわかりやすく視覚化します。")
    col1, col2, col3 = st.columns(3)
    with col1:
        target_date = st.text_input("日付 (例: 20261105)", value=today_str, key="date1")
    with col2:
        venue = st.selectbox("会場", options=list(VENUE_MAP.keys()), index=9, key="venue1")
    with col3:
        race_num = st.number_input("レース番号", min_value=1, max_value=12, value=5, step=1, key="race1")

    if st.button("⚡ プロの思考で予想！", type="primary"):
        with st.spinner("展開と気配をスコア化し、チャートを生成中..."):
            res_dict = predict_race_auto_v9(target_date, venue, race_num)
            if "error" in res_dict:
                st.error(res_dict["error"])
            else:
                st.text_area("AIのスコア解析結果", value=res_dict["text"], height=400)
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
    st.write("1R〜12Rを一気にシミュレーションし、AI予想と「実際の結果・払戻金」を横並びで比較します。")
    col1_bt, col2_bt = st.columns(2)
    with col1_bt:
        bt_date = st.text_input("日付 (例: 20261105)", value=today_str, key="date2")
    with col2_bt:
        bt_venue = st.selectbox("会場", options=list(VENUE_MAP.keys()), index=9, key="venue2")
        
    if st.button("🔄 1R〜12R 一括バックテスト開始", type="primary"):
        progress_text = "バックテスト実行中..."
        my_bar = st.progress(0, text=progress_text)
        
        results = []
        jcd = VENUE_MAP[bt_venue]
        for r_num in range(1, 13):
            my_bar.progress(r_num / 12.0, text=f"{r_num}Rを解析中...")
            res_dict = predict_race_auto_v9(bt_date, bt_venue, r_num, is_backtest=True)
            actual_result, actual_payout = fetch_actual_result(bt_date, jcd, r_num)
            
            if res_dict and "error" not in res_dict:
                res_dict["実際の結果"] = actual_result
                res_dict["払戻金"] = actual_payout
                results.append(res_dict)
            else:
                err_msg = res_dict["error"] if res_dict else "データ未公開"
                results.append({
                    "race_num": f"{r_num}R", "本命": f"【失敗】 {err_msg}", "対抗": "-", "穴": "-",
                    "実際の結果": actual_result, "払戻金": actual_payout
                })
            # 🔥 リクエスト過多を避けるため、1レース終わるごとに1秒休む
            time.sleep(1.0) 
            
        my_bar.empty()
        st.success(f"{bt_venue}全12レースのバックテストが完了しました！")
        df_results = pd.DataFrame(results)
        df_results.set_index("race_num", inplace=True)
        st.dataframe(df_results, use_container_width=True)

with tab3:
    st.header("🎯 全国スナイパー（波乱＆鉄板 爆速検知モード）")
    st.write("全国のレース場を自動スキャンし、「荒れそうな穴レース」と「堅く決まる鉄板レース」の両方をあぶり出します。")
    
    col1_sn, col2_sn = st.columns([1, 3])
    with col1_sn:
        snipe_date = st.text_input("スキャン日付", value=today_str, key="date4")
    with col2_sn:
        target_venues = st.multiselect(
            "スキャン対象の会場（複数選択可）", 
            options=list(VENUE_MAP.keys()), 
            default=["平和島", "戸田", "江戸川", "鳴門", "福岡"]
        )
        
    st.caption("※マルチスレッド（並列処理）により超高速でスキャンします。")
    
    if st.button("📡 スナイパー探知を開始する", type="primary"):
        if not target_venues:
            st.error("スキャン対象の会場を1つ以上選択してください。")
        else:
            hot_races = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            race_targets = [(venue, r_num) for venue in target_venues for r_num in range(1, 13)]
            total_scans = len(race_targets)
            scan_count = 0
            
            def evaluate_race_for_sniper(target):
                venue, r_num = target
                boat_data, env_data, success, _ = scrape_boatrace_data_v9(snipe_date, venue, r_num)
                # 🔥 並行アクセス時も0.3秒待機を入れてBANを回避
                time.sleep(0.3) 
                if not success:
                    return None
                    
                haran_reasons = []
                teppan_reasons = []
                
                if boat_data[1]['級別'] in ['B1', 'B2']:
                    if any(boat_data[w]['級別'] in ['A1', 'A2'] for w in [2, 3, 4]):
                        haran_reasons.append("🚨 イン激弱（1枠B級 / 外にA級）")
                        
                if boat_data[3]['級別'] in ['B1', 'B2'] and boat_data[4]['級別'] in ['A1', 'A2']:
                    haran_reasons.append("🔥 4カドまくり特大チャンス")
                    
                if env_data['風速'] >= 6.0:
                    haran_reasons.append(f"🌪️ 荒れ水面（風速{env_data['風速']}m）")
                    
                if any(boat_data[w]['チルト'] >= 0.5 for w in [4, 5, 6]):
                    haran_reasons.append("🚀 アウト勢に伸び型チルトあり")
                    
                if boat_data[1]['級別'] == 'A1' and boat_data[1]['F'] == 0 and env_data['風速'] <= 3.0:
                    if not any(boat_data[w]['チルト'] >= 0.5 for w in [4, 5, 6]):
                        teppan_reasons.append("💎 イン超鉄板（1枠A1・F無・無風・アウト奇襲無）")
                        if boat_data[2]['級別'] != 'B2':
                            teppan_reasons.append("🛡️ 鉄壁の2コース（壁役が優秀）")

                if haran_reasons:
                    return {
                        "会場": venue, "R": f"{r_num}R", "狙い目": "🔥 波乱(穴)",
                        "探知理由": " ＋ ".join(haran_reasons), "風速": f"{env_data['風速']}m",
                        "1枠": boat_data[1]['級別'], "4枠": boat_data[4]['級別']
                    }
                elif teppan_reasons:
                    return {
                        "会場": venue, "R": f"{r_num}R", "狙い目": "💎 鉄板(本命)",
                        "探知理由": " ＋ ".join(teppan_reasons), "風速": f"{env_data['風速']}m",
                        "1枠": boat_data[1]['級別'], "4枠": boat_data[4]['級別']
                    }
                return None

            # 同時アクセス数は8で固定（上げすぎるとタイムアウトが増える）
            MAX_WORKERS = 8 
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(evaluate_race_for_sniper, t): t for t in race_targets}
                for future in concurrent.futures.as_completed(futures):
                    scan_count += 1
                    status_text.text(f"📡 スキャン中... 完了 {scan_count}/{total_scans} レース")
                    progress_bar.progress(scan_count / total_scans)
                    
                    res = future.result()
                    if res:
                        hot_races.append(res)
                        
            status_text.empty()
            progress_bar.empty()
            
            if hot_races:
                st.success(f"🎯 {len(hot_races)}件の狙い目レースを探知しました！")
                df_hot = pd.DataFrame(hot_races)
                df_hot['R_num'] = df_hot['R'].str.replace('R', '').astype(int)
                df_hot = df_hot.sort_values(by=['会場', 'R_num']).drop(columns=['R_num']).reset_index(drop=True)
                st.dataframe(df_hot, use_container_width=True)
            else:
                st.info("現在指定された会場には、目立った狙い目（波乱・鉄板）のレースはありませんでした。")

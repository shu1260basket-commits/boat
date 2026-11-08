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
st.set_page_config(page_title="ボートレースAI【v11 5年データ学習・一本釣り特化版】", layout="wide")

# 1. 究極のAI（v11）の読み込み
@st.cache_resource
def load_brain():
    with open('boat_ai_brain_v11.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    return (
        saved_data['model_ranker'],
        saved_data['features'],
        saved_data.get('racer_stats', {}),
        saved_data.get('motor_stats', {})
    )

try:
    model_ranker, train_features, racer_stat_dict, motor_stat_dict = load_brain()
except FileNotFoundError:
    st.error("⚠️ `boat_ai_brain_v11.pkl` が見つかりません。先に学習スクリプトを実行してください。")
    st.stop()

VENUE_MAP = {
    "桐生": "01", "戸田": "02", "江戸川": "03", "平和島": "04", "多摩川": "05", "浜名湖": "06",
    "蒲郡": "07", "常滑": "08", "津": "09", "三国": "10", "びわこ": "11", "住之江": "12",
    "尼崎": "13", "鳴門": "14", "丸亀": "15", "児島": "16", "宮島": "17", "徳山": "18",
    "下関": "19", "若松": "20", "芦屋": "21", "福岡": "22", "唐津": "23", "大村": "24"
}

session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20))
session.headers.update({'User-Agent': 'Mozilla/5.0'})

def scrape_boatrace_data_v11(target_date, venue, race_num):
    clean_date = str(target_date).replace("-", "").replace("/", "")
    jcd = VENUE_MAP[venue]
    
    url_list = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={int(race_num)}&jcd={jcd}&hd={clean_date}"
    url_before = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={int(race_num)}&jcd={jcd}&hd={clean_date}"
    
    boat_data = {}
    env_data = {'風速': 0.0, '波高': 0.0, '気温': 20.0, '水温': 20.0}
    
    try:
        res_list = session.get(url_list, timeout=15)
        res_list.encoding = 'utf-8'
        if res_list.status_code != 200: return None, None, False, f"通信ブロック ({res_list.status_code})"

        soup_list = BeautifulSoup(res_list.text, 'html.parser')
        tbodies_list = soup_list.find_all('tbody', class_='is-fs12')
        if len(tbodies_list) < 6: return None, None, False, "休催日 または データ未公開"
            
        for waku, tbody in enumerate(tbodies_list[:6], 1):
            text_data = tbody.text.replace('\n', ' ').replace('\r', ' ')
            toban_match = re.search(r'(\d{4})\s*/\s*([A-B][1-2])', text_data)
            toban = int(toban_match.group(1)) if toban_match else 0
            
            try: motor = int(re.findall(r'\b(\d{1,3})\b', tbody.text)[-6])
            except: motor = 0
            
            boat_data[waku] = {
                '登番': toban, 'モーター': motor, '展示': 0.0, '進入コース': waku, '平均ST': 0.17
            }
            
        time.sleep(0.5) 
        res_before = session.get(url_before, timeout=15)
        res_before.encoding = 'utf-8'
        soup_before = BeautifulSoup(res_before.text, 'html.parser')
        before_text = soup_before.text.replace('\n', ' ')
        
        wind_match = re.search(r'風速.*?(\d+)m', before_text)
        wave_match = re.search(r'波高.*?(\d+)cm', before_text)
        temp_match = re.search(r'気温.*?(\d+\.\d+)℃', before_text)
        water_match = re.search(r'水温.*?(\d+\.\d+)℃', before_text)
        
        if wind_match: env_data['風速'] = float(wind_match.group(1))
        if wave_match: env_data['波高'] = float(wave_match.group(1))
        if temp_match: env_data['気温'] = float(temp_match.group(1))
        if water_match: env_data['水温'] = float(water_match.group(1))
            
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
                    
        return boat_data, env_data, True, url_before
    except:
        return None, None, False, "通信エラー"

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
                    payout_val = max([int(n) for n in numbers if int(n) >= 100]) if numbers else 0
                    return result_str, payout_val
        return "データなし(未発走・休催)", 0
    except:
        return "通信エラー", 0

def predict_race_auto_v11(target_date, venue, race_num, is_backtest=False):
    boat_data, env_data, success, msg = scrape_boatrace_data_v11(target_date, venue, race_num)
    if not success: return {"error": msg}
    
    input_list = []
    for w in range(1, 7):
        input_dict = {
            '枠番': w, '登番': boat_data[w]['登番'], 'モーター': boat_data[w]['モーター'],
            '展示': boat_data[w]['展示'], '進入コース': boat_data[w]['進入コース'],
            '平均ST': boat_data[w]['平均ST'], '風速': env_data['風速'], '波高': env_data['波高'],
            '気温': 20.0, '水温': 20.0, '温度差': 0.0, '1着率': 0.0, '2連対率': 0.0, '3連対率': 0.0
        }
        ordered_dict = {col: input_dict.get(col, 0.0) for col in train_features}
        input_list.append(ordered_dict)

    X_pred = pd.DataFrame(input_list)[train_features]
    
    booster = getattr(model_ranker, 'booster_', getattr(model_ranker, '_Booster', None))
    if booster and hasattr(booster, 'pandas_categorical') and booster.pandas_categorical:
        for item in booster.pandas_categorical:
            col_name = item[0] if isinstance(item, (list, tuple)) else item.get('col_name')
            cats = item[2] if isinstance(item, (list, tuple)) else item.get('categories')
            if col_name and col_name in X_pred.columns:
                X_pred[col_name] = pd.Categorical(X_pred[col_name], categories=cats)

    for col in X_pred.columns:
        if str(X_pred[col].dtype) == 'object': X_pred[col] = pd.to_numeric(X_pred[col], errors='coerce').fillna(0.0)

    # 1. AI純度100%のランキングスコア
    ai_raw_scores = model_ranker.predict(X_pred)
    
    score_details = {}
    graph_data = [] 
    for i in range(6):
        waku = i + 1
        score_details[waku] = f"AI期待値スコア: {ai_raw_scores[i]:>6.3f}"
        graph_data.append({"枠": f"{waku}枠", "項目": "AI純粋期待値", "スコア": round(ai_raw_scores[i], 3)})
        
    # 🔥 修正1: Softmax関数で各艇の「1着になる確率」を正確に算出
    exp_scores = np.exp(ai_raw_scores - np.max(ai_raw_scores))
    win_probs = exp_scores / np.sum(exp_scores)
    
    top_indices = np.argsort(win_probs)[::-1][:4]
    t1, t2, t3, t4 = [i + 1 for i in top_indices]
    
    # 1着候補の勝率
    top1_win_prob = win_probs[top_indices[0]] * 100.0

    # 3連単の確率算出 (Plackett-Luceモデル)
    combos = list(itertools.permutations([0, 1, 2, 3, 4, 5], 3))
    raw_probs = []
    for (r1, r2, r3) in combos:
        p1 = win_probs[r1]
        p2 = win_probs[r2] / max(0.0001, 1.0 - win_probs[r1])
        p3 = win_probs[r3] / max(0.0001, 1.0 - win_probs[r1] - win_probs[r2])
        raw_probs.append((f"{r1+1}-{r2+1}-{r3+1}", p1 * p2 * p3 * 100))
        
    prob_sorted = sorted(raw_probs, key=lambda x: x[1], reverse=True)
    
    # 🔥 修正2: 買い目ロジック（1着確率を基準に評価）
    bets = []
    bet_type_name = ""
    
    if top1_win_prob >= 40.0: 
        bet_type_name = "💎 超・鉄板（激絞り4点）"
        bets = [f"{t1}-{t2}-{t3}", f"{t1}-{t2}-{t4}", f"{t1}-{t3}-{t2}", f"{t1}-{t3}-{t4}"]
    elif top1_win_prob >= 25.0: 
        bet_type_name = "🎯 本命＆逆転（フォーメーション8点）"
        for second in [t2, t3, t4]:
            for third in [t2, t3, t4]:
                if second != third: bets.append(f"{t1}-{second}-{third}")
        for third in [t3, t4]:
            bets.append(f"{t2}-{t1}-{third}")
    else:
        bets = []
        bet_type_name = "⚠️ 混戦・波乱（完全見送り推奨）"

    is_ken = (top1_win_prob < 25.0) or (env_data['風速'] >= 6.0)
    if is_ken:
        bets = []
        bet_type_name = "⚠️ 悪条件・自信度低下（完全見送り推奨）"

    is_easy_hit = (top1_win_prob >= 40.0) and not is_ken

    if is_backtest:
        return {
            "race_num": f"{race_num}R",
            "本命": prob_sorted[0][0],
            "total_bets": bets,
            "bet_type": bet_type_name,
            "is_ken": is_ken,
            "is_easy_hit": is_easy_hit,
            "top1_win_prob": top1_win_prob
        }
    
    result_text = f"🌐 【v11 5年データ学習・純粋AIロジック】解析完了！\n"
    result_text += f"🌪️ 水面環境: {venue} / 風速 {env_data['風速']}m\n"
    result_text += f"🎯 AI 1位抜け確率: {top1_win_prob:.1f}%\n\n"
    
    if is_ken:
        result_text += f"⚠️ 【完全見送り推奨】\n理由: {bet_type_name}\n※無駄な資金を溶かさないために参加を見送ってください。\n\n"
    else:
        result_text += f"✨ 【AI投資判定】 {bet_type_name} で勝負可能！\n\n"
    
    result_text += "📊 各艇のAI純粋期待値スコア\n"
    for i in range(1, 7):
        detail = score_details[i]
        result_text += f"{i}枠: 【{detail}】 (1着率 {win_probs[i-1]*100:.1f}%)\n"
        
    result_text += f"\n🎯 AI自信のピンポイント予想（3連単 上位8点）\n"
    result_text += "-" * 50 + "\n"
    for i in range(min(8, len(prob_sorted))):
        result_text += f"第{i+1}推奨: 【{prob_sorted[i][0]}】 (確率 {prob_sorted[i][1]:.1f}%)\n"
        
    result_text += f"\n🎯 激絞り買い目指示\n"
    result_text += "-" * 50 + "\n"
    if bets:
        result_text += f"👉 指示買い目（{len(bets)}点）: {', '.join(bets)}\n"
        result_text += "※これ以上の点数を買うとトリガミのリスクが高まります。\n"
    else:
        result_text += "👉 指示買い目: なし（ケン）\n"
    result_text += "-" * 50 + "\n"

    return {"text": result_text, "graph_data": pd.DataFrame(graph_data)}

# Streamlitの画面デザイン
st.title("🚤 ボートレースAI【v11 5年データ学習・一本釣り特化版】")

tab1, tab2, tab3 = st.tabs(["🚤 通常予想モード", "📊 厳格バックテスト", "🎯 全国スナイパー"])
today_str = datetime.datetime.now().strftime('%Y%m%d')

with tab1:
    st.write("人間の主観を排除し、5年分のデータから導き出した「AI純度100%」の期待値で勝負します。")
    col1, col2, col3 = st.columns(3)
    with col1: target_date = st.text_input("日付 (例: 20261105)", value=today_str, key="date1")
    with col2: venue = st.selectbox("会場", options=list(VENUE_MAP.keys()), index=9, key="venue1")
    with col3: race_num = st.number_input("レース番号", min_value=1, max_value=12, value=5, step=1, key="race1")

    if st.button("⚡ v11 純粋AI予想！", type="primary"):
        with st.spinner("AIが5年分の記憶から最適な展開を導き出しています..."):
            res_dict = predict_race_auto_v11(target_date, venue, race_num)
            if "error" in res_dict:
                st.error(res_dict["error"])
            else:
                st.text_area("AIのスコア解析結果", value=res_dict["text"], height=450)
                df_graph = res_dict["graph_data"]
                fig = px.bar(
                    df_graph, x='枠', y='スコア', color='枠', barmode='group', template="plotly_dark",
                    text_auto='.2f', color_discrete_sequence=['#ffffff', '#aaaaaa', '#ff4444', '#44aaff', '#ffee00', '#44ff44']
                )
                fig.update_layout(title="📊 AI純粋期待値スコア", xaxis_title=None, yaxis_title="評価スコア", legend_title="枠番")
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.write("勝率が低い・荒れる・ガミるレースを徹底的に「見送り」にし、本当に勝てるレースだけを厳選投資した場合のシミュレーションです。")
    col1_bt, col2_bt, col3_bt = st.columns(3)
    with col1_bt: bt_date = st.text_input("日付 (例: 20261105)", value=today_str, key="date2")
    with col2_bt: bt_venue = st.selectbox("会場", options=list(VENUE_MAP.keys()), index=9, key="venue2")
    with col3_bt: bt_budget = st.number_input("1レースあたりの予算（円）", min_value=800, max_value=50000, value=3000, step=500, key="budget2")
        
    if st.button("🔄 1R〜12R 厳選バックテスト＆資金配分開始", type="primary"):
        progress_text = "厳格フィルター実行中..."
        my_bar = st.progress(0, text=progress_text)
        results = []
        jcd = VENUE_MAP[bt_venue]
        scan_count = 0
        total_scans = 12
        total_invest = 0
        total_return = 0
        
        def run_backtest_single(r_num):
            res_dict = predict_race_auto_v11(bt_date, bt_venue, r_num, is_backtest=True)
            actual_result, actual_payout_val = fetch_actual_result(bt_date, jcd, r_num)
            time.sleep(0.3) 
            
            if res_dict and "error" not in res_dict:
                is_ken = res_dict["is_ken"]
                is_easy_hit = res_dict["is_easy_hit"]
                bets = res_dict["total_bets"]
                
                status = "-"
                profit = 0
                invest = 0
                return_val = 0
                
                # 🔥 修正3: 未発走・中止レースは「見送り（ノーカン）」として投資をゼロにする
                if actual_payout_val == 0:
                    status = "未発走/不明"
                    invest = 0
                    return_val = 0
                elif is_ken or not bets:
                    status = "見送り(AI判定)"
                    invest = 0
                    return_val = 0
                else:
                    num_bets = len(bets)
                    base_amt = max(100, (bt_budget // num_bets // 100) * 100)
                    invest = base_amt * num_bets
                    if actual_result in bets:
                        status = "🎯 的中！"
                        if is_easy_hit: status = "💎 的中(高確信)！"
                        return_val = int(actual_payout_val * (base_amt / 100))
                        profit = return_val - invest
                    else:
                        status = "❌ ハズレ"
                        profit = -invest
                
                return {
                    "R": f"{r_num}R", "投資判定": res_dict["bet_type"], "買い目": f"{len(bets)}点買い" if bets else "-",
                    "実際の結果": actual_result, "払戻金(100円)": f"¥{actual_payout_val:,}" if actual_payout_val > 0 else "-",
                    "判定": status, "収支": profit, "投資": invest, "回収": return_val
                }
            else:
                err_msg = res_dict["error"] if res_dict else "データ未公開"
                return {"R": f"{r_num}R", "投資判定": "-", "買い目": f"【失敗】 {err_msg}", "実際の結果": actual_result, "払戻金(100円)": "-", "判定": "-", "収支": 0, "投資": 0, "回収": 0}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(run_backtest_single, r): r for r in range(1, 13)}
            for future in concurrent.futures.as_completed(futures):
                scan_count += 1
                my_bar.progress(scan_count / total_scans, text=f"厳選スキャン中... 完了 {scan_count}/12 レース")
                res = future.result()
                results.append(res)
                total_invest += res["投資"]
                total_return += res["回収"]
                
        my_bar.empty()
        net_profit = total_return - total_invest
        roi = (total_return / total_invest * 100) if total_invest > 0 else 0
        
        st.success(f"{bt_venue}全12レースの厳選バックテストが完了しました！")
        df_results = pd.DataFrame(results)
        df_results['R_num'] = df_results['R'].str.replace('R', '').astype(int)
        df_results = df_results.sort_values(by='R_num').drop(columns=['R_num']).set_index("R")
        
        df_display = df_results[['投資判定', '買い目', '実際の結果', '払戻金(100円)', '判定', '収支']].copy()
        df_display['収支'] = df_display['収支'].apply(lambda x: f"+{x:,}円" if x > 0 else f"{x:,}円" if x < 0 else "-")
        st.dataframe(df_display, use_container_width=True)
        
        st.markdown("### 📊 厳選トータル収支（見送り回避・激絞り投資シミュレーション）")
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        col_sum1.metric("総投資額", f"¥{total_invest:,}")
        col_sum2.metric("総払戻額", f"¥{total_return:,}")
        col_sum3.metric("回収率", f"{roi:.1f}%", f"{net_profit:,}円")

with tab3:
    st.header("🎯 全国スナイパー（超・鉄板レース一本釣りモード）")
    st.write("全国のレース場から、勝率・回収率を極限まで高められる「超・鉄板（激絞り4点で仕留められる）」レースだけを検索します。")
    
    col1_sn, col2_sn = st.columns([1, 3])
    with col1_sn: snipe_date = st.text_input("スキャン日付", value=today_str, key="date4")
    with col2_sn: target_venues = st.multiselect("スキャン対象の会場", options=list(VENUE_MAP.keys()), default=["平和島", "戸田", "江戸川", "鳴門", "福岡"])
    
    if st.button("📡 一本釣り探知を開始する", type="primary"):
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
                res_dict = predict_race_auto_v11(snipe_date, venue, r_num, is_backtest=True)
                time.sleep(0.3) 
                if not res_dict or "error" in res_dict: return None
                
                if res_dict["is_easy_hit"]:
                    return {
                        "会場": venue, "R": f"{r_num}R", "狙い目": "💎 激絞り・鉄板",
                        "AI確率(1位)": f"{res_dict['top1_win_prob']:.1f}%",
                        "指示買い目": "4点買い"
                    }
                return None

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(evaluate_race_for_sniper, t): t for t in race_targets}
                for future in concurrent.futures.as_completed(futures):
                    scan_count += 1
                    status_text.text(f"📡 鉄板スキャン中... 完了 {scan_count}/{total_scans} レース")
                    progress_bar.progress(scan_count / total_scans)
                    res = future.result()
                    if res: hot_races.append(res)
                        
            status_text.empty()
            progress_bar.empty()
            if hot_races:
                st.success(f"🎯 {len(hot_races)}件の【超・鉄板レース】を探知しました！ここにだけ投資してください。")
                df_hot = pd.DataFrame(hot_races)
                df_hot['R_num'] = df_hot['R'].str.replace('R', '').astype(int)
                df_hot = df_hot.sort_values(by=['会場', 'R_num']).drop(columns=['R_num']).reset_index(drop=True)
                st.dataframe(df_hot, use_container_width=True)
            else:
                st.info("現在指定された会場には、打つべき確実な鉄板レースはありません。（無駄打ち回避成功）")

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

st.set_page_config(page_title="ボートレースAI【v14 投資家特化・ケリー基準搭載版】", layout="wide")

@st.cache_resource
def load_brain():
    with open('boat_ai_brain_v14_deep.pkl', 'rb') as f:
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
    suji_stats = brain['suji_stats']
except Exception as e:
    st.error(f"⚠️ `boat_ai_brain_v14_deep.pkl` の読み込みエラー: {e}")
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
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

def scrape_boatrace_data_v14(target_date, venue, race_num):
    clean_date = str(target_date).replace("-", "").replace("/", "")
    jcd = VENUE_MAP[venue]
    url_list = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={int(race_num)}&jcd={jcd}&hd={clean_date}"
    url_before = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={int(race_num)}&jcd={jcd}&hd={clean_date}"
    
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
        return None, None, False, f"通信/解析エラー: {str(e)}"

def fetch_realtime_odds(target_date, jcd, race_num):
    """ 公式からリアルタイムの3連単オッズを取得（取得失敗時はAI確率からダミー生成） """
    clean_date = str(target_date).replace("-", "").replace("/", "")
    url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={int(race_num)}&jcd={jcd}&hd={clean_date}"
    odds_dict = {}
    try:
        res = session.get(url, timeout=10)
        res.encoding = 'utf-8'
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            # 簡易スクレイピング： "1-2-3" と "15.4" のような組み合わせを抽出
            text_clean = soup.text.replace('\n', ' ').replace('\r', '')
            matches = re.findall(r'([1-6]-[1-6]-[1-6])\s*(\d+\.\d)', text_clean)
            for combo, odds_val in matches:
                odds_dict[combo] = float(odds_val)
    except:
        pass
    return odds_dict

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
        return "データなし(未発走)", 0
    except:
        return "通信エラー", 0

def predict_race_auto_v14(target_date, venue, race_num, is_backtest=False, total_budget=3000):
    boat_data, env_data, success, msg = scrape_boatrace_data_v14(target_date, venue, race_num)
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
        
        # 🔥 v14: 多次元マスタデータの展開
        c_stats = racer_course_stats.get((toban, course), {'runs':1, '1st':0, '2nd':0, '3rd':0})
        v_stats = racer_venue_stats.get((toban, jcd), {'runs':1, '1st':0, '2nd':0, '3rd':0})
        m_stats = motor_stats.get((jcd, motor), {'runs':1, '1st':0, '2nd':0})
        
        c_runs, v_runs, m_runs = max(1, c_stats['runs']), max(1, v_stats['runs']), max(1, m_stats['runs'])
        inside_st = st_array[i-1] if i > 0 else st_array[i]
        
        input_dict = {
            '枠番': waku, '進入コース': course, '展示': boat_data[waku]['展示'], '平均ST': boat_data[waku]['平均ST'],
            '風速': env_data['風速'], '波高': env_data['波高'],
            'コース1着率': c_stats['1st']/c_runs, 'コース2連対率': (c_stats['1st']+c_stats['2nd'])/c_runs,
            '当地1着率': v_stats['1st']/v_runs, '当地2連対率': (v_stats['1st']+v_stats['2nd'])/v_runs,
            'モーター2連対率': (m_stats['1st']+m_stats['2nd'])/m_runs,
            '内側ST差': boat_data[waku]['平均ST'] - inside_st, '展示平均差': boat_data[waku]['展示'] - tenji_mean
        }
        ordered_dict = {col: input_dict.get(col, 0.0) for col in train_features}
        input_list.append(ordered_dict)

    X_pred = pd.DataFrame(input_list)[train_features]
    
    lgb_scores = model_lgb.predict(X_pred)
    xgb_scores = model_xgb.predict(X_pred)
    z_lgb = (lgb_scores - np.mean(lgb_scores)) / (np.std(lgb_scores) + 1e-9)
    z_xgb = (xgb_scores - np.mean(xgb_scores)) / (np.std(xgb_scores) + 1e-9)
    ensemble_scores = (z_lgb + z_xgb) / 2.0
    
    graph_data = [{"枠": f"{i+1}枠", "項目": "総合期待値", "スコア": round(ensemble_scores[i], 3)} for i in range(6)]
        
    exp_scores = np.exp(ensemble_scores * 1.5)
    win_probs = exp_scores / np.sum(exp_scores)

    combos = list(itertools.permutations([0, 1, 2, 3, 4, 5], 3))
    raw_probs = []
    for (r1, r2, r3) in combos:
        c1, c2, c3 = course_map[r1], course_map[r2], course_map[r3]
        p1 = win_probs[r1]
        
        p2_pl = win_probs[r2] / max(0.0001, 1.0 - win_probs[r1])
        c1_count = max(1, suji_stats['1st_count'].get(c1, 1))
        p2_suji = max(0.01, suji_stats['12_matrix'].get((c1, c2), 0) / c1_count)
        
        p3_pl = win_probs[r3] / max(0.0001, 1.0 - win_probs[r1] - win_probs[r2])
        c12_count = max(1, suji_stats['12_matrix'].get((c1, c2), 1))
        p3_suji = max(0.01, suji_stats['123_matrix'].get((c1, c2, c3), 0) / c12_count)

        score = p1 * (p2_pl * p2_suji) * (p3_pl * p3_suji)
        raw_probs.append((f"{r1+1}-{r2+1}-{r3+1}", score))
        
    total_score = sum(x[1] for x in raw_probs)
    prob_sorted = {c: (s / total_score) for c, s in raw_probs} # 確率(0.0〜1.0)
    
    # 🔥 v14: リアルタイムオッズの取得とケリー基準の計算
    real_odds = fetch_realtime_odds(target_date, VENUE_MAP[venue], race_num)
    
    bets_info = []
    # 上位8点の候補に対して、期待値とケリー基準を計算
    sorted_combos = sorted(prob_sorted.items(), key=lambda x: x[1], reverse=True)[:8]
    
    for combo, prob in sorted_combos:
        # オッズが取れなかった場合は、確率の逆数×0.75をダミーオッズとする（控除率25%を想定）
        odds = real_odds.get(combo, (1.0 / max(0.001, prob)) * 0.75) 
        ev = prob * odds # 期待値 (1.0以上でプラス)
        
        # ケリー基準: f* = (p * b - q) / b  (b = odds - 1)
        b = odds - 1.0
        if b > 0 and ev > 1.2: # 期待値120%以上のみ投資対象とする
            kelly_fraction = (prob * b - (1.0 - prob)) / b
            kelly_fraction = max(0.0, min(0.1, kelly_fraction)) # リスク回避のため最大でも総資金の10%まで
        else:
            kelly_fraction = 0.0
            
        bets_info.append({
            "combo": combo,
            "prob": prob * 100,
            "odds": odds,
            "ev": ev,
            "kelly_fraction": kelly_fraction
        })

    # 🔥 投資判断（波乱フィルター ＆ 期待値フィルター）
    is_ken = False
    ken_reason = ""
    if env_data['風速'] >= 4.0 or env_data['波高'] >= 4.0:
        is_ken, ken_reason = True, "波乱環境（風4m以上 or 波4cm以上）"
    
    # ケリー基準で資金を投じるべき買い目だけを抽出
    active_bets = [b for b in bets_info if b['kelly_fraction'] > 0]
    
    if not active_bets:
        is_ken, ken_reason = True, "期待値(EV)120%を超える美味しい馬券が存在しない"

    invest_dict = {}
    if not is_ken:
        # ケリー基準の比率に従って予算を分配
        total_fraction = sum(b['kelly_fraction'] for b in active_bets)
        for b in active_bets:
            # 予算全体に対する比率に再調整して投資額を決定（100円単位）
            amt = int((b['kelly_fraction'] / total_fraction) * total_budget / 100) * 100
            if amt >= 100: invest_dict[b['combo']] = amt
            
        if not invest_dict:
            is_ken, ken_reason = True, "配分資金が100円に満たない"

    if is_backtest:
        return {
            "race_num": f"{race_num}R",
            "本命": sorted_combos[0][0],
            "total_bets": list(invest_dict.keys()),
            "invest_dict": invest_dict, # { "1-2-3": 1500, "1-3-2": 500 ... }
            "bet_type": "⚠️ ケン" if is_ken else f"🎯 ケリー最適投資（{len(invest_dict)}点）",
            "is_ken": is_ken,
            "is_easy_hit": not is_ken, # EVクリアならすべてEasyHit扱い
            "top1_3t_prob": sorted_combos[0][1] * 100
        }
    
    result_text = f"🌐 【v14 投資家特化・ケリー基準搭載版】解析完了！\n"
    result_text += f"🌪️ 環境: {venue} / 風速 {env_data['風速']}m / 波高 {env_data['波高']}cm\n\n"
    
    if is_ken:
        result_text += f"⚠️ 【AI投資判定: 完全見送り（ケン）】\n"
        result_text += f"⛔ 理由: {ken_reason}\n\n"
    else:
        result_text += f"✨ 【AI投資判定】 期待値クリア！勝負可能！\n\n"
    
    result_text += f"🎯 ケリー基準 オッズ×期待値判定\n"
    result_text += "-" * 50 + "\n"
    for b in bets_info:
        marker = "💰投資対象" if b['kelly_fraction'] > 0 else "❌見送り"
        result_text += f"{b['combo']} | 確率:{b['prob']:.1f}% | オッズ:{b['odds']:.1f}倍 | EV:{b['ev']:.2f} | {marker}\n"
        
    result_text += f"\n🎯 最適資金配分（総予算 {total_budget}円）\n"
    result_text += "-" * 50 + "\n"
    if invest_dict:
        for combo, amt in invest_dict.items():
            result_text += f"👉 {combo} : {amt}円\n"
    else:
        result_text += "👉 投資対象なし\n"
    result_text += "-" * 50 + "\n"

    return {"text": result_text, "graph_data": pd.DataFrame(graph_data)}

st.title("🚤 ボートレースAI【v14 投資家特化・ケリー基準搭載版】")

tab1, tab2, tab3 = st.tabs(["🚤 リアルタイム期待値予想", "📊 厳格バックテスト", "🎯 全国スナイパー"])
today_str = datetime.datetime.now().strftime('%Y%m%d')

with tab1:
    st.write("「コース実績」「モーター機力」「当地相性」を学習し、リアルタイムオッズとケリー基準で最適な資金配分を計算します。")
    col1, col2, col3 = st.columns(3)
    with col1: target_date = st.text_input("日付 (例: 20261105)", value=today_str, key="date1")
    with col2: venue = st.selectbox("会場", options=list(VENUE_MAP.keys()), index=9, key="venue1")
    with col3: race_num = st.number_input("レース番号", min_value=1, max_value=12, value=5, step=1, key="race1")
    
    budget_input = st.number_input("このレースの総予算（円）", min_value=1000, max_value=100000, value=5000, step=1000)

    if st.button("⚡ v14 期待値計算＆資金配分！", type="primary"):
        with st.spinner("多次元マスタとリアルタイムオッズを同期中..."):
            res_dict = predict_race_auto_v14(target_date, venue, race_num, total_budget=budget_input)
            if "error" in res_dict:
                st.error(f"【予想失敗】 {res_dict['error']}")
            else:
                st.text_area("AIの期待値＆資金配分レポート", value=res_dict["text"], height=500)
                df_graph = res_dict["graph_data"]
                fig = px.bar(
                    df_graph, x='枠', y='スコア', color='枠', barmode='group', template="plotly_dark",
                    text_auto='.2f', color_discrete_sequence=['#ffffff', '#aaaaaa', '#ff4444', '#44aaff', '#ffee00', '#44ff44']
                )
                fig.update_layout(title="📊 v14 AI総合能力スコア", xaxis_title=None, yaxis_title="標準化スコア")
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.write("過去のレースに対して、オッズ期待値とケリー基準を用いたシミュレーションを行います。")
    col1_bt, col2_bt, col3_bt = st.columns(3)
    with col1_bt: bt_date = st.text_input("日付 (例: 20261105)", value=today_str, key="date2")
    with col2_bt: bt_venue = st.selectbox("会場", options=list(VENUE_MAP.keys()), index=9, key="venue2")
    with col3_bt: bt_budget = st.number_input("1レースあたりの予算（円）", min_value=1000, max_value=50000, value=5000, step=1000, key="budget2")
        
    if st.button("🔄 v14 金融工学バックテスト開始", type="primary"):
        my_bar = st.progress(0, text="バックテスト実行中...")
        results = []
        jcd = VENUE_MAP[bt_venue]
        scan_count, total_invest, total_return = 0, 0, 0
        
        def run_backtest_single(r_num):
            res_dict = predict_race_auto_v14(bt_date, bt_venue, r_num, is_backtest=True, total_budget=bt_budget)
            actual_result, actual_payout_val = fetch_actual_result(bt_date, jcd, r_num)
            time.sleep(0.3) 
            
            if res_dict and "error" not in res_dict:
                is_ken = res_dict["is_ken"]
                invest_dict = res_dict["invest_dict"]
                
                status, profit, invest, return_val = "-", 0, 0, 0
                
                if actual_payout_val == 0:
                    status = "未発走/中止"
                elif is_ken or not invest_dict:
                    status = "ケン(期待値不足)"
                else:
                    invest = sum(invest_dict.values())
                    if actual_result in invest_dict:
                        status = "🎯 期待値ハント成功！"
                        # 賭けた金額(100円単位)に応じて払戻しを計算
                        bet_amt = invest_dict[actual_result]
                        return_val = int(actual_payout_val * (bet_amt / 100))
                        profit = return_val - invest
                    else:
                        status = "❌ ハズレ"
                        profit = -invest
                
                return {
                    "R": f"{r_num}R", "投資判定": res_dict["bet_type"], "投資総額": invest,
                    "実際の結果": actual_result, "払戻金(100円)": f"¥{actual_payout_val:,}" if actual_payout_val > 0 else "-",
                    "判定": status, "収支": profit, "回収": return_val
                }
            else:
                return {"R": f"{r_num}R", "投資判定": "-", "投資総額": 0, "実際の結果": actual_result, "払戻金(100円)": "-", "判定": "-", "収支": 0, "回収": 0}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(run_backtest_single, r): r for r in range(1, 13)}
            for future in concurrent.futures.as_completed(futures):
                scan_count += 1
                my_bar.progress(scan_count / 12, text=f"スキャン中... 完了 {scan_count}/12 レース")
                res = future.result()
                results.append(res)
                total_invest += res["投資総額"]
                total_return += res["回収"]
                
        my_bar.empty()
        net_profit = total_return - total_invest
        roi = (total_return / total_invest * 100) if total_invest > 0 else 0
        
        st.success(f"{bt_venue}全12レースのバックテストが完了しました！")
        df_results = pd.DataFrame(results)
        df_results['R_num'] = df_results['R'].str.replace('R', '').astype(int)
        df_results = df_results.sort_values(by='R_num').drop(columns=['R_num']).set_index("R")
        
        df_display = df_results[['投資判定', '投資総額', '実際の結果', '払戻金(100円)', '判定', '収支']].copy()
        df_display['収支'] = df_display['収支'].apply(lambda x: f"+{x:,}円" if x > 0 else f"{x:,}円" if x < 0 else "-")
        st.dataframe(df_display, use_container_width=True)
        
        st.markdown("### 📊 v14 投資シミュレーション結果（ケリー基準運用）")
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        col_sum1.metric("総投資額", f"¥{total_invest:,}")
        col_sum2.metric("総払戻額", f"¥{total_return:,}")
        col_sum3.metric("回収率", f"{roi:.1f}%", f"{net_profit:,}円")

with tab3:
    st.header("🎯 全国スナイパー（超・鉄板レース一本釣りモード）")
    st.info("※v14スナイパー機能は、リアルタイムオッズの取得と期待値計算を伴うため、当日開催中のレースにのみ有効です。")

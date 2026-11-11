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
import concurrent.futures 

st.set_page_config(page_title="ボートレースAI【v18 最終形態・スナイパー】", layout="wide")

@st.cache_resource
def load_brain():
    with open('boat_ai_brain_v18_final.pkl', 'rb') as f:
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
    racer_kimarite = brain['racer_kimarite']
    racer_in_fail = brain['racer_in_fail']
    racer_names = brain.get('racer_names', {}) 
    suji_stats = brain['suji_stats']
    COURSE_AVG_1ST = brain['course_avg_1st']
    sweet_spots = brain.get('sweet_spots', {})
except Exception as e:
    st.error(f"⚠️ `boat_ai_brain_v18_final.pkl` の読み込みエラー: {e}")
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

def get_season(month):
    if month in [3, 4, 5]: return "春"
    elif month in [6, 7, 8]: return "夏"
    elif month in [9, 10, 11]: return "秋"
    else: return "冬"

def get_wind_category(wind_speed):
    if wind_speed <= 2: return "0-2m(穏やか)"
    elif wind_speed <= 4: return "3-4m(普通)"
    else: return "5m以上(荒れ)"

def scrape_boatrace_data(target_date, venue, race_num):
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

def show_racer_profile(target_toban):
    has_data = False
    for c in range(1, 7):
        if (target_toban, c) in racer_course_stats and racer_course_stats[(target_toban, c)]['runs'] > 0:
            has_data = True
            break
            
    if not has_data:
        st.warning(f"⚠️ 登番 {target_toban} のデータは過去の集計期間内に存在しません。")
        return
        
    racer_name = racer_names.get(target_toban, "")
    
    total_runs, total_wins, total_value_added, in_runs, in_crash = 0, 0, 0.0, 0, 0
    course_data = []
    
    for c in range(1, 7):
        c_st = racer_course_stats.get((target_toban, c), {'runs':0, '1st':0, '2nd':0, '3rd':0, '4th6th':0})
        runs, wins = c_st['runs'], c_st['1st']
        
        total_runs += runs
        total_wins += wins
        total_value_added += wins - (runs * COURSE_AVG_1ST.get(c, 0.0))
        
        if c == 1:
            in_runs = runs
            in_crash = c_st['4th6th']
            
        win_rate = (wins / runs * 100) if runs > 0 else 0.0
        ren_rate = ((wins + c_st['2nd']) / runs * 100) if runs > 0 else 0.0
        sanren_rate = ((wins + c_st['2nd'] + c_st['3rd']) / runs * 100) if runs > 0 else 0.0
        
        k_stats = racer_kimarite.get((target_toban, c), {'nige':0, 'sashi':0, 'makuri':0, 'makurisashi':0})
        
        course_data.append({
            "進入コース": f"{c}コース",
            "出走回数": runs,
            "1着率": f"{win_rate:.1f}%",
            "2連対率": f"{ren_rate:.1f}%",
            "3連対率": f"{sanren_rate:.1f}%",
            "逃げ": k_stats['nige'],
            "差し": k_stats['sashi'],
            "まくり": k_stats['makuri'],
            "まくり差し": k_stats['makurisashi']
        })
        
    true_skill_score = total_value_added / max(1, total_runs)
    fake_a1_rate = (in_crash / in_runs * 100) if in_runs > 0 else 0.0
    
    in_f_stats = racer_in_fail.get(target_toban, {'runs_1c':0, 'sasare':0, 'makurare':0})
    sasare_rate = (in_f_stats['sasare'] / max(1, in_f_stats['runs_1c']) * 100) if in_f_stats['runs_1c'] > 0 else 0.0
    makurare_rate = (in_f_stats['makurare'] / max(1, in_f_stats['runs_1c']) * 100) if in_f_stats['runs_1c'] > 0 else 0.0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("総出走数 (集計期間)", f"{total_runs}走")
    col2.metric("🏆 真の実力値 (0基準)", f"{true_skill_score:+.3f}", help="プラスなら実力派。")
    col3.metric("🚨 イン崩壊率 (4着以下)", f"{fake_a1_rate:.1f}%", help="インで大敗する確率。")
    col4.metric("💥 1コース負けパターン", f"差され: {sasare_rate:.0f}% / まくられ: {makurare_rate:.0f}%")
    
    df_course = pd.DataFrame(course_data)
    st.dataframe(df_course.set_index("進入コース"), use_container_width=True)

def predict_race_auto_v18(target_date, venue, race_num, is_backtest=False):
    boat_data, env_data, success, msg = scrape_boatrace_data(target_date, venue, race_num)
    if not success: return {"error": msg}
    
    jcd = int(VENUE_MAP[venue])
    target_month = int(str(target_date)[4:6]) if len(str(target_date)) >= 8 else datetime.datetime.now().month
    current_season = get_season(target_month)
    wind_cat = get_wind_category(env_data['風速'])
    
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
        
        total_runs_all, total_value_added, in_crash, in_runs_total = 0, 0.0, 0, 0
        for c in range(1, 7):
            c_st = racer_course_stats.get((toban, c), {'runs':0, '1st':0, '2nd':0, '3rd':0, '4th6th':0})
            runs, wins = c_st['runs'], c_st['1st']
            total_runs_all += runs
            total_value_added += wins - (runs * COURSE_AVG_1ST.get(c, 0.0))
            if c == 1:
                in_runs_total = runs
                in_crash = c_st['4th6th']

        true_skill_score = total_value_added / max(1, total_runs_all)
        fake_a1_rate = (in_crash / in_runs_total) if in_runs_total > 0 else 0.0

        c_stats = racer_course_stats.get((toban, course), {'runs':1, '1st':0, '2nd':0})
        v_stats = racer_venue_stats.get((toban, jcd), {'runs':1, '1st':0, '2nd':0})
        m_stats = motor_stats.get((jcd, motor), {'runs':1, '1st':0, '2nd':0})
        k_stats = racer_kimarite.get((toban, course), {'nige':0, 'sashi':0, 'makuri':0, 'makurisashi':0})
        total_wins = max(1, sum(k_stats.values()))
        in_f_stats = racer_in_fail.get(toban, {'runs_1c':1, 'sasare':0, 'makurare':0})
        
        inside_st = st_array[i-1] if i > 0 else st_array[i]
        
        input_dict = {
            '枠番': waku, '進入コース': course, '展示': boat_data[waku]['展示'], '平均ST': boat_data[waku]['平均ST'],
            '風速': env_data['風速'], '波高': env_data['波高'],
            'コース1着率': c_stats['1st'] / max(1, c_stats['runs']),
            '当地1着率': v_stats['1st'] / max(1, v_stats['runs']),
            'モーター2連対率': (m_stats['1st'] + m_stats['2nd']) / max(1, m_stats['runs']),
            'まくり比率': k_stats['makuri'] / max(1, total_wins),
            '差し比率': (k_stats['sashi'] + k_stats['makurisashi']) / max(1, total_wins),
            'イン差され率': in_f_stats['sasare'] / max(1, in_f_stats['runs_1c']),
            'インまくられ率': in_f_stats['makurare'] / max(1, in_f_stats['runs_1c']),
            '真の実力値': true_skill_score, 'イン崩壊率': fake_a1_rate,
            '内側ST差': boat_data[waku]['平均ST'] - inside_st, 
            '展示平均差': boat_data[waku]['展示'] - tenji_mean 
        }
        ordered_dict = {col: input_dict.get(col, 0.0) for col in train_features}
        input_list.append(ordered_dict)
        
        c_st_display = racer_course_stats.get((toban, course), {'runs':0, '1st':0, '2nd':0, '3rd':0, '4th6th':0})
        runs_display = c_st_display['runs']
        sanren_count = c_st_display['1st'] + c_st_display['2nd'] + c_st_display['3rd']
        boat_data[waku]['true_skill'] = true_skill_score
        boat_data[waku]['sanren_rate'] = (sanren_count / runs_display * 100) if runs_display > 0 else 0.0

    X_pred = pd.DataFrame(input_list)[train_features]
    
    lgb_scores = model_lgb.predict(X_pred)
    xgb_scores = model_xgb.predict(X_pred)
    z_lgb = (lgb_scores - np.mean(lgb_scores)) / (np.std(lgb_scores) + 1e-9)
    z_xgb = (xgb_scores - np.mean(xgb_scores)) / (np.std(xgb_scores) + 1e-9)
    ensemble_scores = (z_lgb + z_xgb) / 2.0
    
    score_details = {}
    for i in range(6):
        waku = i + 1
        t_name = racer_names.get(boat_data[waku]['登番'], "")
        c_actual = boat_data[waku]['進入コース']
        score_details[waku] = f"総合: {ensemble_scores[i]:.2f} (真の実力: {boat_data[waku]['true_skill']:+.3f} / {c_actual}ｺｰｽ3連対率: {boat_data[waku]['sanren_rate']:.1f}%)"
        
    win_probs = np.exp(ensemble_scores * 1.5) / np.sum(np.exp(ensemble_scores * 1.5))
    
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
        
        # 🔥 オッズフィルター的なロジック：大本命の時、スジが合わないものは評価を下げる（過大評価防止）
        if p1 > 0.4 and (p2_suji < 0.10 or p3_suji < 0.05):
             score = score * 0.5 
             
        raw_probs.append((f"{r1+1}-{r2+1}-{r3+1}", score))
        
    total_score = sum(x[1] for x in raw_probs)
    prob_sorted = sorted([(c, (s / total_score) * 100) for c, s in raw_probs], key=lambda x: x[1], reverse=True)
    
    top1_3t_prob = prob_sorted[0][1]
    ai_historical_hit_rate = sweet_spots.get((venue, current_season, wind_cat), 0.0)

    bets = []
    ken_reason = ""
    is_ken = False

    if top1_3t_prob < 12.0: # 基準を厳しくして無駄打ちを減らす
        is_ken = True
        ken_reason = "展開が混戦すぎる（大本命不在によるオッズ割れリスク）"
        bet_type_name = "🎯 展開スジ狙い（フォーメーション8点）"
        bets = [x[0] for x in prob_sorted[:8]]
    else:
        if ai_historical_hit_rate < 20.0:
            is_ken = True
            ken_reason = f"AIの苦手条件（{venue} × {current_season} × {wind_cat} の過去的中率 {ai_historical_hit_rate:.1f}%）"
        elif env_data['風速'] >= 6.0:
            is_ken = True
            ken_reason = "暴風による水面悪化（不確定要素大）"
            
        bet_type_name = "💎 究極スジ・超鉄板（激絞り4点）"
        bets = [x[0] for x in prob_sorted[:4]]

    if is_ken:
        bet_type_name = f"⚠️ 見送り推奨 (仮の買い目: {bet_type_name})"

    is_easy_hit = (top1_3t_prob >= 15.0) and not is_ken

    if is_backtest:
        return {
            "race_num": f"{race_num}R",
            "本命": prob_sorted[0][0],
            "total_bets": bets,
            "bet_type": bet_type_name,
            "is_ken": is_ken,
            "is_easy_hit": is_easy_hit,
            "top1_3t_prob": top1_3t_prob,
            "ai_hit_rate": ai_historical_hit_rate
        }
    
    result_text = f"🌐 【v18 最終形態・メタ学習スナイパー】解析完了！\n"
    result_text += f"🌪️ 水面環境: {venue} / {current_season} / 風速 {env_data['風速']}m ({wind_cat})\n"
    result_text += f"🧠 この条件でのAI過去的中率: {ai_historical_hit_rate:.1f}%\n"
    result_text += f"🎯 AI推奨 3連単1番人気 信頼度: {top1_3t_prob:.1f}%\n\n"
    
    if is_ken:
        result_text += f"⚠️ 【AI判定: ケン（見送り）】\n⛔ 理由: {ken_reason}\n\n"
    else:
        result_text += f"✨ 【AI投資判定】 最強のスイートスポット！勝負可能！\n\n"
    
    result_text += "📊 各艇のAIスコア詳細（1着抜け期待値）\n"
    for i in range(1, 7):
        detail = score_details[i]
        t_name = racer_names.get(boat_data[i]['登番'], "")
        name_str = f" {t_name}" if t_name else ""
        result_text += f"{i}枠{name_str}: 【{detail}】 (1着率 {win_probs[i-1]*100:.1f}%)\n"
        
    result_text += f"\n🎯 【スジ完全同期】3連単 上位8点予想\n"
    result_text += "-" * 50 + "\n"
    for i in range(min(8, len(prob_sorted))):
        result_text += f"第{i+1}推奨: 【{prob_sorted[i][0]}】 (確率 {prob_sorted[i][1]:.1f}%)\n"
        
    result_text += f"\n🎯 激絞り買い目指示\n"
    result_text += "-" * 50 + "\n"
    result_text += f"👉 展開予想買い目（{len(bets)}点）: {', '.join(bets)}\n"
    result_text += "-" * 50 + "\n"

    return {"text": result_text, "boat_data": boat_data}

st.title("🚤 ボートレースAI【v18 最終形態・メタ学習スナイパー】")

tab1, tab2, tab3 = st.tabs(["🚤 通常予想モード", "📊 厳格バックテスト", "🎯 全国スナイパー"])
today_str = datetime.datetime.now().strftime('%Y%m%d')

with tab1:
    st.write("「選手の実力」「スジ展開」に加え、AI自身が「自分の得意な季節・風速・会場」を理解してリスクを回避します。")
    col1, col2, col3 = st.columns(3)
    with col1: target_date = st.text_input("日付 (例: 20261105)", value=today_str, key="date1")
    with col2: venue = st.selectbox("会場", options=list(VENUE_MAP.keys()), index=9, key="venue1")
    with col3: race_num = st.number_input("レース番号", min_value=1, max_value=12, value=5, step=1, key="race1")

    if st.button("⚡ v18 メタ学習AI予想！", type="primary"):
        with st.spinner("AIのスイートスポットを照合し、展開を計算中..."):
            res_dict = predict_race_auto_v18(target_date, venue, race_num)
            if "error" in res_dict:
                st.error(f"【予想失敗】 {res_dict['error']}")
            else:
                st.text_area("AIのスコア解析結果", value=res_dict["text"], height=550)
                
                st.markdown("---")
                st.markdown("### 👤 出走選手のディープ・プロファイル")
                st.write("枠番・選手名をクリックすると、決まり手やイン崩壊率の詳細データが開きます。")
                
                boat_data = res_dict["boat_data"]
                for waku in range(1, 7):
                    toban = boat_data[waku]['登番']
                    r_name = racer_names.get(toban, "")
                    with st.expander(f"🚤 {waku}枠: {r_name} (登番: {toban}) のデータを見る"):
                        show_racer_profile(toban)

with tab2:
    st.write("AIが「ケン」と判定したレースも強制的に資金を投じて、仮に買っていた場合の収支を全て記録します。")
    col1_bt, col2_bt, col3_bt = st.columns(3)
    with col1_bt: bt_date = st.text_input("日付 (例: 20261105)", value=today_str, key="date2")
    with col2_bt: bt_venue = st.selectbox("会場", options=list(VENUE_MAP.keys()), index=9, key="venue2")
    with col3_bt: bt_budget = st.number_input("1レースあたりの予算（円）", min_value=800, max_value=50000, value=3000, step=500, key="budget2")
        
    if st.button("🔄 1R〜12R 全レース強制投資バックテスト開始", type="primary"):
        progress_text = "バックテスト実行中..."
        my_bar = st.progress(0, text=progress_text)
        results = []
        jcd = VENUE_MAP[bt_venue]
        scan_count, total_invest, total_return = 0, 0, 0
        
        def run_backtest_single(r_num):
            res_dict = predict_race_auto_v18(bt_date, bt_venue, r_num, is_backtest=True)
            actual_result, actual_payout_val = fetch_actual_result(bt_date, jcd, r_num)
            time.sleep(0.3) 
            
            if res_dict and "error" not in res_dict:
                is_ken, is_easy_hit, bets = res_dict["is_ken"], res_dict["is_easy_hit"], res_dict["total_bets"]
                status, profit, invest, return_val = "-", 0, 0, 0
                
                if actual_payout_val == 0:
                    status = "未発走/中止"
                else:
                    num_bets = len(bets)
                    base_amt = max(100, (bt_budget // num_bets // 100) * 100) if num_bets > 0 else 0
                    invest = base_amt * num_bets
                    if actual_result in bets and invest > 0:
                        status = "🎯 的中！" if not is_ken else "🎯 的中(AIは見送り判定)"
                        if is_easy_hit: status = "💎 的中(超・鉄板)！"
                        return_val = int(actual_payout_val * (base_amt / 100))
                        profit = return_val - invest
                    else:
                        status = "❌ ハズレ" if not is_ken else "❌ ハズレ(AIは見送り判定)"
                        profit = -invest
                
                return {
                    "R": f"{r_num}R", "投資判定": res_dict["bet_type"], "買い目": f"{len(bets)}点買い",
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
                my_bar.progress(scan_count / 12, text=f"スキャン中... 完了 {scan_count}/12 レース")
                res = future.result()
                results.append(res)
                total_invest += res["投資"]
                total_return += res["回収"]
                
        my_bar.empty()
        net_profit = total_return - total_invest
        roi = (total_return / total_invest * 100) if total_invest > 0 else 0
        
        st.success(f"{bt_venue}全12レースのバックテストが完了しました！")
        df_results = pd.DataFrame(results)
        df_results['R_num'] = df_results['R'].str.replace('R', '').astype(int)
        df_results = df_results.sort_values(by='R_num').drop(columns=['R_num']).set_index("R")
        
        df_display = df_results[['投資判定', '買い目', '実際の結果', '払戻金(100円)', '判定', '収支']].copy()
        df_display['収支'] = df_display['収支'].apply(lambda x: f"+{x:,}円" if x > 0 else f"{x:,}円" if x < 0 else "-")
        st.dataframe(df_display, use_container_width=True)
        
        st.markdown("### 📊 トータル収支（AIケン判定のレースも全て強制購入した結果）")
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
    
    if st.button("📡 メタ学習・一本釣り探知を開始する", type="primary"):
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
                res_dict = predict_race_auto_v18(snipe_date, venue, r_num, is_backtest=True)
                time.sleep(0.3) 
                if not res_dict or "error" in res_dict: return None
                
                if res_dict["is_easy_hit"]:
                    return {
                        "会場": venue, "R": f"{r_num}R", "狙い目": "💎 激絞り・超鉄板",
                        "AI確率(3連単)": f"{res_dict['top1_3t_prob']:.1f}%",
                        "過去実績(的中率)": f"{res_dict['ai_hit_rate']:.1f}%",
                        "指示買い目": "4点買い"
                    }
                return None

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(evaluate_race_for_sniper, t): t for t in race_targets}
                for future in concurrent.futures.as_completed(futures):
                    scan_count += 1
                    status_text.text(f"📡 メタ学習スキャン中... 完了 {scan_count}/{total_scans} レース")
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
                st.info("現在指定された会場には、AIの得意条件に合致する安全な鉄板レースはありません。（無駄打ち回避成功）")

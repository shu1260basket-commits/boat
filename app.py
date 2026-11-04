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
st.set_page_config(page_title="ボートレースAI【Ver 9.0 堅実特化版】", layout="centered")

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
            toban_match = re.search(r'(\d{4})\s*/\s*[A-B][1-2]', text_data)
            if not toban_match: toban_match = re.search(r'(\d{4})', text_data)
            toban = int(toban_match.group(1)) if toban_match else 0
            
            tds = tbody.find_all('td')
            try:
                motor_text = tds[6].text.strip()
                motor = int(re.search(r'^(\d{1,3})', motor_text).group(1))
            except: motor = 0
            boat_data[waku] = {'登番': toban, 'モーター': motor, '展示': 0.0}
            
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
            return None, None, False, f"直前情報が未公開です。展示が終わってから試してください。"
            
        for waku, tbody in enumerate(tbodies_before[:6], 1):
            tenji_match = re.findall(r'[5-7]\.\d{2}', tbody.text)
            if tenji_match:
                boat_data[waku]['展示'] = float(tenji_match[0])
                
        return boat_data, env_data, True, url_before
    except Exception as e:
        return None, None, False, f"読み込み失敗: {str(e)}"

# 3. 予想実行関数（★現実的・堅実重視の補正ロジックを追加）
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
    
    # ★【現実的補正】1号艇の1着確率が極端に低く出すぎるのを防ぎ、インの定石を少し優遇する
    # 1号艇の1着確率に1.2倍の現実的ボーナスを付与（暴走した大穴頭をマイルドにする）
    preds_1st[0] = preds_1st[0] * 1.25
    # 確率の合計を100%（1.0）に再正規化
    preds_1st = preds_1st / np.sum(preds_1st)

    combos = list(itertools.permutations([0, 1, 2, 3, 4, 5], 3))
    raw_probs = []
    for (r1, r2, r3) in combos:
        p = preds_1st[r1] * preds_2nd[r2] * preds_3rd[r3]
        raw_probs.append((f"{r1+1}-{r2+1}-{r3+1}", p))
        
    total_p = sum(x[1] for x in raw_probs)
    final_probs = [(c, (p / total_p) * 100) for c, p in raw_probs]
    
    result_text = f"🌐 【Ver 9.0 堅実・現実的狙い目版】解析完了！\n"
    result_text += f"🌪️ 環境: 風速 {env_data['風速']}m / 波高 {env_data['波高']}cm\n\n"
    
    for i in range(1, 7):
        st = input_dict[f'{i}枠_平均ST']
        m_2ren = input_dict[f'{i}枠_M2連対'] * 100
        result_text += f"{i}枠: 登番{boat_data[i]['登番']} / 平均ST: {st:.2f} / M2連対: {m_2ren:.1f}% / 展示: {boat_data[i]['展示']}\n"
        
    prob_sorted = sorted(final_probs, key=lambda x: x[1], reverse=True)
    top_prob = prob_sorted[0][1]
    
    if top_prob > 15.0:
        race_mood = "🟢【鉄板レース】本命イン信頼度高！"
    elif top_prob > 8.0:
        race_mood = "🟡【本線勝負】上位人気中心の堅実決着"
    else:
        race_mood = "🔴【混戦レース】軸選びに注意が必要"

    result_text += f"\n🎯 現実的・推奨買い目 ({race_mood})\n"
    result_text += "-" * 50 + "\n"
    for i, item in enumerate(prob_sorted[:6]):
        result_text += f"推奨 {i+1}点目: 【{item[0]}】 予想確率: {item[1]:>4.1f}%\n"

    result_text += "-" * 50 + "\n"
    result_text += "💡 【AIの狙い目アドバイス】\n"
    result_text += f"・本命軸: 【{prob_sorted[0][0]}】 (確率 {prob_sorted[0][1]:.1f}%)\n"
    result_text += f"・対抗厚め: 【{prob_sorted[1][0]}】 / 【{prob_sorted[2][0]}】\n"

    return result_text

# Streamlitの画面デザイン
st.title("🚤 ボートレースAI【Ver 9.0】堅実・現実的狙い目版")

today_str = datetime.datetime.now().strftime('%Y%m%d')

col1, col2, col3 = st.columns(3)
with col1:
    target_date = st.text_input("日付 (例: 20261103)", value=today_str)
with col2:
    venue = st.selectbox("会場", options=list(VENUE_MAP.keys()), index=9)
with col3:
    race_num = st.number_input("レース番号", min_value=1, max_value=12, value=5, step=1)

if st.button("⚡ 現実的な買い目を計算！", type="primary"):
    with st.spinner("堅実なロジックで解析中..."):
        res = predict_race_auto_v9(target_date, venue, race_num)
        st.text_area("AIの現実的解析結果", value=res, height=450)

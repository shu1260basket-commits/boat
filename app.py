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
st.set_page_config(page_title="ボートレースAI【決まり手完全連動・進化版】", layout="centered")

# 1. AIと各種辞書の読み込み
@st.cache_resource
def load_brain():
    # 読み込みはすでに成功しているはずなので、ファイル名はこのままか、
    # もしGitHub上で「boat_ai_brain_v9 .pkl」(スペースあり)にしている場合は直してな！
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

# 2. スクレイピング関数（今節の調子・ST・展示を網羅）
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
            rank = toban_match.group(2) if toban_match else "

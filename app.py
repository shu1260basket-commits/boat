import zipfile
import re
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
import xgboost as xgb
import itertools
from google.colab import files
import warnings
import os
warnings.filterwarnings('ignore')

print("🚀 【v20 連対至上主義モデル】AIの目標を『1着』から『3着以内』に根本改修し、学習を開始します...")

zip_file = "boat_history_txt-20261107T064207Z-1-001.zip" 

if not os.path.exists(zip_file):
    raise SystemExit(f"⚠️ エラー: '{zip_file}' が見つかりません。")

COURSE_AVG_1ST = {1: 0.55, 2: 0.16, 3: 0.13, 4: 0.10, 5: 0.05, 6: 0.01}
racer_course_stats = {} 
racer_venue_stats = {}  
motor_stats = {}        
racer_kimarite = {} 
racer_in_fail = {}
racer_names = {} 
suji_1st_count = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0} 
suji_12_matrix, suji_123_matrix = {}, {}

X_all, y_all, groups = [], [], []
eval_races = []

VENUE_NAMES = {
    1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖",
    7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江",
    13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山",
    19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村"
}

def _process_race(p_race, jcd, season, is_eval):
    boats = [p_race['boats'][w] for w in range(1, 7)]
    st_array = [d['st'] for d in boats]
    tenji_array = [d['tenji'] for d in boats]
    tenji_mean = np.mean(tenji_array)
    course_map = {d['waku']-1: d['course'] for d in boats}
    
    c1_win_rate = 0.55
    c2_st = 0.17
    max_dash_makuri = 0.0
    
    for d in boats:
        t, c = d['toban'], d['course']
        c_stats_tmp = racer_course_stats.get((t, c), {'runs':0, '1st':0})
        k_stats_tmp = racer_kimarite.get((t, c), {'makuri':0})
        total_wins_tmp = max(1, sum(racer_kimarite.get((t, c), {'a':0}).values()))
        
        if c == 1:
            c1_win_rate = c_stats_tmp['1st'] / max(1, c_stats_tmp['runs'])
        elif c == 2:
            c2_st = d['st']
        elif c >= 4:
            m_rate = k_stats_tmp.get('makuri', 0) / max(1, total_wins_tmp)
            if m_rate > max_dash_makuri: max_dash_makuri = m_rate

    race_features = []
    for i, d in enumerate(boats):
        waku, toban, motor, course = d['waku'], d['toban'], d['motor'], d['course']
        
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

        c_stats = racer_course_stats.get((toban, course), {'runs':1, '1st':0, '2nd':0, '3rd':0, '4th6th':0})
        v_stats = racer_venue_stats.get((toban, jcd), {'runs':1, '1st':0, '2nd':0})
        m_stats = motor_stats.get((jcd, motor), {'runs':1, '1st':0, '2nd':0})
        k_stats = racer_kimarite.get((toban, course), {'nige':0, 'sashi':0, 'makuri':0, 'makurisashi':0})
        total_wins = max(1, sum(k_stats.values()))
        in_f_stats = racer_in_fail.get(toban, {'runs_1c':1, 'sasare':0, 'makurare':0})
        
        inside_st = st_array[i-1] if i > 0 else st_array[i]
        
        # 🔥 新しい特徴量：3連対率（3着内率）
        sanren_count = c_stats['1st'] + c_stats['2nd'] + c_stats['3rd']
        sanren_rate = (sanren_count / max(1, c_stats['runs']))
        
        feat = {
            '枠番': waku, '進入コース': course, '展示': d['tenji'], '平均ST': d['st'],
            '風速': p_race['wind'], '波高': p_race['wave'],
            'コース1着率': c_stats['1st'] / max(1, c_stats['runs']),
            'コース3連対率': sanren_rate, # 🔥 AIに3連対率を直接学習させる
            '当地1着率': v_stats['1st'] / max(1, v_stats['runs']),
            'モーター2連対率': (m_stats['1st'] + m_stats['2nd']) / max(1, m_stats['runs']),
            'まくり比率': k_stats['makuri'] / max(1, total_wins),
            '差し比率': (k_stats['sashi'] + k_stats['makurisashi']) / max(1, total_wins),
            'イン差され率': in_f_stats['sasare'] / max(1, in_f_stats['runs_1c']),
            'インまくられ率': in_f_stats['makurare'] / max(1, in_f_stats['runs_1c']),
            '真の実力値': true_skill_score, 'イン崩壊率': fake_a1_rate,
            '内側ST差': d['st'] - inside_st, 
            '展示平均差': d['tenji'] - tenji_mean,
            '1コース信頼度': c1_win_rate,
            '2コース壁ST': c2_st,
            'ダッシュ攻め力': max_dash_makuri
        }
        race_features.append(feat)
        
        # 🔥 最重要：学習目標（y_all）の大手術
        # 1〜3着は「1（アタリ）」、4〜6着は「0（ハズレ）」として学習
        if d['rank'] <= 3:
            y_all.append(1)
        else:
            y_all.append(0)
            
        X_all.append(feat)
        
    groups.append(6)
    
    if is_eval:
        actual_1st = next((d['waku'] for d in boats if d['rank'] == 1), 0)
        actual_2nd = next((d['waku'] for d in boats if d['rank'] == 2), 0)
        actual_3rd = next((d['waku'] for d in boats if d['rank'] == 3), 0)
        actual_combo = f"{actual_1st}-{actual_2nd}-{actual_3rd}"
        wind_cat = "0-2m(穏やか)" if p_race['wind'] <= 2 else "3-4m(普通)" if p_race['wind'] <= 4 else "5m以上(荒れ)"
        
        eval_races.append({
            'features': race_features, 'course_map': course_map,
            'actual_combo': actual_combo, 'jcd': jcd, 'season': season, 'wind_cat': wind_cat
        })
        
    for d in boats:
        t, c, m, r = d['toban'], d['course'], d['motor'], d['rank']
        if (t, c) not in racer_course_stats: racer_course_stats[(t, c)] = {'runs':0, '1st':0, '2nd':0, '3rd':0, '4th6th':0}
        racer_course_stats[(t, c)]['runs'] += 1
        if r == 1: racer_course_stats[(t, c)]['1st'] += 1
        elif r == 2: racer_course_stats[(t, c)]['2nd'] += 1
        elif r == 3: racer_course_stats[(t, c)]['3rd'] += 1
        elif r >= 4: racer_course_stats[(t, c)]['4th6th'] += 1
        
        if (t, jcd) not in racer_venue_stats: racer_venue_stats[(t, jcd)] = {'runs':0, '1st':0, '2nd':0}
        racer_venue_stats[(t, jcd)]['runs'] += 1
        if r == 1: racer_venue_stats[(t, jcd)]['1st'] += 1
        elif r == 2: racer_venue_stats[(t, jcd)]['2nd'] += 1
        
        if (jcd, m) not in motor_stats: motor_stats[(jcd, m)] = {'runs':0, '1st':0, '2nd':0}
        motor_stats[(jcd, m)]['runs'] += 1
        if r == 1: motor_stats[(jcd, m)]['1st'] += 1
        elif r == 2: motor_stats[(jcd, m)]['2nd'] += 1
        
    winner = next((d for d in boats if d['rank'] == 1), None)
    second = next((d for d in boats if d['rank'] == 2), None)
    third = next((d for d in boats if d['rank'] == 3), None)
    
    c1 = winner['course'] if winner else 0
    c2 = second['course'] if second else 0
    c3 = third['course'] if third else 0
    
    if c1 > 0:
        suji_1st_count[c1] += 1
        if c2 > 0:
            suji_12_matrix[(c1, c2)] = suji_12_matrix.get((c1, c2), 0) + 1
            if c3 > 0: suji_123_matrix[(c1, c2, c3)] = suji_123_matrix.get((c1, c2, c3), 0) + 1
                
    if winner:
        wt, wc = winner['toban'], winner['course']
        if (wt, wc) not in racer_kimarite: racer_kimarite[(wt, wc)] = {'nige':0, 'sashi':0, 'makuri':0, 'makurisashi':0, 'nuki':0}
        km = p_race['kimarite']
        if km == '逃げ': racer_kimarite[(wt, wc)]['nige'] += 1
        elif km == '差し': racer_kimarite[(wt, wc)]['sashi'] += 1
        elif km == 'まくり': racer_kimarite[(wt, wc)]['makuri'] += 1
        elif km == 'まくり差し': racer_kimarite[(wt, wc)]['makurisashi'] += 1
        elif km in ['抜き', '恵まれ']: racer_kimarite[(wt, wc)]['nuki'] += 1
        
    in_racer = next((d for d in boats if d['course'] == 1), None)
    if in_racer:
        in_t = in_racer['toban']
        if in_t not in racer_in_fail: racer_in_fail[in_t] = {'runs_1c':0, 'sasare':0, 'makurare':0}
        racer_in_fail[in_t]['runs_1c'] += 1
        
        if winner and winner['course'] != 1:
            km = p_race['kimarite']
            if km in ['差し', 'まくり差し']: racer_in_fail[in_t]['sasare'] += 1
            elif km == 'まくり': racer_in_fail[in_t]['makurare'] += 1

with zipfile.ZipFile(zip_file, 'r') as z:
    txt_files = [f for f in z.namelist() if f.endswith('.TXT') or f.endswith('.txt')]
    num_files = len(txt_files)
    
    for idx, txt_file in enumerate(txt_files):
        is_eval_target = (idx >= num_files - 5000)
        jcd_match = re.search(r'(\d{2})\.TXT', txt_file.upper())
        jcd = int(jcd_match.group(1)) if jcd_match else 0
        
        with z.open(txt_file) as f:
            content = f.read().decode('shift_jis', errors='ignore')
        
        lines = content.splitlines()
        season = "不明"
        pending_race = None
        
        path_match = re.search(r'(20\d{2})(\d{2})(\d{2})', txt_file)
        if path_match:
            month = int(path_match.group(2))
            if month in [3, 4, 5]: season = "春"
            elif month in [6, 7, 8]: season = "夏"
            elif month in [9, 10, 11]: season = "秋"
            else: season = "冬"
        
        for line in lines:
            clean_line = line.replace(' ', '').replace(' ', '')
            if season == "不明":
                month_match = re.search(r'(\d{1,2})月', clean_line)
                if not month_match: month_match = re.search(r'20\d{2}/(\d{1,2})/', clean_line)
                if month_match:
                    month = int(month_match.group(1))
                    if month in [3, 4, 5]: season = "春"
                    elif month in [6, 7, 8]: season = "夏"
                    elif month in [9, 10, 11]: season = "秋"
                    elif month in [12, 1, 2]: season = "冬"

            race_num_match = re.search(r'^\s*(\d{1,2})R', line)
            if not race_num_match:
                 race_num_match = re.search(r'第\s*(\d{1,2})\s*レース', line)
                 
            if race_num_match:
                r_num = int(race_num_match.group(1))
                if pending_race and pending_race.get('r_num') != r_num:
                    if len(pending_race['boats']) == 6:
                        _process_race(pending_race, jcd, season, is_eval_target)
                    pending_race = {'boats': {}, 'wind': 0.0, 'wave': 0.0, 'kimarite': '不明', 'r_num': r_num}
                elif not pending_race:
                    pending_race = {'boats': {}, 'wind': 0.0, 'wave': 0.0, 'kimarite': '不明', 'r_num': r_num}
            
            if not pending_race:
                continue
                
            if '風' in line and '波' in line:
                wind_m = re.search(r'風.*?(\d+)m', clean_line)
                if wind_m: pending_race['wind'] = float(wind_m.group(1))
                wave_m = re.search(r'波.*?(\d+)cm', clean_line)
                if wave_m: pending_race['wave'] = float(wave_m.group(1))

            if pending_race['kimarite'] == '不明':
                kima_match = re.search(r'(まくり差し|まくり|差し|逃げ|抜き|恵まれ)', clean_line)
                if kima_match:
                    pending_race['kimarite'] = kima_match.group(1)

            match = re.search(r'^\s*([0-9A-Za-z]+)\s+(\d)\s+(\d{4})\s+(.*?)\s+(\d{1,3})\s+(\d{1,3})\s+(\d\.\d{2})\s+(\d)\s+([FfLl]?\d\.\d{2})', line)
            if match:
                rank_str = match.group(1)
                rank = int(rank_str) if rank_str.isdigit() else 6
                waku = int(match.group(2))
                toban = int(match.group(3))
                raw_name = match.group(4).replace(' ', '').replace(' ', '')
                racer_names[toban] = raw_name
                motor = int(match.group(5))
                course = int(match.group(8))
                st_str = match.group(9).replace('F', '-').replace('L', '-').replace('f', '-').replace('l', '-')
                try: st_val = float(st_str)
                except: st_val = 0.20
                    
                pending_race['boats'][waku] = {
                    'waku': waku, 'toban': toban, 'course': course, 
                    'motor': motor, 'st': st_val, 'tenji': float(match.group(7)), 'rank': rank
                }
        
        if pending_race and len(pending_race['boats']) == 6:
            _process_race(pending_race, jcd, season, is_eval_target)

if len(X_all) == 0:
    raise SystemExit("⚠️ データを1件も読み込めませんでした。ZIPファイルを確認してください。")

df_X = pd.DataFrame(X_all)
y = np.array(y_all)
groups = np.array(groups)
train_features = df_X.columns.tolist()

print(f"✅ 【連対データ構築】 {len(groups)} レース分完了！")

# 🔥 目的変数が 0/1 の2値分類になったため、Rankerではなく Classifier を使用
model_lgb = lgb.LGBMClassifier(
    objective='binary', metric='binary_logloss', 
    learning_rate=0.05, num_leaves=63, max_depth=7, colsample_bytree=0.8, subsample=0.8, verbose=-1
)
model_lgb.fit(df_X, y)

model_xgb = xgb.XGBClassifier(
    objective='binary:logistic', 
    learning_rate=0.05, max_depth=6, n_estimators=150
)
model_xgb.fit(df_X, y)

suji_stats = {'1st_count': suji_1st_count, '12_matrix': suji_12_matrix, '123_matrix': suji_123_matrix}

backtest_results = []
for race in eval_races:
    X_pred = pd.DataFrame(race['features'])[train_features]
    
    # 🔥 predict_proba で「3着以内に入る確率」を直接取得
    lgb_scores = model_lgb.predict_proba(X_pred)[:, 1]
    xgb_scores = model_xgb.predict_proba(X_pred)[:, 1]
    
    # 2つのモデルの確率を平均
    ensemble_scores = (lgb_scores + xgb_scores) / 2.0
    
    # 確率を正規化して、1着確率（win_probs）として扱う
    win_probs = ensemble_scores / np.sum(ensemble_scores)
    
    combos = list(itertools.permutations([0, 1, 2, 3, 4, 5], 3))
    raw_probs = []
    for (r1, r2, r3) in combos:
        c1, c2, c3 = race['course_map'][r1], race['course_map'][r2], race['course_map'][r3]
        p1 = win_probs[r1]
        p2_pl = win_probs[r2] / max(0.0001, 1.0 - win_probs[r1])
        p2_suji = max(0.01, suji_stats['12_matrix'].get((c1, c2), 0) / max(1, suji_stats['1st_count'].get(c1, 1))) 
        p3_pl = win_probs[r3] / max(0.0001, 1.0 - win_probs[r1] - win_probs[r2])
        p3_suji = max(0.01, suji_stats['123_matrix'].get((c1, c2, c3), 0) / max(1, suji_stats['12_matrix'].get((c1, c2), 1)))
        raw_probs.append((f"{r1+1}-{r2+1}-{r3+1}", p1 * (p2_pl * p2_suji) * (p3_pl * p3_suji)))
        
    prob_sorted = sorted([(c, (s / sum(x[1] for x in raw_probs)) * 100) for c, s in raw_probs], key=lambda x: x[1], reverse=True)
    top4_bets = [x[0] for x in prob_sorted[:4]]
    is_hit = 1 if race['actual_combo'] in top4_bets else 0
    
    backtest_results.append({
        '会場': VENUE_NAMES.get(race['jcd'], "不明"),
        '季節': race['season'], '風速条件': race['wind_cat'],
        'AI最高信頼度(%)': prob_sorted[0][1], '的中(Top4)': is_hit
    })

df_results = pd.DataFrame(backtest_results)
pivot_df = df_results.groupby(['会場', '季節', '風速条件']).agg(
    レース数=('的中(Top4)', 'count'), 的中数=('的中(Top4)', 'sum')
).reset_index()

pivot_df['AI的中率(%)'] = (pivot_df['的中数'] / pivot_df['レース数']) * 100
sweet_spot_dict = {(r['会場'], r['季節'], r['風速条件']): r['AI的中率(%)'] for _, r in pivot_df.iterrows()}

brain_v20 = {
    'model_lgb': model_lgb, 'model_xgb': model_xgb, 'features': train_features,
    'racer_course_stats': racer_course_stats, 'racer_venue_stats': racer_venue_stats, 'motor_stats': motor_stats,
    'racer_kimarite': racer_kimarite, 'racer_in_fail': racer_in_fail, 'racer_names': racer_names,
    'suji_stats': suji_stats, 'course_avg_1st': COURSE_AVG_1ST, 'sweet_spots': sweet_spot_dict
}

filename = 'boat_ai_brain_v20.pkl'
with open(filename, 'wb') as f:
    pickle.dump(brain_v20, f)

print(f"\n✨ 【v20完成】「3着以内に入る確率」を直接学習した連対至上主義モデルを保存しました！")
files.download(filename)

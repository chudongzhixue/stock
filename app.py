import streamlit as st
import pandas as pd
import requests
import os
import time
import shutil
import numpy as np
import akshare as ak
import yfinance as yf
from datetime import datetime, timedelta, time as dt_time
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 页面基础设置 ---
st.set_page_config(
    page_title="Alpha 游资系统 Pro + AI",
    page_icon="🐲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 🎨 CSS 样式 ---
st.markdown("""
    <style>
        html, body, p, div, span { font-family: 'Source Sans Pro', sans-serif; color: #0E1117; }
        .block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; }
        
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid #e6e6e6 !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08); 
            background-color: #ffffff; 
            padding: 15px !important;
            border-radius: 12px;
            margin-bottom: 15px;
        }

        .big-price { font-size: 2.2rem; font-weight: 900; line-height: 1.0; letter-spacing: -1px; margin-bottom: 5px; }
        .price-up { color: #d9534f; }
        .price-down { color: #5cb85c; }
        .price-gray { color: #888; }
        
        .stock-name { font-size: 1.1rem; font-weight: bold; color: #222; }
        .stock-code { font-size: 0.8rem; color: #888; margin-left: 5px; }
        
        .strategy-tag { padding: 2px 6px; border-radius: 3px; font-size: 0.75rem; font-weight: bold; color: white; display: inline-block; vertical-align: middle; margin-right: 4px; margin-bottom: 4px;}
        .tag-dragon { background: linear-gradient(45deg, #ff0000, #ff6b6b); }
        .tag-first { background: linear-gradient(45deg, #ff9f43, #ff6b6b); }
        .tag-buy { background-color: #d9534f; }
        .tag-sell { background-color: #5cb85c; }
        .tag-wait { background-color: #999; }
        .tag-special { background-color: #f0ad4e; }
        .tag-purple { background: linear-gradient(45deg, #8e44ad, #c0392b); }

        .cost-range-box { background-color: #f8f9fa; border-left: 3px solid #666; padding: 2px 6px; margin: 5px 0; border-radius: 0 4px 4px 0; font-size: 0.75rem; color: #444; }
        
        .plan-container { font-size: 0.85rem; color: #444; padding: 5px; }
        .plan-title { font-weight: bold; color: #2c3e50; font-size: 0.9rem; margin-bottom: 5px; border-bottom: 1px dashed #ddd; padding-bottom: 3px;}
        .plan-item { margin-bottom: 4px; line-height: 1.4; }
        .highlight-money { color: #d9534f; font-weight: bold; background: #fff5f5; padding: 0 4px; border-radius: 3px; }
        
        .advice-box { margin-top: 5px; padding: 6px; border-radius: 4px; font-weight: bold; text-align: center; font-size: 0.85rem; }
        .advice-buy { background-color: #d9534f; color: white; animation: pulse 2s infinite;}
        .advice-sell { background-color: #5cb85c; color: white; }
        .advice-hold { background-color: #3498db; color: white; }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(217, 83, 79, 0.4); }
            70% { box-shadow: 0 0 0 6px rgba(217, 83, 79, 0); }
            100% { box-shadow: 0 0 0 0 rgba(217, 83, 79, 0); }
        }
        
        .sr-block { padding-top: 6px; border-top: 1px dashed #eee; display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }
        .sr-item { font-size: 0.8rem; font-weight: bold; color: #555; }
        div[data-testid="stButton"] button { width: 100%; }
        
        .streamlit-expanderHeader {
            font-size: 0.9rem !important;
            font-weight: bold !important;
            color: #333 !important;
            background-color: #f8f9fa !important;
            border-radius: 4px !important;
        }
    </style>
""", unsafe_allow_html=True)

DATA_FILE = 'my_stock_plan_v3.csv'
TRAIN_DATA_FILE = 'ai_training_dataset.csv'
VIDEO_DIR = 'training_videos'

# 确保视频目录存在
if not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)

# --- 核心函数 ---

def save_data(df): df.to_csv(DATA_FILE, index=False)

def load_data():
    if not os.path.exists(DATA_FILE):
        df = pd.DataFrame(columns=["code", "name", "s1", "s2", "r1", "r2", "group", "note"])
        df.to_csv(DATA_FILE, index=False)
        return df
    df = pd.read_csv(DATA_FILE, dtype={"code": str})
    expected_cols = ["code", "name", "s1", "s2", "r1", "r2", "group", "note"]
    for col in expected_cols:
        if col not in df.columns: df[col] = 0.0
    df = df[expected_cols]
    df['code'] = df['code'].str.strip()
    df.drop_duplicates(subset=['code'], keep='last', inplace=True)
    return df

# 🔥 AI 数据管理核心 🔥
def load_train_data():
    if not os.path.exists(TRAIN_DATA_FILE):
        # 扩展了字段：包含主力成本、策略类型、视频路径、次日结果等
        cols = ["record_date", "code", "name", "strategy_type", "price_at_entry", 
                "cost_at_entry", "video_path", "note", 
                "next_day_open_pct", "next_day_high_pct", "next_day_close_pct", "result_label"]
        df = pd.DataFrame(columns=cols)
        df.to_csv(TRAIN_DATA_FILE, index=False)
        return df
    return pd.read_csv(TRAIN_DATA_FILE, dtype={"code": str})

def save_train_record_with_video(code, name, price, cost, strategy, video_file, note):
    df = load_train_data()
    today = datetime.now().strftime("%Y-%m-%d")
    
    # 保存视频文件
    video_path = ""
    if video_file is not None:
        # 文件名: 日期_代码_策略.mp4
        file_ext = video_file.name.split('.')[-1]
        safe_name = f"{today}_{code}_{strategy}.{file_ext}"
        video_path = os.path.join(VIDEO_DIR, safe_name)
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
    
    new_record = {
        "record_date": today,
        "code": code,
        "name": name,
        "strategy_type": strategy,
        "price_at_entry": price,
        "cost_at_entry": cost, # 记录当时的主力成本，这对于后续训练至关重要
        "video_path": video_path,
        "note": note,
        "next_day_open_pct": 0.0, # 待回填
        "next_day_high_pct": 0.0, # 待回填
        "next_day_close_pct": 0.0, # 待回填
        "result_label": "⏳ 待验证"
    }
    
    # 覆盖当日同策略记录
    df = df[~((df['record_date'] == today) & (df['code'] == code) & (df['strategy_type'] == strategy))]
    df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
    df.to_csv(TRAIN_DATA_FILE, index=False)
    return True

# 🔥 自动回填逻辑 (Auto-Labeling)
def auto_label_data():
    df = load_train_data()
    if df.empty: return "无数据"
    
    count = 0
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    for index, row in df.iterrows():
        # 如果已经有结果，或者是今天的记录(没法验证)，跳过
        if row['result_label'] != "⏳ 待验证" or row['record_date'] == today_str:
            continue
            
        # 获取该股历史数据来验证
        try:
            # 简单逻辑：取记录日期的下一天数据
            # 实际需获取该股的日线数据
            hist = ak.stock_zh_a_hist(symbol=row['code'], period="daily", adjust="qfq")
            hist['日期'] = pd.to_datetime(hist['日期']).dt.strftime('%Y-%m-%d')
            
            # 找到记录日期的索引
            record_idx = hist[hist['日期'] == row['record_date']].index
            if not record_idx.empty and record_idx[0] + 1 < len(hist):
                next_day = hist.iloc[record_idx[0] + 1]
                
                # 计算次日表现
                open_pct = next_day['开盘'] / next_day['前收盘'] - 1
                high_pct = next_day['最高'] / next_day['前收盘'] - 1
                close_pct = next_day['收盘'] / next_day['前收盘'] - 1 # 也就是涨跌幅
                
                df.at[index, 'next_day_open_pct'] = round(open_pct * 100, 2)
                df.at[index, 'next_day_high_pct'] = round(high_pct * 100, 2)
                df.at[index, 'next_day_close_pct'] = round(close_pct, 2) # akshare涨跌幅本身就是百分比
                
                # 简单自动打标逻辑 (可自定义)
                if close_pct > 5 or high_pct > 8:
                    df.at[index, 'result_label'] = "✅ 成功(大肉)"
                elif close_pct > 0:
                    df.at[index, 'result_label'] = "⭕ 成功(小肉)"
                elif close_pct < -5:
                    df.at[index, 'result_label'] = "❌ 失败(大面)"
                else:
                    df.at[index, 'result_label'] = "➖ 失败(亏损)"
                
                count += 1
        except:
            pass
            
    if count > 0:
        df.to_csv(TRAIN_DATA_FILE, index=False)
    return f"已自动回填 {count} 条历史数据的验证结果"

def delete_single_stock(code_to_delete):
    df = load_data()
    if code_to_delete in df['code'].values:
        df = df[df['code'] != code_to_delete]
        save_data(df)
        return True
    return False

def is_trading_time():
    now = datetime.utcnow() + timedelta(hours=8)
    if now.weekday() >= 5: return False, "周末休市"
    current_time = now.time()
    am_start, am_end = dt_time(9, 15), dt_time(11, 30)
    pm_start, pm_end = dt_time(13, 0), dt_time(15, 0)
    if (am_start <= current_time <= am_end) or (pm_start <= current_time <= pm_end):
        return True, "交易中"
    return False, "非交易时间"

def get_realtime_quotes(code_list):
    if not code_list: return {}
    query_codes = [f"{'sh' if c.startswith(('6', '5')) else 'sz'}{c}" for c in code_list]
    url = f"http://hq.sinajs.cn/list={','.join(query_codes)}"
    try:
        r = requests.get(url, headers={'Referer': 'http://finance.sina.com.cn'}, timeout=3)
        data = {}
        for line in r.text.split('\n'):
            if '="' in line:
                code = line.split('="')[0].split('_')[-1][2:]
                val = line.split('="')[1].strip('";').split(',')
                if len(val) > 30:
                    data[code] = {
                        "name": val[0], "open": float(val[1]), "pre_close": float(val[2]), 
                        "price": float(val[3]), "high": float(val[4]), "low": float(val[5]),
                        "vol": float(val[8]), "amount": float(val[9])
                    }
        return data
    except: return {}

@st.cache_data(ttl=3600)
def get_stock_history_metrics(code):
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=120)).strftime("%Y%m%d") 
    stock_df = None
    try:
        stock_df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
    except: pass
    if stock_df is None or stock_df.empty:
        try:
            y_code = f"{code}.SS" if code.startswith('6') else f"{code}.SZ"
            if code.startswith(('8', '4')): y_code = f"{code}.BJ"
            y_data = yf.download(y_code, period="6mo", progress=False)
            if not y_data.empty:
                y_data = y_data.reset_index()
                y_data.columns = ['日期', '开盘', '最高', '最低', '收盘', '成交量'] if len(y_data.columns)==6 else y_data.columns
                y_data.rename(columns={'Date': '日期', 'Open': '开盘', 'High': '最高', 'Low': '最低', 'Close': '收盘', 'Volume': '成交量'}, inplace=True)
                y_data['涨跌幅'] = y_data['收盘'].pct_change() * 100
                y_data['成交额'] = y_data['收盘'] * y_data['成交量'] 
                stock_df = y_data
        except: pass

    if stock_df is not None and not stock_df.empty:
        try:
            stock_df['MA5'] = stock_df['收盘'].rolling(5).mean()
            stock_df['MA10'] = stock_df['收盘'].rolling(10).mean()
            recent = stock_df.tail(20)
            total_amt = recent['成交额'].sum()
            total_vol = recent['成交量'].sum()
            avg_cost = (total_amt / total_vol) if total_vol > 0 else 0
            if avg_cost > 200: avg_cost /= 100
            stock_df['is_zt'] = stock_df['涨跌幅'] > 9.5
            zt_count = 0
            check_df = stock_df.copy()
            for i in range(len(check_df)-1, -1, -1):
                if check_df.iloc[i]['is_zt']: zt_count += 1
                else: break
            recent_15 = stock_df.tail(20)
            max_streak = 0
            curr_str = 0
            for zt in recent_15['is_zt']:
                if zt: curr_str += 1
                else:
                    max_streak = max(max_streak, curr_str)
                    curr_str = 0
            max_streak = max(max_streak, curr_str)
            recent_60 = stock_df.tail(60)
            max_amount_60d = recent_60['成交额'].max()
            return stock_df, avg_cost, zt_count, check_df.iloc[-2]['is_zt'] if len(check_df) > 1 else False, max_streak, max_amount_60d
        except: return None, 0, 0, False, 0, 0
    return None, 0, 0, False, 0, 0

def format_money(num):
    if pd.isna(num) or num == 0: return "N/A"
    num = float(num)
    if num > 100000000: return f"{num/100000000:.2f}亿"
    if num > 10000: return f"{num/10000:.2f}万"
    return f"{num:.2f}"

def generate_plan_and_advice(code, name, current_price, open_price, pre_close, max_amount_60d, zt_count):
    plan_html = ""
    advice_html = ""
    target_auction_amt = max_amount_60d * 0.05
    exp_open_low = current_price * 1.02
    exp_open_high = current_price * 1.06
    
    plan_html += f"<div class='plan-title'>🎲 {zt_count}进{zt_count+1} 操盘推演</div>"
    plan_html += f"<div class='plan-item'>🎯 <b>竞价目标：</b><span class='highlight-money'>{format_money(target_auction_amt)}</span> (60日最大成交5%)</div>"
    plan_html += f"<div class='plan-item'>📊 <b>理想开盘：</b>{exp_open_low:.2f} ~ {exp_open_high:.2f} (+2%~+6%)</div>"
    plan_html += "<hr style='margin:4px 0; border-top:1px dashed #ddd;'>"
    plan_html += "<div class='plan-item'>1. <b>🔥 弱转强(买点)：</b>高开>3%，竞价金额达标，开盘分时均线支撑不破。</div>"
    plan_html += "<div class='plan-item'>2. <b>❄️ 不及预期(卖点)：</b>低开/平开，竞价无量，开盘迅速跌破均线。</div>"
    plan_html += "<div class='plan-item'>3. <b>🔒 缩量锁仓：</b>竞价/开盘直接涨停(一字/秒板)，量能极小。👉 **持有不动**。</div>"

    trading_active, _ = is_trading_time()
    
    if trading_active and open_price > 0:
        advice_text = ""
        advice_class = ""
        pct = (current_price - pre_close) / pre_close * 100
        open_pct = (open_price - pre_close) / pre_close * 100
        if current_price >= (pre_close * 1.098):
            advice_text = "🔒 涨停锁仓"
            advice_class = "advice-hold"
        elif open_pct > 2 and current_price > open_price and pct > 5:
            advice_text = "🔴 弱转强：关注确认"
            advice_class = "advice-buy"
        elif open_pct < 0 and current_price < open_price:
            advice_text = "🟢 不及预期：离场"
            advice_class = "advice-sell"
        elif current_price < pre_close:
            advice_text = "🟢 水下震荡：观望"
            advice_class = "advice-sell"
        else:
            advice_text = "🔵 盘中震荡"
            advice_class = "advice-hold"
        advice_html = f"<div class='advice-box {advice_class}'>{advice_text}</div>"
    
    return plan_html, advice_html

def ai_strategy_engine(info, history_df, smart_cost, zt_count, yesterday_zt, max_streak):
    price = info['price']
    pre_close = info['pre_close']
    high = info['high']
    pct_chg = ((price - pre_close) / pre_close) * 100
    day_vwap = info['amount'] / info['vol'] if info['vol'] > 0 else price
    if history_df is None or history_df.empty: return "数据加载中...", "tag-wait"
    try:
        ma5 = history_df.iloc[-1]['MA5']
        ma10 = history_df.iloc[-1]['MA10']
    except: return "数据错误", "tag-wait"

    if max_streak >= 4:
        if zt_count > 0: return f"🔥 妖股加速 ({zt_count}板)", "tag-dragon"
        elif pct_chg > 5.0: return "🦁 龙头二波", "tag-purple"
        elif pct_chg < -5.0 and price > ma10: return "🐲 龙头首阴", "tag-special"
        else: return "💀 龙头退潮", "tag-sell"

    if zt_count >= 2: return f"🚀 {zt_count}连板持筹", "tag-dragon"
    if not yesterday_zt and pct_chg > 9.5: return "🚀 首板启动", "tag-first"
    if yesterday_zt and zt_count < 2:
        if 2 < pct_chg < 9.0 and price > day_vwap: return "🚀 1进2 接力", "tag-buy"
        if pct_chg > 9.0: return "🚀 秒板/一字", "tag-dragon"
    high_pct = ((high - pre_close) / pre_close) * 100
    if high_pct > 7 and pct_chg < 3 and price > ma5: return "👆 仙人指路", "tag-special"
    if pct_chg > 0 and price > day_vwap: return "💪 趋势向上", "tag-wait"
    if pct_chg < 0 and price < day_vwap: return "🤢 弱势调整", "tag-wait"
    return "😴 观望", "tag-wait"

def prefetch_all_data(stock_codes):
    results = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_code = {executor.submit(get_stock_history_metrics, code): code for code in stock_codes}
        for future in as_completed(future_to_code):
            code = future_to_code[future]
            try: results[code] = future.result()
            except: results[code] = (None, 0, 0, False, 0, 0)
    return results

# --- 主界面 ---
st.title("Alpha 游资系统 Pro + AI")
enable_refresh = st.sidebar.toggle("⚡ 智能实时刷新", value=True)
trading_active, status_msg = is_trading_time()
status_color = "green" if trading_active else "gray"
st.sidebar.markdown(f"当前状态: <span style='color:{status_color};font-weight:bold'>{status_msg}</span>", unsafe_allow_html=True)

if st.sidebar.button("🧹 强制刷新数据"):
    st.cache_data.clear()
    st.rerun()

# 🔥🔥🔥 核心：AI 训练数据收集区 🔥🔥🔥
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 AI 模型训练 (数据采集)")

# 自动计算验证逻辑
if st.sidebar.button("🔄 自动回填历史结果 (Auto-Label)"):
    msg = auto_label_data()
    st.toast(msg)
    time.sleep(1)
    st.rerun()

# 录入表单
with st.sidebar.form("ai_data_form"):
    train_code = st.text_input("股票代码", help="输入你想记录的个股代码")
    # 🔥 1. 战法策略选择 (标准化)
    strategy_options = [
        "🐲 (1) 龙头掘金 (机构波段)",
        "🚀 (2) 1进2 / 2进3 (接力)",
        "📉 (3) 涨停回调 (低吸)",
        "🌊 (4) 趋势低吸 (5日线战法)",
        "🔥 (5) 短线情绪 (跟随大游资)"
    ]
    train_strategy = st.selectbox("核心战法", strategy_options)
    
    # 🔥 2. 视频上传 (多模态)
    uploaded_video = st.file_uploader("上传思路视频 (MP4/MOV)", type=['mp4', 'mov'])
    
    # 备注
    train_note = st.text_area("补充思路 (可选)", placeholder="例如：竞价抢筹，板块效应强...")
    
    if st.form_submit_button("💾 记录并冻结数据"):
        if train_code:
            # 获取当前实时数据
            q_data = get_realtime_quotes([train_code])
            curr_price = q_data.get(train_code, {}).get('price', 0)
            c_name = q_data.get(train_code, {}).get('name', '未知')
            
            # 获取当前技术指标 (作为特征冻结)
            _, cost, _, _, _, _ = get_stock_history_metrics(train_code)
            
            if curr_price > 0:
                save_train_record_with_video(train_code, c_name, curr_price, cost, train_strategy, uploaded_video, train_note)
                st.toast(f"✅ 数据已录入：{c_name} | {train_strategy}")
            else:
                st.error("无法获取当前价格，请检查代码")
        else:
            st.warning("请输入代码")

# 显示今日数据
train_df = load_train_data()
today_str = datetime.now().strftime("%Y-%m-%d")
if not train_df.empty:
    with st.sidebar.expander("📊 查看训练数据集", expanded=False):
        st.dataframe(train_df[['record_date', 'name', 'strategy_type', 'result_label']], hide_index=True)

# 备份功能
st.sidebar.markdown("---")
with st.sidebar.expander("📂 数据备份", expanded=False):
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            st.download_button("⬇️ 自选股备份", f, file_name=f"stock_backup.csv", mime="text/csv")
    if os.path.exists(TRAIN_DATA_FILE):
        with open(TRAIN_DATA_FILE, "rb") as f:
            st.download_button("⬇️ 训练集备份", f, file_name=f"ai_dataset.csv", mime="text/csv")
            
    uploaded_file = st.file_uploader("⬆️ 恢复自选股", type=["csv"])
    if uploaded_file is not None:
        try:
            pd.read_csv(uploaded_file, dtype={"code": str}).to_csv(DATA_FILE, index=False)
            st.success("成功！")
            st.rerun()
        except: st.error("错误")

st.sidebar.markdown("---")

df = load_data()

with st.sidebar.expander("➕ 添加/编辑 个股", expanded=True):
    code_in = st.text_input("代码 (6位数)", key="cin").strip()
    if 'calc_s1' not in st.session_state: 
        for k in ['s1','s2','r1','r2']: st.session_state[f'calc_{k}'] = 0.0
    if st.button("⚡ 智能计算支撑压力"):
        if code_in:
            with st.spinner("计算中..."):
                hist, cost, zt, _, max_streak, _ = get_stock_history_metrics(code_in)
                if hist is not None:
                    last = hist.iloc[-1]
                    pivot = (last['最高']+last['最低']+last['收盘'])/3
                    st.session_state.calc_r1 = round(2*pivot - last['最低'], 2)
                    st.session_state.calc_s1 = round(2*pivot - last['最高'], 2)
                    st.session_state.calc_r2 = round(pivot + (last['最高'] - last['最低']), 2)
                    st.session_state.calc_s2 = round(pivot - (last['最高'] - last['最低']), 2)
                    st.success(f"识别结果：{zt}连板 (曾{max_streak}板)")
    
    with st.form("add"):
        c1,c2=st.columns(2)
        s1=c1.number_input("支撑1", value=float(st.session_state.calc_s1))
        s2=c1.number_input("支撑2", value=float(st.session_state.calc_s2))
        r1=c2.number_input("压力1", value=float(st.session_state.calc_r1))
        r2=c2.number_input("压力2", value=float(st.session_state.calc_r2))
        existing_groups = df['group'].unique().tolist() if not df.empty else ["默认"]
        if "默认" not in existing_groups: existing_groups.insert(0, "默认")
        select_options = ["✍️ 新建/手动输入"] + existing_groups
        selected_grp = st.selectbox("选择或新建分组", select_options, index=1 if len(select_options)>1 else 0)
        final_grp = st.text_input("输入新分组名称", "龙头") if selected_grp == "✍️ 新建/手动输入" else selected_grp
        note=st.text_area("笔记 (可选)")
        if st.form_submit_button("💾 保存") and code_in:
            name=""
            if code_in in df.code.values: name=df.loc[df.code==code_in,'name'].values[0]
            new={"code":code_in,"name":name,"s1":s1,"s2":s2,"r1":r1,"r2":r2,"group":final_grp,"note":note}
            if code_in in df.code.values: df.loc[df.code==code_in, list(new.keys())]=list(new.values())
            else: df=pd.concat([df,pd.DataFrame([new])],ignore_index=True)
            save_data(df)
            st.rerun()

@st.dialog("📈 个股详情", width="large")
def view_chart_modal(code, name):
    st.subheader(f"{name} ({code})")
    import time; ts = int(time.time()); mid = "1" if code.startswith(('6','5','9')) else "0"
    t1, t2 = st.tabs(["分时图", "日线图"])
    with t1: st.image(f"https://webquotepic.eastmoney.com/GetPic.aspx?nid={mid}.{code}&imageType=r&t={ts}", use_container_width=True)
    with t2: st.image(f"https://webquotepic.eastmoney.com/GetPic.aspx?nid={mid}.{code}&imageType=k&t={ts}", use_container_width=True)

if not df.empty:
    quotes = get_realtime_quotes(df['code'].tolist())
    with st.spinner("🚀 正在极速分析游资数据..."):
        batch_strategy_data = prefetch_all_data(df['code'].unique().tolist())

    def get_dist_html(target, current):
        try: target=float(target); current=float(current)
        except: return ""
        if target == 0: return ""
        d = ((current - target) / target) * 100
        col = "#d9534f" if abs(d)<1.0 else "#f0ad4e" if abs(d)<3.0 else "#999"
        return f"<span style='color:{col}; font-weight:bold;'>({d:+.1f}%)</span>"

    all_groups_for_popover = df['group'].unique().tolist()
    if "默认" not in all_groups_for_popover: all_groups_for_popover.insert(0, "默认")

    for group in df['group'].unique():
        st.subheader(f"📂 {group}")
        group_df = df[df['group'] == group]
        rows = [r for _, r in group_df.iterrows()]
        
        for i in range(0, len(rows), 4):
            cols = st.columns(4)
            chunk = rows[i:i+4]
            for j, row in enumerate(chunk):
                code = row['code']
                info = quotes.get(code, {})
                price = info.get('price', 0)
                open_p = info.get('open', 0)
                pre_close = info.get('pre_close', 0)
                name = info.get('name', code)
                chg = ((price-pre_close)/pre_close)*100 if pre_close else 0
                price_color = "price-up" if chg > 0 else ("price-down" if chg < 0 else "price-gray")
                
                hist_df, cost_low, zt_count, yesterday_zt, max_streak, max_amt_60d = batch_strategy_data.get(code, (None, 0, 0, False, 0, 0))
                strategy_text, strategy_class = ai_strategy_engine(info, hist_df, cost_low, zt_count, yesterday_zt, max_streak)
                
                with cols[j]:
                    with st.container(border=True):
                        col_name, col_grp_btn, col_del_btn = st.columns([5, 1, 1])
                        with col_name: st.markdown(f"<div style='white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'><span class='stock-name'>{name}</span> <span class='stock-code'>{code}</span></div>", unsafe_allow_html=True)
                        with col_grp_btn:
                            with st.popover("🏷️"):
                                new_grp = st.selectbox("组", ["(不变)"]+all_groups_for_popover, key=f"g_{code}")
                                if st.button("OK", key=f"ok_{code}"): 
                                    if new_grp!="(不变)":
                                        df.loc[df.code==code,'group']=new_grp
                                        save_data(df)
                                        st.rerun()
                        with col_del_btn: 
                             if st.button("🗑️", key=f"d_{code}"):
                                delete_single_stock(code)
                                st.rerun()

                        st.markdown(f"<div class='big-price {price_color}'>{price:.2f}</div>", unsafe_allow_html=True)
                        zt_badge = f"<span style='background:#ff0000;color:white;padding:1px 4px;border-radius:3px;font-size:0.8rem;margin-left:5px'>{zt_count}连板</span>" if zt_count>=2 else ""
                        st.markdown(f"<div style='font-weight:bold; margin-bottom:8px;'>{chg:+.2f}% {zt_badge}</div>", unsafe_allow_html=True)
                        st.markdown(f"<span class='strategy-tag {strategy_class}'>{strategy_text}</span>", unsafe_allow_html=True)
                        if cost_low>0: st.markdown(f"<div class='cost-range-box'>主力: {cost_low:.2f}</div>", unsafe_allow_html=True)
                        
                        r1, r2, s1, s2 = float(row['r1']), float(row['r2']), float(row['s1']), float(row['s2'])
                        st.markdown(f"""
                        <div class='sr-block'>
                            <div class='sr-item'><span style='color:#d9534f'>R2</span> {r2:.2f}{get_dist_html(r2, price)}</div>
                            <div class='sr-item'><span style='color:#5cb85c'>S1</span> {s1:.2f}{get_dist_html(s1, price)}</div>
                            <div class='sr-item'><span style='color:#f0ad4e'>R1</span> {r1:.2f}{get_dist_html(r1, price)}</div>
                            <div class='sr-item'><span style='color:#4cae4c'>S2</span> {s2:.2f}{get_dist_html(s2, price)}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        if str(row['note']) not in ['nan', '']: st.caption(f"📝 {row['note']}")
                        
                        if 1 <= zt_count <= 3 or strategy_text == "🚀 首板启动":
                            with st.expander(f"🎲 点击推演: {zt_count}进{zt_count+1}"):
                                plan_html, advice_html = generate_plan_and_advice(code, name, price, open_p, pre_close, max_amt_60d, zt_count)
                                st.markdown(f"<div class='plan-container'>{plan_html}</div>", unsafe_allow_html=True)
                                if advice_html: st.markdown(advice_html, unsafe_allow_html=True)

                        st.markdown('<div style="height:5px"></div>', unsafe_allow_html=True)
                        if st.button("📈 看图", key=f"btn_{code}"): view_chart_modal(code, name)

else: st.info("👈 请在左侧添加股票")

if enable_refresh and trading_active:
    time.sleep(3)
    st.rerun()
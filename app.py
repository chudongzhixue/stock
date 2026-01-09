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
    page_title="Alpha 游资系统 Pro (Cloud)",
    page_icon="🐲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 尝试连接 Google Sheets (云端同步) ---
try:
    from streamlit_gsheets import GSheetsConnection
    # 检查是否配置了 secrets
    if "connections" in st.secrets and "gsheets" in st.secrets["connections"]:
        USE_CLOUD_DB = True
        conn = st.connection("gsheets", type=GSheetsConnection)
    else:
        USE_CLOUD_DB = False
except:
    USE_CLOUD_DB = False

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
        
        .strategy-badge { 
            padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; color: white; 
            display: inline-block; vertical-align: middle; margin-right: 4px; margin-bottom: 4px;
            background-color: #333;
        }
        .bg-dragon { background: linear-gradient(45deg, #d32f2f, #ef5350); }
        .bg-relay { background: linear-gradient(45deg, #f57c00, #ffb74d); }
        .bg-low { background: linear-gradient(45deg, #1976d2, #42a5f5); }
        .bg-trend { background: linear-gradient(45deg, #388e3c, #66bb6a); }
        .bg-mood { background: linear-gradient(45deg, #7b1fa2, #ab47bc); }
        .bg-auto { background-color: #7f8c8d; }

        .cost-range-box { background-color: #f8f9fa; border-left: 3px solid #666; padding: 2px 6px; margin: 5px 0; border-radius: 0 4px 4px 0; font-size: 0.75rem; color: #444; }
        
        .plan-container { font-size: 0.85rem; color: #444; padding: 5px; }
        .plan-title { font-weight: bold; color: #2c3e50; font-size: 0.9rem; margin-bottom: 5px; border-bottom: 1px dashed #ddd; padding-bottom: 3px;}
        .plan-item { margin-bottom: 4px; line-height: 1.4; }
        .highlight-money { color: #d9534f; font-weight: bold; background: #fff5f5; padding: 0 4px; border-radius: 3px; }
        .highlight-support { color: #2980b9; font-weight: bold; background: #eaf2f8; padding: 0 4px; border-radius: 3px; }
        
        .advice-box { margin-top: 5px; padding: 8px; border-radius: 4px; font-weight: bold; text-align: center; font-size: 0.9rem; border: 1px solid #eee; }
        .advice-buy { background-color: #fff3f3; color: #d9534f; border-color: #d9534f; animation: pulse 2s infinite;}
        .advice-sell { background-color: #f0f9f0; color: #5cb85c; border-color: #5cb85c; }
        .advice-hold { background-color: #f0f8ff; color: #3498db; border-color: #3498db; }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(217, 83, 79, 0.2); }
            70% { box-shadow: 0 0 0 5px rgba(217, 83, 79, 0); }
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

if not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)

STRATEGY_OPTIONS = [
    "🤖 自动判断 (Auto)",
    "🐲 龙头掘金 (机构波段)",
    "🚀 连板接力 (1进2/2进3)",
    "📉 涨停回调 (低吸)",
    "🌊 趋势低吸 (5日/10日线)",
    "🔥 短线情绪 (游资跟随)"
]

# --- 🔥 核心：双模态数据引擎 (Cloud + Local) ---

def load_data():
    """读取自选股配置 (优先云端)"""
    default_cols = ["code", "name", "s1", "s2", "r1", "r2", "group", "strategy", "note"]
    
    if USE_CLOUD_DB:
        try:
            # ttl=10 防止触发 Google API 频率限制
            df = conn.read(worksheet="stock_config", ttl=10)
            
            # 🔥🔥🔥 修复核心：清理股票代码格式 🔥🔥🔥
            # 1. 转为字符串 2. 删掉 .0 3. 补齐6位
            df['code'] = df['code'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)
            
            # 🔥🔥🔥 数据清洗：防止空值报错 🔥🔥🔥
            # 填充文本列
            for col in ['name', 'group', 'strategy', 'note']:
                if col in df.columns:
                    df[col] = df[col].fillna("")
            
            # 填充数字列
            for col in ['s1', 's2', 'r1', 'r2']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
            # 补全缺失列
            for col in default_cols:
                if col not in df.columns:
                    df[col] = 0.0 if col not in ['name','group','strategy','note'] else ""
                    
            return df[default_cols]
        except Exception as e:
            st.error(f"云端读取失败，降级为本地模式: {e}")
    
    # 本地 CSV 兜底
    if not os.path.exists(DATA_FILE):
        df = pd.DataFrame(columns=default_cols)
        df.to_csv(DATA_FILE, index=False)
        return df
    
    df = pd.read_csv(DATA_FILE, dtype={"code": str})
    if "strategy" not in df.columns:
        df["strategy"] = "🤖 自动判断 (Auto)"
        save_data_local(df)
        
    expected_cols = ["code", "name", "s1", "s2", "r1", "r2", "group", "strategy", "note"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    df = df[expected_cols]
    df['code'] = df['code'].str.strip()
    df.drop_duplicates(subset=['code'], keep='last', inplace=True)
    return df

def save_data(df):
    """保存自选股配置 (双向同步)"""
    if USE_CLOUD_DB:
        try:
            conn.update(worksheet="stock_config", data=df)
            st.toast("☁️ 云端同步成功！")
        except:
            st.error("云端保存失败，仅保存本地")
    
    # 永远备份一份本地 CSV
    df.to_csv(DATA_FILE, index=False)

def save_data_local(df):
    df.to_csv(DATA_FILE, index=False)

def load_train_data():
    """读取 AI 训练数据"""
    cols = ["record_date", "code", "name", "strategy_type", "price_at_entry", 
            "cost_at_entry", "video_path", "note", 
            "next_day_open_pct", "next_day_high_pct", "next_day_close_pct", "result_label"]
    
    if USE_CLOUD_DB:
        try:
            df = conn.read(worksheet="ai_dataset", ttl=10) # 这里的 ttl 也要加上
            df['code'] = df['code'].astype(str).str.zfill(6)
            return df
        except: pass
        
    if not os.path.exists(TRAIN_DATA_FILE):
        df = pd.DataFrame(columns=cols)
        df.to_csv(TRAIN_DATA_FILE, index=False)
        return df
    return pd.read_csv(TRAIN_DATA_FILE, dtype={"code": str})

def save_train_data(df):
    """保存 AI 训练数据"""
    if USE_CLOUD_DB:
        try:
            conn.update(worksheet="ai_dataset", data=df)
            st.toast("☁️ AI数据已上云！")
        except: pass
    df.to_csv(TRAIN_DATA_FILE, index=False)

def save_train_record_with_video(code, name, price, cost, strategy, video_file, note):
    df = load_train_data()
    today = datetime.now().strftime("%Y-%m-%d")
    
    video_path = ""
    if video_file is not None:
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
        "cost_at_entry": cost,
        "video_path": video_path,
        "note": note,
        "next_day_open_pct": 0.0, 
        "next_day_high_pct": 0.0, 
        "next_day_close_pct": 0.0, 
        "result_label": "⏳ 待验证"
    }
    
    df = df[~((df['record_date'] == today) & (df['code'] == code))]
    df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
    save_train_data(df)
    return True

def auto_label_data():
    df = load_train_data()
    if df.empty: return "无数据"
    
    count = 0
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    for index, row in df.iterrows():
        if row['result_label'] != "⏳ 待验证" or row['record_date'] == today_str:
            continue
            
        try:
            hist = ak.stock_zh_a_hist(symbol=row['code'], period="daily", adjust="qfq")
            hist['日期'] = pd.to_datetime(hist['日期']).dt.strftime('%Y-%m-%d')
            
            record_idx = hist[hist['日期'] == row['record_date']].index
            if not record_idx.empty and record_idx[0] + 1 < len(hist):
                next_day = hist.iloc[record_idx[0] + 1]
                
                close_pct = next_day['收盘'] / next_day['前收盘'] - 1
                
                df.at[index, 'next_day_open_pct'] = round((next_day['开盘']/next_day['前收盘']-1)*100, 2)
                df.at[index, 'next_day_high_pct'] = round((next_day['最高']/next_day['前收盘']-1)*100, 2)
                df.at[index, 'next_day_close_pct'] = round(close_pct*100, 2)
                
                if close_pct > 0.05:
                    df.at[index, 'result_label'] = "✅ 成功(大肉)"
                elif close_pct > 0:
                    df.at[index, 'result_label'] = "⭕ 成功(小肉)"
                elif close_pct < -0.05:
                    df.at[index, 'result_label'] = "❌ 失败(大面)"
                else:
                    df.at[index, 'result_label'] = "➖ 失败(亏损)"
                
                count += 1
        except:
            pass
            
    if count > 0:
        save_train_data(df)
    return f"已回填 {count} 条结果"

def delete_single_stock(code_to_delete):
    df = load_data()
    if code_to_delete in df['code'].values:
        df = df[df['code'] != code_to_delete]
        save_data(df)
        return True
    return False

# --- 辅助功能 ---

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
                y_data['换手率'] = 0.0
                stock_df = y_data
        except: pass

    if stock_df is not None and not stock_df.empty:
        try:
            stock_df['MA5'] = stock_df['收盘'].rolling(5).mean()
            stock_df['MA10'] = stock_df['收盘'].rolling(10).mean()
            stock_df['MA20'] = stock_df['收盘'].rolling(20).mean()
            
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
            
            last_turnover = stock_df.iloc[-1]['换手率'] if '换手率' in stock_df.columns else 0.0
            
            return stock_df, avg_cost, zt_count, check_df.iloc[-2]['is_zt'] if len(check_df) > 1 else False, max_streak, max_amount_60d, last_turnover
        except: return None, 0, 0, False, 0, 0, 0
    return None, 0, 0, False, 0, 0, 0

def format_money(num):
    if pd.isna(num) or num == 0: return "N/A"
    num = float(num)
    if num > 100000000: return f"{num/100000000:.2f}亿"
    if num > 10000: return f"{num/10000:.2f}万"
    return f"{num:.2f}"

# --- 🔥 AI 实时操盘大脑 ---

def evaluate_strategy_realtime(strategy_name, info, history_df, avg_cost, zt_count, max_streak, max_amount_60d, turnover):
    if history_df is None or history_df.empty: return "数据不足", "bg-auto", ""
    
    price = info['price']
    open_p = info['open']
    pre_close = info['pre_close']
    pct_chg = ((price - pre_close) / pre_close) * 100
    open_pct = ((open_p - pre_close) / pre_close) * 100
    
    ma5 = history_df.iloc[-1]['MA5']
    ma10 = history_df.iloc[-1]['MA10']
    ma20 = history_df.iloc[-1]['MA20']
    
    advice = "观察"
    style = "advice-hold"
    badge_style = "bg-auto"
    
    if "龙头掘金" in strategy_name:
        badge_style = "bg-dragon"
        if price > avg_cost and price > ma10:
            if pct_chg < -3: advice = "🟢 回调洗盘: 吸"; style = "advice-buy"
            elif pct_chg > 5: advice = "🔴 加速: 持"; style = "advice-hold"
            else: advice = "🔵 趋势好: 持"; style = "advice-hold"
        elif price < ma10: advice = "⚠️ 破10日: 减"; style = "advice-sell"

    elif "连板接力" in strategy_name:
        badge_style = "bg-relay"
        threshold_open = 3.0 if turnover > 15 else 1.0
        
        if open_pct > threshold_open and price > open_p:
            if pct_chg > 9.5: advice = "🔒 涨停锁仓"; style = "advice-hold"
            else: advice = "🔥 弱转强: 买"; style = "advice-buy"
        elif open_pct < -2:
            advice = "❄️ 不及预期: 撤"; style = "advice-sell"
        elif price < pre_close:
            advice = "🟢 水下: 观望"; style = "advice-sell"
        else:
            advice = "🔵 分歧: 等"; style = "advice-hold"

    elif "涨停回调" in strategy_name:
        badge_style = "bg-low"
        dist_ma10 = (price - ma10) / ma10
        if -0.02 < dist_ma10 < 0.02: advice = "🎯 踩10日线: 吸"; style = "advice-buy"
        elif price < ma10: advice = "🚫 破位: 止"; style = "advice-sell"
        else: advice = "🔵 等回落"; style = "advice-hold"

    elif "趋势低吸" in strategy_name:
        badge_style = "bg-trend"
        if price > ma5: advice = "🔴 5日上: 持"; style = "advice-hold"
        elif price < ma5 and price > ma10: advice = "⚠️ 破5日: 减"; style = "advice-sell"
        else: advice = "🟢 破位: 清"; style = "advice-sell"

    elif "短线情绪" in strategy_name:
        badge_style = "bg-mood"
        if pct_chg > 7: advice = "🔥 高潮: 止盈"; style = "advice-sell"
        elif pct_chg < -5: advice = "❄️ 冰点: 博弈"; style = "advice-buy"
        else: advice = "🔵 跟随"; style = "advice-hold"

    else:
        badge_style = "bg-auto"
        if zt_count >= 2: advice = f"🚀 {zt_count}连板"; style = "advice-hold"
        elif pct_chg > 5: advice = "🔴 强势"; style = "advice-hold"
        else: advice = "🔵 观察"; style = "advice-hold"

    return advice, style, badge_style

def generate_plan_details(strategy_name, code, current_price, pre_close, max_amount_60d, turnover, ma5, ma10, ma20):
    html = ""
    
    if "连板" in strategy_name or "龙头" in strategy_name or "情绪" in strategy_name:
        target_auction_amt = max_amount_60d * 0.05
        base_open_pct = 2.0 if turnover < 10 else 4.0 
        exp_open_low = current_price * (1 + base_open_pct/100)
        exp_open_high = current_price * (1 + (base_open_pct+4)/100)
        
        html += f"<div class='plan-item'>🎯 <b>竞价目标：</b><span class='highlight-money'>{format_money(target_auction_amt)}</span></div>"
        html += f"<div class='plan-item'>📊 <b>理想开盘：</b>{exp_open_low:.2f}~{exp_open_high:.2f}</div>"
        html += "<hr style='margin:4px 0; border-top:1px dashed #ddd;'>"
        html += "<div class='plan-item'>1. <b>弱转强：</b>竞价达标，开盘不破均线 👉 买入。</div>"
        html += "<div class='plan-item'>2. <b>不及预期：</b>低开/平开，无量下杀 👉 卖出。</div>"
    
    elif "低吸" in strategy_name or "回调" in strategy_name or "趋势" in strategy_name:
        support_price = ma10 if ma10 > 0 else (ma5 if ma5 > 0 else current_price * 0.95)
        buy_zone_high = support_price * 1.01
        buy_zone_low = support_price * 0.99
        
        html += f"<div class='plan-item'>🛡️ <b>关键支撑：</b><span class='highlight-support'>{support_price:.2f}</span></div>"
        html += f"<div class='plan-item'>🎯 <b>伏击区间：</b>{buy_zone_low:.2f} ~ {buy_zone_high:.2f}</div>"
        html += "<hr style='margin:4px 0; border-top:1px dashed #ddd;'>"
        html += "<div class='plan-item'>1. <b>黄金坑：</b>缩量回踩支撑 👉 低吸。</div>"
        html += "<div class='plan-item'>2. <b>破位：</b>有效跌破支撑 👉 止损。</div>"
    
    else:
        html += "<div class='plan-item'>🤖 暂无特定战法，请观察盘口。</div>"
        
    return html

def prefetch_all_data(stock_codes):
    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor: 
        future_to_code = {executor.submit(get_stock_history_metrics, code): code for code in stock_codes}
        for future in as_completed(future_to_code):
            code = future_to_code[future]
            try: results[code] = future.result()
            except: results[code] = (None, 0, 0, False, 0, 0, 0)
    return results

# --- 主界面 ---
st.title("Alpha 游资系统 Pro + AI")

# 🔥 核心初始化：确保 trading_active 有定义
trading_active, trading_status_msg = is_trading_time()

status_msg = "☁️ 云端同步中" if USE_CLOUD_DB else "💾 本地模式 (请注意备份)"
st.sidebar.markdown(f"系统状态: **{status_msg}**")
st.sidebar.markdown(f"市场状态: **{trading_status_msg}**")

enable_refresh = st.sidebar.toggle("⚡ 智能实时刷新", value=True)

if st.sidebar.button("🧹 强制刷新数据"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 AI 模型训练")

if st.sidebar.button("🔄 自动回填历史结果"):
    msg = auto_label_data()
    st.toast(msg)
    time.sleep(1)
    st.rerun()

with st.sidebar.form("ai_data_form"):
    train_code = st.text_input("股票代码")
    train_strategy = st.selectbox("核心战法", STRATEGY_OPTIONS)
    uploaded_video = st.file_uploader("上传视频", type=['mp4', 'mov'])
    train_note = st.text_area("补充思路")
    
    if st.form_submit_button("💾 记录数据"):
        if train_code:
            q_data = get_realtime_quotes([train_code])
            curr_price = q_data.get(train_code, {}).get('price', 0)
            c_name = q_data.get(train_code, {}).get('name', '未知')
            _, cost, _, _, _, _, _ = get_stock_history_metrics(train_code)
            
            if curr_price > 0:
                save_train_record_with_video(train_code, c_name, curr_price, cost, train_strategy, uploaded_video, train_note)
                st.toast(f"✅ 已记录：{c_name}")
        else:
            st.warning("请输入代码")

train_df = load_train_data()
if not train_df.empty:
    with st.sidebar.expander("📊 查看数据集", expanded=False):
        st.dataframe(train_df[['record_date', 'name', 'strategy_type']], hide_index=True)

st.sidebar.markdown("---")
with st.sidebar.expander("📂 数据备份", expanded=False):
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            st.download_button("⬇️ 自选股备份", f, "stock_backup.csv")
    if os.path.exists(TRAIN_DATA_FILE):
        with open(TRAIN_DATA_FILE, "rb") as f:
            st.download_button("⬇️ 训练集备份", f, "ai_dataset.csv")
    
    uploaded_file = st.file_uploader("⬆️ 恢复自选股", type=["csv"])
    if uploaded_file is not None:
        pd.read_csv(uploaded_file, dtype={"code": str}).to_csv(DATA_FILE, index=False)
        st.rerun()

st.sidebar.markdown("---")

df = load_data()

with st.sidebar.expander("➕ 添加/编辑 个股", expanded=True):
    code_in = st.text_input("代码 (6位数)", key="cin").strip()
    
    if 'calc_s1' not in st.session_state:
        st.session_state.calc_s1 = 0.0
        st.session_state.calc_s2 = 0.0
        st.session_state.calc_r1 = 0.0
        st.session_state.calc_r2 = 0.0

    if st.button("⚡ 智能计算"):
        if code_in:
            with st.spinner("计算中..."):
                hist, cost, zt, _, max_streak, _, _ = get_stock_history_metrics(code_in)
                if hist is not None:
                    last = hist.iloc[-1]
                    pivot = (last['最高']+last['最低']+last['收盘'])/3
                    st.session_state.calc_r1 = round(2*pivot - last['最低'], 2)
                    st.session_state.calc_s1 = round(2*pivot - last['最高'], 2)
                    st.session_state.calc_r2 = round(pivot + (last['最高'] - last['最低']), 2)
                    st.session_state.calc_s2 = round(pivot - (last['最高'] - last['最低']), 2)
                    st.success(f"识别结果：{zt}连板")
    
    with st.form("add"):
        col1, col2 = st.columns(2)
        s1 = col1.number_input("支撑1", value=float(st.session_state.calc_s1))
        s2 = col1.number_input("支撑2", value=float(st.session_state.calc_s2))
        r1 = col2.number_input("压力1", value=float(st.session_state.calc_r1))
        r2 = col2.number_input("压力2", value=float(st.session_state.calc_r2))
        
        new_grp = st.selectbox("分组", ["默认"] + [g for g in df['group'].unique() if g!="默认"])
        new_strategy = st.selectbox("绑定战法", STRATEGY_OPTIONS)
        note = st.text_area("笔记")
        
        if st.form_submit_button("💾 保存"):
            if code_in:
                name = ""
                if code_in in df.code.values:
                    name = df.loc[df.code==code_in, 'name'].values[0]
                
                new_entry = {
                    "code": code_in, "name": name, 
                    "s1": s1, "s2": s2, "r1": r1, "r2": r2, 
                    "group": new_grp, "strategy": new_strategy, "note": note
                }
                
                if code_in in df.code.values:
                    for k, v in new_entry.items():
                        df.loc[df.code==code_in, k] = v
                else:
                    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                
                save_data(df)
                st.rerun()

if not df.empty:
    quotes = get_realtime_quotes(df['code'].tolist())
    with st.spinner("🚀 正在分析..."):
        batch_data = prefetch_all_data(df['code'].unique().tolist())

    def get_dist_html(target, current):
        try: target=float(target); current=float(current)
        except: return ""
        if target == 0: return ""
        d = ((current - target) / target) * 100
        col = "#d9534f" if abs(d)<1.0 else "#f0ad4e" if abs(d)<3.0 else "#999"
        return f"<span style='color:{col}; font-weight:bold;'>({d:+.1f}%)</span>"

    all_groups = df['group'].unique().tolist()
    if "默认" not in all_groups: all_groups.insert(0, "默认")

    for group in df['group'].unique():
        st.subheader(f"📂 {group}")
        group_df = df[df['group'] == group]
        rows = [r for _, r in group_df.iterrows()]
        
        for i in range(0, len(rows), 4):
            cols = st.columns(4)
            chunk = rows[i:i+4]
            for j, row in enumerate(chunk):
                code = row['code']
                assigned_strategy = row.get('strategy', "🤖 自动判断 (Auto)")
                info = quotes.get(code, {})
                price = info.get('price', 0)
                pre_close = info.get('pre_close', 0)
                name = info.get('name', code)
                chg = ((price-pre_close)/pre_close)*100 if pre_close else 0
                price_color = "price-up" if chg > 0 else ("price-down" if chg < 0 else "price-gray")
                
                hist_df, cost_low, zt_count, _, _, max_amt_60d, last_to = batch_data.get(code, (None, 0, 0, False, 0, 0, 0))
                
                ma5 = hist_df.iloc[-1]['MA5'] if hist_df is not None else 0
                ma10 = hist_df.iloc[-1]['MA10'] if hist_df is not None else 0
                ma20 = hist_df.iloc[-1]['MA20'] if hist_df is not None else 0
                
                ai_advice, ai_style, badge_style = evaluate_strategy_realtime(assigned_strategy, info, hist_df, cost_low, zt_count, 0, max_amt_60d, last_to)
                
                with cols[j]:
                    with st.container(border=True):
                        c1, c2, c3 = st.columns([5, 1, 1])
                        with c1: st.markdown(f"<div style='white-space:nowrap;overflow:hidden;'><span class='stock-name'>{name}</span> <span class='stock-code'>{code}</span></div>", unsafe_allow_html=True)
                        with c2:
                            with st.popover("🏷️"):
                                n_grp = st.selectbox("分组", all_groups, key=f"ng_{code}", index=all_groups.index(group) if group in all_groups else 0)
                                n_strat = st.selectbox("战法", STRATEGY_OPTIONS, key=f"ns_{code}", index=STRATEGY_OPTIONS.index(assigned_strategy) if assigned_strategy in STRATEGY_OPTIONS else 0)
                                if st.button("更新", key=f"up_{code}"):
                                    df.loc[df.code==code, 'group'] = n_grp; df.loc[df.code==code, 'strategy'] = n_strat; save_data(df); st.rerun()
                        with c3:
                            if st.button("🗑️", key=f"del_{code}"): delete_single_stock(code); st.rerun()

                        st.markdown(f"<div class='big-price {price_color}'>{price:.2f}</div>", unsafe_allow_html=True)
                        zt_badge = f"<span style='background:#ff0000;color:white;padding:1px 4px;border-radius:3px;font-size:0.8rem;margin-left:5px'>{zt_count}连板</span>" if zt_count>=2 else ""
                        st.markdown(f"<div style='font-weight:bold; margin-bottom:8px;'>{chg:+.2f}% {zt_badge}</div>", unsafe_allow_html=True)
                        st.markdown(f"<span class='strategy-badge {badge_style}'>{assigned_strategy.split(' ')[0]}</span>", unsafe_allow_html=True)
                        
                        if trading_active:
                            st.markdown(f"<div class='advice-box {ai_style}'>{ai_advice}</div>", unsafe_allow_html=True)
                        
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
                        
                        with st.expander("🎲 操盘推演"):
                            st.markdown(generate_plan_details(assigned_strategy, code, price, pre_close, max_amt_60d, last_to, ma5, ma10, ma20), unsafe_allow_html=True)

                        st.markdown('<div style="height:5px"></div>', unsafe_allow_html=True)
                        if st.button("📈 看图", key=f"btn_{code}"): view_chart_modal(code, name)

else: st.info("👈 请在左侧添加股票")

if enable_refresh and trading_active:
    time.sleep(3)
    st.rerun()
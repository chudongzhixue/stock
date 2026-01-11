import streamlit as st
import pandas as pd
import requests
import os
import time
import json
import numpy as np
import akshare as ak
import yfinance as yf
import google.generativeai as genai
from datetime import datetime, timedelta, time as dt_time
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 页面基础设置 ---
st.set_page_config(
    page_title="Alpha 游资系统 (AI完全体)",
    page_icon="🐲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. 连接数据库 (Google Sheets) ---
try:
    from streamlit_gsheets import GSheetsConnection
    if "connections" in st.secrets and "gsheets" in st.secrets["connections"]:
        USE_CLOUD_DB = True
        conn = st.connection("gsheets", type=GSheetsConnection)
    else:
        USE_CLOUD_DB = False
except:
    USE_CLOUD_DB = False

# --- 2. 连接 AI 大脑 (Gemini) ---
try:
    if "gemini" in st.secrets and "api_key" in st.secrets["gemini"]:
        genai.configure(api_key=st.secrets["gemini"]["api_key"])
        USE_AI = True
    else:
        USE_AI = False
except:
    USE_AI = False

# --- 🎨 CSS 样式 (保留原汁原味) ---
st.markdown("""
    <style>
        html, body, p, div, span { font-family: 'Source Sans Pro', sans-serif; }
        .block-container { padding-top: 1rem !important; }
        
        /* 操盘卡片样式 */
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
        
        /* 标签与建议 */
        .strategy-badge { padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; color: white; background-color: #333; margin-right: 4px; }
        .bg-dragon { background: linear-gradient(45deg, #d32f2f, #ef5350); }
        .bg-relay { background: linear-gradient(45deg, #f57c00, #ffb74d); }
        .bg-low { background: linear-gradient(45deg, #1976d2, #42a5f5); }
        .bg-trend { background: linear-gradient(45deg, #388e3c, #66bb6a); }
        .bg-mood { background: linear-gradient(45deg, #7b1fa2, #ab47bc); }
        
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
        
        /* 计划推演 */
        .plan-item { margin-bottom: 4px; line-height: 1.4; font-size: 0.85rem; color: #444; }
        .highlight-money { color: #d9534f; font-weight: bold; background: #fff5f5; padding: 0 4px; border-radius: 3px; }
        .highlight-support { color: #2980b9; font-weight: bold; background: #eaf2f8; padding: 0 4px; border-radius: 3px; }

        /* AI 报告样式 */
        .report-box { border: 1px solid #ddd; padding: 20px; border-radius: 10px; background: #f9f9f9; margin-bottom: 20px;}
    </style>
""", unsafe_allow_html=True)

DATA_FILE = 'my_stock_plan_v3.csv'
LEARNED_STRATEGY_FILE = 'learned_strategies.csv'
STRATEGY_OPTIONS = [
    "🤖 自动判断 (Auto)",
    "🐲 龙头掘金 (机构波段)",
    "🚀 连板接力 (1进2/2进3)",
    "📉 涨停回调 (低吸)",
    "🌊 趋势低吸 (5日/10日线)",
    "🔥 短线情绪 (游资跟随)"
]

# --- 核心函数 (保留 v7.4 所有逻辑) ---

def load_data():
    """读取自选股配置"""
    default_cols = ["code", "name", "s1", "s2", "r1", "r2", "group", "strategy", "note"]
    if USE_CLOUD_DB:
        try:
            df = conn.read(worksheet="stock_config", ttl=10)
            df['code'] = df['code'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)
            for col in ['name', 'group', 'strategy', 'note']:
                if col in df.columns: df[col] = df[col].fillna("")
            for col in ['s1', 's2', 'r1', 'r2']:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            for col in default_cols:
                if col not in df.columns: df[col] = 0.0 if col not in ['name','group','strategy','note'] else ""
            return df[default_cols]
        except Exception: pass
    
    if not os.path.exists(DATA_FILE):
        df = pd.DataFrame(columns=default_cols)
        df.to_csv(DATA_FILE, index=False)
        return df
    
    df = pd.read_csv(DATA_FILE, dtype={"code": str})
    expected_cols = ["code", "name", "s1", "s2", "r1", "r2", "group", "strategy", "note"]
    for col in expected_cols:
        if col not in df.columns: df[col] = 0.0
    return df

def save_data(df):
    """保存自选股配置"""
    if USE_CLOUD_DB:
        try: conn.update(worksheet="stock_config", data=df)
        except: pass
    df.to_csv(DATA_FILE, index=False)

def delete_single_stock(code_to_delete):
    df = load_data()
    if code_to_delete in df['code'].values:
        df = df[df['code'] != code_to_delete]
        save_data(df)
        return True
    return False

def get_learned_strategies():
    """读取 AI 学习到的战法"""
    cols = ["date", "strategy_name", "core_logic", "buy_condition", "sell_condition", "visual_pattern"]
    if USE_CLOUD_DB:
        try:
            df = conn.read(worksheet="learned_strategies", ttl=10)
            return df
        except: pass
    if not os.path.exists(LEARNED_STRATEGY_FILE):
        return pd.DataFrame(columns=cols)
    return pd.read_csv(LEARNED_STRATEGY_FILE)

def save_learned_strategy(record):
    """保存 AI 新学会的战法"""
    df = get_learned_strategies()
    new_df = pd.DataFrame([record])
    df = pd.concat([df, new_df], ignore_index=True)
    if USE_CLOUD_DB:
        try: conn.update(worksheet="learned_strategies", data=df)
        except: pass
    df.to_csv(LEARNED_STRATEGY_FILE, index=False)

# --- 行情与计算 ---

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
    try:
        stock_df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        stock_df['MA5'] = stock_df['收盘'].rolling(5).mean()
        stock_df['MA10'] = stock_df['收盘'].rolling(10).mean()
        stock_df['MA20'] = stock_df['收盘'].rolling(20).mean()
        
        recent = stock_df.tail(20)
        total_amt = recent['成交额'].sum(); total_vol = recent['成交量'].sum()
        avg_cost = (total_amt / total_vol) if total_vol > 0 else 0
        if avg_cost > 200: avg_cost /= 100
        
        stock_df['is_zt'] = (stock_df['收盘'].pct_change() * 100) > 9.5
        zt_count = 0
        for i in range(len(stock_df)-1, -1, -1):
            if stock_df.iloc[i]['is_zt']: zt_count += 1
            else: break
            
        recent_60 = stock_df.tail(60)
        max_amount_60d = recent_60['成交额'].max()
        last_turnover = stock_df.iloc[-1]['换手率']
        return stock_df, avg_cost, zt_count, max_amount_60d, last_turnover
    except: return None, 0, 0, 0, 0

def evaluate_strategy_realtime(strategy_name, info, history_df, avg_cost, zt_count, turnover):
    if history_df is None: return "数据不足", "bg-auto", ""
    price = info['price']; pre_close = info['pre_close']
    pct_chg = ((price - pre_close) / pre_close) * 100
    ma5 = history_df.iloc[-1]['MA5']; ma10 = history_df.iloc[-1]['MA10']
    
    advice = "观察"; style = "advice-hold"; badge_style = "bg-auto"
    
    if "龙头" in strategy_name:
        badge_style = "bg-dragon"
        if price > avg_cost and price > ma10:
            if pct_chg < -3: advice = "🟢 回调洗盘: 吸"; style = "advice-buy"
            elif pct_chg > 5: advice = "🔴 加速: 持"; style = "advice-hold"
            else: advice = "🔵 趋势好: 持"; style = "advice-hold"
        elif price < ma10: advice = "⚠️ 破10日: 减"; style = "advice-sell"
    elif "连板" in strategy_name:
        badge_style = "bg-relay"
        if pct_chg > 9.5: advice = "🔒 涨停锁仓"; style = "advice-hold"
        elif price > pre_close * 1.03: advice = "🔥 弱转强: 买"; style = "advice-buy"
        elif price < pre_close: advice = "🟢 水下: 观望"; style = "advice-sell"
    elif "回调" in strategy_name or "低吸" in strategy_name:
        badge_style = "bg-low"
        if abs((price - ma10)/ma10) < 0.02: advice = "🎯 踩10日线: 吸"; style = "advice-buy"
        elif price < ma10: advice = "🚫 破位: 止"; style = "advice-sell"
    else:
        if zt_count >= 2: advice = f"🚀 {zt_count}连板"; style = "advice-hold"
        elif pct_chg > 5: advice = "🔴 强势"; style = "advice-hold"

    return advice, style, badge_style

def generate_plan_details(strategy_name, code, current_price, max_amount_60d, turnover, ma5, ma10):
    html = ""
    target_auction_amt = max_amount_60d * 0.05
    
    if "连板" in strategy_name or "龙头" in strategy_name:
        html += f"<div class='plan-item'>🎯 <b>竞价目标：</b><span class='highlight-money'>{format_money(target_auction_amt)}</span></div>"
        html += "<div class='plan-item'>1. <b>弱转强：</b>竞价达标，开盘不破均线 👉 买入。</div>"
        html += "<div class='plan-item'>2. <b>不及预期：</b>低开/平开，无量下杀 👉 卖出。</div>"
    elif "低吸" in strategy_name or "回调" in strategy_name:
        support_price = ma10 if ma10 > 0 else current_price * 0.95
        html += f"<div class='plan-item'>🛡️ <b>关键支撑：</b><span class='highlight-support'>{support_price:.2f}</span></div>"
        html += "<div class='plan-item'>1. <b>黄金坑：</b>缩量回踩支撑 👉 低吸。</div>"
    else:
        html += "<div class='plan-item'>🤖 暂无特定战法，请观察盘口。</div>"
    return html

def format_money(num):
    if pd.isna(num) or num == 0: return "N/A"
    num = float(num)
    if num > 100000000: return f"{num/100000000:.2f}亿"
    if num > 10000: return f"{num/10000:.2f}万"
    return f"{num:.2f}"

def prefetch_all_data(stock_codes):
    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor: 
        future_to_code = {executor.submit(get_stock_history_metrics, code): code for code in stock_codes}
        for future in as_completed(future_to_code):
            code = future_to_code[future]
            try: results[code] = future.result()
            except: results[code] = (None, 0, 0, 0, 0)
    return results

# --- 新增：AI 视频处理函数 ---
def process_video_with_gemini(video_file, user_prompt):
    """上传视频给 Gemini 并获取战法"""
    if not USE_AI: return None
    
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(video_file.getbuffer())
    
    status = st.empty()
    status.info("📤 视频上传中，请稍候...")
    
    try:
        video_upload = genai.upload_file(path=temp_path)
        while video_upload.state.name == "PROCESSING":
            time.sleep(2)
            video_upload = genai.get_file(video_upload.name)
        
        if video_upload.state.name == "FAILED":
            status.error("❌ 视频处理失败")
            return None
            
        status.info("🧠 AI 正在深度分析操盘逻辑 (耗时约 10-20 秒)...")
        
        system_prompt = """
        你是一位顶级游资操盘手。请分析这段复盘视频。
        总结出一套可执行的策略，严格返回如下 JSON 格式 (不要 Markdown):
        {
            "strategy_name": "策略名",
            "core_logic": "核心逻辑",
            "buy_condition": "买入条件",
            "sell_condition": "卖出/止损条件",
            "visual_pattern": "K线或分时形态特征"
        }
        """
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content([video_upload, system_prompt, user_prompt])
        genai.delete_file(video_upload.name)
        status.empty()
        return response.text
    except Exception as e:
        status.error(f"AI 调用出错: {e}")
        return None

# --- 图表弹窗 (解决 NameError 关键) ---
@st.dialog("📈 个股详情", width="large")
def view_chart_modal(code, name):
    st.subheader(f"{name} ({code})")
    ts = int(time.time())
    mid = "1" if code.startswith(('6','5','9')) else "0"
    t1, t2 = st.tabs(["分时图", "日线图"])
    with t1: st.image(f"https://webquotepic.eastmoney.com/GetPic.aspx?nid={mid}.{code}&imageType=r&t={ts}", use_container_width=True)
    with t2: st.image(f"https://webquotepic.eastmoney.com/GetPic.aspx?nid={mid}.{code}&imageType=k&t={ts}", use_container_width=True)

# --- 主程序 ---

# 初始化 Session
if 'calc_s1' not in st.session_state:
    st.session_state.calc_s1 = 0.0; st.session_state.calc_s2 = 0.0
    st.session_state.calc_r1 = 0.0; st.session_state.calc_r2 = 0.0

trading_active, trading_status_msg = is_trading_time()

# 侧边栏：状态 & 添加股票 (全局保留)
with st.sidebar:
    st.title("控制台")
    st.markdown(f"市场: **{trading_status_msg}**")
    status_icon = "☁️" if USE_CLOUD_DB else "💾"
    ai_icon = "🧠" if USE_AI else "🚫"
    st.markdown(f"数据: {status_icon} | AI: {ai_icon}")
    
    st.divider()
    
    with st.expander("➕ 添加/编辑 个股", expanded=False):
        code_in = st.text_input("代码 (6位数)", key="cin").strip()
        if st.button("⚡ 智能计算"):
            if code_in:
                with st.spinner("计算中..."):
                    hist, _, zt, _, _ = get_stock_history_metrics(code_in)
                    if hist is not None:
                        last = hist.iloc[-1]
                        pivot = (last['最高']+last['最低']+last['收盘'])/3
                        st.session_state.calc_r1 = round(2*pivot - last['最低'], 2)
                        st.session_state.calc_s1 = round(2*pivot - last['最高'], 2)
                        st.session_state.calc_r2 = round(pivot + (last['最高'] - last['最低']), 2)
                        st.session_state.calc_s2 = round(pivot - (last['最高'] - last['最低']), 2)
                        st.success(f"识别结果：{zt}连板")
        
        with st.form("add"):
            c1, c2 = st.columns(2)
            s1 = c1.number_input("S1", value=float(st.session_state.calc_s1))
            s2 = c1.number_input("S2", value=float(st.session_state.calc_s2))
            r1 = c2.number_input("R1", value=float(st.session_state.calc_r1))
            r2 = c2.number_input("R2", value=float(st.session_state.calc_r2))
            new_strategy = st.selectbox("战法", STRATEGY_OPTIONS)
            note = st.text_area("笔记")
            
            if st.form_submit_button("💾 保存"):
                if code_in:
                    df = load_data()
                    name = ""
                    if code_in in df.code.values: name = df.loc[df.code==code_in, 'name'].values[0]
                    new_entry = {"code": code_in, "name": name, "s1": s1, "s2": s2, "r1": r1, "r2": r2, "group": "默认", "strategy": new_strategy, "note": note}
                    
                    if code_in in df.code.values:
                        for k, v in new_entry.items(): df.loc[df.code==code_in, k] = v
                    else:
                        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                    save_data(df)
                    st.rerun()
    
    if st.button("🧹 刷新数据"):
        st.cache_data.clear()
        st.rerun()

# 主界面：多标签页结构
st.title("Alpha 游资系统 (Pro + AI)")

tab1, tab2, tab3 = st.tabs(["📈 实战看板", "🎓 AI 视频悟道", "📚 战法知识库"])

# --- Tab 1: 实战看板 (您原来的功能) ---
with tab1:
    df = load_data()
    if not df.empty:
        quotes = get_realtime_quotes(df['code'].tolist())
        batch_data = prefetch_all_data(df['code'].unique().tolist())

        def get_dist_html(target, current):
            try: target=float(target); current=float(current)
            except: return ""
            if target == 0: return ""
            d = ((current - target) / target) * 100
            col = "#d9534f" if abs(d)<1.0 else "#f0ad4e" if abs(d)<3.0 else "#999"
            return f"<span style='color:{col}; font-weight:bold;'>({d:+.1f}%)</span>"

        # 遍历展示
        rows = [r for _, r in df.iterrows()]
        for i in range(0, len(rows), 4):
            cols = st.columns(4)
            chunk = rows[i:i+4]
            for j, row in enumerate(chunk):
                code = row['code']
                strategy = row['strategy']
                info = quotes.get(code, {})
                price = info.get('price', 0)
                name = info.get('name', code)
                pre_close = info.get('pre_close', 0)
                chg = ((price-pre_close)/pre_close)*100 if pre_close else 0
                
                hist_df, cost, zt_cnt, max_amt, turnover = batch_data.get(code, (None, 0, 0, 0, 0))
                ma5 = hist_df.iloc[-1]['MA5'] if hist_df is not None else 0
                ma10 = hist_df.iloc[-1]['MA10'] if hist_df is not None else 0
                
                advice, style, badge = evaluate_strategy_realtime(strategy, info, hist_df, cost, zt_cnt, turnover)
                
                with cols[j]:
                    with st.container(border=True):
                        c1, c2 = st.columns([4, 1])
                        with c1: st.markdown(f"**{name}** `{code}`")
                        with c2: 
                            if st.button("🗑️", key=f"d_{code}"): delete_single_stock(code); st.rerun()
                        
                        p_col = "price-up" if chg > 0 else "price-down"
                        st.markdown(f"<div class='big-price {p_col}'>{price:.2f}</div>", unsafe_allow_html=True)
                        st.markdown(f"**{chg:+.2f}%** <span class='strategy-badge {badge}'>{strategy[:2]}</span>", unsafe_allow_html=True)
                        
                        st.markdown(f"<div class='advice-box {style}'>{advice}</div>", unsafe_allow_html=True)
                        
                        r1, r2, s1, s2 = row['r1'], row['r2'], row['s1'], row['s2']
                        st.markdown(f"""
                        <div class='sr-block'>
                            <div class='sr-item'><span style='color:#d9534f'>R2</span> {r2:.2f}{get_dist_html(r2, price)}</div>
                            <div class='sr-item'><span style='color:#5cb85c'>S1</span> {s1:.2f}{get_dist_html(s1, price)}</div>
                            <div class='sr-item'><span style='color:#f0ad4e'>R1</span> {r1:.2f}{get_dist_html(r1, price)}</div>
                            <div class='sr-item'><span style='color:#4cae4c'>S2</span> {s2:.2f}{get_dist_html(s2, price)}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("🎲 操盘计划"):
                            st.markdown(generate_plan_details(strategy, code, price, max_amt, turnover, ma5, ma10), unsafe_allow_html=True)
                        
                        if st.button("📈 看图", key=f"b_{code}"): view_chart_modal(code, name)
    else:
        st.info("👈 请在左侧添加股票")

# --- Tab 2: AI 视频悟道 (新功能) ---
with tab2:
    st.header("🎓 AI 视频操盘学徒")
    if not USE_AI:
        st.warning("⚠️ 请先在 secrets.toml 中配置 [gemini] api_key 才能使用 AI 功能")
    else:
        st.markdown("上传游资复盘视频 (MP4)，AI 将自动总结核心战法。")
        v_file = st.file_uploader("上传视频", type=['mp4', 'mov'])
        v_note = st.text_input("提示词 (例如: 重点关注弱转强逻辑)", value="提取买卖点逻辑")
        
        if v_file and st.button("🚀 开始 AI 分析"):
            res_text = process_video_with_gemini(v_file, v_note)
            if res_text:
                try:
                    clean_json = res_text.replace("```json", "").replace("```", "").strip()
                    s_data = json.loads(clean_json)
                    
                    st.success("✅ AI 悟道成功！")
                    with st.container(border=True):
                        st.subheader(f"🛡️ {s_data.get('strategy_name', '未命名')}")
                        st.markdown(f"**核心逻辑:** {s_data.get('core_logic')}")
                        c1, c2 = st.columns(2)
                        with c1: 
                            st.markdown("### 🔴 买入条件")
                            st.info(s_data.get('buy_condition'))
                        with c2:
                            st.markdown("### 🟢 卖出条件")
                            st.warning(s_data.get('sell_condition'))
                        
                        if st.button("💾 存入战法库"):
                            rec = {
                                "date": datetime.now().strftime("%Y-%m-%d"),
                                "strategy_name": s_data.get('strategy_name'),
                                "core_logic": s_data.get('core_logic'),
                                "buy_condition": s_data.get('buy_condition'),
                                "sell_condition": s_data.get('sell_condition'),
                                "visual_pattern": s_data.get('visual_pattern')
                            }
                            save_learned_strategy(rec)
                            st.toast("战法已保存！")
                except:
                    st.error("AI 返回格式解析失败，请重试")
                    st.text(res_text)

# --- Tab 3: 战法知识库 ---
with tab3:
    st.header("📚 游资战法知识库")
    sdf = get_learned_strategies()
    if not sdf.empty:
        for i, r in sdf.iterrows():
            with st.container(border=True):
                st.markdown(f"### {r['strategy_name']} <small style='color:grey'>{r['date']}</small>", unsafe_allow_html=True)
                st.markdown(f"> **逻辑:** {r['core_logic']}")
                st.markdown(f"**🔴 买:** {r['buy_condition']} | **🟢 卖:** {r['sell_condition']}")
    else:
        st.info("暂无战法，请去 Tab 2 上传视频进行学习。")
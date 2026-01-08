import streamlit as st
import pandas as pd
import requests
import os
import time
import numpy as np
import akshare as ak
import yfinance as yf
from datetime import datetime, timedelta, time as dt_time
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 页面基础设置 ---
st.set_page_config(
    page_title="Alpha 游资操盘系统",
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
            border: 1px solid #eee !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05); 
            background-color: #ffffff; 
            padding: 12px !important;
            border-radius: 10px;
            margin-bottom: 12px;
        }

        .big-price { font-size: 3.2rem; font-weight: 900; line-height: 1.0; letter-spacing: -2px; margin-bottom: 5px; }
        .price-up { color: #d9534f; }
        .price-down { color: #5cb85c; }
        .price-gray { color: #888; }
        
        .stock-name { font-size: 1.1rem; font-weight: bold; color: #333; }
        .stock-code { font-size: 0.9rem; color: #999; margin-left: 5px; }
        
        /* 🔥 策略标签体系升级 */
        .strategy-tag { padding: 3px 6px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; color: white; display: inline-block; vertical-align: middle; margin-right: 5px; }
        .tag-dragon { background: linear-gradient(45deg, #ff0000, #ff6b6b); } /* 龙头红 */
        .tag-buy { background-color: #d9534f; }
        .tag-sell { background-color: #5cb85c; }
        .tag-wait { background-color: #999; }
        .tag-special { background-color: #f0ad4e; } /* 橙色：特殊关注 */
        .tag-purple { background: linear-gradient(45deg, #8e44ad, #c0392b); } /* 紫色：妖股二波/震荡 */

        .cost-range-box { background-color: #f8f9fa; border-left: 3px solid #666; padding: 3px 8px; margin: 8px 0; border-radius: 0 4px 4px 0; font-size: 0.85rem; color: #444; }
        
        .sr-block { padding-top: 6px; border-top: 1px dashed #eee; display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }
        .sr-item { font-size: 0.85rem; font-weight: bold; color: #555; }
        
        [data-testid="column"] .stButton button { padding: 0px 8px; min-height: 0px; height: 32px; border: none; background: transparent; font-size: 1.1rem; color: #888; transition: all 0.2s; }
        button[kind="secondary"]:hover { color: #d9534f !important; background: #fff5f5 !important; }
        div[data-testid="stPopover"] button { padding: 0px 8px; min-height: 0px; height: 32px; border: none; background: transparent; font-size: 1.1rem; color: #888; }
        div[data-testid="stPopover"] button:hover { color: #007bff !important; background: #f0f8ff !important; }
        
        .view-chart-btn button { width: 100%; border-radius: 4px; font-weight: bold; margin-top: 8px; background-color: #f0f2f6; color: #31333F; height: auto; padding: 0.5rem; }
        .view-chart-btn button:hover { background-color: #e0e2e6; }
    </style>
""", unsafe_allow_html=True)

DATA_FILE = 'my_stock_plan_v3.csv'

# --- 核心函数 ---

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

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
    initial_count = len(df)
    df.drop_duplicates(subset=['code'], keep='last', inplace=True)
    if len(df) < initial_count: save_data(df)
    return df

def delete_single_stock(code_to_delete):
    df = load_data()
    if code_to_delete in df['code'].values:
        df = df[df['code'] != code_to_delete]
        save_data(df)
        return True
    return False

# 🔥 时区修复：强制使用北京时间 (UTC+8)
def is_trading_time():
    # 云服务器通常是 UTC 时间，所以要 +8 小时
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

# 🔥 数据获取升级：计算"历史最大连板数" (Max Streak)
@st.cache_data(ttl=3600)
def get_stock_history_metrics(code):
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=100)).strftime("%Y%m%d")
    stock_df = None
    
    try:
        stock_df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
    except: pass
        
    if stock_df is None or stock_df.empty:
        try:
            y_code = f"{code}.SS" if code.startswith('6') else f"{code}.SZ"
            if code.startswith(('8', '4')): y_code = f"{code}.BJ"
            y_data = yf.download(y_code, period="3mo", progress=False)
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
            if total_vol > 0:
                avg_cost = total_amt / total_vol
                if avg_cost > 200: avg_cost /= 100 
            else: avg_cost = 0
            
            stock_df['is_zt'] = stock_df['涨跌幅'] > 9.5
            
            # 1. 计算当前连板 (Current Streak)
            zt_count = 0
            check_df = stock_df.copy()
            for i in range(len(check_df)-1, -1, -1):
                if check_df.iloc[i]['is_zt']: zt_count += 1
                else: break
            
            # 2. 🔥 计算过去15天内的最大连板高度 (Max Streak)
            # 这就是"龙头记忆"：只要近期辉煌过，就不能当成杂毛
            recent_15_days = stock_df.tail(15)
            max_streak = 0
            current_streak_temp = 0
            for zt in recent_15_days['is_zt']:
                if zt: current_streak_temp += 1
                else:
                    max_streak = max(max_streak, current_streak_temp)
                    current_streak_temp = 0
            max_streak = max(max_streak, current_streak_temp)
            
            return stock_df, avg_cost, zt_count, check_df.iloc[-2]['is_zt'] if len(check_df) > 1 else False, max_streak
        except: return None, 0, 0, False, 0
    return None, 0, 0, False, 0

# 🧠 游资策略引擎 (逻辑大升级：引入 max_streak)
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

    # --- 👑 优先级 1: 龙头/妖股判定 (身份压制) ---
    # 逻辑：如果过去15天内有过4连板以上，它就是"龙"，哪怕今天断板也是"困龙"，不是"虫"
    if max_streak >= 4:
        if zt_count > 0:
            return f"🔥 妖股加速 ({zt_count}板)", "tag-dragon"
        elif pct_chg > 5.0:
            # 雷科防务这种情况会落在这里：曾6板，今断板但大涨 -> 龙头震荡
            return "🦁 龙头震荡/二波", "tag-purple"
        elif pct_chg < -5.0 and price > ma10:
            return "🐲 龙头首阴(反核)", "tag-special"
        elif price > day_vwap:
            return "🦁 龙头承接", "tag-special"
        else:
            return "💀 龙头退潮", "tag-sell"

    # --- 优先级 2: 普通连板接力 ---
    if zt_count >= 2:
        return f"🚀 {zt_count}连板持筹", "tag-dragon"
    
    if yesterday_zt and zt_count < 2:
        if 2 < pct_chg < 9.0 and price > day_vwap: return "🚀 1进2 接力", "tag-buy"
        if pct_chg > 9.0: return "🚀 秒板/一字", "tag-dragon"
    
    # --- 优先级 3: 形态/趋势 ---
    high_pct = ((high - pre_close) / pre_close) * 100
    if high_pct > 7 and pct_chg < 3 and price > ma5: return "👆 仙人指路", "tag-special"
    
    if pct_chg > 0 and price > day_vwap:
        return "💪 趋势向上", "tag-wait"
    if pct_chg < 0 and price < day_vwap:
        return "🤢 弱势调整", "tag-wait"
        
    return "😴 观望", "tag-wait"

# 并发预加载
def prefetch_all_data(stock_codes):
    results = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_code = {executor.submit(get_stock_history_metrics, code): code for code in stock_codes}
        for future in as_completed(future_to_code):
            code = future_to_code[future]
            try: results[code] = future.result()
            except: results[code] = (None, 0, 0, False, 0)
    return results

# --- 主界面 ---
st.title("Alpha 游资系统")
enable_refresh = st.sidebar.toggle("⚡ 智能实时刷新", value=True)
trading_active, status_msg = is_trading_time()
status_color = "green" if trading_active else "gray"
st.sidebar.markdown(f"当前状态: <span style='color:{status_color};font-weight:bold'>{status_msg}</span>", unsafe_allow_html=True)

if st.sidebar.button("🧹 强制刷新数据"):
    st.cache_data.clear()
    st.toast("正在重新拉取数据...")
    time.sleep(1)
    st.rerun()
if st.button('🔄 全局刷新', type="primary"): st.rerun()

df = load_data()

# 添加个股区域
with st.sidebar.expander("➕ 添加/编辑 个股", expanded=True):
    code_in = st.text_input("代码 (6位数)", key="cin").strip()
    if 'calc_s1' not in st.session_state: 
        for k in ['s1','s2','r1','r2']: st.session_state[f'calc_{k}'] = 0.0
    if st.button("⚡ 智能计算支撑压力"):
        if code_in:
            with st.spinner("计算中..."):
                hist, cost, zt, _, max_streak = get_stock_history_metrics(code_in)
                if hist is not None:
                    last = hist.iloc[-1]
                    pivot = (last['最高']+last['最低']+last['收盘'])/3
                    st.session_state.calc_r1 = round(2*pivot - last['最低'], 2)
                    st.session_state.calc_s1 = round(2*pivot - last['最高'], 2)
                    st.session_state.calc_r2 = round(pivot + (last['最高'] - last['最低']), 2)
                    st.session_state.calc_s2 = round(pivot - (last['最高'] - last['最低']), 2)
                    # 提示信息更加精准
                    status_text = f"当前{zt}连板" if zt > 0 else "断板"
                    high_status = f" (曾{max_streak}连板妖股)" if max_streak >= 4 else ""
                    st.success(f"识别结果：{status_text}{high_status}")
    
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
                name = info.get('name', code)
                pre = info.get('pre_close', 0)
                chg = ((price-pre)/pre)*100 if pre else 0
                price_color = "price-up" if chg > 0 else ("price-down" if chg < 0 else "price-gray")
                
                # 🔥 传入 max_streak 参数
                hist_df, cost_low, zt_count, yesterday_zt, max_streak = batch_strategy_data.get(code, (None, 0, 0, False, 0))
                strategy_text, strategy_class = ai_strategy_engine(info, hist_df, cost_low, zt_count, yesterday_zt, max_streak)
                
                with cols[j]:
                    with st.container(border=True):
                        col_name, col_grp_btn, col_del_btn = st.columns([5, 1, 1])
                        with col_name: st.markdown(f"<div style='white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'><span class='stock-name'>{name}</span> <span class='stock-code'>{code}</span></div>", unsafe_allow_html=True)
                        with col_grp_btn:
                            with st.popover("🏷️", help="修改分组"):
                                st.markdown(f"##### 修改 【{name}】")
                                new_group_select = st.selectbox("选择分组", ["(不变)"] + all_groups_for_popover, key=f"grp_sel_{code}")
                                new_group_input = st.text_input("或新分组", key=f"grp_inp_{code}")
                                final_new_group = new_group_input.strip() if new_group_input.strip() else (new_group_select if new_group_select != "(不变)" else None)
                                if st.button("✅ 确认", key=f"cfm_{code}"):
                                    if final_new_group and final_new_group != group:
                                        df.loc[df.code == code, 'group'] = final_new_group
                                        save_data(df)
                                        st.rerun()
                        with col_del_btn:
                            if st.button("🗑️", key=f"del_{code}"):
                                if delete_single_stock(code):
                                    st.rerun()
                        
                        st.markdown(f"<div class='big-price {price_color}'>{price:.2f}</div>", unsafe_allow_html=True)
                        zt_badge = f"<span style='background:#ff0000;color:white;padding:1px 4px;border-radius:3px;font-size:0.8rem;margin-left:5px'>{zt_count}连板</span>" if zt_count>=2 else ""
                        st.markdown(f"<div style='font-weight:bold; margin-bottom:8px;'>{chg:+.2f}% {zt_badge}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='margin-bottom:8px'><span class='strategy-tag {strategy_class}'>{strategy_text}</span></div>", unsafe_allow_html=True)
                        if cost_low > 0: st.markdown(f"<div class='cost-range-box'>主力成本: {cost_low:.2f}</div>", unsafe_allow_html=True)
                        
                        r1, r2 = float(row['r1']), float(row['r2'])
                        s1, s2 = float(row['s1']), float(row['s2'])
                        st.markdown(f"""
                        <div class='sr-block'>
                            <div class='sr-item'><span style='color:#d9534f'>R2</span> {r2:.2f}{get_dist_html(r2, price)}</div>
                            <div class='sr-item'><span style='color:#5cb85c'>S1</span> {s1:.2f}{get_dist_html(s1, price)}</div>
                            <div class='sr-item'><span style='color:#f0ad4e'>R1</span> {r1:.2f}{get_dist_html(r1, price)}</div>
                            <div class='sr-item'><span style='color:#4cae4c'>S2</span> {s2:.2f}{get_dist_html(s2, price)}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        if str(row['note']) not in ['nan', '']: st.caption(f"📝 {row['note']}")
                        st.markdown('<div class="view-chart-btn">', unsafe_allow_html=True)
                        if st.button("📈 看图", key=f"btn_{code}"): view_chart_modal(code, name)
                        st.markdown('</div>', unsafe_allow_html=True)
else: st.info("👈 请在左侧添加股票")

if enable_refresh and trading_active:
    time.sleep(3)
    st.rerun()
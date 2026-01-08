import streamlit as st
import pandas as pd
import requests
import os
import time
import numpy as np
import akshare as ak
from datetime import datetime, timedelta

# --- 页面基础设置 ---
st.set_page_config(
    page_title="Alpha 游资操盘系统",
    page_icon="🐲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 🎨 CSS 样式 (融合版) ---
st.markdown("""
    <style>
        html, body, p, div, span { font-family: 'Source Sans Pro', sans-serif; color: #0E1117; }
        .block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; }
        
        /* 卡片容器 */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid #eee !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05); 
            background-color: #ffffff; 
            padding: 12px !important;
            border-radius: 8px;
            margin-bottom: 10px;
        }

        /* 价格大字 */
        .big-price {
            font-size: 3.2rem; font-weight: 900; line-height: 1.0; letter-spacing: -2px; margin-bottom: 5px;
        }
        .price-up { color: #d9534f; }
        .price-down { color: #5cb85c; }
        .price-gray { color: #888; }
        
        .stock-name { font-size: 1.1rem; font-weight: bold; color: #333; }
        .stock-code { font-size: 0.9rem; color: #999; margin-left: 5px; }
        
        /* 策略标签体系 */
        .strategy-tag {
            padding: 4px 8px; border-radius: 4px; font-size: 0.85rem; font-weight: bold; 
            color: white; display: inline-block; vertical-align: middle; margin-right: 5px;
        }
        .tag-dragon { background: linear-gradient(45deg, #ff0000, #ff6b6b); } /* 龙头/妖股 */
        .tag-buy { background-color: #d9534f; } /* 买入/持有 */
        .tag-sell { background-color: #5cb85c; } /* 卖出/减仓 */
        .tag-wait { background-color: #999; } /* 观望 */
        .tag-special { background-color: #f0ad4e; } /* 特殊形态 */

        /* 成本区间样式 */
        .cost-range-box {
            background-color: #f8f9fa; border-left: 4px solid #666;
            padding: 4px 8px; margin: 8px 0; border-radius: 0 4px 4px 0;
            font-size: 0.85rem; color: #444;
        }
        
        /* 支撑压力 */
        .sr-block {
            padding-top: 6px; border-top: 1px dashed #eee;
            display: grid; grid-template-columns: 1fr 1fr; gap: 4px;
        }
        .sr-item { font-size: 0.85rem; font-weight: bold; color: #555; }
        
        /* 按钮微调 - 让看图按钮更显眼 */
        div[data-testid="stButton"] button {
            width: 100%; border-radius: 4px; font-weight: bold; margin-top: 5px;
        }
    </style>
""", unsafe_allow_html=True)

DATA_FILE = 'my_stock_plan_v3.csv'

# --- 核心功能函数 ---

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
    df.drop_duplicates(subset=['code'], keep='last', inplace=True)
    return df

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

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

# 🔥 获取历史数据 & 智能计算 (含缓存)
@st.cache_data(ttl=3600)
def get_stock_history_metrics(code):
    try:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=100)).strftime("%Y%m%d")
        try:
            stock_df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        except: return None, 0, 0, 0
        
        if stock_df.empty: return None, 0, 0, 0
        
        # 1. 计算均线
        stock_df['MA5'] = stock_df['收盘'].rolling(5).mean()
        stock_df['MA10'] = stock_df['收盘'].rolling(10).mean()
        stock_df['MA20'] = stock_df['收盘'].rolling(20).mean()
        
        # 2. 计算主力成本 (近20日VWAP)
        recent = stock_df.tail(20)
        total_amt = recent['成交额'].sum()
        total_vol = recent['成交量'].sum()
        smart_cost = total_amt / (total_vol * 100) if total_vol > 0 else 0
        
        # 3. 识别连板数
        stock_df['is_zt'] = stock_df['涨跌幅'] > 9.5
        zt_count = 0
        
        # 排除当天(如果是盘中)，只看历史收盘的连板
        today_str = datetime.now().strftime("%Y-%m-%d")
        check_df = stock_df.copy()
        if str(check_df.iloc[-1]['日期']) == today_str:
            check_df = check_df.iloc[:-1]
            
        for i in range(len(check_df)-1, -1, -1):
            if check_df.iloc[i]['is_zt']: zt_count += 1
            else: break
            
        return stock_df, smart_cost, zt_count, check_df.iloc[-1]['is_zt'] if not check_df.empty else False
    except:
        return None, 0, 0, 0

# 🧠 大游资策略引擎 (核心逻辑)
def ai_strategy_engine(info, history_df, smart_cost, zt_count, yesterday_zt):
    price = info['price']
    pre_close = info['pre_close']
    high = info['high']
    
    pct_chg = ((price - pre_close) / pre_close) * 100
    day_vwap = info['amount'] / info['vol'] if info['vol'] > 0 else price
    
    if history_df is None or history_df.empty:
        return "数据不足", "tag-wait"

    ma5 = history_df.iloc[-1]['MA5']
    ma10 = history_df.iloc[-1]['MA10']
    ma20 = history_df.iloc[-1]['MA20']

    # --- 策略 1: 妖股/连板锁仓 ---
    if zt_count >= 2:
        if pct_chg > 9.5: return f"💎 {zt_count+1}板锁仓", "tag-dragon"
        elif price > day_vwap and price > ma5: return f"🔥 妖股持筹 ({zt_count}板)", "tag-dragon"
        elif price < ma5: return "💀 断板止盈", "tag-sell"

    # --- 策略 2: 连板接力 (1进2, 2进3) ---
    if yesterday_zt and zt_count < 3:
        if 2 < pct_chg < 9.0 and price > day_vwap: return f"🚀 {zt_count}进{zt_count+1} 接力", "tag-buy"
    
    # --- 策略 3: 龙头首阴 ---
    if zt_count >= 3 and pct_chg < -3 and price > ma10: return "🐲 龙头首阴(博反包)", "tag-special"

    # --- 策略 4: 仙人指路 ---
    high_pct = ((high - pre_close) / pre_close) * 100
    if high_pct > 7 and pct_chg < 3 and price > ma20: return "👆 仙人指路", "tag-special"

    # --- 策略 5: 趋势低吸 ---
    if price > ma20 and ma10 > ma20:
        dist_ma10 = abs(price - ma10) / ma10
        if dist_ma10 < 0.02: return "🌊 MA10 低吸", "tag-buy"

    # --- 默认逻辑 ---
    if pct_chg > 9.8: return "🚀 涨停持股", "tag-dragon"
    if price > day_vwap: return "💪 强势整理", "tag-wait"
    if price < day_vwap: return "👀 弱势观望", "tag-wait"
    
    return "😴 观望", "tag-wait"

# --- 侧边栏 ---
st.sidebar.title("Control Panel")
auto_refresh = st.sidebar.toggle("🔥 实时刷新 (3s)", value=True)
st.sidebar.caption("游资模式已激活：自动匹配龙头、连板、首阴策略")
st.sidebar.markdown("---")

df = load_data()

with st.sidebar.expander("➕ 添加个股", expanded=True):
    code_in = st.text_input("代码", key="cin")
    if 'calc_s1' not in st.session_state: 
        for k in ['s1','s2','r1','r2']: st.session_state[f'calc_{k}'] = 0.0
    
    if st.button("⚡ 智能计算"):
        if code_in:
            with st.spinner("游资算法计算中..."):
                hist, cost, zt, _ = get_stock_history_metrics(code_in)
                if hist is not None:
                    last = hist.iloc[-1]
                    pivot = (last['最高']+last['最低']+last['收盘'])/3
                    st.session_state.calc_r1 = round(2*pivot - last['最低'], 2)
                    st.session_state.calc_s1 = round(2*pivot - last['最高'], 2)
                    st.session_state.calc_r2 = round(pivot + (last['最高'] - last['最低']), 2)
                    st.session_state.calc_s2 = round(pivot - (last['最高'] - last['最低']), 2)
                    st.success(f"已识别：{zt}连板妖股" if zt>=2 else "计算完成")
                else:
                    st.error("无法获取历史数据")

    with st.form("add"):
        c1,c2=st.columns(2)
        s1=c1.number_input("支撑1", value=float(st.session_state.calc_s1))
        s2=c1.number_input("支撑2", value=float(st.session_state.calc_s2))
        r1=c2.number_input("压力1", value=float(st.session_state.calc_r1))
        r2=c2.number_input("压力2", value=float(st.session_state.calc_r2))
        grp=st.text_input("分组","默认")
        note=st.text_area("笔记")
        if st.form_submit_button("保存") and code_in:
            name=""
            if code_in in df.code.values: name=df.loc[df.code==code_in,'name'].values[0]
            new={"code":code_in,"name":name,"s1":s1,"s2":s2,"r1":r1,"r2":r2,"group":grp,"note":note}
            if code_in in df.code.values: 
                df.loc[df.code==code_in, list(new.keys())]=list(new.values())
            else: 
                df=pd.concat([df,pd.DataFrame([new])],ignore_index=True)
            save_data(df)
            st.rerun()

if not df.empty:
    with st.sidebar.expander("🗑️ 删除"):
        if st.button("删除选中"): pass

# 🔥 弹窗函数 (保留这个功能！)
@st.dialog("📈 个股详情", width="large")
def view_chart_modal(code, name):
    st.subheader(f"{name} ({code})")
    import time; ts = int(time.time()); mid = "1" if code.startswith(('6','5','9')) else "0"
    t1, t2 = st.tabs(["分时图", "日线图"])
    with t1: st.image(f"https://webquotepic.eastmoney.com/GetPic.aspx?nid={mid}.{code}&imageType=r&t={ts}", use_container_width=True)
    with t2: st.image(f"https://webquotepic.eastmoney.com/GetPic.aspx?nid={mid}.{code}&imageType=k&t={ts}", use_container_width=True)

# --- 主界面 ---
st.title("Alpha 游资系统")
if st.button('🔄 全局刷新', type="primary"): st.rerun()

if not df.empty:
    quotes = get_realtime_quotes(df['code'].tolist())
    
    def get_dist_html(target, current):
        try: target=float(target); current=float(current)
        except: return ""
        if target == 0: return ""
        d = ((current - target) / target) * 100
        col = "#d9534f" if abs(d)<1.0 else "#f0ad4e" if abs(d)<3.0 else "#999"
        return f"<span style='color:{col}; font-weight:bold;'>({d:+.1f}%)</span>"

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
                
                # 🔥 调用游资策略
                hist_df, cost_low, zt_count, yesterday_zt = get_stock_history_metrics(code)
                strategy_text, strategy_class = ai_strategy_engine(info, hist_df, cost_low, zt_count, yesterday_zt)
                
                with cols[j]:
                    with st.container(border=True):
                        # 头部
                        st.markdown(f"<div><span class='stock-name'>{name}</span> <span class='stock-code'>{code}</span></div>", unsafe_allow_html=True)
                        
                        # 价格大字
                        st.markdown(f"<div class='big-price {price_color}'>{price:.2f}</div>", unsafe_allow_html=True)
                        
                        # 连板数 + 涨跌
                        zt_badge = f"<span style='background:#ff0000;color:white;padding:1px 4px;border-radius:3px;font-size:0.8rem;margin-left:5px'>{zt_count}连板</span>" if zt_count>=2 else ""
                        st.markdown(f"<div style='font-weight:bold; margin-bottom:8px;'>{chg:+.2f}% {zt_badge}</div>", unsafe_allow_html=True)
                        
                        # 游资策略标签
                        st.markdown(f"<div style='margin-bottom:8px'><span class='strategy-tag {strategy_class}'>{strategy_text}</span></div>", unsafe_allow_html=True)
                        
                        if cost_low > 0:
                            st.markdown(f"<div class='cost-range-box'>主力成本: {cost_low:.2f}</div>", unsafe_allow_html=True)

                        # S/R
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
                        
                        if str(row['note']) not in ['nan', '']:
                            st.caption(f"📝 {row['note']}")
                        
                        # 🔥 弹窗看图按钮 (保留！)
                        if st.button("📈 看图", key=f"btn_{code}"):
                            view_chart_modal(code, name)

else: st.info("请在左侧添加股票")

if auto_refresh:
    time.sleep(3)
    st.rerun()
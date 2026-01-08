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

# --- 🎨 CSS 样式 (优化图标按钮布局) ---
st.markdown("""
    <style>
        html, body, p, div, span { font-family: 'Source Sans Pro', sans-serif; color: #0E1117; }
        .block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; }
        
        /* 卡片容器 */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid #eee !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05); 
            background-color: #ffffff; 
            padding: 12px !important;
            border-radius: 10px;
            margin-bottom: 12px;
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
        
        /* 策略标签 */
        .strategy-tag {
            padding: 3px 6px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; 
            color: white; display: inline-block; vertical-align: middle; margin-right: 5px;
        }
        .tag-dragon { background: linear-gradient(45deg, #ff0000, #ff6b6b); }
        .tag-buy { background-color: #d9534f; }
        .tag-sell { background-color: #5cb85c; }
        .tag-wait { background-color: #999; }
        .tag-special { background-color: #f0ad4e; }

        /* 成本区间 */
        .cost-range-box {
            background-color: #f8f9fa; border-left: 3px solid #666;
            padding: 3px 8px; margin: 8px 0; border-radius: 0 4px 4px 0;
            font-size: 0.85rem; color: #444;
        }
        
        /* 支撑压力 */
        .sr-block {
            padding-top: 6px; border-top: 1px dashed #eee;
            display: grid; grid-template-columns: 1fr 1fr; gap: 4px;
        }
        .sr-item { font-size: 0.85rem; font-weight: bold; color: #555; }
        
        /* --- 🔥 核心UI调整：右上角图标按钮组 --- */
        
        /* 1. 让操作列的按钮紧凑排列 */
        [data-testid="column"] .stButton button {
             padding: 0px 8px;
             min-height: 0px;
             height: 32px; /* 固定高度 */
             border: none;
             background: transparent;
             font-size: 1.1rem;
             color: #888;
             transition: all 0.2s;
        }
        
        /* 2. 删除按钮鼠标悬停 */
        button[kind="secondary"]:hover {
            color: #d9534f !important; /* 红色 */
            background: #fff5f5 !important;
        }

        /* 3. popover 按钮 (分组图标) 样式 */
        div[data-testid="stPopover"] button {
             padding: 0px 8px;
             min-height: 0px;
             height: 32px;
             border: none;
             background: transparent;
             font-size: 1.1rem;
             color: #888;
        }
        /* popover 悬停效果 */
        div[data-testid="stPopover"] button:hover {
             color: #007bff !important; /* 蓝色 */
             background: #f0f8ff !important;
        }
        
        /* 底部看图按钮保持原样 */
        .view-chart-btn button {
             width: 100%; border-radius: 4px; font-weight: bold; margin-top: 8px;
             background-color: #f0f2f6; color: #31333F; height: auto; padding: 0.5rem;
        }
        .view-chart-btn button:hover { background-color: #e0e2e6; }

    </style>
""", unsafe_allow_html=True)

DATA_FILE = 'my_stock_plan_v3.csv'

# --- 核心数据管理函数 ---

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

# 单只股票删除功能
def delete_single_stock(code_to_delete):
    df = load_data()
    if code_to_delete in df['code'].values:
        df = df[df['code'] != code_to_delete]
        save_data(df)
        return True
    return False

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

# 获取历史数据 & 智能计算 (含缓存)
@st.cache_data(ttl=3600)
def get_stock_history_metrics(code):
    try:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=100)).strftime("%Y%m%d")
        try:
            stock_df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        except: return None, 0, 0, 0
        
        if stock_df.empty: return None, 0, 0, 0
        
        stock_df['MA5'] = stock_df['收盘'].rolling(5).mean()
        stock_df['MA10'] = stock_df['收盘'].rolling(10).mean()
        stock_df['MA20'] = stock_df['收盘'].rolling(20).mean()
        
        recent = stock_df.tail(20)
        total_amt = recent['成交额'].sum()
        total_vol = recent['成交量'].sum()
        smart_cost = total_amt / (total_vol * 100) if total_vol > 0 else 0
        
        stock_df['is_zt'] = stock_df['涨跌幅'] > 9.5
        zt_count = 0
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

# 🧠 大游资策略引擎
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

    if zt_count >= 2:
        if pct_chg > 9.5: return f"💎 {zt_count+1}板锁仓", "tag-dragon"
        elif price > day_vwap and price > ma5: return f"🔥 妖股持筹 ({zt_count}板)", "tag-dragon"
        elif price < ma5: return "💀 断板止盈", "tag-sell"

    if yesterday_zt and zt_count < 3:
        if 2 < pct_chg < 9.0 and price > day_vwap: return f"🚀 {zt_count}进{zt_count+1} 接力", "tag-buy"
    
    if zt_count >= 3 and pct_chg < -3 and price > ma10: return "🐲 龙头首阴", "tag-special"

    high_pct = ((high - pre_close) / pre_close) * 100
    if high_pct > 7 and pct_chg < 3 and price > ma20: return "👆 仙人指路", "tag-special"

    if price > ma20 and ma10 > ma20:
        dist_ma10 = abs(price - ma10) / ma10
        if dist_ma10 < 0.02: return "🌊 MA10 低吸", "tag-buy"

    if pct_chg > 9.8: return "🚀 涨停持股", "tag-dragon"
    if price > day_vwap: return "💪 强势整理", "tag-wait"
    if price < day_vwap: return "👀 弱势观望", "tag-wait"
    
    return "😴 观望", "tag-wait"

# --- 侧边栏 ---
st.sidebar.title("Control Panel")
auto_refresh = st.sidebar.toggle("🔥 实时刷新 (3s)", value=True)
st.sidebar.markdown("---")

df = load_data()

# 🔥 添加个股 + 快捷分组
with st.sidebar.expander("➕ 添加/编辑 个股", expanded=True):
    code_in = st.text_input("代码 (6位数)", key="cin")
    
    if 'calc_s1' not in st.session_state: 
        for k in ['s1','s2','r1','r2']: st.session_state[f'calc_{k}'] = 0.0
    
    if st.button("⚡ 智能计算支撑压力"):
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
                    st.success(f"已识别：{zt}连板" if zt>=2 else "计算完成")
    
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
        
        if selected_grp == "✍️ 新建/手动输入":
            final_grp = st.text_input("输入新分组名称", "龙头")
        else:
            final_grp = selected_grp
            
        note=st.text_area("笔记 (可选)")
        
        if st.form_submit_button("💾 保存") and code_in:
            name=""
            if code_in in df.code.values: name=df.loc[df.code==code_in,'name'].values[0]
            new={"code":code_in,"name":name,"s1":s1,"s2":s2,"r1":r1,"r2":r2,"group":final_grp,"note":note}
            if code_in in df.code.values: 
                df.loc[df.code==code_in, list(new.keys())]=list(new.values())
            else: 
                df=pd.concat([df,pd.DataFrame([new])],ignore_index=True)
            save_data(df)
            st.success(f"{code_in} 已保存到【{final_grp}】")
            time.sleep(0.5)
            st.rerun()

# 弹窗看图
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

    # 获取所有分组列表，用于 popover 选择
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
                
                hist_df, cost_low, zt_count, yesterday_zt = get_stock_history_metrics(code)
                strategy_text, strategy_class = ai_strategy_engine(info, hist_df, cost_low, zt_count, yesterday_zt)
                
                with cols[j]:
                    with st.container(border=True):
                        # 🔥🔥🔥 核心UI升级：右上角图标操作区 🔥🔥🔥
                        # 使用 5:1:1 的比例，将名字、分组图标、删除图标排成一行
                        col_name, col_grp_btn, col_del_btn = st.columns([5, 1, 1])
                        
                        with col_name:
                            # 显示股票名称和代码
                            st.markdown(f"<div style='white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'><span class='stock-name'>{name}</span> <span class='stock-code'>{code}</span></div>", unsafe_allow_html=True)
                        
                        with col_grp_btn:
                            # 🔥 分组修改弹窗 (Popover)
                            # 点击这个 🏷️ 图标，会弹出一个小浮窗
                            with st.popover("🏷️", help="修改分组"):
                                st.markdown(f"##### 修改 【{name}】 的分组")
                                # 选择已有分组
                                new_group_select = st.selectbox("选择已有分组", ["(不变)"] + all_groups_for_popover, key=f"grp_sel_{code}")
                                # 手动输入新分组
                                new_group_input = st.text_input("或输入新分组名称", key=f"grp_inp_{code}")
                                
                                # 确定最终的新分组名称
                                final_new_group = None
                                if new_group_input.strip():
                                    final_new_group = new_group_input.strip()
                                elif new_group_select != "(不变)":
                                    final_new_group = new_group_select
                                    
                                # 确认修改按钮
                                if st.button("✅ 确认修改", key=f"confirm_grp_{code}"):
                                    if final_new_group and final_new_group != group:
                                        # 更新 Dataframe 并保存
                                        df.loc[df.code == code, 'group'] = final_new_group
                                        save_data(df)
                                        st.toast(f"已将 {name} 移动到 【{final_new_group}】 分组")
                                        time.sleep(0.5)
                                        st.rerun() # 刷新页面
                                    elif final_new_group == group:
                                         st.toast("分组未发生变化")
                                    else:
                                         st.toast("请选择或输入新的分组名称")

                        with col_del_btn:
                            # 🔥 直接删除按钮 (垃圾桶图标)
                            if st.button("🗑️", key=f"del_{code}", help="删除个股"):
                                if delete_single_stock(code):
                                    st.toast(f"{name} 已删除")
                                    time.sleep(0.5)
                                    st.rerun()

                        # --- 下面是卡片主体内容 ---
                        
                        # 价格大字
                        st.markdown(f"<div class='big-price {price_color}'>{price:.2f}</div>", unsafe_allow_html=True)
                        
                        # 连板数 + 涨跌幅
                        zt_badge = f"<span style='background:#ff0000;color:white;padding:1px 4px;border-radius:3px;font-size:0.8rem;margin-left:5px'>{zt_count}连板</span>" if zt_count>=2 else ""
                        st.markdown(f"<div style='font-weight:bold; margin-bottom:8px;'>{chg:+.2f}% {zt_badge}</div>", unsafe_allow_html=True)
                        
                        # 策略标签
                        st.markdown(f"<div style='margin-bottom:8px'><span class='strategy-tag {strategy_class}'>{strategy_text}</span></div>", unsafe_allow_html=True)
                        
                        # 主力成本
                        if cost_low > 0:
                            st.markdown(f"<div class='cost-range-box'>主力成本: {cost_low:.2f}</div>", unsafe_allow_html=True)

                        # 支撑压力
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
                        
                        # 笔记
                        if str(row['note']) not in ['nan', '']:
                            st.caption(f"📝 {row['note']}")
                        
                        # 看图按钮 (增加了一个class方便CSS定位)
                        st.markdown('<div class="view-chart-btn">', unsafe_allow_html=True)
                        if st.button("📈 看图", key=f"btn_{code}"):
                            view_chart_modal(code, name)
                        st.markdown('</div>', unsafe_allow_html=True)

else: st.info("👈 请在左侧添加股票")

if auto_refresh:
    time.sleep(3)
    st.rerun()
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
import yt_dlp
from datetime import datetime, timedelta, time as dt_time
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 页面基础设置 ---
st.set_page_config(
    page_title="Alpha 游资系统 (Pro + AI)",
    page_icon="🐲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 连接服务 ---
try:
    from streamlit_gsheets import GSheetsConnection
    if "connections" in st.secrets and "gsheets" in st.secrets["connections"]:
        USE_CLOUD_DB = True
        conn = st.connection("gsheets", type=GSheetsConnection)
    else: USE_CLOUD_DB = False
except: USE_CLOUD_DB = False

try:
    if "gemini" in st.secrets and "api_key" in st.secrets["gemini"]:
        genai.configure(api_key=st.secrets["gemini"]["api_key"])
        USE_AI = True
    else: USE_AI = False
except: USE_AI = False

# --- 🎨 CSS 样式 (恢复四宫格样式) ---
st.markdown("""
    <style>
        html, body, p, div, span { font-family: 'Source Sans Pro', sans-serif; }
        .block-container { padding-top: 1rem !important; }
        
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
        
        .strategy-badge { padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; color: white; background-color: #333; margin-right: 4px; }
        .bg-dragon { background: linear-gradient(45deg, #d32f2f, #ef5350); }
        .bg-relay { background: linear-gradient(45deg, #f57c00, #ffb74d); }
        .bg-low { background: linear-gradient(45deg, #1976d2, #42a5f5); }
        .bg-trend { background: linear-gradient(45deg, #388e3c, #66bb6a); }
        
        .advice-box { margin-top: 5px; padding: 8px; border-radius: 4px; font-weight: bold; text-align: center; font-size: 0.9rem; border: 1px solid #eee; }
        .advice-buy { background-color: #fff3f3; color: #d9534f; border-color: #d9534f; }
        .advice-sell { background-color: #f0f9f0; color: #5cb85c; border-color: #5cb85c; }
        .advice-hold { background-color: #f0f8ff; color: #3498db; border-color: #3498db; }
        
        /* 🔥 恢复支撑压力位四宫格样式 */
        .sr-block { padding-top: 6px; border-top: 1px dashed #eee; display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }
        .sr-item { font-size: 0.8rem; font-weight: bold; color: #555; }
        
        .plan-item { margin-bottom: 4px; line-height: 1.4; font-size: 0.85rem; color: #444; }
        .highlight-money { color: #d9534f; font-weight: bold; background: #fff5f5; padding: 0 4px; border-radius: 3px; }
        .highlight-support { color: #2980b9; font-weight: bold; background: #eaf2f8; padding: 0 4px; border-radius: 3px; }
    </style>
""", unsafe_allow_html=True)

DATA_FILE = 'my_stock_plan_v3.csv'
LEARNED_STRATEGY_FILE = 'learned_strategies.csv'
STRATEGY_OPTIONS = ["🤖 自动判断 (Auto)", "🐲 龙头掘金", "🚀 连板接力", "📉 涨停回调", "🌊 趋势低吸", "🔥 短线情绪"]

# --- 核心数据函数 ---
def load_data():
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
        except: pass
    if not os.path.exists(DATA_FILE): return pd.DataFrame(columns=default_cols)
    return pd.read_csv(DATA_FILE, dtype={"code": str})

def save_data(df):
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
    cols = ["date", "strategy_name", "core_logic", "buy_condition", "sell_condition", "visual_pattern"]
    if USE_CLOUD_DB:
        try: return conn.read(worksheet="learned_strategies", ttl=10)
        except: pass
    if not os.path.exists(LEARNED_STRATEGY_FILE): return pd.DataFrame(columns=cols)
    return pd.read_csv(LEARNED_STRATEGY_FILE)

def save_learned_strategy(record):
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
    t = now.time()
    if (dt_time(9,15)<=t<=dt_time(11,30)) or (dt_time(13,0)<=t<=dt_time(15,0)): return True, "交易中"
    return False, "非交易时间"

def get_realtime_quotes(code_list):
    if not code_list: return {}
    q_codes = [f"{'sh' if c.startswith(('6','5')) else 'sz'}{c}" for c in code_list]
    url = f"http://hq.sinajs.cn/list={','.join(q_codes)}"
    try:
        r = requests.get(url, headers={'Referer': 'http://sina.com.cn'}, timeout=3)
        data = {}
        for line in r.text.split('\n'):
            if '="' in line:
                code = line.split('="')[0].split('_')[-1][2:]
                val = line.split('="')[1].strip('";').split(',')
                if len(val)>30:
                    data[code] = {"name": val[0], "open": float(val[1]), "pre_close": float(val[2]), "price": float(val[3])}
        return data
    except: return {}

@st.cache_data(ttl=3600)
def get_stock_history_metrics(code):
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=(datetime.now()-timedelta(days=120)).strftime("%Y%m%d"), adjust="qfq")
        df['MA5'] = df['收盘'].rolling(5).mean(); df['MA10'] = df['收盘'].rolling(10).mean()
        recent = df.tail(20)
        cost = (recent['成交额'].sum()/recent['成交量'].sum()) if recent['成交量'].sum()>0 else 0
        if cost>200: cost/=100
        zt_count = 0
        df['is_zt'] = df['收盘'].pct_change()>0.095
        for i in range(len(df)-1,-1,-1):
            if df.iloc[i]['is_zt']: zt_count+=1
            else: break
        return df, cost, zt_count, df.tail(60)['成交额'].max(), df.iloc[-1]['换手率']
    except: return None, 0, 0, 0, 0

def evaluate_strategy_realtime(strategy, info, hist, cost, zt_cnt, turnover):
    if hist is None: return "数据不足", "bg-auto", ""
    p, pre = info['price'], info['pre_close']
    pct = (p-pre)/pre*100
    ma5, ma10 = hist.iloc[-1]['MA5'], hist.iloc[-1]['MA10']
    
    advice, style, badge = "观察", "advice-hold", "bg-auto"
    if "龙头" in strategy:
        badge = "bg-dragon"
        if p>cost and p>ma10:
            if pct<-3: advice, style = "🟢 回调洗盘: 吸", "advice-buy"
            elif pct>5: advice, style = "🔴 加速: 持", "advice-hold"
        elif p<ma10: advice, style = "⚠️ 破10日: 减", "advice-sell"
    elif "连板" in strategy:
        badge = "bg-relay"
        if pct>9.5: advice, style = "🔒 涨停锁仓", "advice-hold"
        elif p>pre*1.03: advice, style = "🔥 弱转强: 买", "advice-buy"
    elif "回调" in strategy:
        badge = "bg-low"
        if abs((p-ma10)/ma10)<0.02: advice, style = "🎯 踩10日线: 吸", "advice-buy"
        elif p<ma10: advice, style = "🚫 破位: 止", "advice-sell"
    return advice, style, badge

def generate_plan_details(strategy, code, price, max_amt, turnover, ma5, ma10):
    html = ""
    target_amt = max_amt * 0.05
    if "连板" in strategy or "龙头" in strategy:
        html += f"<div class='plan-item'>🎯 <b>竞价目标：</b><span class='highlight-money'>{target_amt/10000:.2f}万</span></div>"
        html += "<div class='plan-item'>1. <b>弱转强：</b>竞价达标，开盘不破均线 👉 买入。</div>"
    elif "低吸" in strategy:
        sup = ma10 if ma10>0 else price*0.95
        html += f"<div class='plan-item'>🛡️ <b>支撑：</b><span class='highlight-support'>{sup:.2f}</span></div>"
        html += "<div class='plan-item'>1. <b>黄金坑：</b>缩量回踩支撑 👉 低吸。</div>"
    return html

def prefetch_all_data(stock_codes):
    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor: 
        future_to_code = {executor.submit(get_stock_history_metrics, code): code for code in stock_codes}
        for future in as_completed(future_to_code):
            code = future_to_code[future]
            try: results[code] = future.result()
            except: results[code] = (None, 0, 0, 0, 0)
    return results

def get_dist_html(target, current):
    try: target=float(target); current=float(current)
    except: return ""
    if target == 0: return ""
    d = ((current - target) / target) * 100
    col = "#d9534f" if abs(d)<1.0 else "#f0ad4e" if abs(d)<3.0 else "#999"
    return f"<span style='color:{col}; font-weight:bold;'>({d:+.1f}%)</span>"

# --- 视频分析模块 ---
def process_video_url_or_file(input_type, file_obj, url, user_prompt):
    if not USE_AI: return None
    status = st.empty()
    temp_path = "temp_ai_video.mp4"
    
    if input_type == "Link (链接)":
        if not url:
            status.error("❌ 请输入链接")
            return None
        status.info(f"🕸️ 正在抓取视频... (请稍候)")
        try:
            ydl_opts = {'format': 'best[ext=mp4]/best', 'outtmpl': temp_path, 'quiet': True, 'overwrites': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            status.error(f"❌ 下载失败: {e}"); return None
    else:
        if not file_obj:
            status.error("❌ 请上传文件"); return None
        with open(temp_path, "wb") as f: f.write(file_obj.getbuffer())

    try:
        status.info("📤 正在上传给 AI 大脑...")
        video_upload = genai.upload_file(path=temp_path)
        while video_upload.state.name == "PROCESSING":
            time.sleep(2); video_upload = genai.get_file(video_upload.name)
        if video_upload.state.name == "FAILED":
            status.error("❌ AI 处理失败"); return None
            
        status.info("🧠 AI 正在深度分析...")
        system_prompt = """
        你是一位顶级游资操盘手。请分析这段视频。
        总结出一套可执行的策略，严格返回如下 JSON 格式 (纯文本):
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
        if os.path.exists(temp_path): os.remove(temp_path)
        status.empty()
        return response.text
    except Exception as e:
        status.error(f"AI 出错: {e}"); return None

@st.dialog("📈 个股详情", width="large")
def view_chart_modal(code, name):
    st.subheader(f"{name} ({code})")
    ts = int(time.time())
    mid = "1" if code.startswith(('6','5','9')) else "0"
    t1, t2 = st.tabs(["分时图", "日线图"])
    with t1: st.image(f"https://webquotepic.eastmoney.com/GetPic.aspx?nid={mid}.{code}&imageType=r&t={ts}", use_container_width=True)
    with t2: st.image(f"https://webquotepic.eastmoney.com/GetPic.aspx?nid={mid}.{code}&imageType=k&t={ts}", use_container_width=True)

# --- 主程序 ---
if 'calc_s1' not in st.session_state:
    st.session_state.calc_s1 = 0.0; st.session_state.calc_s2 = 0.0
    st.session_state.calc_r1 = 0.0; st.session_state.calc_r2 = 0.0

trading_active, trading_status_msg = is_trading_time()

with st.sidebar:
    st.title("控制台")
    st.markdown(f"市场: **{trading_status_msg}**")
    status_icon = "☁️" if USE_CLOUD_DB else "💾"
    ai_icon = "🧠" if USE_AI else "🚫"
    st.markdown(f"数据: {status_icon} | AI: {ai_icon}")
    st.divider()
    
    # 🔥 恢复：添加/编辑股票时可选择分组
    with st.expander("➕ 添加/编辑 个股"):
        code_in = st.text_input("代码", key="cin").strip()
        if st.button("⚡ 计算"):
            if code_in:
                hist, _, zt, _, _ = get_stock_history_metrics(code_in)
                if hist is not None:
                    last = hist.iloc[-1]; pivot = (last['最高']+last['最低']+last['收盘'])/3
                    st.session_state.calc_r1 = round(2*pivot - last['最低'], 2)
                    st.session_state.calc_s1 = round(2*pivot - last['最高'], 2)
                    st.session_state.calc_r2 = round(pivot + (last['最高'] - last['最低']), 2)
                    st.session_state.calc_s2 = round(pivot - (last['最高'] - last['最低']), 2)
                    st.success(f"{zt}连板")
        with st.form("add"):
            c1, c2 = st.columns(2)
            s1 = c1.number_input("S1", value=float(st.session_state.calc_s1))
            s2 = c1.number_input("S2", value=float(st.session_state.calc_s2))
            r1 = c2.number_input("R1", value=float(st.session_state.calc_r1))
            r2 = c2.number_input("R2", value=float(st.session_state.calc_r2))
            
            # 🔥 恢复：分组选择功能
            df_temp = load_data()
            existing_groups = list(df_temp['group'].unique())
            if "默认" not in existing_groups: existing_groups.insert(0, "默认")
            
            group_sel = st.selectbox("分组", existing_groups + ["➕ 新建分组..."])
            if group_sel == "➕ 新建分组...":
                group_val = st.text_input("输入新分组名")
            else:
                group_val = group_sel
            
            strat = st.selectbox("战法", STRATEGY_OPTIONS)
            
            if st.form_submit_button("💾 保存"):
                if code_in:
                    df = load_data(); name = ""
                    if code_in in df.code.values: name = df.loc[df.code==code_in, 'name'].values[0]
                    # 如果没有输入新组名，回退到默认
                    final_group = group_val if group_val else "默认"
                    
                    new_entry = {"code": code_in, "name": name, "s1": s1, "s2": s2, "r1": r1, "r2": r2, "group": final_group, "strategy": strat, "note": ""}
                    if code_in in df.code.values:
                        for k, v in new_entry.items(): df.loc[df.code==code_in, k] = v
                    else: df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                    save_data(df); st.rerun()
    if st.button("🧹 刷新"): st.cache_data.clear(); st.rerun()

st.title("Alpha 游资系统 (Pro + AI)")
tab1, tab2, tab3 = st.tabs(["📈 实战看板", "🎓 AI 视频悟道", "📚 战法知识库"])

with tab1:
    df = load_data()
    if not df.empty:
        quotes = get_realtime_quotes(df['code'].tolist())
        batch_data = prefetch_all_data(df['code'].unique().tolist())
        
        # 🔥 恢复：按分组循环显示 (龙头掘金、涨停回调...)
        all_groups = df['group'].unique()
        for group in all_groups:
            st.subheader(f"📂 {group}")
            group_df = df[df['group'] == group]
            
            rows = [r for _, r in group_df.iterrows()]
            for i in range(0, len(rows), 4):
                cols = st.columns(4)
                for j, row in enumerate(rows[i:i+4]):
                    code = row['code']; strat = row['strategy']
                    info = quotes.get(code, {}); p = info.get('price', 0); name = info.get('name', code)
                    hist, cost, zt, max_amt, tn = batch_data.get(code, (None, 0, 0, 0, 0))
                    adv, sty, bdg = evaluate_strategy_realtime(strat, info, hist, cost, zt, tn)
                    
                    with cols[j]:
                        with st.container(border=True):
                            c1, c2 = st.columns([4, 1])
                            with c1: st.markdown(f"**{name}** `{code}`")
                            with c2: 
                                if st.button("🗑️", key=f"d_{code}"): delete_single_stock(code); st.rerun()
                            p_col = "price-up" if p > info.get('pre_close',0) else "price-down"
                            st.markdown(f"<div class='big-price {p_col}'>{p:.2f}</div>", unsafe_allow_html=True)
                            st.markdown(f"<span class='strategy-badge {bdg}'>{strat[:2]}</span> {adv}", unsafe_allow_html=True)
                            
                            # 🔥 恢复：四宫格支撑压力位显示
                            r1, r2, s1, s2 = row['r1'], row['r2'], row['s1'], row['s2']
                            st.markdown(f"""
                            <div class='sr-block'>
                                <div class='sr-item'><span style='color:#d9534f'>R2</span> {r2}{get_dist_html(r2, p)}</div>
                                <div class='sr-item'><span style='color:#5cb85c'>S1</span> {s1}{get_dist_html(s1, p)}</div>
                                <div class='sr-item'><span style='color:#f0ad4e'>R1</span> {r1}{get_dist_html(r1, p)}</div>
                                <div class='sr-item'><span style='color:#4cae4c'>S2</span> {s2}{get_dist_html(s2, p)}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            with st.expander("🎲 计划"):
                                st.markdown(generate_plan_details(strat, code, p, max_amt, tn, 0, 0), unsafe_allow_html=True)
                            if st.button("📈 看图", key=f"b_{code}"): view_chart_modal(code, name)
    else:
        st.info("👈 请在左侧添加股票")

with tab2:
    st.header("🎓 AI 视频操盘学徒")
    if not USE_AI: st.warning("⚠️ 请配置 secrets.toml 中的 [gemini] api_key")
    else:
        st.markdown("支持 B站 / YouTube 链接，或直接上传文件。")
        input_method = st.radio("选择来源", ["Link (链接)", "File (上传文件)"], horizontal=True)
        url_input = ""; file_input = None
        if input_method == "Link (链接)": url_input = st.text_input("🔗 粘贴视频链接 (B站/YouTube)")
        else: file_input = st.file_uploader("📂 上传视频", type=['mp4', 'mov'])
        note_input = st.text_input("💡 提示词 (可选)", value="重点分析主力买点逻辑")
        
        if st.button("🚀 开始 AI 分析"):
            res = process_video_url_or_file(input_method, file_input, url_input, note_input)
            if res:
                try:
                    s_data = json.loads(res.replace("```json", "").replace("```", "").strip())
                    st.success("✅ AI 悟道成功！")
                    with st.container(border=True):
                        st.subheader(f"🛡️ {s_data.get('strategy_name', '未命名')}")
                        st.info(f"**核心逻辑:** {s_data.get('core_logic')}")
                        st.write(f"**🔴 买入:** {s_data.get('buy_condition')}")
                        st.write(f"**🟢 卖出:** {s_data.get('sell_condition')}")
                        if st.button("💾 存入战法库"):
                            save_learned_strategy({"date": datetime.now().strftime("%Y-%m-%d"), **s_data})
                            st.toast("保存成功！")
                except: st.error("解析失败"); st.text(res)

with tab3:
    st.header("📚 知识库")
    sdf = get_learned_strategies()
    if not sdf.empty:
        for i, r in sdf.iterrows():
            with st.container(border=True):
                st.markdown(f"### {r['strategy_name']} <small>{r['date']}</small>", unsafe_allow_html=True)
                st.write(r['core_logic'])
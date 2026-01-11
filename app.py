import streamlit as st
import pandas as pd
import requests
import os
import time
import json
import numpy as np
import akshare as ak
import google.generativeai as genai
import yt_dlp
from datetime import datetime, timedelta, time as dt_time
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 页面基础设置 ---
st.set_page_config(
    page_title="Alpha 游资系统 (双引擎)",
    page_icon="🐲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 连接服务 ---
try:
    from streamlit_gsheets import GSheetsConnection
    if "connections" in st.secrets and "gsheets" in st.secrets["connections"]:
        USE_CLOUD_DB = True; conn = st.connection("gsheets", type=GSheetsConnection)
    else: USE_CLOUD_DB = False
except: USE_CLOUD_DB = False

try:
    if "gemini" in st.secrets and "api_key" in st.secrets["gemini"]:
        genai.configure(api_key=st.secrets["gemini"]["api_key"])
        USE_AI = True
    else: USE_AI = False
except: USE_AI = False

# --- 🎨 CSS 样式 ---
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
        .sr-block { padding-top: 6px; border-top: 1px dashed #eee; display: grid; grid-template-columns: 1fr 1fr; gap: 4px; margin-top: 8px;}
        .sr-item { font-size: 0.8rem; font-weight: bold; color: #555; }
        .signal-box { margin-top: 10px; padding: 8px; background: #f8f9fa; border-radius: 6px; border-left: 4px solid #ddd; font-size: 0.9rem;}
        .sig-buy { border-left-color: #d9534f; background: #fff5f5; }
        .sig-sell { border-left-color: #28a745; background: #f0f9f0; }
        .sig-wait { border-left-color: #17a2b8; background: #f0f8ff; }
        .strategy-badge { padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; background: #333; color: white; margin-right: 5px; }
        .bg-dragon { background: linear-gradient(45deg, #d32f2f, #ef5350); }
        .bg-relay { background: linear-gradient(45deg, #f57c00, #ffb74d); }
        .bg-low { background: linear-gradient(45deg, #1976d2, #42a5f5); }
        .bg-ai { background: linear-gradient(45deg, #6a11cb, #2575fc); }
    </style>
""", unsafe_allow_html=True)

DATA_FILE = 'my_stock_plan_v3.csv'
LEARNED_LOGIC_FILE = 'comprehensive_logic_v2.csv'
BUILTIN_STRATEGIES = ["自动观察", "🐲 龙头掘金", "🚀 连板接力", "📉 涨停回调", "🌊 趋势低吸"]

# --- 数据管理 ---
def load_data():
    default_cols = ["code", "name", "s1", "s2", "r1", "r2", "group", "strategy", "note"]
    if USE_CLOUD_DB:
        try:
            df = conn.read(worksheet="stock_config", ttl=5)
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

def get_learned_logics():
    cols = ["date", "strategy_name", "trend_logic", "intraday_logic", "python_code"]
    if USE_CLOUD_DB:
        try: return conn.read(worksheet="comprehensive_logic_v2", ttl=10)
        except: pass
    if not os.path.exists(LEARNED_LOGIC_FILE): return pd.DataFrame(columns=cols)
    return pd.read_csv(LEARNED_LOGIC_FILE)

def save_learned_logic(record):
    df = get_learned_logics()
    new_df = pd.DataFrame([record])
    df = pd.concat([df, new_df], ignore_index=True)
    if USE_CLOUD_DB:
        try: conn.update(worksheet="comprehensive_logic_v2", data=df)
        except: pass
    df.to_csv(LEARNED_LOGIC_FILE, index=False)

# --- 辅助函数 ---
def is_trading_time():
    now = datetime.utcnow() + timedelta(hours=8)
    if now.weekday() >= 5: return False, "周末休市"
    t = now.time()
    if (dt_time(9,15)<=t<=dt_time(11,30)) or (dt_time(13,0)<=t<=dt_time(15,0)): return True, "交易中"
    return False, "非交易时间"

def get_dist_html(target, current):
    try: target=float(target); current=float(current)
    except: return ""
    if target == 0: return ""
    d = ((current - target) / target) * 100
    col = "#d9534f" if abs(d)<1.0 else "#f0ad4e" if abs(d)<3.0 else "#999"
    return f"<span style='color:{col}; font-weight:bold;'>({d:+.1f}%)</span>"

# --- 🔥 全维度数据获取 ---
@st.cache_data(ttl=60)
def get_stock_data_bundle(code):
    bundle = {"daily": None, "minute": None, "info": {}}
    try:
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=90)).strftime("%Y%m%d")
        daily = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
        
        if not daily.empty:
            daily['MA5'] = daily['收盘'].rolling(5).mean()
            daily['MA10'] = daily['收盘'].rolling(10).mean()
            daily['MA20'] = daily['收盘'].rolling(20).mean()
            daily['VOL_MA5'] = daily['成交量'].rolling(5).mean()
            bundle['daily'] = daily
            
            last = daily.iloc[-1]
            bundle['info'] = {
                "name": code,
                "price": last['收盘'],
                "pct": last['涨跌幅'],
                "pre_close": last['收盘'] / (1 + last['涨跌幅']/100)
            }

        minute = ak.stock_zh_a_hist_min_em(symbol=code, period='1', adjust='qfq')
        if not minute.empty:
            minute['MA_PRICE'] = (minute['close'] * minute['volume']).cumsum() / minute['volume'].cumsum()
            bundle['minute'] = minute
        return bundle
    except: return None

# --- 🔥 核心1：内置硬编码策略引擎 (恢复这个功能！) ---
def evaluate_builtin_strategy(strategy, bundle):
    if not bundle or bundle['daily'] is None: return "数据不足", "bg-auto"
    
    daily = bundle['daily']
    info = bundle['info']
    last = daily.iloc[-1]
    p = info['price']; pct = info['pct']
    ma5 = last['MA5']; ma10 = last['MA10']
    
    # 默认值
    text = "观察"; badge_class = "bg-auto"
    
    if "龙头" in strategy:
        badge_class = "bg-dragon"
        if p > ma5 and p > ma10:
            if pct < -3: text = "🟢 回调洗盘: 吸"
            elif pct > 5: text = "🔴 加速: 持"
            else: text = "🔵 趋势良好"
        elif p < ma10: text = "⚠️ 破10日: 减"
            
    elif "连板" in strategy:
        badge_class = "bg-relay"
        if pct > 9.5: text = "🔒 涨停锁仓"
        elif p > info['pre_close'] * 1.03: text = "🔥 弱转强: 买"
        elif p < info['pre_close']: text = "🟢 水下: 观望"
        
    elif "回调" in strategy or "低吸" in strategy:
        badge_class = "bg-low"
        if abs((p - ma10)/ma10) < 0.02: text = "🎯 踩10日线: 吸"
        elif p < ma10: text = "🚫 破位: 止"
        else: text = "🔵 等回落"
        
    return text, badge_class

# --- 🔥 核心2：AI 动态逻辑执行引擎 ---
def execute_ai_logic(bundle, logic_code):
    if not bundle or bundle['daily'] is None: return "数据不足", "sig-wait"
    daily_df = bundle['daily']; minute_df = bundle['minute']
    try:
        local_scope = {}
        exec(logic_code, globals(), local_scope)
        if 'analyze' in local_scope:
            signal, reason = local_scope['analyze'](daily_df, minute_df)
            if signal == "BUY": return f"🚀 {reason}", "sig-buy"
            if signal == "SELL": return f"⚠️ {reason}", "sig-sell"
            if signal == "WAIT": return f"👀 {reason}", "sig-wait"
        return "逻辑未触发", "sig-wait"
    except Exception as e: return f"运行错误: {str(e)[:20]}", "sig-wait"

# --- AI 学习模块 ---
def process_video_comprehensive(file_obj, url, input_type, note):
    if not USE_AI: return None
    status = st.empty()
    temp_path = "temp.mp4"
    if input_type == "Link (链接)":
        try:
            status.info("🕸️ 正在抓取视频...")
            ydl_opts = {'format': 'best[ext=mp4]/best', 'outtmpl': temp_path, 'quiet': True, 'overwrites': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([url])
        except Exception as e: status.error(f"下载失败: {e}"); return None
    else:
        with open(temp_path, "wb") as f: f.write(file_obj.getbuffer())

    try:
        status.info("🧠 AI 正在进行【日线趋势+量价+分时】三维建模...")
        video_upload = genai.upload_file(path=temp_path)
        while video_upload.state.name == "PROCESSING": time.sleep(2); video_upload = genai.get_file(video_upload.name)
        
        system_prompt = """
        你是一位顶级游资操盘手。请分析视频，总结出一套【多周期共振】的交易系统。
        
        请编写一个 Python 函数 `analyze(daily_df, minute_df)`:
        - daily_df 列名: '收盘','开盘','最高','最低','成交量','MA5','MA10','MA20','VOL_MA5'
        - minute_df 列名: 'close','open','high','low','volume','MA_PRICE'(均价线)
        - minute_df 可能为 None (如果未开盘)，需处理。
        
        函数返回元组: (SIGNAL, REASON)
        - SIGNAL: "BUY", "SELL", "WAIT"
        - REASON: 简短中文理由
        
        示例代码逻辑：
        def analyze(daily_df, minute_df):
            last_day = daily_df.iloc[-1]
            if last_day['收盘'] < last_day['MA5']: return "WAIT", "日线破位"
            if minute_df is not None and not minute_df.empty:
                if minute_df.iloc[-1]['close'] > minute_df.iloc[-1]['MA_PRICE']: return "BUY", "分时强势"
            return "WAIT", "等待分时确认"

        请严格返回 JSON (纯文本):
        {
            "strategy_name": "策略名",
            "trend_logic": "日线逻辑描述",
            "intraday_logic": "分时逻辑描述",
            "python_code": "def analyze(daily_df, minute_df):\\n    #..."
        }
        """
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content([video_upload, system_prompt, note])
        genai.delete_file(video_upload.name)
        if os.path.exists(temp_path): os.remove(temp_path)
        status.empty()
        return response.text
    except Exception as e: status.error(f"AI Error: {e}"); return None

@st.dialog("📈 个股详情", width="large")
def view_chart_modal(code, name):
    st.subheader(f"{name} ({code})")
    ts = int(time.time())
    mid = "1" if code.startswith(('6','5','9')) else "0"
    t1, t2 = st.tabs(["分时图", "日线图"])
    with t1: st.image(f"https://webquotepic.eastmoney.com/GetPic.aspx?nid={mid}.{code}&imageType=r&t={ts}", use_container_width=True)
    with t2: st.image(f"https://webquotepic.eastmoney.com/GetPic.aspx?nid={mid}.{code}&imageType=k&t={ts}", use_container_width=True)

# --- 主程序 ---
trading_active, trading_status_msg = is_trading_time()
if 'calc_s1' not in st.session_state:
    st.session_state.calc_s1 = 0.0; st.session_state.calc_s2 = 0.0
    st.session_state.calc_r1 = 0.0; st.session_state.calc_r2 = 0.0

with st.sidebar:
    st.title("控制台")
    with st.expander("➕ 添加/编辑 个股 (手动)", expanded=True):
        code_in = st.text_input("代码", key="cin").strip()
        if st.button("⚡ 智能计算 R/S"):
            if code_in:
                end = datetime.now().strftime("%Y%m%d")
                start = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
                hist = ak.stock_zh_a_hist(symbol=code_in, period="daily", start_date=start, end_date=end, adjust="qfq")
                if not hist.empty:
                    last = hist.iloc[-1]; pivot = (last['最高']+last['最低']+last['收盘'])/3
                    st.session_state.calc_r1 = round(2*pivot - last['最低'], 2)
                    st.session_state.calc_s1 = round(2*pivot - last['最高'], 2)
                    st.session_state.calc_r2 = round(pivot + (last['最高'] - last['最低']), 2)
                    st.session_state.calc_s2 = round(pivot - (last['最高'] - last['最低']), 2)
                    st.success("计算完成")

        with st.form("add"):
            c1, c2 = st.columns(2)
            s1 = c1.number_input("S1", value=float(st.session_state.calc_s1))
            s2 = c1.number_input("S2", value=float(st.session_state.calc_s2))
            r1 = c2.number_input("R1", value=float(st.session_state.calc_r1))
            r2 = c2.number_input("R2", value=float(st.session_state.calc_r2))
            
            df_temp = load_data()
            groups = list(df_temp['group'].unique())
            if "默认" not in groups: groups.insert(0, "默认")
            grp = st.selectbox("分组", groups + ["➕ 新建..."])
            grp_val = st.text_input("新分组名") if grp == "➕ 新建..." else grp
            
            learned = get_learned_logics()
            opts = BUILTIN_STRATEGIES + (learned['strategy_name'].tolist() if not learned.empty else [])
            strat = st.selectbox("绑定战法", opts)
            
            if st.form_submit_button("💾 保存"):
                if code_in:
                    df = load_data(); name = ""
                    try:
                        info = ak.stock_zh_a_spot_em()
                        name = info[info['代码']==code_in]['名称'].values[0]
                    except: name = code_in
                    final_grp = grp_val if grp_val else "默认"
                    new_entry = {"code": code_in, "name": name, "s1": s1, "s2": s2, "r1": r1, "r2": r2, "group": final_grp, "strategy": strat, "note": ""}
                    if code_in in df.code.values:
                        for k, v in new_entry.items(): df.loc[df.code==code_in, k] = v
                    else: df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                    save_data(df); st.rerun()
    if st.button("🧹 刷新数据"): st.cache_data.clear(); st.rerun()

st.title("Alpha 游资系统 (Ultimate)")
tab1, tab2, tab3 = st.tabs(["🔭 实战看板", "🎓 AI 深度训练", "🧠 策略逻辑库"])

with tab1:
    df = load_data()
    df_logics = get_learned_logics()
    if not df.empty:
        try:
            spot = ak.stock_zh_a_spot_em(); spot = spot[['代码','名称','最新价','涨跌幅']]
            spot.columns = ['code','name','price','pct']
        except: spot = pd.DataFrame()

        all_groups = df['group'].unique()
        for group in all_groups:
            st.subheader(f"📂 {group}")
            group_df = df[df['group'] == group]
            rows = [r for _, r in group_df.iterrows()]
            for i in range(0, len(rows), 4):
                cols = st.columns(4)
                for j, row in enumerate(rows[i:i+4]):
                    code = row['code']; strat = row['strategy']
                    price = 0; pct = 0; name = row['name']
                    if not spot.empty:
                        s_row = spot[spot['code']==code]
                        if not s_row.empty:
                            price = s_row.iloc[0]['price']; pct = s_row.iloc[0]['pct']; name = s_row.iloc[0]['name']
                    
                    with cols[j]:
                        with st.container(border=True):
                            c1, c2 = st.columns([3, 1])
                            with c1: st.markdown(f"**{name}** `{code}`")
                            with c2: 
                                if st.button("🗑️", key=f"d_{code}"): delete_single_stock(code); st.rerun()
                            p_col = "price-up" if pct > 0 else "price-down"
                            st.markdown(f"<div class='big-price {p_col}'>{price} <small>{pct:+.2f}%</small></div>", unsafe_allow_html=True)

                            # 🔥 核心修正：双引擎分流
                            bundle = get_stock_data_bundle(code)
                            
                            # A. 如果是内置策略 (龙头/连板...)
                            if strat in BUILTIN_STRATEGIES:
                                builtin_text, badge_class = evaluate_builtin_strategy(strat, bundle)
                                st.markdown(f"<span class='strategy-badge {badge_class}'>{strat[:5]}</span> {builtin_text}", unsafe_allow_html=True)
                            
                            # B. 如果是 AI 学习的策略
                            elif not df_logics.empty and strat in df_logics['strategy_name'].values:
                                st.markdown(f"<span class='strategy-badge bg-ai'>AI战法</span> {strat}", unsafe_allow_html=True)
                                if bundle:
                                    logic_code = df_logics[df_logics['strategy_name']==strat].iloc[0]['python_code']
                                    res_text, res_class = execute_ai_logic(bundle, logic_code)
                                    st.markdown(f"<div class='signal-box {res_class}'><b>🤖:</b> {res_text}</div>", unsafe_allow_html=True)

                            r1, r2, s1, s2 = row['r1'], row['r2'], row['s1'], row['s2']
                            st.markdown(f"""
                            <div class='sr-block'>
                                <div class='sr-item'><span style='color:#d9534f'>R2</span> {r2}{get_dist_html(r2, price)}</div>
                                <div class='sr-item'><span style='color:#5cb85c'>S1</span> {s1}{get_dist_html(s1, price)}</div>
                                <div class='sr-item'><span style='color:#f0ad4e'>R1</span> {r1}{get_dist_html(r1, price)}</div>
                                <div class='sr-item'><span style='color:#4cae4c'>S2</span> {s2}{get_dist_html(s2, price)}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if st.button("📈 看图", key=f"b_{code}"): view_chart_modal(code, name)

with tab2:
    st.header("🎓 训练 AI：多周期共振")
    input_method = st.radio("来源", ["Link (链接)", "File (文件)"], horizontal=True)
    url_input = ""; file_input = None
    if input_method == "Link (链接)": url_input = st.text_input("🔗 视频链接")
    else: file_input = st.file_uploader("📂 上传视频")
    note = st.text_input("提示词", value="重点分析：日线趋势和分时买点的配合")
    
    if st.button("🚀 开始深度学习"):
        res = process_video_comprehensive(file_input, url_input, input_method.split(" ")[0], note)
        if res:
            try:
                data = json.loads(res.replace("```json","").replace("```","").replace("python","").strip())
                st.success(f"✅ 学会战法：{data['strategy_name']}")
                with st.expander("查看 AI 逻辑代码"): st.code(data['python_code'], language='python')
                if st.button("💾 存入库"):
                    save_learned_logic({"date": datetime.now().strftime("%Y-%m-%d"), **data})
                    st.toast("保存成功！去侧边栏应用吧")
            except: st.error("解析失败"); st.write(res)

with tab3:
    st.header("🧠 策略逻辑库")
    ldf = get_learned_logics()
    if not ldf.empty: st.dataframe(ldf[['strategy_name', 'trend_logic', 'intraday_logic']], use_container_width=True)
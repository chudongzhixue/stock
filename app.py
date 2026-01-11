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
    page_title="Alpha 游资系统 (完全体)",
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

# --- 🎨 CSS 样式 (融合了 v11 的四宫格 和 v14 的 AI标签) ---
st.markdown("""
    <style>
        html, body, p, div, span { font-family: 'Source Sans Pro', sans-serif; }
        .block-container { padding-top: 1rem !important; }
        
        /* 卡片基础 */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid #e6e6e6 !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08); 
            background-color: #ffffff; 
            padding: 15px !important;
            border-radius: 12px;
            margin-bottom: 15px;
        }
        
        /* 价格显示 */
        .big-price { font-size: 2.2rem; font-weight: 900; line-height: 1.0; letter-spacing: -1px; margin-bottom: 5px; }
        .price-up { color: #d9534f; }
        .price-down { color: #5cb85c; }
        
        /* 支撑压力位四宫格 (回归) */
        .sr-block { padding-top: 6px; border-top: 1px dashed #eee; display: grid; grid-template-columns: 1fr 1fr; gap: 4px; margin-top: 8px;}
        .sr-item { font-size: 0.8rem; font-weight: bold; color: #555; }
        
        /* AI 信号盒子 (新) */
        .signal-box { margin-top: 10px; padding: 8px; background: #f8f9fa; border-radius: 6px; border-left: 4px solid #ddd; font-size: 0.9rem;}
        .sig-buy { border-left-color: #d9534f; background: #fff5f5; }
        .sig-sell { border-left-color: #28a745; background: #f0f9f0; }
        .sig-wait { border-left-color: #17a2b8; background: #f0f8ff; }
        
        .strategy-badge { padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; background: #333; color: white; margin-right: 5px; }
    </style>
""", unsafe_allow_html=True)

DATA_FILE = 'my_stock_plan_v3.csv'
LEARNED_LOGIC_FILE = 'comprehensive_logic_v2.csv'

# --- 数据管理 (保留所有手动输入的字段) ---
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
            # 补齐缺失列
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
    # 这里的逻辑库升级了，包含日线+分时的逻辑
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

# --- 🔥 全维度数据获取 (Data Bundle) ---
@st.cache_data(ttl=60) # 1分钟缓存，保证分时图新鲜
def get_stock_data_bundle(code):
    """
    一次性获取：
    1. 日线数据 (判断趋势+量价)
    2. 分时数据 (判断盘口意图)
    """
    bundle = {"daily": None, "minute": None, "info": {}}
    try:
        # 1. 日线 (取过去60天)
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=90)).strftime("%Y%m%d")
        daily = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
        
        if not daily.empty:
            # 计算关键指标
            daily['MA5'] = daily['收盘'].rolling(5).mean()
            daily['MA10'] = daily['收盘'].rolling(10).mean()
            daily['MA20'] = daily['收盘'].rolling(20).mean()
            daily['VOL_MA5'] = daily['成交量'].rolling(5).mean()
            bundle['daily'] = daily
            
            # 基础信息
            last = daily.iloc[-1]
            bundle['info'] = {
                "name": code, # 暂存
                "price": last['收盘'],
                "pct": last['涨跌幅'],
                "pre_close": last['收盘'] / (1 + last['涨跌幅']/100)
            }

        # 2. 分时 (当天分钟级)
        minute = ak.stock_zh_a_hist_min_em(symbol=code, period='1', adjust='qfq')
        if not minute.empty:
            minute['MA_PRICE'] = (minute['close'] * minute['volume']).cumsum() / minute['volume'].cumsum() # 分时均价线
            bundle['minute'] = minute
            
        return bundle
    except:
        return None

# --- 🔥 核心：多周期逻辑执行引擎 ---
def execute_comprehensive_logic(bundle, logic_code):
    """
    执行 AI 写的代码，同时传入 日线df 和 分时df
    """
    if not bundle or bundle['daily'] is None: return "数据不足", "sig-wait"
    
    daily_df = bundle['daily']
    minute_df = bundle['minute'] # 可能为空(未开盘)
    
    try:
        local_scope = {}
        exec(logic_code, globals(), local_scope)
        
        if 'analyze' in local_scope:
            # AI 函数签名: analyze(daily_df, minute_df)
            signal, reason = local_scope['analyze'](daily_df, minute_df)
            
            if signal == "BUY": return f"🚀 {reason}", "sig-buy"
            if signal == "SELL": return f"⚠️ {reason}", "sig-sell"
            if signal == "WAIT": return f"👀 {reason}", "sig-wait"
            
        return "逻辑未触发", "sig-wait"
    except Exception as e:
        return f"运行错误: {str(e)[:20]}", "sig-wait"

# --- AI 学习模块 (Prompt: 多周期共振) ---
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
        
        # 🔥🔥🔥 终极 Prompt：多周期共振
        system_prompt = """
        你是一位顶级游资操盘手。请分析视频，总结出一套【多周期共振】的交易系统。
        一个完整的逻辑必须包含：
        1. 日线趋势 (Trend): 比如"站上5日线"、"多头排列"。
        2. 量价关系 (Volume): 比如"缩量回调"、"倍量突破"。
        3. 分时意图 (Intraday): 比如"分时承接有力"、"均价线上方运行"。

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
            # 1. 先看日线趋势 (必须满足)
            if last_day['收盘'] < last_day['MA5']:
                return "WAIT", "日线破位"
            
            # 2. 再看量能 (必须满足)
            if last_day['成交量'] > last_day['VOL_MA5'] * 2: # 异常放量
                 return "WAIT", "高位异常放量"
                 
            # 3. 最后看分时 (如果开盘了)
            if minute_df is not None and not minute_df.empty:
                last_min = minute_df.iloc[-1]
                if last_min['close'] > last_min['MA_PRICE']:
                    return "BUY", "趋势向上且分时站稳均价线"
            
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

# --- 辅助 UI 函数 ---
def get_dist_html(target, current):
    try: target=float(target); current=float(current)
    except: return ""
    if target == 0: return ""
    d = ((current - target) / target) * 100
    col = "#d9534f" if abs(d)<1.0 else "#f0ad4e" if abs(d)<3.0 else "#999"
    return f"<span style='color:{col}; font-weight:bold;'>({d:+.1f}%)</span>"

@st.dialog("📈 个股详情", width="large")
def view_chart_modal(code, name):
    st.subheader(f"{name} ({code})")
    ts = int(time.time())
    mid = "1" if code.startswith(('6','5','9')) else "0"
    t1, t2 = st.tabs(["分时图", "日线图"])
    with t1: st.image(f"https://webquotepic.eastmoney.com/GetPic.aspx?nid={mid}.{code}&imageType=r&t={ts}", use_container_width=True)
    with t2: st.image(f"https://webquotepic.eastmoney.com/GetPic.aspx?nid={mid}.{code}&imageType=k&t={ts}", use_container_width=True)

# --- 主程序 ---
trading_active, _ = is_trading_time()

# 初始化 Session State
if 'calc_s1' not in st.session_state:
    st.session_state.calc_s1 = 0.0; st.session_state.calc_s2 = 0.0
    st.session_state.calc_r1 = 0.0; st.session_state.calc_r2 = 0.0

# 侧边栏：这里保留了所有手动功能！
with st.sidebar:
    st.title("控制台")
    with st.expander("➕ 添加/编辑 个股 (手动)", expanded=True):
        code_in = st.text_input("代码", key="cin").strip()
        
        # 1. 智能计算器 (保留)
        if st.button("⚡ 智能计算 R/S"):
            if code_in:
                # 简单获取日线算枢轴点
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
            # 2. 手动输入框 (保留)
            c1, c2 = st.columns(2)
            s1 = c1.number_input("S1", value=float(st.session_state.calc_s1))
            s2 = c1.number_input("S2", value=float(st.session_state.calc_s2))
            r1 = c2.number_input("R1", value=float(st.session_state.calc_r1))
            r2 = c2.number_input("R2", value=float(st.session_state.calc_r2))
            
            # 3. 分组选择 (保留)
            df_temp = load_data()
            groups = list(df_temp['group'].unique())
            if "默认" not in groups: groups.insert(0, "默认")
            grp = st.selectbox("分组", groups + ["➕ 新建..."])
            grp_val = st.text_input("新分组名") if grp == "➕ 新建..." else grp
            
            # 4. 战法选择 (包含AI学到的)
            learned = get_learned_logics()
            opts = ["自动观察"] + (learned['strategy_name'].tolist() if not learned.empty else [])
            strat = st.selectbox("绑定战法 (AI/手动)", opts)
            
            if st.form_submit_button("💾 保存"):
                if code_in:
                    df = load_data()
                    name = ""
                    # 尝试获取名字
                    try:
                        import akshare as ak
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

# --- Tab 1: 看板 (显示四宫格 + AI 信号) ---
with tab1:
    df = load_data()
    df_logics = get_learned_logics()
    
    if not df.empty:
        # 获取所有名字用于显示
        try:
            spot = ak.stock_zh_a_spot_em(); spot = spot[['代码','名称','最新价','涨跌幅']]
            spot.columns = ['code','name','price','pct']
        except: spot = pd.DataFrame()

        # 遍历分组
        all_groups = df['group'].unique()
        for group in all_groups:
            st.subheader(f"📂 {group}")
            group_df = df[df['group'] == group]
            
            rows = [r for _, r in group_df.iterrows()]
            for i in range(0, len(rows), 4):
                cols = st.columns(4)
                for j, row in enumerate(rows[i:i+4]):
                    code = row['code']; strat = row['strategy']
                    
                    # 1. 基础行情
                    price = 0; pct = 0; name = row['name']
                    if not spot.empty:
                        s_row = spot[spot['code']==code]
                        if not s_row.empty:
                            price = s_row.iloc[0]['price']; pct = s_row.iloc[0]['pct']; name = s_row.iloc[0]['name']
                    
                    with cols[j]:
                        with st.container(border=True):
                            # 标题行
                            c1, c2 = st.columns([3, 1])
                            with c1: st.markdown(f"**{name}** `{code}`")
                            with c2: 
                                if st.button("🗑️", key=f"d_{code}"): delete_single_stock(code); st.rerun()
                            
                            # 价格大字
                            p_col = "price-up" if pct > 0 else "price-down"
                            st.markdown(f"<div class='big-price {p_col}'>{price} <small>{pct:+.2f}%</small></div>", unsafe_allow_html=True)
                            
                            # 战法标签
                            st.markdown(f"<span class='strategy-badge'>{strat[:5]}</span>", unsafe_allow_html=True)

                            # 🔥 四宫格 (您的旧爱回归)
                            r1, r2, s1, s2 = row['r1'], row['r2'], row['s1'], row['s2']
                            st.markdown(f"""
                            <div class='sr-block'>
                                <div class='sr-item'><span style='color:#d9534f'>R2</span> {r2}{get_dist_html(r2, price)}</div>
                                <div class='sr-item'><span style='color:#5cb85c'>S1</span> {s1}{get_dist_html(s1, price)}</div>
                                <div class='sr-item'><span style='color:#f0ad4e'>R1</span> {r1}{get_dist_html(r1, price)}</div>
                                <div class='sr-item'><span style='color:#4cae4c'>S2</span> {s2}{get_dist_html(s2, price)}</div>
                            </div>
                            """, unsafe_allow_html=True)

                            # 🔥 AI 实时分析 (日线+分时共振)
                            if strat != "自动观察" and not df_logics.empty and strat in df_logics['strategy_name'].values:
                                # 获取数据包
                                bundle = get_stock_data_bundle(code)
                                logic_code = df_logics[df_logics['strategy_name']==strat].iloc[0]['python_code']
                                
                                # 执行代码
                                res_text, res_class = execute_comprehensive_logic(bundle, logic_code)
                                
                                st.markdown(f"""
                                <div class='signal-box {res_class}'>
                                    <b>🤖 AI 研判:</b> {res_text}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            if st.button("📈 看图", key=f"b_{code}"): view_chart_modal(code, name)

# --- Tab 2: 训练 (日线+分时) ---
with tab2:
    st.header("🎓 训练 AI：多周期共振")
    st.info("AI 将学会：在大趋势（日线）正确的前提下，如何通过分时图找买点。")
    
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
                with st.expander("查看 AI 逻辑代码"):
                    st.code(data['python_code'], language='python')
                if st.button("💾 存入库"):
                    save_learned_logic({"date": datetime.now().strftime("%Y-%m-%d"), **data})
                    st.toast("保存成功！去侧边栏应用吧")
            except: st.error("解析失败"); st.write(res)

with tab3:
    st.header("🧠 策略逻辑库")
    ldf = get_learned_logics()
    if not ldf.empty:
        st.dataframe(ldf[['strategy_name', 'trend_logic', 'intraday_logic']], use_container_width=True)
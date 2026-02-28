import os
import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import requests
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV

# ==========================================
# 1. 全局配置与请求拦截补丁
# ==========================================
st.set_page_config(page_title="A股慢牛健康指标量化模型", layout="wide")

REAL_COOKIE = "qgqp_b_id=8aef636eb69282130f7e8f79da8f6e20; st_nvi=5cZX0CEB3Ba439P2f2vn262be; nid18=0211112583013a1150f6ce06028f1406; nid18_create_time=1766415090157; gviem=-NZ2ghisJi5Bmw8khF6La759c; gviem_create_time=1766415090157; websitepoptg_api_time=1772086010018; st_si=34181324749368; st_asi=delete; fullscreengg=1; fullscreengg2=1; wsc_checkuser_ok=1; st_pvi=23050207989242; st_sp=2025-04-04%2021%3A51%3A55; st_inirUrl=https%3A%2F%2Fcn.bing.com%2F; st_sn=2; st_psi=20260226141436855-111000300841-0119960545" 
    # 2. 定义全局伪装头
def apply_request_patch(cookie_str):
    HEADERS_PATCH = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": "https://quote.eastmoney.com/center/grid_list.html",
        "Cookie": cookie_str,
        "Connection": "close"# 强制短连接，防止东财对长连接的审计
    }
    # 3. 拦截 requests 库，强制注入 Header和关闭 SSL 验证 (部分数据源可能存在证书问题)
    _old_get = requests.get
    def new_get(url, **kwargs):
        if any(domain in url for domain in ["eastmoney.com", "akshare", "sina"]):
            kwargs['headers'] = HEADERS_PATCH
            kwargs['verify'] = False # 绕过可能的 SSL 证书阻拦
        return _old_get(url, **kwargs)
    requests.get = new_get

# ==========================================
# 1. 页面配置与全局设置
# ==========================================

st.title("📈 A股慢牛健康度与宏观多因子动态分析面板 (2026 增强版)")
st.markdown("""
本工作站基于**滚动岭回归**与**蒙特卡洛模拟**，结合开源宏观/金融数据源（AkShare），为您提供可交互的A股上证指数走势预测与健康度分析。
""")

with st.expander("🔑 数据源访问授权", expanded=True):
    st.subheader("东方财富反爬校验")
    user_cookie = st.text_input("东方财富 Cookie，请输入完整的浏览器 Cookie 字符串(默认可能为错):", value=REAL_COOKIE, type="password",help="请从浏览器开发者工具F12中获取东财的完整 Cookie,1打开浏览器（Chrome/Edge），登录东方财富网（或直接打开股吧/行情页面）。2按 F12 打开开发者工具，切换到 Network (网络) 标签。3刷新页面，随便找一个请求，在 Request Headers 中找到 Cookie 这一项。4复制那一长串字符串。")
    apply_request_patch(user_cookie)
    st.write("✅ 已应用请求补丁，正在验证数据源访问...")

# ==========================================
# 2. 核心数据获取与清洗Pipeline
# ==========================================
@st.cache_data(ttl=3600)
def fetch_and_align_data(start_date, end_date):
    with st.status("🚀 正在调度7大回归因子与展示数据...", expanded=True) as status:
        try:

            # 环境清理，防止代理干扰
            os.environ['NO_PROXY'] = '*'

            # 1. 股市基准 (收盘价, 成交量, 波动率)
            st.write("🔍 正在连接：东方财富服务器 (AkShare 行情)...")
            df_index = ak.stock_zh_index_daily_em(symbol="sh000001")
            st.success("✅ 东方财富连接成功")
            df_index['date'] = pd.to_datetime(df_index['date'])
            df_index.set_index('date', inplace=True)
            df_index = df_index.loc[~df_index.index.duplicated(keep='last')]
            df_index.sort_index(inplace=True)

            df_index = df_index[['close', 'volume']].rename(columns={'close': 'Close'})
            df_index['Return'] = df_index['Close'].pct_change()
            df_index['Vol_20d'] = df_index['Return'].rolling(20).std()

            # 2. 美债 & 中债 -> 计算利差
            # 由于近期美债数据接口不稳定，增加异常处理逻辑，回退至模拟值或手动输入
            try:
                st.write("🔍 正在连接：新浪美债数据源...")
                us_bond = ak.bond_gb_us_sina(symbol="美国10年期国债")
                st.success("✅ 美债数据获取成功")
                us_bond['date'] = pd.to_datetime(us_bond['date'])
                us_bond = us_bond.set_index('date')[['close']].rename(columns={'close': 'DGS10'})
                us_bond = us_bond.loc[~us_bond.index.duplicated(keep='last')]
                us_bond.sort_index(inplace=True)

            except Exception as e:
                st.error(f"⚠️ 美债数据获取失败: {e}")
                st.warning(f"美债接口微调 ({e})，启用 4.05% 模拟值2026/02/26，或使用手动修改")
                df_index['DGS10'] = 4.05
            
            # 同样增加中债数据的异常处理，回退至模拟值或手动输入
            try:
                st.write("🔍 正在连接：新浪国债数据源...")
                cn_bond = ak.bond_gb_zh_sina(symbol="中国10年期国债")
                st.success("✅ 国债数据获取成功")
                cn_bond['date'] = pd.to_datetime(cn_bond['date'])
                cn_bond = cn_bond.set_index('date')[['close']].rename(columns={'close': 'CN10Y'})
                cn_bond = cn_bond.loc[~cn_bond.index.duplicated(keep='last')]
                cn_bond.sort_index(inplace=True)

            except Exception as e:
                st.warning(f"⚠️ 中债接口提取异常: {e}，回退至基准值 1.8%，2026/02/26")
                cn_bond = pd.DataFrame(index=df_index.index)
                cn_bond['CN10Y'] = 1.8

            # 3. 宏观 M1-M2 剪刀差
            try:
                st.write("🔍 正在连接：中国M1-M2宏观数据源...")
                m1_m2_data = ak.macro_china_money_supply()
                st.success("✅ M1-M2 数据获取成功")
                m1_m2_data['date'] = pd.to_datetime(m1_m2_data.iloc[:, 0].astype(str).str.replace('年', '-').str.replace('月份', '-01'))
                m1_m2_data.set_index('date', inplace=True)
                m1_m2_data = m1_m2_data.loc[~m1_m2_data.index.duplicated(keep='last')]
                m1_m2_data.sort_index(inplace=True)
                
                m1_m2_data['M1_M2_Spread'] = m1_m2_data.iloc[:, 5].astype(float) - m1_m2_data.iloc[:, 2].astype(float)
                m1_m2_clean = m1_m2_data[['M1_M2_Spread']]

            except Exception as e:
                st.error(f"⚠️ M1-M2 数据获取失败: {e}")
                st.warning(f"M1-M2 接口微调 ({e})，启用 -1.2% 模拟值2026/02/26，或使用手动修改")
                m1_m2_clean = -1.2 


            # 4. 估值因子 (PE百分位, PB -> 倒推 ROE)
            try:
                # 使用 funddb 接口获取中证全指估值
                st.write("🔍 正在连接：中国中证全指宏观数据源...")
                df_pe_raw = ak.stock_a_ttm_lyr()
                df_pb_raw = ak.stock_a_all_pb()
                st.success("✅ 中证全指估值数据获取成功")
                df_pe_raw['date'] = pd.to_datetime(df_pe_raw['date'])
                df_pe_raw.set_index('date', inplace=True)
                df_pe_raw = df_pe_raw.loc[~df_pe_raw.index.duplicated(keep='last')]
                df_pe_raw.sort_index(inplace=True)
                
                df_pb_raw['date'] = pd.to_datetime(df_pb_raw['date'])
                df_pb_raw.set_index('date', inplace=True)
                df_pb_raw = df_pb_raw.loc[~df_pb_raw.index.duplicated(keep='last')]
                df_pb_raw.sort_index(inplace=True)

                # 合并估值数据
                val_df = pd.concat([df_pe_raw[['middlePETTM', 'quantileInRecent10YearsMiddlePeTtm']],df_pb_raw[['middlePB']]], axis=1, join='inner')

                # 计算核心因子：
                # 1. PE_Ptile: 使用最近10年滚动市盈率中位数的分位数
                # 2. ROE: 利用 PB(中位数) / PE(中位数) 倒推全A股整体的盈利能力
                val_df['PE_Percentile'] = val_df['quantileInRecent10YearsMiddlePeTtm'].astype(float)
                val_df['ROE'] = val_df['middlePB'].astype(float) / val_df['middlePETTM'].astype(float)
                val_df = val_df[['PE_Percentile', 'ROE']]
                # 处理极端值或空值
                val_df.replace([np.inf, -np.inf], np.nan, inplace=True)

            except Exception as e:
                st.warning(f"估值数据拉取异常，使用默认安全值50及0.08填充: {e}")
                val_df = pd.DataFrame(index=df_index.index)
                val_df['PE_Percentile'] = 50.0
                val_df['ROE'] = 0.08

            # 5. 展示专用因子: CPI, PPI
            try:
                st.write("🔍 正在连接其他数据源...")
                cpi_df = ak.macro_china_cpi_monthly()
                ppi_df = ak.macro_china_ppi_yearly()
                st.success("✅ 其他数据获取成功")

                cpi_val = 0.0
                cpi_date = None
                
                if not cpi_df.empty:
                    i = -1
                    while abs(i) <= len(cpi_df):
                        cpi_val_col2 = cpi_df.iloc[i, 1]
                        cpi_val_col3 = cpi_df.iloc[i, 2]
                        if pd.notna(cpi_val_col3):
                            cpi_val = float(cpi_val_col3) 
                            cpi_date = cpi_val_col2
                            break
                        i -= 1 

                
                ppi_val = 0.0
                ppi_date = None

                if not ppi_df.empty:
                    i = -1
                    while abs(i) <= len(ppi_df):
                        ppi_val_col2 = ppi_df.iloc[i, 1]
                        ppi_val_col3 = ppi_df.iloc[i, 2]
                        if pd.notna(ppi_val_col3):
                            ppi_date = ppi_val_col2
                            ppi_val = float(ppi_val_col3)
                            break
                        i -= 1               
            except Exception as e:
                st.error(f"⚠️ 其他数据源获取失败: {e}")
                st.warning(f"其他数据源接口微调 ({e})，启用模拟值2026/02/26，或使用手动修改")
                cpi_val = 0.7
                ppi_val = -2.2
                ppi_date = cpi_date = datetime.datetime.today().strftime('%Y-%m-%d')

            # === 数据合并、填充与截取 ===
            df = df_index.join([us_bond, cn_bond, m1_m2_clean, val_df[['PE_Percentile', 'ROE']]], how='left')
            df = df.ffill().bfill()

            # 特征工程计算
            df['Sino_US_Spread'] = df['CN10Y'].astype(float) - df['DGS10'].astype(float)
            df['Target_5d'] = df['Close'].pct_change(5).shift(-5) # 未来5日收益率
            df['volume'] = np.log1p(df['volume'])

            
            mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
            final_df = df.loc[mask]
            status.update(label="✅ 模型训练已就绪", state="complete", expanded=False)

            return final_df, cpi_date, cpi_val,ppi_date,ppi_val

        except Exception as e:
            st.error(f"⚠️ 数据源同步失败: {e}")
            return pd.DataFrame(), None, 0.0, None, 0.0

# ==========================================
# 3. 侧边栏交互面板 (参数与干预)
# ==========================================
st.sidebar.header("⚙️ 动态干预与时间维度")

default_start = datetime.date.today() - datetime.timedelta(days=1000)
default_end = datetime.date.today()

start_date_obj = st.sidebar.date_input("开始日期", value=default_start)
end_date_obj = st.sidebar.date_input("结束日期", value=default_end)

train_window = st.sidebar.slider("滚动回归训练窗口 (交易日)", 60, 500, 300, 10)
forget_factor = st.sidebar.slider(r"时间遗忘因子衰减度 ($\lambda$)", 0.0, 5.0, 2.0, 0.1, help="建议 1.5-2.5,值越大，越重视近期数据。0代表所有历史平权对待。")

st.sidebar.subheader("外部宏观因子干预 (情景测试)")
use_manual = st.sidebar.checkbox("开启最新因子手动干预", value=False)
manual_us10y = st.sidebar.number_input("美债10Y (%)", value=4.05, step=0.01)
manual_cn10y = st.sidebar.number_input("中债10Y (%)", value=1.85, step=0.01)
manual_m2 = st.sidebar.number_input("M1-M2 剪刀差 (%)", value=-1.20, step=0.01, help="M1同比 - M2同比")
manual_pe_ptile = st.sidebar.number_input("全A PE百分位 (%)", value=45.0, step=0.1, max_value=100.0,min_value=0.0, help="当前全A市盈率在历史上的百分位位置，过高可能预示估值过热")

# ==========================================
# 4. 回归计算引擎 (遗忘因子岭回归)
# ==========================================
if ((end_date_obj - start_date_obj).days >= 30) and start_date_obj >= default_start and end_date_obj <= default_end:
    raw_data, cpi_date, cpi_latest, ppi_date, ppi_latest = fetch_and_align_data(start_date_obj.strftime('%Y%m%d'), end_date_obj.strftime('%Y%m%d'))
    
    if not raw_data.empty:
        df = raw_data.copy()
        
        # 1. 人工干预最新数据
        if use_manual:
            df.loc[df.index[-1], 'DGS10'] = manual_us10y
            df.loc[df.index[-1], 'CN10Y'] = manual_cn10y
            df.loc[df.index[-1], 'Sino_US_Spread'] = manual_cn10y - manual_us10y
            df.loc[df.index[-1], 'M1_M2_Spread'] = manual_m2
            df.loc[df.index[-1], 'PE_Percentile'] = manual_pe_ptile

        # 2. 定义回归七大因子 (包含衍生因子)
        # 注: CN10Y 本身已整合进 Sino_US_Spread, 这里保留七项核心
        features = ['Vol_20d', 'DGS10', 'Sino_US_Spread', 'M1_M2_Spread', 'volume', 'PE_Percentile', 'ROE']
        # features = ['Vol_20d', 'DGS10', 'M1_M2_Spread', 'volume', 'ROE']
        # 剔除无法训练的末尾5天 (这5天只有特征,没有Target)
        trainable_df = df.iloc[:-5].dropna(subset=['Target_5d'])
        predict_df = df.iloc[-5:] # 最新这5天，只用于预测

        corr_matrix = trainable_df[features].corr()
        st.write("### 因子间相关性矩阵 (检查共线性)")
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))

        X_train_full = trainable_df[features]
        y_train_full = trainable_df['Target_5d'] 

        predictions, health_scores, actuals, test_dates = [], [], [], []
        
        # 1. 选择模型
        # model = LassoCV(alphas=np.logspace(-8, -1, 30), cv=5, max_iter=20000)
        # model = ElasticNetCV(l1_ratio=0.5, alphas=np.logspace(-8, -1, 30), cv=5)
        model = RidgeCV(alphas=np.logspace(-2, 2, 20),cv=5)        
        
        # 3. 滚动训练 (指数衰减样本权重)
        progress_text = "正在执行带遗忘因子的滚动回归..."
        my_bar = st.progress(0, text=progress_text)
        total_steps = len(trainable_df) - train_window
        
        for idx, i in enumerate(range(train_window, len(trainable_df))):
            # 获取当前窗口数据
            X_win = X_train_full.iloc[i-train_window : i]
            y_win = y_train_full.iloc[i-train_window : i]
            # y_win = y_win * 100 # 放大目标变量，增强模型对微小收益率的敏感度

            # 构造遗忘权重 (Exponential Decay Weight)
            weights = np.exp(np.linspace(-forget_factor, 0, train_window))
            
            scaler = StandardScaler()
            X_win_scaled = scaler.fit_transform(X_win)
            if i == train_window:
                # 检查是否有因子的方差为 0
                print("各因子标准差:\n", X_win.std()) 
                # 检查 y 是否有足够的波动
                print("收益率标准差:", y_win.std())
            
            model.fit(X_win_scaled, y_win, sample_weight=weights)
            
            # 预测第 i 天
            X_curr = scaler.transform(X_train_full.iloc[i:i+1])
            pred_return = model.predict(X_curr)[0]
            curr_pe_ptile = X_train_full['PE_Percentile'].iloc[i]
            
            # --- 健康分重构逻辑 ---
            # 基础评分：模型预测期望收益映射
            base_score = 50 + (pred_return * 1000)
            
            # 过热惩罚：当全 A PE百分位大于70%时，每超1%扣除0.5分；大于90%每超1%扣除1.5分
            penalty = 0
            if curr_pe_ptile > 90:
                penalty = (90 - 70) * 0.5 + (curr_pe_ptile - 90) * 1.5
            elif curr_pe_ptile > 70:
                penalty = (curr_pe_ptile - 70) * 0.5
                
            score = base_score - penalty
            
            health_scores.append(np.clip(score, 0, 100))
            predictions.append(pred_return)
            actuals.append(y_win.iloc[-1]) # 对应的实际未来5日收益
            test_dates.append(trainable_df.index[i])
            
            if idx % 50 == 0:
                my_bar.progress(idx / total_steps, text=progress_text)

        print("因子预测力诊断 (IC):")
        print(trainable_df[features].corrwith(trainable_df['Target_5d']))
        # 在循环最后
        coeffs = dict(zip(features, model.coef_))
        print(f"当前系数分配: {coeffs}")
        model.alpha_        
        my_bar.empty()
        
        # 4. 预测最后 5 天与未来 5 天
        future_dates = [] #list(predict_df.index)
        future_preds = []
        future_scores = []
        future_prices = []
        
        last_actual_price = df['Close'].iloc[-6] # 最后一个有完整 5 日收益的数据点的价格

        # (A) 预测缺失的最近 5 天
        for i in range(len(predict_df)):
                        
            curr_row = predict_df.iloc[i:i+1]
            curr_date = predict_df.index[i]
            # 这5天虽然没有Target，但特征是已知的，不算真正的外推           
            # 预测 5 日收益率            
            curr_x = scaler.transform(curr_row[features])
            pred_ret_5d = model.predict(curr_x)[0]

            # 计算预测点位 (基于 drift 漂移)
            daily_drift = pred_ret_5d / 5
            pred_price = last_actual_price * (1 + daily_drift * (i + 1))

            pe_ptile = predict_df['PE_Percentile'].iloc[i]
            base_score = 50 + (pred_ret_5d * 1000)
            penalty = max(0, pe_ptile - 70) * 0.5 if pe_ptile <= 90 else (10 + (pe_ptile - 90) * 1.5)
            
            future_dates.append(curr_date)
            future_prices.append(pred_price)
            future_preds.append(pred_ret_5d)
            future_scores.append(np.clip(base_score - penalty, 0, 100))


        # (B) 外推预测未来 5 天 (假设当前宏观因子保持不变，仅时间偏移)
        last_actual_price = predict_df['Close'].iloc[-1] # 外推起点价格为最后一个已知价格
        last_known_features = predict_df.iloc[-1:]
        
        future_to_predict_days = 5
        step_count = 0

        for i in range(1, 10):# 周末跳过，实际外推7天，约等于5个交易日
            
            if step_count >= future_to_predict_days:break
            curr_date = predict_df.index[-1] + pd.Timedelta(days=i)
            if curr_date.weekday() >= 5: continue # 跳过周末
            step_count += 1
            # 使用最新的特征推演未来
            curr_date = last_known_features.index[0] + pd.Timedelta(days=i)
            # 预测 5 日收益率
            curr_x = scaler.transform(last_known_features[features])
            pred_ret_5d = model.predict(curr_x)[0] # 这里的 pred_ret_5d 是基于最后已知特征的未来5日收益预测，虽然特征不变，但时间推进会影响预测结果（如果模型对时间敏感的话）。
            
            # 计算预测点位 (基于 drift 漂移)
            daily_drift = pred_ret_5d / 5
            pred_price = last_actual_price * (1 + daily_drift * (step_count))
            
            pe_ptile = last_known_features['PE_Percentile'].iloc[0]
            base_score = 50 + (pred_ret_5d * 1000)
            penalty = max(0, pe_ptile - 70) * 0.5 if pe_ptile <= 90 else (10 + (pe_ptile - 90) * 1.5)
            
            noise = np.random.normal(0, 2)# 给未来的健康分增加一定的不确定性抖动 (模拟随机游走)
            final_score = np.clip(base_score - penalty + noise, 0, 100)

            future_dates.append(curr_date)
            future_prices.append(pred_price)
            future_preds.append(pred_ret_5d)
            future_scores.append(final_score)
        
        # ==========================================
        # 5. 可视化与展示
        # ==========================================
        res_df = pd.DataFrame({'Close': df.loc[test_dates, 'Close'], 'Health_Score': health_scores}, index=test_dates)
        future_df = pd.DataFrame({'Future_Score': future_scores, 'Expected_Ret': future_preds, 'Expected_Price': future_prices}, index=future_dates)
        
        st.header("实时监测指标大屏" + f" (截至 {future_df.index[-1].strftime('%Y-%m-%d')})")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("中证全指市盈率(PE)分位", f"{df['PE_Percentile'].iloc[-1]*100:.1f}%", delta="估值过热警报" if df['PE_Percentile'].iloc[-1] *100> 75 else "估值健康", delta_color="inverse")
        col2.metric("中全指中位ROE", f"{df['ROE'].iloc[-1]*100:.2f}%")
        col3.metric("M1_M2_剪刀差", f"{df['M1_M2_Spread'].iloc[-1]:.2f}%")
        col4.metric(f"{cpi_date}CPI同比", f"{cpi_latest}%")
        col5.metric(f"{ppi_date}PPI同比", f"{ppi_latest}%")
        col6.metric("5日未来预测价格", f"{future_df['Expected_Price'].iloc[-1]:.2f}")
        
        tab1, tab2, tab3 = st.tabs(["📊 宏观-估值综合健康追踪与预测", "🎲 布朗运动路径推演", "📥 因子库底稿下载"])
        
        with tab1:
            st.subheader("指数走势与综合健康评分 (含近/远期预测)")
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # 历史走势与评分
            fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Close'], name='上证指数', line=dict(color='red', width=2)), secondary_y=False)
            fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Health_Score'], name='历史健康评分', fill='tozeroy', fillcolor='rgba(0,176,246,0.3)', line=dict(color='rgba(0,176,246,1)')), secondary_y=True)
            
            #未来走势
            # 1. 预测点位 (红色虚线)
            fig.add_trace(go.Scatter(x=future_df.index, y=future_df['Expected_Price'], name='SSE 预测路径', line=dict(color='red', width=2, dash='dash')), secondary_y=False)
        
            # 预测区域 (最近缺失5天 + 未来30天)
            fig.add_trace(go.Scatter(x=future_df.index, y=future_df['Future_Score'], name='模型外推预测评分 (含不确定性)', line=dict(color='orange', width=2, dash='dot')), secondary_y=True)
            # 2. 预测健康分 (橙色实线)
            fig.add_trace(go.Scatter(x=future_df.index, y=future_df['Future_Score'], name='预测健康分', line=dict(color='orange', width=2)), secondary_y=True)

            # 警戒线
            fig.add_hline(y=75, line_dash="dash", line_color="orange",annotation_text="过热线 (75)", annotation_position="bottom right", secondary_y=True)
            fig.add_hline(y=50, line_dash="solid", line_color="gray", annotation_text="均衡中枢 (50)",annotation_position="bottom right", secondary_y=True)
            fig.add_hline(y=40, line_dash="dash", line_color="green", annotation_text="低估线 (40)",annotation_position="bottom right", secondary_y=True)

            fig.update_layout(height=600, hovermode="x unified", title="融合估值惩罚的遗忘因子动态回归", legend=dict(orientation="h", y=1.05), margin=dict(l=50, r=50, t=50, b=50))
            st.plotly_chart(fig, width='stretch')
            
        sim_days = 60
        with tab2:
            st.subheader(f"基于最新因子的布朗运动模拟 (未来 {sim_days} 天)")
            latest_close = df['Close'].iloc[-1]
            latest_score = res_df['Health_Score'].iloc[-1]
            
            # 计算预期漂移与波动
            recent_vol = df['Return'].tail(60).std() * np.sqrt(252)
            drift_adj = (latest_score - 50) / 100 * 0.1 
            mu = df['Return'].mean() * 252 + drift_adj
            
            dt = 1/252
            sim_paths = 2000
            paths = np.zeros((sim_days, sim_paths))
            paths[0] = latest_close
            
            for t in range(1, sim_days):
                paths[t] = paths[t-1] * np.exp((mu - 0.5 * recent_vol**2)*dt + recent_vol * np.sqrt(dt) * np.random.standard_normal(sim_paths))
            
            fig_mc = go.Figure()
            fig_mc = go.Figure()
            # 画前 100 条路径作为展示
            for i in range(min(100, sim_paths)):
                fig_mc.add_trace(go.Scatter(y=paths[:, i], mode='lines', line=dict(color='gray', width=1), opacity=0.1, showlegend=False))
            
            p_5, p_50, p_95 = np.percentile(paths, 5, axis=1), np.percentile(paths, 50, axis=1), np.percentile(paths, 95, axis=1)
            
            fig_mc.add_trace(go.Scatter(y=p_95, mode='lines', name='95% 乐观边界', line=dict(color='red', dash='dash')))
            fig_mc.add_trace(go.Scatter(y=p_50, mode='lines', name='50% 稳健中枢', line=dict(color='blue', width=3)))
            fig_mc.add_trace(go.Scatter(y=p_5, mode='lines', name='5% 悲观支撑', line=dict(color='green', dash='dash')))
            fig_mc.update_layout(height=450, title="未来指数可能运行路径与概率区间", xaxis_title="未来交易天数", yaxis_title="指数点位")
            st.plotly_chart(fig_mc, width='stretch')
            
            st.success(f"**模拟测算结论：** 未来 {sim_days} 天后，大概率（90%置信度）落在 **{p_5[-1]:.0f} 点** 到 **{p_95[-1]:.0f} 点** 之间，中枢目标位 **{p_50[-1]:.0f} 点**。")

        with tab3:
            st.write(f"**模型历史预测 IC 值:** `{pd.Series(predictions).corr(pd.Series(actuals)):.4f}` *(注：在量化多因子中，IC值绝对值 > 0.03 即被认为具有有效预测能力> 0.1: 属于极强的预测因子，非常罕见。越接近 1: 预测越准；越接近 -1: 预测越反向（也可以用）；接近 0: 模型在瞎猜。)*")
            st.dataframe(df.tail(100).sort_index(ascending=False), width='stretch')
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Factor_Database')
                res_df.to_excel(writer, sheet_name='Historical_Scores')
                future_df.to_excel(writer, sheet_name='Future_Prediction')
            
            st.download_button("📥 导出量化追踪 Excel", output.getvalue(), f"Quant_Report_{datetime.date.today()}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.warning("请选择有效的日期区间开始分析。")
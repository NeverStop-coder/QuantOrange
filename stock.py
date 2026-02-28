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

# ==========================================
# 第一部分：源头攻坚 - 全局请求拦截补丁
# ==========================================

# 1. 在这里填入你从浏览器中抓取的真实 Cookie
REAL_COOKIE = "qgqp_b_id=8aef636eb69282130f7e8f79da8f6e20; st_nvi=5cZX0CEB3Ba439P2f2vn262be; nid18=0211112583013a1150f6ce06028f1406; nid18_create_time=1766415090157; gviem=-NZ2ghisJi5Bmw8khF6La759c; gviem_create_time=1766415090157; websitepoptg_api_time=1772086010018; st_si=34181324749368; st_asi=delete; fullscreengg=1; fullscreengg2=1; wsc_checkuser_ok=1; st_pvi=23050207989242; st_sp=2025-04-04%2021%3A51%3A55; st_inirUrl=https%3A%2F%2Fcn.bing.com%2F; st_sn=2; st_psi=20260226141436855-111000300841-0119960545" 

def apply_request_patch(cookie_str):
    # 2. 定义全局伪装头
    HEADERS_PATCH = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": "https://quote.eastmoney.com/center/grid_list.html",
        "Cookie": cookie_str,
        "Connection": "close" # 强制短连接，防止东财对长连接的审计
    }
    # 3. 拦截 requests 库，强制注入 Header
    _old_get = requests.get
    def new_get(url, **kwargs):
        if "eastmoney.com" in url or "akshare" in url:
            kwargs['headers'] = HEADERS_PATCH
            kwargs['verify'] = False # 绕过可能的 SSL 证书阻拦
        return _old_get(url, **kwargs)
    
    requests.get = new_get

# ==========================================
# 0. 页面配置与全局设置
# ==========================================
st.set_page_config(page_title="A股慢牛健康指标量化模型", layout="wide")
st.title("📈 A股慢牛健康度与宏观多因子动态分析面板 (2026版)")
st.markdown("""
本工作站基于**滚动岭回归**与**蒙特卡洛模拟**，结合开源宏观/金融数据源（AkShare），为您提供可交互的A股上证指数走势预测与健康度分析。
""")

with st.expander("🔑 数据源访问授权", expanded=True):
    st.subheader("东方财富反爬校验")
    user_cookie = st.text_input(
        "请输入完整的浏览器 Cookie 字符串(默认可能为错):", 
        value=REAL_COOKIE,
        type="password", 
        help="请从浏览器开发者工具F12中获取东财的完整 Cookie,1打开浏览器（Chrome/Edge），登录东方财富网（或直接打开股吧/行情页面）。2按 F12 打开开发者工具，切换到 Network (网络) 标签。3刷新页面，随便找一个请求，在 Request Headers 中找到 Cookie 这一项。4复制那一长串字符串。"
    )
    if not user_cookie:
        st.info("💡 请先输入 Cookie 授权后，系统将自动开始获取实时行情并运行岭回归模型。")
        # st.stop()  # 停止执行后续代码，直到输入 Cookie


# 运行到这里说明已有 Cookie，应用补丁
run_button = st.button("🚀 授权并启动量化模型")
i = 0 # 控制首次点击后不重复提示输入 Cookie
if not run_button:
        st.info("💡 默认 Cookie 已填入。请检查无误后，点击上方按钮启动。")
        REAL_COOKIE = user_cookie  # 更新全局 Cookie 变量
        if i == 0:
            i += 1
            # st.stop()

apply_request_patch(user_cookie)

# ==========================================
# 1. 数据获取模块 (支持缓存防重复请求)
# ==========================================
@st.cache_data(ttl=3600)
def load_data(start_date, end_date):
    """
    【国产化平替版】彻底移除 FRED 依赖，绕开网络握手失败问题
    """
    with st.spinner("🚀 正在通过增强型链路拉取东财行情..."):
        """
        ✅ 正在应用全局伪装补丁
        """
        try:

            # 环境清理，防止代理干扰
            os.environ['NO_PROXY'] = '*'

            # --- 1. 获取上证指数 (国内源) ---
            st.write("🔍 正在连接：东方财富服务器 (AkShare 行情)...")
            df_index = ak.stock_zh_index_daily_em(symbol="sh000001")
            st.success("✅ 东方财富连接成功")
            df_index['date'] = pd.to_datetime(df_index['date'])
            df_index.set_index('date', inplace=True)

            mask = (df_index.index >= pd.Timestamp(start_date)) & (df_index.index <= pd.Timestamp(end_date))
            df = df_index.loc[mask, ['close']].copy()
            df.columns = ['Close']
            df['Return'] = df['Close'].pct_change()
            df['Vol_20d'] = df['Return'].rolling(20).std() * np.sqrt(252)

            # --- 2. 核心修改：使用 AkShare 直接获取美债 (避开 FRED) ---
            try:
                
                st.write("🔍 正在连接：美债数据源...")
                us_bond = ak.bond_gb_us_sina(symbol="美国10年期国债") 
                st.success("✅ 美债数据获取成功")
                # 解析日期
                us_bond['date'] = pd.to_datetime(us_bond['date'])
                us_bond.set_index('date', inplace=True)

                us_bond = us_bond[['close']].rename(columns={'close': 'DGS10'})

            except Exception as e:
                st.error(f"⚠️ 美债数据获取失败: {e}")
                st.warning(f"美债接口微调 ({e})，启用 4.05% 模拟值2026/02/26，或使用手动修改")
                df = df_index.copy()
                df['DGS10'] = 4.05
            
            # 3. 获取中国 10 年期国债 (动态替换固定值)
            try:
                # 使用你找到的新接口获取中债数据
                cn_bond = ak.bond_gb_zh_sina(symbol="中国10年期国债")
                cn_bond['date'] = pd.to_datetime(cn_bond['date'])
                cn_bond.set_index('date', inplace=True)

                cn_bond = cn_bond[['close']].rename(columns={'close': 'CN10Y'})
            
            except Exception as e:
                st.warning(f"⚠️ 中债接口提取异常: {e}，回退至基准值 1.8%，2026/02/26")
                cn_bond = pd.DataFrame(index=us_bond.index)
                cn_bond['CN10Y'] = 1.8


            # --- 3. 获取 M2 数据 (增强清洗) ---
            try:
                st.write("🔍 正在连接：中国M2宏观数据源...")
                m2_data = ak.macro_china_money_supply()
                st.success("✅ M2 数据获取成功")
                date_col = [c for c in m2_data.columns if '时间' in c or '月份' in c][0]
                m2_col = [c for c in m2_data.columns if 'M2' in c and '同比' in c][0]
                m2_df = m2_data[[date_col, m2_col]].copy()
                m2_df[date_col] = m2_df[date_col].astype(str).str.replace('年', '-').str.replace('月份', '')
                m2_df[date_col] = pd.to_datetime(m2_df[date_col], errors='coerce')
                # m2_df.dropna(subset=[date_col], inplace=True)
                m2_df.set_index(date_col, inplace=True)
                
                m2_df = m2_df[[m2_col]].rename(columns={m2_col: 'M2_YoY'})
                
            except Exception:
                st.error(f"⚠️ M2 数据获取失败: {e}")
                st.warning(f"M2 接口微调 ({e})，启用 9.0% 模拟值，或使用手动修改")
                df['M2_YoY'] = 9.0 

            # 4. 后处理
            # 4. 多表对齐 (核心重构点)
            # 以 A 股交易日为主表，合并美债和中债
            df = df_index.join(us_bond, how='left').join(cn_bond, how='left').join(m2_df, how='left')
            
            # 填充缺失值 (处理由于 1000 天限制或节假日导致的空缺)
            # 使用前向填充（ffill）确保利差计算连续
            

            df['Close'] = df['close'].astype(float) # A股收盘价
            df['DGS10'] = df['DGS10'].ffill().bfill().astype(float) # 美债收益率
            df['CN10Y'] = df['CN10Y'].ffill().bfill().astype(float) # 中债收益率
            df['M2_YoY'] = df['M2_YoY'].bfill().fillna(9.0) # 先向前填充，再用默认值填充剩余缺失

            # 5. 计算真实的中美利差
            # 逻辑：中国国债收益率 - 美国国债收益率
            df['Sino_US_Spread'] = df['CN10Y'] - df['DGS10']

            df['Return'] = df['Close'].pct_change()


            # 年化波动率
            df['Vol_20d'] = df['Return'].rolling(20).std() * np.sqrt(252)
            # 预测目标：未来 5 天累积收益
            df['Target_5d'] = df['Return'].shift(-5).rolling(5).sum()

            # 筛选用户选择的时间区间
            # 7. 筛选用户选择的时间区间
            mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
            final_df = df.loc[mask].dropna(subset=['Close', 'Vol_20d','Sino_US_Spread','DGS10','CN10Y','M2_YoY','Target_5d'])
            
            if len(final_df) < 10:
                st.error("数据区间过短或数据源返回不足，请调整日期。")
                
            return final_df
            
        except Exception as e:
            st.error(f"服务器或数据连接错误: {e}")
            return pd.DataFrame()

# ==========================================
# 2. 侧边栏：参数配置与干预面板
# ==========================================
st.sidebar.header("⚙️ 模型参数与人工干预配置")

# 时间窗口配置
# 默认显示过去 3 年的数据
default_start = datetime.datetime.now() - datetime.timedelta(days=1000)
default_end = datetime.datetime.now()

# 获取用户输入的日期对象
start_date_obj = st.sidebar.date_input("开始日期", value=default_start)
end_date_obj = st.sidebar.date_input("结束日期", value=default_end)
if start_date_obj > end_date_obj or end_date_obj > datetime.datetime.now().date():
    st.sidebar.error("⚠️ 开始日期不能晚于结束日期！")
    start_date_obj = default_start
    end_date_obj = default_end

if (end_date_obj - start_date_obj).days < 30 or (end_date_obj - start_date_obj).days > 1000:
    st.sidebar.warning("⚠️ 建议至少选择30天以上及1000天以下的时间范围以获得稳定的模型训练效果。")
    start_date_obj = default_start
    end_date_obj = default_end
# --- 2. 关键：将日期对象转换为 AkShare 兼容的字符串格式 ---
# 这样即便用户在控件里选了，我们也能拿到最新的字符串变量
start_str = start_date_obj.strftime('%Y%m%d')
end_str = end_date_obj.strftime('%Y%m%d')

# --- 3. 动态文本反馈 (解决你提到的“文本没变化”问题) ---
st.sidebar.write(f"📅 当前选择范围：")
st.sidebar.info(f"{start_str} 至 {end_str}")
date_range = st.sidebar.date_input("分析数据区间", [start_str, end_str])

# 模型参数说明与设置
st.sidebar.subheader("量化模型参数")
train_window = st.sidebar.slider("滚动回归训练窗口 (交易日)", min_value=60, max_value=500, value=250, step=10, 
                                 help="窗口越小对近期数据越敏感，但容易过拟合；窗口越大越平滑。")
cv_alphas = st.sidebar.multiselect("RidgeCV 惩罚系数 (Alpha)", [0.1, 1.0, 10.0, 100.0, 500.0], default=[0.1, 1.0, 10.0],
                                   help="L2正则化强度，模型会自动在选定值中寻找最优解以防止过拟合。")

# 宏观参数手动干预 (Scenario Analysis)
st.sidebar.subheader("外部宏观因子干预 (情景测试)")
st.sidebar.markdown("开启后，最新一日的因子数据将替换为您的设定值，观察健康得分变化。")
use_manual_macro = st.sidebar.checkbox("开启宏观参数手动干预", value=False)

manual_us10y = st.sidebar.number_input("美联储10年期收益率预期 (%)", value=4.05, step=0.01, 
                                       help="当前 10 年期美债收益率基准,影响外资流向及成长股估值定价分母。")
manual_m2 = st.sidebar.number_input("国内M2同比增速预期 (%)", value=9.00, step=0.01,
                                    help="代表国内广义流动性支持力度。")
manual_cn10y = st.sidebar.number_input("国内10年期国债收益率预期 (%)", value=1.80, step=0.01,
                                    help="当前 10 年期中债收益率基准，影响贷款利率、企业融资成本与资产定价。")

# 蒙特卡洛参数
st.sidebar.subheader("蒙特卡洛区间模拟参数")
sim_days = st.sidebar.slider("预测未来交易天数", min_value=20, max_value=252, value=60)
sim_paths = st.sidebar.selectbox("模拟路径数量", [1000, 5000, 10000], index=1)

# ==========================================
# 3. 核心计算模块
# ==========================================
if len(date_range) == 2:
    raw_data = load_data(date_range[0], date_range[1])
    
    if not raw_data.empty:
        
        st.success(f"✅ 已成功加载从 {date_range[0]} 到 {date_range[1]} 的 {len(raw_data)} 条交易记录")
        
        df = raw_data.copy()
        
        # 应用人工干预参数
        if use_manual_macro:
            df.loc[df.index[-1], 'DGS10'] = manual_us10y
            df.loc[df.index[-1], 'CN10Y'] = manual_cn10y
            df.loc[df.index[-1], 'Sino_US_Spread'] = manual_cn10y - manual_us10y
            df.loc[df.index[-1], 'M2_YoY'] = manual_m2

        # 准备因子
        # 1. 更新准备因子：加入 CN10Y (10年中债)
        # 因子含义：波动率(情绪)、美债(全球锚)、利差(流向)、中债(内因)、M2(总量)
        features = ['Vol_20d', 'DGS10', 'CN10Y', 'Sino_US_Spread', 'M2_YoY']

        X = df[features]
        y = df['Target_5d']
        
        predictions, health_scores, actuals, test_dates = [], [], [], []
        scaler = StandardScaler()
        model = RidgeCV(alphas=cv_alphas, cv=None)
        
        # 时序滚动交叉验证计算
        progress_text = "正在执行时序滚动交叉验证..."
        my_bar = st.progress(0, text=progress_text)
        
        total_steps = len(df) - train_window - 5
        for idx, i in enumerate(range(train_window, len(df) - 5)):
            X_train = X.iloc[i-train_window : i]
            y_train = y.iloc[i-train_window : i]
            
            X_train_scaled = scaler.fit_transform(X_train)
            model.fit(X_train_scaled, y_train)
            
            X_current = scaler.transform(X.iloc[i:i+1])
            pred_return = model.predict(X_current)[0]
            
            # 健康分映射 (50分基准)
            score = 50 + (pred_return * 1000)
            health_scores.append(np.clip(score, 0, 100))
            predictions.append(pred_return)
            actuals.append(y.iloc[i])
            test_dates.append(df.index[i])
            
            if idx % 50 == 0:
                my_bar.progress(idx / total_steps, text=progress_text)
                
        my_bar.empty()
        
        # 结果打包
        res_df = pd.DataFrame({
            'Close': df.loc[test_dates, 'Close'],
            'Predicted_Ret': predictions,
            'Actual_Ret': actuals,
            'Health_Score': health_scores,
            'Surplus': np.array(health_scores) - 50  # 计算健康分盈余
        }, index=test_dates)

        # ==========================================
        # 4. 可视化与 UI 展现 (Plotly 交互图表)
        # ==========================================
        tab1, tab2, tab3 = st.tabs(["📊 动态监控大屏", "🎲 蒙特卡洛预测", "📥 数据验证与下载"])
        
        with tab1:
            st.subheader("指数走势 vs 慢牛健康盈余 (Health Score Surplus)")
            st.info("**图表说明：** 下方的蓝色/红色面积图代表**【健康分盈余】**。当盈余>0时，说明宏观因子支撑股市上涨；若指数上涨但盈余缩小，提示背离风险；盈余<0提示基本面恶化。")
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # K线/收盘价走势
            fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Close'], name='上证指数', line=dict(color='red', width=2)), secondary_y=False)
            
            # --- 副轴：美国 10 年期国债 (DGS10) ---
            fig.add_trace(go.Scatter(x=res_df.index, y=raw_data['DGS10']*10, name="美债 10Y (%) 10倍", line=dict(color='#FF4B4B', width=1.5, dash='dot'), ),secondary_y=True)# 红色点划线

            # --- 副轴：中国 10 年期国债 (CN10Y) ---
            fig.add_trace(go.Scatter(x=res_df.index, y=raw_data['CN10Y']*10, name="中债 10Y (%) 10倍", line=dict(color='#00CC96', width=1.5), ),secondary_y=True)# 绿色实线
            
            # 健康分盈余 (以50分为中轴绘制区域图)
            fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Health_Score'], name='健康评分', 
                                     fill='tozeroy', fillcolor='rgba(0,176,246,0.3)', line=dict(color='rgba(0,176,246,1)')), secondary_y=True)
            
            # 过热和过冷警戒线
            fig.add_hline(y=75, line_dash="dash", line_color="orange",annotation_text="过热线 (75)", annotation_position="top right", secondary_y=True)
            fig.add_hline(y=50, line_dash="solid", line_color="gray", annotation_text="均衡中枢 (50)",annotation_position="bottom right", secondary_y=True)
            fig.add_hline(y=40, line_dash="dash", line_color="green", annotation_text="低估线 (40)",annotation_position="bottom right", secondary_y=True)

            fig.update_layout(height=600, margin=dict(l=50, r=50, t=50, b=50), hovermode="x unified",legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig.update_yaxes(title_text="上证指数", secondary_y=False)
            fig.update_yaxes(title_text="慢牛健康得分 (0-100)/利率x10", range=[0, 100], secondary_y=True)
            
            st.plotly_chart(fig, width='stretch')
            
            # 指标卡片显示最新状态
            latest_score = res_df['Health_Score'].iloc[-1]
            latest_close = res_df['Close'].iloc[-1]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("当前上证指数", f"{latest_close:.2f}")
            col2.metric("最新健康评分", f"{latest_score:.1f}", delta=f"{res_df['Health_Score'].iloc[-1] - res_df['Health_Score'].iloc[-2]:.1f}")
            col3.metric("当前美债10年期收益率", f"{df['DGS10'].iloc[-1]:.2f}%")
            col4.metric("最新模型测算状态", "过热预警" if latest_score > 75 else ("底部区域" if latest_score < 40 else "健康慢牛区间"))

        with tab2:
            st.subheader(f"蒙特卡洛概率推演 (未来 {sim_days} 个交易日)")
            st.markdown(r"基于最新健康评分动态调整指数的预期漂移率 $\mu$ 和历史波动率 $\sigma$，进行布朗运动路径模拟。")
            
            #if st.button("🚀 运行蒙特卡洛模拟"):
            recent_vol = df['Return'].tail(60).std() * np.sqrt(252)
            # 根据健康分动态微调预期收益率漂移 (健康分越高，预期年化略高)
            base_cagr = df['Return'].mean() * 252
            drift_adj = (latest_score - 50) / 100 * 0.1 
            mu = base_cagr + drift_adj
            
            dt = 1/252
            paths = np.zeros((sim_days, sim_paths))
            paths[0] = latest_close
            
            for t in range(1, sim_days):
                rand = np.random.standard_normal(sim_paths)
                paths[t] = paths[t-1] * np.exp((mu - 0.5 * recent_vol**2)*dt + recent_vol * np.sqrt(dt) * rand)
            
            # 绘制模拟路径图
            fig_mc = go.Figure()
            # 画前 100 条路径作为展示
            for i in range(min(100, sim_paths)):
                fig_mc.add_trace(go.Scatter(y=paths[:, i], mode='lines', line=dict(color='gray', width=1), opacity=0.1, showlegend=False))
            
            # 添加均值和概率分位数线
            p_5 = np.percentile(paths, 5, axis=1)
            p_50 = np.percentile(paths, 50, axis=1)
            p_95 = np.percentile(paths, 95, axis=1)
            
            fig_mc.add_trace(go.Scatter(y=p_95, mode='lines', name='95% 乐观边界', line=dict(color='red', dash='dash')))
            fig_mc.add_trace(go.Scatter(y=p_50, mode='lines', name='50% 稳健中枢', line=dict(color='blue', width=3)))
            fig_mc.add_trace(go.Scatter(y=p_5, mode='lines', name='5% 悲观支撑', line=dict(color='green', dash='dash')))
            
            fig_mc.update_layout(height=450, title="未来指数可能运行路径与概率区间", xaxis_title="未来交易天数", yaxis_title="指数点位")
            st.plotly_chart(fig_mc, width='stretch')
            
            st.success(f"**模拟测算结论：** 未来 {sim_days} 天后，大概率（90%置信度）落在 **{p_5[-1]:.0f} 点** 到 **{p_95[-1]:.0f} 点** 之间，中枢目标位 **{p_50[-1]:.0f} 点**。")

        with tab3:
            st.subheader("交叉验证结果与数据导出")
            ic_value = res_df['Predicted_Ret'].corr(res_df['Actual_Ret'])
            st.write(f"**模型预测与真实值 IC 校验系数:** `{ic_value:.4f}` *(注：在量化多因子中，IC值绝对值 > 0.03 即被认为具有有效预测能力)*")
            
            st.dataframe(res_df.tail(100).sort_index(ascending=False), width='stretch')
            
            # Excel 导出功能
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                res_df.to_excel(writer, sheet_name='Model_Results')
                df.tail(500).to_excel(writer, sheet_name='Raw_Factors')
            excel_data = output.getvalue()
            
            st.download_button(
                label="📥 一键下载包含公式与结果的 Excel 报表",
                data=excel_data,
                file_name=f"SlowBull_Quant_Report_{datetime.date.today()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
else:
    st.warning("请选择有效的日期区间开始分析。")
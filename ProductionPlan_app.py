import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ページ設定
st.set_page_config(page_title="生産計画最適化システム", layout="wide")

def solve_optimization(df, alpha_C, alpha_S, alpha_I, wS_raw, wI_raw):
    """最適化問題を解く"""
    # データ整形
    df["積出日付"] = pd.to_datetime(df["積出日付"])
    df["銘柄"] = df["銘柄"].str.strip()
    
    P_MAP = {
        "セメント": "cement",
        "砕石2005": "agg2005",
        "塊鉱6080": "lump6080",
        "塊鉱4060": "lump4060",
        "塊鉱2040": "lump2040",
        "塊鉱1030": "lump1030",
        "粉鉱": "fines",
    }
    df = df[df["銘柄"].isin(P_MAP.keys())].copy()
    df["p"] = df["銘柄"].map(P_MAP)
    
    t0 = df["積出日付"].min().normalize()
    t1 = df["積出日付"].max()
    T = pd.date_range(t0, t1, freq="h")
    T_index = {t:i for i,t in enumerate(T)}
    delta = 1
    
    # 累積需要 D[p,t]
    dem = df.groupby(["p","積出日付"], as_index=False)["数量"].sum()
    P = list(P_MAP.values())
    D = {(p,t):0.0 for p in P for t in T}
    for p in P:
        s = pd.Series(0.0, index=T)
        if p in dem["p"].unique():
            tmp = dem[dem["p"]==p].set_index("積出日付")["数量"].reindex(T, fill_value=0.0)
            s = tmp.cumsum()
        for t in T:
            D[(p,t)] = float(s.loc[t])
    
    # 定数・能力
    B = 1600.0
    cap_DG = 770.0
    cap_EF = 727.0
    
    K_Y = {"C_m":9000.0, "S_m":7000.0, "I_m":11000.0}
    
    Q_S = 260.0
    Q_I = 520.0
    
    K_P = {
        "cement":20000.0, "lump6080":21500.0, "lump4060":15000.0,
        "lump2040":26500.0, "lump1030":15000.0, "agg2005":19200.0, "fines":30000.0
    }
    
    # 正規化
    normS = sum(wS_raw.values()); wS = {k:v/normS for k,v in wS_raw.items()}
    normI = sum(wI_raw.values()); wI = {k:v/normI for k,v in wI_raw.items()}
    
    # 稼働フラグ
    def work_flag(ts: pd.Timestamp) -> int:
        if ts.weekday() == 4:  # Friday
            return 0
        if ts.hour in (8,9,10):
            return 0
        return 1
    
    workY = {t:int(work_flag(t)) for t in T}
    workP = {t:int(work_flag(t)) for t in T}
    workB = {t:int(work_flag(t)) for t in T}
    
    # モデル
    m = pulp.LpProblem("Mine_to_Port_With_Shipping", pulp.LpMinimize)
    
    # 変数
    x_DG = pulp.LpVariable.dicts("x_DG", T, lowBound=0)
    x_EF = pulp.LpVariable.dicts("x_EF", T, lowBound=0)
    
    S_Y = {r: pulp.LpVariable.dicts(f"S_Y_{r}", T, lowBound=0) for r in ["C_m","S_m","I_m"]}
    y = {r: pulp.LpVariable.dicts(f"y_{r}", T, lowBound=0) for r in ["C_m","S_m","I_m"]}
    z = {r: pulp.LpVariable.dicts(f"z_{r}", T, lowBound=0, upBound=1, cat=pulp.LpBinary)
         for r in ["C_m","S_m","I_m"]}
    
    RPS = pulp.LpVariable.dicts("RPS_Sm_buf", T, lowBound=0)
    RPI = pulp.LpVariable.dicts("RPI_Im_buf", T, lowBound=0)
    uS  = pulp.LpVariable.dicts("uS_proc", T, lowBound=0)
    uI  = pulp.LpVariable.dicts("uI_proc", T, lowBound=0)
    
    S_P = {p: pulp.LpVariable.dicts(f"S_P_{p}", T, lowBound=0) for p in P}
    SHIP = {p: pulp.LpVariable.dicts(f"ship_{p}", T, lowBound=0) for p in P}
    
    Xi = {(p,t): pulp.LpVariable(f"xi_{p}_{T_index[t]}", lowBound=0) for p in P for t in T}
    
    # 採掘能力
    for t in T:
        m += x_DG[t] <= cap_DG * workY[t]
        m += x_EF[t] <= cap_EF * workY[t]
    
    # 山元在庫
    for i,t in enumerate(T):
        prodC = alpha_C * x_EF[t]
        prodS = alpha_S * x_EF[t]
        prodI = alpha_I * x_EF[t]
        if i == 0:
            SYC_prev = K_Y["C_m"]/2.0
            SYS_prev = K_Y["S_m"]/2.0
            SYI_prev = K_Y["I_m"]/2.0
        else:
            tprev = T[i-1]
            SYC_prev = S_Y["C_m"][tprev]
            SYS_prev = S_Y["S_m"][tprev]
            SYI_prev = S_Y["I_m"][tprev]
        m += S_Y["C_m"][t] == SYC_prev + x_DG[t] + prodC - y["C_m"][t]
        m += S_Y["S_m"][t] == SYS_prev + prodS - y["S_m"][t]
        m += S_Y["I_m"][t] == SYI_prev + prodI - y["I_m"][t]
        m += S_Y["C_m"][t] <= K_Y["C_m"]
        m += S_Y["S_m"][t] <= K_Y["S_m"]
        m += S_Y["I_m"][t] <= K_Y["I_m"]
    
    # ベルト能力 + 同時停止
    for t in T:
        m += y["C_m"][t] + y["S_m"][t] + y["I_m"][t] <= B * workB[t]
        m += y["C_m"][t] <= B * z["C_m"][t]
        m += y["S_m"][t] <= B * z["S_m"][t]
        m += y["I_m"][t] <= B * z["I_m"][t]
        m += z["C_m"][t] + z["S_m"][t] + z["I_m"][t] <= workB[t]
    
    # 港頭原料バッファと処理能力
    for i,t in enumerate(T):
        t_arr = T[i-delta] if i-delta >= 0 else None
        arrS = y["S_m"][t_arr] if t_arr is not None else 0
        arrI = y["I_m"][t_arr] if t_arr is not None else 0
        if i == 0:
            RPS_prev = 0; RPI_prev = 0
        else:
            tprev = T[i-1]
            RPS_prev = RPS[tprev]
            RPI_prev = RPI[tprev]
        m += RPS[t] == RPS_prev + arrS - uS[t]
        m += RPI[t] == RPI_prev + arrI - uI[t]
        m += uS[t] <= Q_S * workP[t]
        m += uI[t] <= Q_I * workP[t]
        m += uS[t] <= (RPS_prev + arrS)
        m += uI[t] <= (RPI_prev + arrI)
    
    # 港頭 製品在庫 + 出荷
    for i,t in enumerate(T):
        t_arr = T[i-delta] if i-delta >= 0 else None
        cement_from_Cm = y["C_m"][t_arr] if t_arr is not None else 0
        make_S = {p: wS.get(p,0.0) * uS[t] for p in P}
        make_I = {p: wI.get(p,0.0) * uI[t] for p in P}
        for p in P:
            if i == 0:
                SP_prev = K_P[p]/2
            else:
                tprev = T[i-1]
                SP_prev = S_P[p][tprev]
            add_cement_arr = cement_from_Cm if p=="cement" else 0
            m += S_P[p][t] == SP_prev + add_cement_arr + make_S[p] + make_I[p] - SHIP[p][t]
            m += S_P[p][t] <= K_P[p]
    
    # 需要(累積出荷で満たす)
    for p in P:
        for i,t in enumerate(T):
            m += pulp.lpSum(SHIP[p][tt] for tt in T[:i+1]) + Xi[(p,t)] >= D[(p,t)]
            m += pulp.lpSum(SHIP[p][tt] for tt in T[:i+1]) <= D[(p,t)]
    
    # 目的:不足総量の最小化
    m += pulp.lpSum(Xi[(p,t)] for p in P for t in T)
    
    # 求解
    m.solve(pulp.PULP_CBC_CMD(msg=0))
    
    status = pulp.LpStatus[m.status]
    shortfall = sum((Xi[(p,t)].value() or 0.0) for p in P for t in T)
    
    # 結果を時系列データフレームに変換
    rows = []
    for t in T:
        row = {
            "time": t,
            "x_DG": x_DG[t].value() or 0.0,
            "x_EF": x_EF[t].value() or 0.0,
            "y_C_m": y["C_m"][t].value() or 0.0,
            "y_S_m": y["S_m"][t].value() or 0.0,
            "y_I_m": y["I_m"][t].value() or 0.0,
            "S_Y_C_m": S_Y["C_m"][t].value() or 0.0,
            "S_Y_S_m": S_Y["S_m"][t].value() or 0.0,
            "S_Y_I_m": S_Y["I_m"][t].value() or 0.0,
            "S_P_cement": S_P["cement"][t].value() or 0.0,
        }
        rows.append(row)
    
    result_df = pd.DataFrame(rows).sort_values("time")
    
    return result_df, status, shortfall


def plot_results(result_df):
    """結果をプロットする"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("坑道稼働状態", "ベルト稼働状態", "山元在庫 (セメント, 砕石, 鉄鋼)"),
        vertical_spacing=0.20,
        row_heights=[0.33, 0.33, 0.34]
    )
    
    # 稼働状態の計算
    result_df['DG_working'] = (result_df['x_DG'] > 0).astype(int)
    result_df['EF_working'] = (result_df['x_EF'] > 0).astype(int)
    result_df['C_working'] = (result_df['y_C_m'] > 0).astype(int)
    result_df['S_working'] = (result_df['y_S_m'] > 0).astype(int)
    result_df['I_working'] = (result_df['y_I_m'] > 0).astype(int)
    
    # 1日分の範囲を計算(最初の24時間)
    start_time = result_df["time"].min()
    end_time = start_time + pd.Timedelta(hours=24)
    
    # 1. 坑道稼働状態ガントチャート
    fig.add_trace(
        go.Bar(
            x=result_df["time"], 
            y=result_df['DG_working'],
            name="DG坑稼働",
            marker_color='blue',
            base=1,
            width=3600000,
            showlegend=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=result_df["time"], 
            y=result_df['EF_working'],
            name="EF坑稼働",
            marker_color='green',
            base=0,
            width=3600000,
            showlegend=True
        ),
        row=1, col=1
    )
    
    # 2. ベルト搬送稼働状態ガントチャート
    fig.add_trace(
        go.Bar(
            x=result_df["time"], 
            y=result_df['C_working'],
            name="セメントベルト稼働",
            marker_color='red',
            base=2,
            width=3600000,
            showlegend=True
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=result_df["time"], 
            y=result_df['S_working'],
            name="砕石ベルト稼働",
            marker_color='orange',
            base=1,
            width=3600000,
            showlegend=True
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=result_df["time"], 
            y=result_df['I_working'],
            name="鉄鋼ベルト稼働",
            marker_color='purple',
            base=0,
            width=3600000,
            showlegend=True
        ),
        row=2, col=1
    )
    
    # 3. 山元在庫
    fig.add_trace(
        go.Scatter(x=result_df["time"], y=result_df["S_Y_C_m"], name="セメント在庫", line=dict(color="darkred", width=2)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=result_df["time"], y=result_df["S_Y_S_m"], name="砕石在庫", line=dict(color="darkorange", width=2)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=result_df["time"], y=result_df["S_Y_I_m"], name="鉄鋼在庫", line=dict(color="darkviolet", width=2)),
        row=3, col=1
    )
    
    # 坑道稼働状態グラフの設定(1日分表示、スクロールバー付き)
    fig.update_xaxes(
        row=1, col=1,
        range=[start_time, end_time],
        rangeslider=dict(visible=True, thickness=0.08),
        type='date'
    )
    fig.update_yaxes(
        title_text="坑道", 
        row=1, col=1, 
        tickmode='array', 
        tickvals=[0.5, 1.5],
        ticktext=['EF坑', 'DG坑']
    )
    
    # ベルト稼働状態グラフの設定(1日分表示、スクロールバー付き)
    fig.update_xaxes(
        row=2, col=1,
        range=[start_time, end_time],
        rangeslider=dict(visible=True, thickness=0.08),
        type='date'
    )
    fig.update_yaxes(
        title_text="ベルト", 
        row=2, col=1,
        tickmode='array',
        tickvals=[0.5, 1.5, 2.5],
        ticktext=['鉄鋼', '砕石', 'セメント']
    )
    
    # 山元在庫グラフの設定(1日分表示、スクロールバー付き)
    fig.update_xaxes(
        title_text="時刻",
        row=3, col=1,
        range=[start_time, end_time],
        rangeslider=dict(visible=True, thickness=0.08),
        type='date'
    )
    fig.update_yaxes(title_text="在庫量 (t)", row=3, col=1)
    
    fig.update_layout(
        height=1400, 
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig


def create_cement_payment_table(result_df, demand_df):
    """受払い表_セメントを作成する"""
    # demand_dfのセメントデータのみ抽出
    cement_demand = demand_df[demand_df["銘柄"].str.strip() == "セメント"].copy()
    cement_demand["積出日付"] = pd.to_datetime(cement_demand["積出日付"])
    cement_demand["date"] = cement_demand["積出日付"].dt.date
    
    # result_dfから日付ごとのデータを抽出
    result_df_copy = result_df.copy()
    result_df_copy["date"] = pd.to_datetime(result_df_copy["time"]).dt.date
    
    # 日付のリストを取得
    dates = sorted(result_df_copy["date"].unique())
    
    payment_rows = []
    
    for date in dates:
        # 朝6:00の在庫を取得
        morning_6am = pd.Timestamp(date) + pd.Timedelta(hours=6)
        morning_stock_row = result_df_copy[result_df_copy["time"] == morning_6am]
        
        if len(morning_stock_row) > 0:
            morning_stock = morning_stock_row.iloc[0]["S_P_cement"]
        else:
            # 6:00のデータがない場合は、その日の最初のデータを使用
            day_data = result_df_copy[result_df_copy["date"] == date]
            if len(day_data) > 0:
                morning_stock = day_data.iloc[0]["S_P_cement"]
            else:
                morning_stock = 0.0
        
        # LBC引取(その日のy_C_mの合計)
        lbc_total = result_df_copy[result_df_copy["date"] == date]["y_C_m"].sum()
        
        # その日のセメント出荷データを取得
        day_shipments = cement_demand[cement_demand["date"] == date].sort_values("積出日付")
        
        # 出荷量と開始時間を取得
        shipment_1 = day_shipments.iloc[0]["数量"] if len(day_shipments) >= 1 else 0.0
        time_1 = day_shipments.iloc[0]["積出日付"].strftime("%H:%M") if len(day_shipments) >= 1 else ""
        
        shipment_2 = day_shipments.iloc[1]["数量"] if len(day_shipments) >= 2 else 0.0
        time_2 = day_shipments.iloc[1]["積出日付"].strftime("%H:%M") if len(day_shipments) >= 2 else ""
        
        # 計算
        after_ship_1 = morning_stock - shipment_1 + lbc_total
        after_ship_2 = morning_stock - shipment_1 - shipment_2 + lbc_total
        
        payment_rows.append({
            "日付": str(date),
            "朝在庫": morning_stock,
            "LBC引取": lbc_total,
            "出荷量1": shipment_1,
            "開始時間1": time_1,
            "朝在-出荷1": after_ship_1,
            "出荷量2": shipment_2,
            "開始時間2": time_2,
            "朝在-出荷2": after_ship_2
        })
    
    payment_df = pd.DataFrame(payment_rows)
    return payment_df


# メインアプリケーション
st.title("🏔️ 生産計画最適化システム")

# レイアウト: 2列に分割
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📁 データ入力")
    uploaded_file = st.file_uploader("demand.csv をアップロード", type=["csv"])
    
    if uploaded_file is not None:
        # CSVファイル読み込み
        df = pd.read_csv(uploaded_file)
        
        st.success("✅ ファイルが読み込まれました")
        st.write("**データプレビュー:**")
        st.dataframe(df.head(10), height=300)
        
        # パラメータ調整セクション
        st.divider()
        st.subheader("⚙️ パラメータ調整")
        st.write("**EF坑からの産出比率(合計=1.0)**")
        
        # セッションステートの初期化
        if 'alpha_C' not in st.session_state:
            st.session_state.alpha_C = 0.194
            st.session_state.alpha_S = 0.144
            st.session_state.alpha_I = 0.662
        
        if 'wS_cement' not in st.session_state:
            st.session_state.wS_cement = 0.019
            st.session_state.wS_agg2005 = 0.785
            st.session_state.wS_lump2040 = 0.197
        
        if 'wI_cement' not in st.session_state:
            st.session_state.wI_cement = 0.052
            st.session_state.wI_lump6080 = 0.213
            st.session_state.wI_lump4060 = 0.104
            st.session_state.wI_lump2040 = 0.216
            st.session_state.wI_lump1030 = 0.110
            st.session_state.wI_agg2005 = 0.056
            st.session_state.wI_fines = 0.250
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            alpha_C = st.number_input(
                "α_C (セメント)",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.alpha_C,
                step=0.01,
                format="%.3f",
                key="input_alpha_C"
            )
        
        with col_b:
            alpha_S = st.number_input(
                "α_S (砕石)",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.alpha_S,
                step=0.01,
                format="%.3f",
                key="input_alpha_S"
            )
        
        with col_c:
            alpha_I = st.number_input(
                "α_I (鉄鋼)",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.alpha_I,
                step=0.01,
                format="%.3f",
                key="input_alpha_I"
            )
        
        # 合計値の表示と検証
        total_alpha = alpha_C + alpha_S + alpha_I
        
        if abs(total_alpha - 1.0) > 0.001:
            st.warning(f"⚠️ 合計値: {total_alpha:.3f} (合計は1.0である必要があります)")
        else:
            st.success(f"✅ 合計値: {total_alpha:.3f}")
            st.session_state.alpha_C = alpha_C
            st.session_state.alpha_S = alpha_S
            st.session_state.alpha_I = alpha_I
        
        st.divider()
        
        # wS_raw パラメータ調整
        st.write("**砕石処理 (S) の製品比率**")
        
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            wS_cement = st.number_input(
                "セメント",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.wS_cement,
                step=0.001,
                format="%.3f",
                key="input_wS_cement"
            )
        
        with col_s2:
            wS_agg2005 = st.number_input(
                "砕石2005",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.wS_agg2005,
                step=0.001,
                format="%.3f",
                key="input_wS_agg2005"
            )
        
        with col_s3:
            wS_lump2040 = st.number_input(
                "塊鉱2040",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.wS_lump2040,
                step=0.001,
                format="%.3f",
                key="input_wS_lump2040"
            )
        
        total_wS = wS_cement + wS_agg2005 + wS_lump2040
        
        if total_wS > 0:
            st.info(f"📊 合計: {total_wS:.3f} (正規化後に使用されます)")
            st.session_state.wS_cement = wS_cement
            st.session_state.wS_agg2005 = wS_agg2005
            st.session_state.wS_lump2040 = wS_lump2040
        else:
            st.error("❌ すべてのパラメータが0です")
        
        st.divider()
        
        # wI_raw パラメータ調整
        st.write("**鉄鋼処理 (I) の製品比率**")
        
        col_i1, col_i2, col_i3, col_i4 = st.columns(4)
        
        with col_i1:
            wI_cement = st.number_input(
                "セメント",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.wI_cement,
                step=0.001,
                format="%.3f",
                key="input_wI_cement"
            )
            wI_lump2040 = st.number_input(
                "塊鉱2040",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.wI_lump2040,
                step=0.001,
                format="%.3f",
                key="input_wI_lump2040"
            )
        
        with col_i2:
            wI_lump6080 = st.number_input(
                "塊鉱6080",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.wI_lump6080,
                step=0.001,
                format="%.3f",
                key="input_wI_lump6080"
            )
            wI_lump1030 = st.number_input(
                "塊鉱1030",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.wI_lump1030,
                step=0.001,
                format="%.3f",
                key="input_wI_lump1030"
            )
        
        with col_i3:
            wI_lump4060 = st.number_input(
                "塊鉱4060",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.wI_lump4060,
                step=0.001,
                format="%.3f",
                key="input_wI_lump4060"
            )
            wI_agg2005 = st.number_input(
                "砕石2005",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.wI_agg2005,
                step=0.001,
                format="%.3f",
                key="input_wI_agg2005"
            )
        
        with col_i4:
            wI_fines = st.number_input(
                "粉鉱",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.wI_fines,
                step=0.001,
                format="%.3f",
                key="input_wI_fines"
            )
        
        total_wI = wI_cement + wI_lump6080 + wI_lump4060 + wI_lump2040 + wI_lump1030 + wI_agg2005 + wI_fines
        
        if total_wI > 0:
            st.info(f"📊 合計: {total_wI:.3f} (正規化後に使用されます)")
            st.session_state.wI_cement = wI_cement
            st.session_state.wI_lump6080 = wI_lump6080
            st.session_state.wI_lump4060 = wI_lump4060
            st.session_state.wI_lump2040 = wI_lump2040
            st.session_state.wI_lump1030 = wI_lump1030
            st.session_state.wI_agg2005 = wI_agg2005
            st.session_state.wI_fines = wI_fines
        else:
            st.error("❌ すべてのパラメータが0です")
        
        st.divider()
        
        # 最適化実行ボタン
        if st.button("🚀 最適化を実行", type="primary"):
            if abs(total_alpha - 1.0) > 0.001:
                st.error("❌ αパラメータの合計が1.0になるように調整してください")
            elif total_wS <= 0:
                st.error("❌ 砕石処理の製品比率がすべて0です")
            elif total_wI <= 0:
                st.error("❌ 鉄鋼処理の製品比率がすべて0です")
            else:
                with st.spinner("最適化計算中..."):
                    try:
                        # パラメータを辞書形式で準備
                        wS_raw = {
                            "cement": st.session_state.wS_cement,
                            "agg2005": st.session_state.wS_agg2005,
                            "lump2040": st.session_state.wS_lump2040
                        }
                        
                        wI_raw = {
                            "cement": st.session_state.wI_cement,
                            "lump6080": st.session_state.wI_lump6080,
                            "lump4060": st.session_state.wI_lump4060,
                            "lump2040": st.session_state.wI_lump2040,
                            "lump1030": st.session_state.wI_lump1030,
                            "agg2005": st.session_state.wI_agg2005,
                            "fines": st.session_state.wI_fines
                        }
                        
                        result_df, status, shortfall = solve_optimization(
                            df, 
                            st.session_state.alpha_C, 
                            st.session_state.alpha_S, 
                            st.session_state.alpha_I,
                            wS_raw,
                            wI_raw
                        )
                        st.session_state['result_df'] = result_df
                        st.session_state['original_result_df'] = result_df.copy()
                        st.session_state['demand_df'] = df.copy()  # demand_dfを保存
                        st.session_state['status'] = status
                        st.session_state['shortfall'] = shortfall
                        st.session_state['used_alpha_C'] = st.session_state.alpha_C
                        st.session_state['used_alpha_S'] = st.session_state.alpha_S
                        st.session_state['used_alpha_I'] = st.session_state.alpha_I
                        st.session_state['used_wS_raw'] = wS_raw.copy()
                        st.session_state['used_wI_raw'] = wI_raw.copy()
                        # 編集用データフレームもリセット
                        if 'edited_result_df' in st.session_state:
                            del st.session_state['edited_result_df']
                        st.success("✅ 最適化完了!")
                    except Exception as e:
                        st.error(f"❌ エラーが発生しました: {str(e)}")

with col2:
    st.header("📊 最適化結果")
    
    if 'result_df' in st.session_state:
        # 使用したパラメータの表示
        if 'used_alpha_C' in st.session_state:
            with st.expander("📌 使用パラメータを表示", expanded=False):
                st.write("**EF坑産出比率:**")
                st.write(f"- α_C (セメント) = {st.session_state['used_alpha_C']:.3f}")
                st.write(f"- α_S (砕石) = {st.session_state['used_alpha_S']:.3f}")
                st.write(f"- α_I (鉄鋼) = {st.session_state['used_alpha_I']:.3f}")
                
                st.divider()
                
                if 'used_wS_raw' in st.session_state:
                    st.write("**砕石処理 (S) 製品比率:**")
                    for k, v in st.session_state['used_wS_raw'].items():
                        st.write(f"- {k}: {v:.3f}")
                
                st.divider()
                
                if 'used_wI_raw' in st.session_state:
                    st.write("**鉄鋼処理 (I) 製品比率:**")
                    for k, v in st.session_state['used_wI_raw'].items():
                        st.write(f"- {k}: {v:.3f}")
        
        # ステータス表示
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("求解ステータス", st.session_state['status'])
        with col_m2:
            st.metric("総不足量", f"{st.session_state['shortfall']:.2f} t")
        
        # タブで結果を切り替え
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📋 LBC引取表", "⛏️ 立杭抜出計画", "✏️ 稼働状態編集", "📋 受払い表_セメント", "📦 山元在庫", "📊 日別集計"])
        
        with tab1:
            st.subheader("ベルト稼働状態")
            # ベルト稼働状態グラフのみ表示
            result_df_copy = st.session_state['result_df'].copy()
            result_df_copy['C_working'] = (result_df_copy['y_C_m'] > 0).astype(int)
            result_df_copy['S_working'] = (result_df_copy['y_S_m'] > 0).astype(int)
            result_df_copy['I_working'] = (result_df_copy['y_I_m'] > 0).astype(int)
            
            start_time = result_df_copy["time"].min()
            end_time = start_time + pd.Timedelta(hours=24)
            
            fig_belt = go.Figure()
            
            fig_belt.add_trace(
                go.Bar(
                    x=result_df_copy["time"], 
                    y=result_df_copy['C_working'],
                    name="セメントベルト稼働",
                    marker_color='red',
                    base=2,
                    width=3600000,
                    showlegend=True
                )
            )
            
            fig_belt.add_trace(
                go.Bar(
                    x=result_df_copy["time"], 
                    y=result_df_copy['S_working'],
                    name="砕石ベルト稼働",
                    marker_color='orange',
                    base=1,
                    width=3600000,
                    showlegend=True
                )
            )
            
            fig_belt.add_trace(
                go.Bar(
                    x=result_df_copy["time"], 
                    y=result_df_copy['I_working'],
                    name="鉄鋼ベルト稼働",
                    marker_color='purple',
                    base=0,
                    width=3600000,
                    showlegend=True
                )
            )
            
            fig_belt.update_xaxes(
                title_text="時刻",
                range=[start_time, end_time],
                rangeslider=dict(visible=True, thickness=0.08),
                type='date'
            )
            fig_belt.update_yaxes(
                title_text="ベルト",
                tickmode='array',
                tickvals=[0.5, 1.5, 2.5],
                ticktext=['鉄鋼', '砕石', 'セメント']
            )
            
            fig_belt.update_layout(
                height=600,
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_belt, use_container_width=True)
        
        with tab2:
            st.subheader("坑道稼働状態")
            # 坑道稼働状態グラフのみ表示
            result_df_copy = st.session_state['result_df'].copy()
            result_df_copy['DG_working'] = (result_df_copy['x_DG'] > 0).astype(int)
            result_df_copy['EF_working'] = (result_df_copy['x_EF'] > 0).astype(int)
            
            start_time = result_df_copy["time"].min()
            end_time = start_time + pd.Timedelta(hours=24)
            
            fig_mine = go.Figure()
            
            fig_mine.add_trace(
                go.Bar(
                    x=result_df_copy["time"], 
                    y=result_df_copy['DG_working'],
                    name="DG坑稼働",
                    marker_color='blue',
                    base=1,
                    width=3600000,
                    showlegend=True
                )
            )
            
            fig_mine.add_trace(
                go.Bar(
                    x=result_df_copy["time"], 
                    y=result_df_copy['EF_working'],
                    name="EF坑稼働",
                    marker_color='green',
                    base=0,
                    width=3600000,
                    showlegend=True
                )
            )
            
            fig_mine.update_xaxes(
                title_text="時刻",
                range=[start_time, end_time],
                rangeslider=dict(visible=True, thickness=0.08),
                type='date'
            )
            fig_mine.update_yaxes(
                title_text="坑道",
                tickmode='array', 
                tickvals=[0.5, 1.5],
                ticktext=['EF坑', 'DG坑']
            )
            
            fig_mine.update_layout(
                height=600,
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_mine, use_container_width=True)
        
        with tab5:
            st.subheader("山元在庫 (セメント, 砕石, 鉄鋼)")
            # 山元在庫グラフのみ表示
            result_df_copy = st.session_state['result_df'].copy()
            
            start_time = result_df_copy["time"].min()
            end_time = start_time + pd.Timedelta(hours=24)
            
            fig_stock = go.Figure()
            
            fig_stock.add_trace(
                go.Scatter(
                    x=result_df_copy["time"], 
                    y=result_df_copy["S_Y_C_m"], 
                    name="セメント在庫", 
                    line=dict(color="darkred", width=2)
                )
            )
            fig_stock.add_trace(
                go.Scatter(
                    x=result_df_copy["time"], 
                    y=result_df_copy["S_Y_S_m"], 
                    name="砕石在庫", 
                    line=dict(color="darkorange", width=2)
                )
            )
            fig_stock.add_trace(
                go.Scatter(
                    x=result_df_copy["time"], 
                    y=result_df_copy["S_Y_I_m"], 
                    name="鉄鋼在庫", 
                    line=dict(color="darkviolet", width=2)
                )
            )
            
            fig_stock.update_xaxes(
                title_text="時刻",
                range=[start_time, end_time],
                rangeslider=dict(visible=True, thickness=0.08),
                type='date'
            )
            fig_stock.update_yaxes(title_text="在庫量 (t)")
            
            fig_stock.update_layout(
                height=600,
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_stock, use_container_width=True)
        
        with tab6:
            # 日別集計の計算
            result_df = st.session_state['result_df'].copy()
            result_df['date'] = result_df['time'].dt.date
            
            daily_summary = result_df.groupby('date').agg({
                'x_DG': 'sum',
                'x_EF': 'sum',
                'y_C_m': 'sum',
                'y_S_m': 'sum',
                'y_I_m': 'sum'
            }).reset_index()
            
            daily_summary.columns = ['日付', 'DG坑採掘量(t)', 'EF坑採掘量(t)', 
                                    'セメントベルト搬送量(t)', '砕石ベルト搬送量(t)', '鉄鋼ベルト搬送量(t)']
            
            # 合計行を追加
            daily_summary['総ベルト搬送量(t)'] = (daily_summary['セメントベルト搬送量(t)'] + 
                                                 daily_summary['砕石ベルト搬送量(t)'] + 
                                                 daily_summary['鉄鋼ベルト搬送量(t)'])
            daily_summary['総採掘量(t)'] = daily_summary['DG坑採掘量(t)'] + daily_summary['EF坑採掘量(t)']
            
            st.subheader("日別集計表")
            st.dataframe(
                daily_summary.style.format({
                    'DG坑採掘量(t)': '{:.2f}',
                    'EF坑採掘量(t)': '{:.2f}',
                    'セメントベルト搬送量(t)': '{:.2f}',
                    '砕石ベルト搬送量(t)': '{:.2f}',
                    '鉄鋼ベルト搬送量(t)': '{:.2f}',
                    '総ベルト搬送量(t)': '{:.2f}',
                    '総採掘量(t)': '{:.2f}'
                }),
                use_container_width=True,
                height=400
            )
            
            # 日別グラフ
            st.subheader("日別採掘量")
            fig_daily_mining = go.Figure()
            fig_daily_mining.add_trace(go.Bar(
                x=daily_summary['日付'], 
                y=daily_summary['DG坑採掘量(t)'],
                name='DG坑',
                marker_color='blue'
            ))
            fig_daily_mining.add_trace(go.Bar(
                x=daily_summary['日付'], 
                y=daily_summary['EF坑採掘量(t)'],
                name='EF坑',
                marker_color='green'
            ))
            fig_daily_mining.update_layout(
                barmode='stack',
                xaxis_title='日付',
                yaxis_title='採掘量 (t)',
                height=400
            )
            st.plotly_chart(fig_daily_mining, use_container_width=True)
            
            # 日別ベルト搬送量グラフ
            st.subheader("日別ベルト搬送量")
            fig_daily_belt = go.Figure()
            fig_daily_belt.add_trace(go.Bar(
                x=daily_summary['日付'], 
                y=daily_summary['セメントベルト搬送量(t)'],
                name='セメント',
                marker_color='red'
            ))
            fig_daily_belt.add_trace(go.Bar(
                x=daily_summary['日付'], 
                y=daily_summary['砕石ベルト搬送量(t)'],
                name='砕石',
                marker_color='orange'
            ))
            fig_daily_belt.add_trace(go.Bar(
                x=daily_summary['日付'], 
                y=daily_summary['鉄鋼ベルト搬送量(t)'],
                name='鉄鋼',
                marker_color='purple'
            ))
            fig_daily_belt.update_layout(
                barmode='stack',
                xaxis_title='日付',
                yaxis_title='搬送量 (t)',
                height=400
            )
            st.plotly_chart(fig_daily_belt, use_container_width=True)
            
            # CSVダウンロード
            csv_daily = daily_summary.to_csv(index=False, encoding="utf-8")
            st.download_button(
                label="📥 日別集計をCSVでダウンロード",
                data=csv_daily,
                file_name="daily_summary.csv",
                mime="text/csv"
            )
        
        with tab3:
            st.subheader("✏️ 稼働状態の編集")
            
            # 編集用データの初期化
            if 'edited_result_df' not in st.session_state:
                st.session_state['edited_result_df'] = st.session_state['result_df'].copy()
            
            edited_df = st.session_state['edited_result_df'].copy()
            
            # 日付選択
            edited_df['date'] = pd.to_datetime(edited_df['time']).dt.date
            available_dates = sorted(edited_df['date'].unique())
            
            selected_date = st.selectbox(
                "編集する日付を選択",
                options=available_dates,
                format_func=lambda x: str(x)
            )
            
            # 選択した日付のデータをフィルタ
            date_data = edited_df[edited_df['date'] == selected_date].copy()
            date_data = date_data.sort_values('time').reset_index(drop=True)
            
            st.write(f"**{selected_date} の稼働状態編集**")
            
            # 編集可能なデータフレームを作成
            display_columns = ['time', 'x_DG', 'x_EF', 'y_C_m', 'y_S_m', 'y_I_m']
            edit_data = date_data[display_columns].copy()
            edit_data.columns = ['時刻', 'DG坑採掘量(t/h)', 'EF坑採掘量(t/h)', 
                                 'セメントベルト(t/h)', '砕石ベルト(t/h)', '鉄鋼ベルト(t/h)']
            
            # データエディタで編集
            edited_table = st.data_editor(
                edit_data,
                use_container_width=True,
                height=600,
                column_config={
                    "時刻": st.column_config.DatetimeColumn(
                        "時刻",
                        format="YYYY-MM-DD HH:mm",
                        disabled=True
                    ),
                    "DG坑採掘量(t/h)": st.column_config.NumberColumn(
                        "DG坑採掘量(t/h)",
                        min_value=0,
                        max_value=1000,
                        step=1,
                        format="%.2f"
                    ),
                    "EF坑採掘量(t/h)": st.column_config.NumberColumn(
                        "EF坑採掘量(t/h)",
                        min_value=0,
                        max_value=1000,
                        step=1,
                        format="%.2f"
                    ),
                    "セメントベルト(t/h)": st.column_config.NumberColumn(
                        "セメントベルト(t/h)",
                        min_value=0,
                        max_value=2000,
                        step=1,
                        format="%.2f"
                    ),
                    "砕石ベルト(t/h)": st.column_config.NumberColumn(
                        "砕石ベルト(t/h)",
                        min_value=0,
                        max_value=2000,
                        step=1,
                        format="%.2f"
                    ),
                    "鉄鋼ベルト(t/h)": st.column_config.NumberColumn(
                        "鉄鋼ベルト(t/h)",
                        min_value=0,
                        max_value=2000,
                        step=1,
                        format="%.2f"
                    )
                },
                hide_index=True
            )
            
            # 適用ボタン
            col_apply, col_reset = st.columns(2)
            
            with col_apply:
                if st.button("✅ 変更を適用", type="primary", key="apply_edit"):
                    # 編集したデータを元のデータフレームに反映
                    for idx, row in date_data.iterrows():
                        time_val = row['time']
                        table_idx = date_data[date_data['time'] == time_val].index[0]
                        
                        edited_df.loc[edited_df['time'] == time_val, 'x_DG'] = edited_table.iloc[table_idx]['DG坑採掘量(t/h)']
                        edited_df.loc[edited_df['time'] == time_val, 'x_EF'] = edited_table.iloc[table_idx]['EF坑採掘量(t/h)']
                        edited_df.loc[edited_df['time'] == time_val, 'y_C_m'] = edited_table.iloc[table_idx]['セメントベルト(t/h)']
                        edited_df.loc[edited_df['time'] == time_val, 'y_S_m'] = edited_table.iloc[table_idx]['砕石ベルト(t/h)']
                        edited_df.loc[edited_df['time'] == time_val, 'y_I_m'] = edited_table.iloc[table_idx]['鉄鋼ベルト(t/h)']
                    
                    st.session_state['edited_result_df'] = edited_df
                    st.session_state['result_df'] = edited_df.drop(columns=['date'])
                    st.success("✅ 変更を適用しました")
                    st.rerun()
            
            with col_reset:
                if st.button("🔄 元に戻す", key="reset_edit"):
                    # 最適化結果に戻す
                    if 'original_result_df' in st.session_state:
                        st.session_state['result_df'] = st.session_state['original_result_df'].copy()
                        st.session_state['edited_result_df'] = st.session_state['original_result_df'].copy()
                        st.success("🔄 最適化結果に戻しました")
                        st.rerun()
            
            # プレビューグラフ
            st.subheader("📊 編集後のプレビュー")
            preview_df = st.session_state['edited_result_df'].copy()
            
            fig_preview = make_subplots(
                rows=2, cols=1,
                subplot_titles=("坑道採掘量", "ベルト搬送量"),
                vertical_spacing=0.15
            )
            
            # 坑道採掘量
            fig_preview.add_trace(
                go.Scatter(x=preview_df['time'], y=preview_df['x_DG'], 
                          name='DG坑', line=dict(color='blue')),
                row=1, col=1
            )
            fig_preview.add_trace(
                go.Scatter(x=preview_df['time'], y=preview_df['x_EF'], 
                          name='EF坑', line=dict(color='green')),
                row=1, col=1
            )
            
            # ベルト搬送量
            fig_preview.add_trace(
                go.Scatter(x=preview_df['time'], y=preview_df['y_C_m'], 
                          name='セメント', line=dict(color='red')),
                row=2, col=1
            )
            fig_preview.add_trace(
                go.Scatter(x=preview_df['time'], y=preview_df['y_S_m'], 
                          name='砕石', line=dict(color='orange')),
                row=2, col=1
            )
            fig_preview.add_trace(
                go.Scatter(x=preview_df['time'], y=preview_df['y_I_m'], 
                          name='鉄鋼', line=dict(color='purple')),
                row=2, col=1
            )
            
            fig_preview.update_xaxes(title_text="時刻", row=2, col=1)
            fig_preview.update_yaxes(title_text="採掘量 (t/h)", row=1, col=1)
            fig_preview.update_yaxes(title_text="搬送量 (t/h)", row=2, col=1)
            fig_preview.update_layout(height=800, showlegend=True)
            
            st.plotly_chart(fig_preview, use_container_width=True)
        
        with tab4:
            st.subheader("📋 受払い表_セメント")
            
            if 'demand_df' in st.session_state:
                # 受払い表の作成
                payment_table = create_cement_payment_table(
                    st.session_state['result_df'], 
                    st.session_state['demand_df']
                )
                
                # 表示
                st.dataframe(
                    payment_table.style.format({
                        "朝在庫": "{:.2f}",
                        "LBC引取": "{:.2f}",
                        "出荷量1": "{:.2f}",
                        "朝在-出荷1": "{:.2f}",
                        "出荷量2": "{:.2f}",
                        "朝在-出荷2": "{:.2f}"
                    }),
                    use_container_width=True,
                    height=600
                )
                
                # CSVダウンロード
                csv_payment = payment_table.to_csv(index=False, encoding="utf-8")
                st.download_button(
                    label="📥 受払い表をCSVでダウンロード",
                    data=csv_payment,
                    file_name="cement_payment_table.csv",
                    mime="text/csv"
                )
            else:
                st.warning("⚠️ demand.csvデータが見つかりません")
        
        # 結果データのダウンロード
        csv = st.session_state['result_df'].to_csv(index=False, encoding="utf-8")
        st.download_button(
            label="📥 時系列結果をCSVでダウンロード",
            data=csv,
            file_name="optimization_result.csv",
            mime="text/csv"
        )
    else:
        st.info("👈 左側でdemand.csvをアップロードし、最適化を実行してください")
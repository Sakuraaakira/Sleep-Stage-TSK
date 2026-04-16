import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import tempfile
import time
import wfdb
from config import Config
from drrh_model import DrrhTSKFC
from data_loader import load_mit_bih_data
from preprocessing import extract_paper_features
from sklearn.preprocessing import OneHotEncoder

# --- 1. 页面配置 ---
st.set_page_config(page_title="Drrh-TSK 睡眠监测可视化系统", layout="wide", page_icon="🌙")

# --- 2. 睡眠阶段医学特征库 (百科全书) ---
SLEEP_ENCYCLOPEDIA = {
    'Wake': {
        'name': '清醒期 (Wakefulness)',
        'nature': '人体处于觉醒状态，精神活跃或闭目放松。',
        'waves': '主要表现为 Alpha 波 (8-13Hz) 或 Beta 波 (>13Hz)。',
        'clinical': '波幅较低，频率较快。闭眼时 Alpha 波明显，睁眼或思考时转为 Beta 波。'
    },
    'S1': {
        'name': '浅睡 1 期 (Stage 1 / N1)',
        'nature': '入睡过渡阶段，容易被惊醒。',
        'waves': '主要表现为 Theta 波 (4-8Hz)。',
        'clinical': '频率变慢，波幅稍增，常伴有缓慢的眼球转动。'
    },
    'S2': {
        'name': '浅睡 2 期 (Stage 2 / N2)',
        'nature': '正式进入睡眠，占总睡眠时间比例最高。',
        'waves': '特征性表现：睡眠锭 (Sleep Spindles) 和 K-复合波。',
        'clinical': '睡眠锭频率在 12-14Hz 左右，是保护睡眠不被外界干扰的关键机制。'
    },
    'S3': {
        'name': '深睡 3 期 (Stage 3 / N3)',
        'nature': '深度睡眠阶段，身体开始进行修复和生长激素分泌。',
        'waves': '主要表现为 Delta 波 (0.5-4Hz)。',
        'clinical': '高波幅、低频率。慢波睡眠 (SWS) 的开始。'
    },
    'S4': {
        'name': '深睡 4 期 (Stage 4 / N4)',
        'nature': '最深层的睡眠，极难唤醒。',
        'waves': 'Delta 波占比超过 50%。',
        'clinical': '人体代谢最低点，对恢复体力和智力至关重要。'
    },
    'REM': {
        'name': '快速眼动期 (REM Sleep)',
        'nature': '梦境发生的阶段，大脑活跃度接近清醒。',
        'waves': '锯齿波 (Sawtooth Waves)，看起来与清醒期相似。',
        'clinical': '伴随眼球快速转动和全身肌肉松弛，称为“矛盾睡眠”。'
    }
}


# --- 3. 模型训练与缓存 (针对18个数据集) ---
@st.cache_resource
def get_trained_model():
    data_dir = "data"
    all_X, all_y = [], []
    records = ['slp01a', 'slp01b', 'slp02a', 'slp02b', 'slp03', 'slp04', 'slp14', 'slp16', 'slp32', 'slp37', 'slp41',
               'slp45', 'slp48', 'slp59', 'slp60', 'slp61', 'slp66', 'slp67x']

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, rec in enumerate(records):
        try:
            status_text.text(f"正在读取原始数据训练模型: {rec}...")
            epochs, labels, fs = load_mit_bih_data(data_dir, rec)
            if len(epochs) > 0:
                X_f = extract_paper_features(epochs, fs=fs, n_components=Config.N_COMPONENTS)
                all_X.append(X_f)
                all_y.append(labels)
            progress_bar.progress((i + 1) / len(records))
        except:
            pass

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    enc = OneHotEncoder(sparse_output=False)
    T = enc.fit_transform(y.reshape(-1, 1))

    model = DrrhTSKFC(Config)
    status_text.text("模型正在进行深度模糊推理学习 (Drrh-TSK-FC)...")
    model.train(X, T)
    status_text.text("✅ 模型训练完成！系统就绪。")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    return model


# --- 4. 睡眠评分逻辑 ---
def calculate_sleep_score(stages):
    counts = pd.Series(stages).value_counts(normalize=True).to_dict()
    wake_ratio = counts.get(0, 0)
    sleep_efficiency = (1 - wake_ratio) * 100
    deep_sleep_ratio = counts.get(3, 0) + counts.get(4, 0)
    rem_ratio = counts.get(5, 0)

    score = (sleep_efficiency * 0.7) + (deep_sleep_ratio * 100 * 0.2) + (rem_ratio * 100 * 0.1)
    score = np.clip(score, 15, 100)

    if wake_ratio > 0.3:
        advice = "检测到频繁觉醒。建议睡前进行冥想，减少蓝光刺激。"
    elif deep_sleep_ratio < 0.1:
        advice = "深睡眠不足。深睡是体力恢复的关键，建议保持卧室全黑。"
    else:
        advice = "睡眠质量良好，请继续保持规律的作息。"
    return int(score), advice


# --- 5. 主程序界面 ---
def main():
    st.sidebar.title("🌙 睡眠数据中心")
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    # 加载/训练模型
    model = get_trained_model()
    
    st.sidebar.divider()
    
    # --- 关键部分：在这里增加了数据来源选择 ---
    data_source = st.sidebar.radio("选择数据来源", ["使用系统示例", "上传本地文件"])
    
    target_record = None
    target_dir = "data" # 默认指向项目里的 data 文件夹
    
    if data_source == "使用系统示例":
        # 这里的名字必须和你 GitHub data 文件夹里的文件名对应
        sample_list = ['slp01a', 'slp01b', 'slp02a', 'slp67x'] 
        selected_sample = st.sidebar.selectbox("选择一个受试者记录", sample_list)
        if st.sidebar.button("分析该示例", type="primary"):
            target_record = selected_sample 
            
    else: # 上传模式
        hea_file = st.sidebar.file_uploader("上传 .hea 文件", type=['hea'])
        dat_file = st.sidebar.file_uploader("上传 .dat 文件", type=['dat'])
        if hea_file and dat_file:
            if st.sidebar.button("开始分析上传数据", type="primary"):
                hb = hea_file.name.split('.')[0]
                db = dat_file.name.split('.')[0]
                if hb != db:
                    st.error("❌ 错误：文件名不匹配！")
                else:
                    tmpdir = tempfile.mkdtemp()
                    with open(os.path.join(tmpdir, hea_file.name), "wb") as f: f.write(hea_file.getbuffer())
                    with open(os.path.join(tmpdir, dat_file.name), "wb") as f: f.write(dat_file.getbuffer())
                    target_record = hb
                    target_dir = tmpdir

    # 执行分析逻辑
    if target_record:
        with st.spinner(f'正在对 {target_record} 进行生理信号深度分析...'):
            try:
                import wfdb
                path_to_read = os.path.join(target_dir, target_record)
                record = wfdb.rdrecord(path_to_read)
                
                # 自动识别 EEG 通道
                eeg_idx = 1
                for i, name in enumerate(record.sig_name):
                    if 'EEG' in name.upper(): eeg_idx = i; break
                
                signal, fs = record.p_signal[:, eeg_idx], record.fs
                el = int(30 * fs)
                epochs = [signal[i:i + el] for i in range(0, len(signal) - el, el)]
                X_f = extract_paper_features(epochs, fs=fs, n_components=Config.N_COMPONENTS)
                y_pred = model.predict(X_f)
                score, advice = calculate_sleep_score(y_pred)

                st.session_state.analysis_results = {
                    'y_pred': y_pred, 'epochs': epochs, 'score': score, 
                    'advice': advice, 'hb': target_record,
                    'stage_map': {0:'Wake', 1:'S1', 2:'S2', 3:'S3', 4:'S4', 5:'REM'}
                }
            except Exception as e:
                st.error(f"分析出错: {e}")

    # 下面的渲染逻辑保持不变...
    if st.session_state.analysis_results:
        res = st.session_state.analysis_results
        # ... 这里接你之前 app.py 里的可视化渲染代码 ...

    # --- 渲染分析结果 ---
    if st.session_state.analysis_results:
        res = st.session_state.analysis_results
        y_p, eps, sm = res['y_pred'], res['epochs'], res['stage_map']

        st.header(f"📋 个人睡眠质量分析报告 - {res['hb']}")

        # A. 指标卡片
        c1, c2, c3 = st.columns(3)
        c1.metric("睡眠得分", f"{res['score']} 分")
        c2.metric("监测时长", f"{len(y_p) * 0.5 / 60:.2f} 小时")
        c3.success("分析状态：完成")
        st.warning(f"**专家建议：** {res['advice']}")

        # B. 睡眠结构图 (Hypnogram)
        st.divider()
        st.subheader("📊 睡眠结构图 (Hypnogram)")
        df = pd.DataFrame({'Time (min)': [i * 0.5 for i in range(len(y_p))], 'Stage': [sm[s] for s in y_p]})
        fig_hyp = go.Figure(go.Scatter(x=df['Time (min)'], y=df['Stage'], mode='lines+markers', line_shape='hv',
                                       line=dict(color='#636EFA')))
        fig_hyp.update_yaxes(categoryorder='array', categoryarray=['S4', 'S3', 'S2', 'S1', 'REM', 'Wake'])
        fig_hyp.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_hyp, use_container_width=True)

        # C. 统计与占比
        col_pie, col_table = st.columns([1, 1])
        with col_pie:
            st.subheader("🕒 阶段占比")
            fig_pie = px.pie(df, names='Stage', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_table:
            st.subheader("📑 阶段时长统计")
            stats = df['Stage'].value_counts().reset_index()
            stats.columns = ['阶段', '计数']
            stats['时长(min)'] = stats['计数'] * 0.5
            st.table(stats[['阶段', '时长(min)']])

        # D. 局部刷新：脑电波片段分析 (使用 st.fragment)
        @st.fragment()
        def eeg_analysis_section():
            st.divider()
            st.header("🔍 脑电波片段深度分析 (Visual EEG Analysis)")
            st.info("💡 提示：拖动滑块可查看不同时间点的原始 EEG 信号。")
            col_left, col_right = st.columns([2, 1])
            with col_left:
                st.subheader("1. 选择分析片段")
                s_idx = st.slider("监测片段选择", 0, len(y_p) - 1, 0, key="eeg_slider")
                curr_sig = eps[s_idx]
                curr_stage = sm[y_p[s_idx]]
                st.line_chart(pd.DataFrame(curr_sig, columns=["Voltage (μV)"]), height=300)
                st.caption(f"当前：第 {s_idx + 1} 片段 - 识别结果: {curr_stage}")
            with col_right:
                st.subheader("2. 阶段性质与特征")
                info = SLEEP_ENCYCLOPEDIA.get(curr_stage, {})
                st.success(f"**结果：{info.get('name')}**")
                st.write(f"**性质：** {info.get('nature')}")
                st.write(f"**波形：** {info.get('waves')}")
                st.info(f"**临床特征：** {info.get('clinical')}")

        eeg_analysis_section()

        # E. 典型阶段快照 (Tabs)
        st.divider()
        st.subheader("📂 典型阶段波形快照 (Subject Snapshots)")
        t1, t2, t3, t4 = st.tabs(["Wake 样本", "S2 样本", "REM 样本", "深睡样本"])

        def find_idx(val):
            idx_list = np.where(y_p == val)[0]
            return idx_list[0] if len(idx_list) > 0 else None

        for tab, val, name in zip([t1, t2, t3, t4], [0, 2, 5, 3], ["Wake", "S2", "REM", "深睡"]):
            with tab:
                idx = find_idx(val)
                if idx is not None:
                    st.write(f"在第 {idx + 1} 片段检测到典型的 {name} 波形：")
                    st.line_chart(eps[idx][:1200], height=150)
                else:
                    st.write(f"本记录中未检测到 {name} 阶段。")
    else:
        st.info("请在左侧侧边栏上传 .hea 和 .dat 文件开始分析。")


if __name__ == "__main__":
    main()

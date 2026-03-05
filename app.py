import streamlit as st
import numpy as np
import soundfile as sf
import io
import pandas as pd

st.set_page_config(page_title="EQ Lab - 1/3オクターブEQ", layout="wide")

# -----------------------------
# 1/3オクターブ中心周波数
# -----------------------------
THIRD_OCT_BANDS = np.array([
    20, 25, 31.5, 40, 50, 63, 80, 100,
    125, 160, 200, 250, 315, 400, 500,
    630, 800, 1000, 1250, 1600, 2000,
    2500, 3150, 4000, 5000, 6300,
    8000, 10000, 12500, 16000
])

# セッション状態の初期化
if "eq_values" not in st.session_state:
    st.session_state.eq_values = [0] * len(THIRD_OCT_BANDS)

# -----------------------------
# 音声読み込み
# -----------------------------
def load_audio(file):
    data, sr = sf.read(file)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr

# -----------------------------
# 1/3オクターブEQ算出
# -----------------------------
def calculate_13oct_eq(src, rec, sr):
    min_len = min(len(src), len(rec))
    src = src[:min_len]
    rec = rec[:min_len]

    fft_src = np.fft.rfft(src)
    fft_rec = np.fft.rfft(rec)

    mag_src = np.abs(fft_src) + 1e-9
    mag_rec = np.abs(fft_rec) + 1e-9

    freqs = np.fft.rfftfreq(len(src), 1 / sr)

    eq_values = []

    for fc in THIRD_OCT_BANDS:
        f_low = fc / (2 ** (1/6))
        f_high = fc * (2 ** (1/6))

        idx = np.where((freqs >= f_low) & (freqs <= f_high))[0]
        if len(idx) == 0:
            eq_values.append(0)
            continue

        ratio = np.mean(mag_rec[idx] / mag_src[idx])
        gain_db = 20 * np.log10(ratio)
        # 算出したゲインを逆にする（補正用）
        correction_db = -gain_db
        correction_db = int(np.clip(round(correction_db), -30, 30))

        eq_values.append(correction_db)

    return eq_values

# -----------------------------
# EQ適用
# -----------------------------
def apply_eq(src, sr, eq_values):
    fft = np.fft.rfft(src)
    freqs = np.fft.rfftfreq(len(src), 1 / sr)

    eq_curve = np.ones_like(freqs)

    for fc, gain in zip(THIRD_OCT_BANDS, eq_values):
        f_low = fc / (2 ** (1/6))
        f_high = fc * (2 ** (1/6))
        idx = np.where((freqs >= f_low) & (freqs <= f_high))[0]
        eq_curve[idx] *= 10 ** (gain / 20)

    processed = np.fft.irfft(fft * eq_curve)
    
    # 正規化（クリッピング防止）
    max_val = np.max(np.abs(processed))
    if max_val > 1e-9:
        processed /= max_val

    return processed

# -----------------------------
# メインUI
# -----------------------------
st.title("🎧 EQ Lab")
st.write("音源を比較してEQ設定を算出したり、別の音声に適用したりできるよ！")

tab1, tab2 = st.tabs(["📊 比較・解析", "✨ EQ適用"])

# --- タブ1: 比較・解析 ---
with tab1:
    st.header("1/3オクターブEQ解析")
    st.write("基準となる「音源」と、実際に録音された「録音」を比較して、その差を埋めるEQ設定を見つけるよ。")
    
    col1, col2 = st.columns(2)
    with col1:
        source_file = st.file_uploader("🎼 基準音源 (.wav)", type=["wav"], key="src_upload")
    with col2:
        recorded_file = st.file_uploader("🎤 録音ファイル (.wav)", type=["wav"], key="rec_upload")

    if source_file and recorded_file:
        if st.button("EQを解析する"):
            with st.spinner("解析中..."):
                src, sr = load_audio(source_file)
                rec, _ = load_audio(recorded_file)
                
                # EQ算出
                st.session_state.eq_values = calculate_13oct_eq(src, rec, sr)
                st.success("解析が終わったよ！現在のEQ設定に保存したよ。")
                st.rerun()

    # 現在のEQ設定を表示
    st.subheader("📊 現在の解析結果")
    
    if any(v != 0 for v in st.session_state.eq_values):
        # テキストで一覧表示
        text_output = ""
        for fc, g in zip(THIRD_OCT_BANDS, st.session_state.eq_values):
            text_output += f"{fc:>6} Hz : {g:+d} dB | "
            if int(fc) % 1000 == 0 or fc == 80 or fc == 800: # 適当なところで改行
                text_output += "\n"
        
        st.code(text_output)
    else:
        st.info("まだ解析されてないよ。上のボタンから解析してね。")

    if st.button("設定をリセット"):
        st.session_state.eq_values = [0] * len(THIRD_OCT_BANDS)
        st.rerun()

# --- タブ2: EQ適用 ---
with tab2:
    st.header("別の音声に適用")
    st.write("「比較・解析」タブで決まった（または自分で調整した）EQ設定を、好きな音声ファイルに適用できるよ。")
    
    target_file = st.file_uploader("🎵 適用したい音声ファイル (.wav)", type=["wav"], key="target_upload")
    
    if target_file:
        if st.button("EQを適用して生成"):
            with st.spinner("処理中..."):
                data, sr = load_audio(target_file)
                processed = apply_eq(data, sr, st.session_state.eq_values)
                
                st.success("適用完了！")
                
                # 再生
                st.subheader("🔊 プレビュー")
                buffer = io.BytesIO()
                sf.write(buffer, processed, sr, format="WAV")
                st.audio(buffer.getvalue(), format="audio/wav")
                
                # ダウンロードボタン
                st.download_button(
                    label="💾 処理後のファイルをダウンロード",
                    data=buffer.getvalue(),
                    file_name="processed_audio.wav",
                    mime="audio/wav"
                )
    else:
        st.info("適用したいファイルをアップロードしてね。")

# 共通: 現在のEQ設定のプレビュー（グラフなど）
st.divider()
st.subheader("📈 現在のEQカーブ")
df_eq = pd.DataFrame({
    "Frequency (Hz)": [str(f) for f in THIRD_OCT_BANDS],
    "Gain (dB)": st.session_state.eq_values
})
st.line_chart(df_eq.set_index("Frequency (Hz)"))

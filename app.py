import streamlit as st
import numpy as np
import soundfile as sf

st.set_page_config(page_title="1/3オクターブEQ解析", layout="centered")

st.title("🎧 1/3オクターブEQ設定算出ツール")
st.write("音源と録音を比較して、1/3オクターブEQ設定（-30〜30 dB）を出力します")

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
        gain_db = int(np.clip(round(gain_db), -30, 30))

        eq_values.append(gain_db)

    return eq_values

# -----------------------------
# UI
# -----------------------------
source_file = st.file_uploader("🎼 音源.wav を選択", type=["wav"])
recorded_file = st.file_uploader("🎤 録音.wav を選択", type=["wav"])

if source_file and recorded_file:
    st.success("ファイルが選択されました")

    src, sr = load_audio(source_file)
    rec, _ = load_audio(recorded_file)

    eq_values = calculate_13oct_eq(src, rec, sr)

    st.subheader("📊 1/3オクターブEQ設定結果")

    eq_table = {
        "中心周波数 (Hz)": THIRD_OCT_BANDS,
        "EQ設定 (dB)": eq_values
    }

    st.table(eq_table)

    st.subheader("📄 テキスト表示")
    text_output = ""
    for fc, g in zip(THIRD_OCT_BANDS, eq_values):
        text_output += f"{fc:>6} Hz : {g:+d} dB\n"

    st.text(text_output)

else:
    st.info("音源.wav と 録音.wav を両方アップロードしてください")

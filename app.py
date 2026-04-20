import streamlit as st
from PIL import Image
from ultralytics import YOLO

# 1. 網頁標題與外觀設定
st.set_page_config(page_title="愛玉檢測系統", layout="centered")
st.title("🌱 愛玉品質與成熟度 AI 檢測系統")
st.write("請上傳愛玉的照片，系統將自動框出目標並判斷結果。")

# 2. 載入模型
@st.cache_resource 
def load_model():
    return YOLO('best.pt')

model = load_model()

# 3. 建立檔案上傳區塊
uploaded_file = st.file_uploader("點擊或拖曳上傳圖片...", type=["jpg", "jpeg", "png"])

# 4. 執行預測
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="原始上傳照片", use_column_width=True)
    
    if st.button("🚀 開始 AI 辨識"):
        with st.spinner('AI 正在努力運算中...'):
            # 使用模型預測，設定信心門檻 0.90
            results = model.predict(source=image, conf=0.90)
            res_plotted = results[0].plot()
            res_rgb = res_plotted[:, :, ::-1]
            
            st.success('辨識完成！')
            st.image(res_rgb, caption="AI 辨識結果", use_column_width=True)
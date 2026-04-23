import streamlit as st
from PIL import Image
from ultralytics import YOLO

# 1. 網頁標題與外觀設定
st.set_page_config(page_title="愛玉檢測系統", layout="centered")
st.title("🌱 愛玉品質與成熟度 AI 檢測系統")
st.write("請上傳愛玉的照片，系統將自動判斷結果。")

# 2. 載入模型
@st.cache_resource 
def load_model():
    return YOLO('best.pt')

model = load_model()

# 3. 建立檔案上傳區塊
uploaded_file = st.file_uploader("點擊或拖曳上傳圖片...", type=["jpg", "jpeg", "png"])

# 4. 執行預測
if uploaded_file is not None:
    # 強制轉換格式以防手機照片出錯
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="原始上傳照片", use_container_width=True)
    
    if st.button("🚀 開始 AI 辨識"):
        with st.spinner('AI 正在努力運算中...'):
            
            # 執行預測
            results = model.predict(source=image, conf=0.25)
            res_plotted = results[0].plot()
            res_rgb = res_plotted[:, :, ::-1] 
            
            st.success('辨識完成！')
            st.image(res_rgb, caption="AI 辨識結果", use_container_width=True)
            
            st.markdown("### 📊 AI 判定報告：")
            
            # 【智慧判斷 A】：如果是「影像分類」模型
            if results[0].probs is not None:
                top1_index = results[0].probs.top1
                class_name = model.names[top1_index]
                confidence = float(results[0].probs.top1conf)
                st.info(f"👉 整張圖片判定為：**{class_name}** (AI 把握度：{confidence:.1%})")
                
            # 【智慧判斷 B】：如果是「物件偵測」模型
            elif results[0].boxes is not None and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    st.info(f"👉 偵測到目標：**{class_name}** (AI 把握度：{confidence:.1%})")
                    
            # 【智慧判斷 C】：真的什麼都沒看到
            else:
                st.warning("⚠️ AI 看不到任何目標。可能是照片特徵不明顯，請換一張試試看！")

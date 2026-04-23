import os
import sys

# 🚀 終極防線：在載入任何 AI 模型之前，強制把會導致系統崩潰的標準版 OpenCV 刪除！
os.system(f"{sys.executable} -m pip uninstall -y opencv-python opencv-contrib-python")

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

# 💡 新增：成熟度標籤翻譯字典
# 這裡請把你訓練時設定的標籤名稱 (等號左邊) 對應到想顯示的中文 (等號右邊)
# 如果你訓練時直接用中文命名，那它就會直接顯示原本的中文。
label_translator = {
    "immature": "🟢 未成熟",
    "mature": "🟡 成熟",
    "overripe": "🟤 過熟",
    "peel": "果皮 (未分類成熟度)",
    # 可以在下面繼續新增你訓練用的標籤...
}

# 3. 建立檔案上傳區塊
uploaded_file = st.file_uploader("點擊或拖曳上傳圖片...", type=["jpg", "jpeg", "png"])

# 4. 執行預測
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="原始上傳照片", use_container_width=True)
    
    st.markdown("---")
    st.write("🔧 **進階設定**")
    conf_threshold = st.slider("調整 AI 敏感度 (數值越低越容易框出東西，但也容易誤判背景)", min_value=0.01, max_value=0.99, value=0.25, step=0.01)
    
    if st.button("🚀 開始 AI 辨識"):
        with st.spinner('AI 正在努力運算中...'):
            
            results = model.predict(source=image, conf=conf_threshold)
            res_plotted = results[0].plot()
            res_rgb = res_plotted[:, :, ::-1] 
            
            st.success('辨識完成！')
            st.image(res_rgb, caption="AI 辨識結果", use_container_width=True)
            
            st.markdown("### 📊 AI 判定報告：")
            
            # 【智慧判斷 A】：如果是「影像分類」模型
            if results[0].probs is not None:
                top1_index = results[0].probs.top1
                original_class = model.names[top1_index]
                # 使用翻譯機，如果字典裡沒有，就顯示原本的標籤名
                display_class = label_translator.get(original_class.lower(), original_class)
                confidence = float(results[0].probs.top1conf)
                st.info(f"👉 系統判定成熟度為：**{display_class}** (AI 把握度：{confidence:.1%})")
                
            # 【智慧判斷 B】：如果是「物件偵測」模型
            elif results[0].boxes is not None and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    original_class = model.names[class_id]
                    # 使用翻譯機
                    display_class = label_translator.get(original_class.lower(), original_class)
                    confidence = float(box.conf[0])
                    st.info(f"👉 偵測到目標：**{display_class}** (AI 把握度：{confidence:.1%})")
                    
            # 【智慧判斷 C】：真的什麼都沒看到
            else:
                st.warning("⚠️ AI 看不到任何目標。試著把上方的「AI 敏感度」拉低一點再試一次看看！")

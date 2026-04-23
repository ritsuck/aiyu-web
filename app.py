import os
import sys

# 🚀 終極防線：強制解決 Streamlit 雲端的 OpenCV 套件衝突
os.system(f"{sys.executable} -m pip uninstall -y opencv-python opencv-contrib-python")

import streamlit as st
from PIL import Image
from ultralytics import YOLO

# 1. 網頁標題與外觀設定
st.set_page_config(page_title="愛玉成熟度辨識系統", layout="centered")
st.title("🌱 愛玉成熟度 AI 影像分類系統")
st.write("本系統採用影像分類模型，自動分析整張照片的愛玉成熟度機率。")

# 2. 載入模型 (請確認你的模型檔案名稱為 best.pt)
@st.cache_resource 
def load_model():
    # 這裡會載入你的分類模型
    return YOLO('best.pt')

model = load_model()

# 🔍 側邊欄偵測器：確認分類模型內的類別
st.sidebar.title("🕵️ 模型類別檢查")
st.sidebar.info(f"模型辨識類別：\n\n{model.names}")

# 💡 成熟度標籤翻譯字典 (根據你提供的類別名稱對齊)
label_translator = {
    "77non_mature": "🟢 未成熟 (77以下)",
    "78_84mature": "🟡 成熟 (78-84)",
    "85over_mature": "🟤 過熟 (85以上)"
}

# 3. 建立檔案上傳區塊
uploaded_file = st.file_uploader("上傳愛玉果實照片...", type=["jpg", "jpeg", "png"])

# 4. 執行分類預測
if uploaded_file is not None:
    # 顯示上傳的照片
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="待檢測照片", use_container_width=True)
    
    if st.button("🚀 開始分析成熟度"):
        with st.spinner('AI 正在計算各類別機率...'):
            
            # 執行預測
            results = model.predict(source=image)
            result = results[0]

            # 檢查是否為分類模型結果 (probs)
            if result.probs is not None:
                st.success('分析完成！')
                
                # 獲取最高機率的類別索引與數值
                top1_idx = result.probs.top1
                top1_conf = result.probs.top1conf.item()
                top1_name = model.names[top1_idx]
                
                # 翻譯名稱
                display_name = label_translator.get(top1_name, top1_name)

                # 顯示主結果
                st.subheader(f"判定結果：{display_class if 'display_class' in locals() else display_name}")
                st.metric(label="最高信心值", value=f"{top1_conf:.2%}")

                # 顯示所有類別的機率分佈
                st.write("---")
                st.write("📊 **各成熟度詳細機率：**")
                
                # 遍歷所有類別並顯示機率
                for idx, name in model.names.items():
                    conf = result.probs.data[idx].item()
                    translated = label_translator.get(name, name)
                    st.write(f"- {translated}: **{conf:.2%}**")
                    st.progress(conf) # 用進度條呈現機率感
                    
            else:
                st.error("⚠️ 偵測失敗：這顆模型似乎不是「影像分類」模型，請檢查你的模型檔案。")

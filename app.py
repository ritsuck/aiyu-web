# 4. 執行預測
if uploaded_file is not None:
    # 強制轉換格式以防手機照片出錯
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="原始上傳照片", use_container_width=True)
    
    st.markdown("---")
    st.write("🔧 **進階設定**")
    # 新增一個滑桿，讓你可以自由調整 AI 的敏感度 (預設 0.25，最低 0.01)
    conf_threshold = st.slider("調整 AI 敏感度 (數值越低越容易框出東西，但也容易誤判背景)", min_value=0.01, max_value=0.99, value=0.25, step=0.01)
    
    if st.button("🚀 開始 AI 辨識"):
        with st.spinner('AI 正在努力運算中...'):
            
            # 這裡的 conf 換成了你滑桿拉到的數值
            results = model.predict(source=image, conf=conf_threshold)
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
                st.warning("⚠️ AI 看不到任何目標。試著把上方的「AI 敏感度」拉低一點再試一次看看！")

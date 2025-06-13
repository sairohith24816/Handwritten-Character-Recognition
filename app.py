import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import joblib
import cv2
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="MNIST Digit Recognition", layout="wide")

st.title('🎯 MNIST Digit Recognition')

if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
    st.session_state.prediction_result = None
    st.session_state.canvas_key = 0

st.markdown("---")  

left_col, right_col = st.columns([1.2, 0.8])

with left_col:
    st.subheader('✏️ Draw a digit (0-9)')
    
    st.markdown("""
    <style>
    .canvas-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    canvas_result = st_canvas(
        fill_color='black',
        stroke_width=15,
        stroke_color='white',
        background_color='black',
        height=450,
        width=450,
        drawing_mode='freedraw',
        key=f'canvas_{st.session_state.canvas_key}',
    )
    
    model_type = st.selectbox('🤖 Select Model', [
        'KNN', 
        'Logistic Regression',
        'SVM',
        'Random Forest',
        'Gradient Boosting',
        'Naive Bayes'
    ])
    
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        clear_button = st.button('🧹 Clear', use_container_width=True)
    with btn_col2:
        predict_button = st.button('🔮 Predict', use_container_width=True)

with right_col:
    st.subheader('📊 Prediction Results')
    
    if clear_button:
        st.session_state.processed_image = None
        st.session_state.prediction_result = None  # Clear prediction results
        st.session_state.canvas_key += 1  # Force canvas to reset by changing key
        st.rerun()
    
    if predict_button and canvas_result.image_data is not None:
        img_data = canvas_result.image_data[:, :, :3]  
        
        gray_img = np.dot(img_data, [0.2989, 0.5870, 0.1140])
        
        if np.std(gray_img) > 0:  
            gray_img = ((gray_img - gray_img.min()) / (gray_img.max() - gray_img.min()) * 255).astype(np.uint8)
            
            if gray_img.mean() > 127:  
                gray_img = 255 - gray_img
            
            img = Image.fromarray(gray_img)
            img = img.resize((28, 28), Image.Resampling.LANCZOS)
            
            img_arr = np.array(img)
            
            img_arr = cv2.threshold(img_arr, 50, 255, cv2.THRESH_BINARY)[1]
            
            st.session_state.processed_image = img_arr
            
            st.write("**Processed Images:**")
            col1, col2 = st.columns(2)
            with col1:
                st.image(gray_img, width=150, caption="Original Drawing")
            with col2:
                st.image(img_arr, width=150, caption="MNIST Format (28x28)")
            
            if model_type == 'KNN':
                try:
                    model = joblib.load('models/best_knn.joblib')
                    scaler = joblib.load('models/knn_scaler.joblib')
                    x = img_arr.reshape(1, -1) / 255.0
                    x_scaled = scaler.transform(x)
                    pred = model.predict(x_scaled)
                    proba = model.predict_proba(x_scaled)
                    predicted_digit = int(pred[0])
                    confidence = np.max(proba)
                    
                    st.session_state.prediction_result = {
                        'model': 'KNN',
                        'digit': predicted_digit,
                        'confidence': confidence,
                        'probabilities': proba[0]
                    }
                except Exception as e:
                    st.error(f"Error loading KNN model: {e}")
            elif model_type == 'Logistic Regression':
                try:
                    model = joblib.load('models/best_logistic_regression.joblib')
                    scaler = joblib.load('models/logistic_regression_scaler.joblib')
                    x = img_arr.reshape(1, -1) / 255.0
                    x_scaled = scaler.transform(x)
                    pred = model.predict(x_scaled)
                    proba = model.predict_proba(x_scaled)
                    predicted_digit = int(pred[0])
                    confidence = np.max(proba)
                    
                    st.session_state.prediction_result = {
                        'model': 'Logistic Regression',
                        'digit': predicted_digit,
                        'confidence': confidence,
                        'probabilities': proba[0]
                    }
                except Exception as e:
                    st.error(f"Error loading Logistic Regression model: {e}")
            elif model_type == 'SVM':
                try:
                    model = joblib.load('models/best_svm.joblib')
                    scaler = joblib.load('models/svm_scaler.joblib')
                    x = img_arr.reshape(1, -1) / 255.0
                    x_scaled = scaler.transform(x)
                    pred = model.predict(x_scaled)
                    proba = model.predict_proba(x_scaled)
                    predicted_digit = int(pred[0])
                    confidence = np.max(proba)
                    
                    st.session_state.prediction_result = {
                        'model': 'SVM',
                        'digit': predicted_digit,
                        'confidence': confidence,
                        'probabilities': proba[0]
                    }
                except Exception as e:
                    st.error(f"Error loading SVM model: {e}")
            elif model_type == 'Random Forest':
                try:
                    model = joblib.load('models/best_random_forest.joblib')
                    x = img_arr.reshape(1, -1) / 255.0
                    pred = model.predict(x)
                    proba = model.predict_proba(x)
                    predicted_digit = int(pred[0])
                    confidence = np.max(proba)
                    
                    st.session_state.prediction_result = {
                        'model': 'Random Forest',
                        'digit': predicted_digit,
                        'confidence': confidence,
                        'probabilities': proba[0]
                    }
                except Exception as e:
                    st.error(f"Error loading Random Forest model: {e}")
            elif model_type == 'Gradient Boosting':
                try:
                    model = joblib.load('models/best_gradient_boosting.joblib')
                    x = img_arr.reshape(1, -1) / 255.0
                    pred = model.predict(x)
                    proba = model.predict_proba(x)
                    predicted_digit = int(pred[0])
                    confidence = np.max(proba)
                    
                    st.session_state.prediction_result = {
                        'model': 'Gradient Boosting',
                        'digit': predicted_digit,
                        'confidence': confidence,
                        'probabilities': proba[0]
                    }
                except Exception as e:
                    st.error(f"Error loading Gradient Boosting model: {e}")
            elif model_type == 'Naive Bayes':
                try:
                    model = joblib.load('models/best_naive_bayes.joblib')
                    try:
                        scaler = joblib.load('models/naive_bayes_scaler.joblib')
                        x = img_arr.reshape(1, -1) / 255.0
                        x_scaled = scaler.transform(x)
                        pred = model.predict(x_scaled)
                        proba = model.predict_proba(x_scaled)
                    except FileNotFoundError:
                        x = img_arr.reshape(1, -1) / 255.0
                        pred = model.predict(x)
                        proba = model.predict_proba(x)
                    predicted_digit = int(pred[0])
                    confidence = np.max(proba)
                    
                    st.session_state.prediction_result = {
                        'model': 'Naive Bayes',
                        'digit': predicted_digit,
                        'confidence': confidence,
                        'probabilities': proba[0]
                    }
                except Exception as e:
                    st.error(f"Error loading Naive Bayes model: {e}")
        else:
            st.warning('Please draw something on the canvas first!')
    
    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        
        st.write("**Prediction Results:**")
        st.metric(
            label=f"{result['model']} Prediction",
            value=f"Digit: {result['digit']}",
            delta=f"Confidence: {result['confidence']*100:.1f}%"
        )
        
        st.write("**Top 3 Confidence Scores:**")
        probabilities = result['probabilities']
        
        top_3_indices = np.argsort(probabilities)[-3:][::-1] 
        
        for i, digit_idx in enumerate(top_3_indices):
            confidence = probabilities[digit_idx]
            rank = ["🥇", "🥈", "🥉"][i]  # Medal emojis for top 3
            st.write(f"{rank} Digit {digit_idx}: {confidence*100:.1f}%")
            st.progress(float(confidence))
    
    if st.session_state.processed_image is not None:
        with st.expander("Debug Info"):
            img_arr = st.session_state.processed_image
            st.write(f"Image shape: {img_arr.shape}")
            st.write(f"Min pixel value: {img_arr.min()}")
            st.write(f"Max pixel value: {img_arr.max()}")
            st.write(f"Mean pixel value: {img_arr.mean():.2f}")
            st.write("(MNIST format: black background ~0, white digits ~255)")


# reduce print statements while model training
# the results must be erased after clicking clear in canavas
# and the canvas must be reset

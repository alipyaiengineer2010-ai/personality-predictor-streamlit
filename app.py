import streamlit as st
import joblib
import pandas as pd

# ۱. تنظیمات صفحه
st.set_page_config(
    page_title="Personality Predictor AI",
    page_icon="🧠",
    layout="wide"
)

# ۲. بارگذاری مدل


@st.cache_resource
def load_model():
    return joblib.load('personality_model.pkl')


model = load_model()

# ۳. عنوان داشبورد
st.title("🧠 Personality Prediction Dashboard")
st.markdown(
    "تشخیص تیپ شخصیتی **(Introvert / Extrovert)** بر اساس الگوهای رفتاری با هوش مصنوعی:")
st.markdown("---")

# ۴. ورودی‌ها در سایدبار
st.sidebar.header("📊 Social & Behavioral Profile")

time_alone = st.sidebar.slider(
    "1. Time Spent Alone (Hours/Day)", min_value=0, max_value=12, value=4)
friends_circle = st.sidebar.slider(
    "2. Friends Circle Size", min_value=0, max_value=30, value=5)
social_attendance = st.sidebar.slider(
    "3. Social Event Attendance (per month)", min_value=0, max_value=20, value=3)
going_outside = st.sidebar.slider(
    "4. Going Outside (days per week)", min_value=0, max_value=7, value=3)
post_freq = st.sidebar.slider(
    "5. Social Media Post Frequency (per week)", min_value=0, max_value=20, value=2)

stage_fear = st.sidebar.selectbox(
    "6. Stage Fear (Speaking in public)?", ["No", "Yes"])
drained = st.sidebar.selectbox(
    "7. Feeling drained after socializing?", ["No", "Yes"])

# تبدیل بله/خیر به 0 و 1
stage_fear_val = 1 if stage_fear == "Yes" else 0
drained_val = 1 if drained == "Yes" else 0

# ساخت دیتافریم ورودی برای مدل
input_data = pd.DataFrame({
    'Time_spent_Alone': [time_alone],
    'Stage_fear': [stage_fear_val],
    'Social_event_attendance': [social_attendance],
    'Going_outside': [going_outside],
    'Drained_after_socializing': [drained_val],
    'Friends_circle_size': [friends_circle],
    'Post_frequency': [post_freq]
})

# دیکشنری نگاشت اعداد به کلمات خوانا
label_map = {0: "Introvert", 1: "Extrovert"}

# ۵. نمایش خلاصه و خروجی
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Summary of Responses")
    st.write(f"• **Time Alone:** {time_alone} hrs/day")
    st.write(f"• **Friends Count:** {friends_circle}")
    st.write(f"• **Social Events:** {social_attendance} / month")
    st.write(f"• **Going Outside:** {going_outside} days/week")
    st.write(f"• **Post Frequency:** {post_freq} posts/week")
    st.write(f"• **Stage Fear:** {stage_fear}")
    st.write(f"• **Drained After Socializing:** {drained}")

    predict_btn = st.button("🚀 Predict Personality", use_container_width=True)

with col2:
    st.subheader("🎯 Result")
    if predict_btn:
        raw_pred = model.predict(input_data)[0]
        # تبدیل کد عددی به نام متنی
        pred_label = label_map.get(raw_pred, str(raw_pred))

        if pred_label == "Introvert":
            st.success(
                f"### Predicted Personality: **{pred_label} (درون‌گرا)** 🧘‍♂️")
        elif pred_label == "Extrovert":
            st.success(
                f"### Predicted Personality: **{pred_label} (برون‌گرا)** 🎉")
        else:
            st.success(f"### Predicted Personality: **{pred_label}**")

        # رسم نمودار احتمالاتی
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_data)[0]
            classes = [label_map.get(c, str(c)) for c in model.classes_]
            prob_df = pd.DataFrame(
                {'Personality': classes, 'Confidence': probs})
            st.bar_chart(prob_df.set_index('Personality'))
    else:
        st.info("روی دکمه‌ی **Predict Personality** کلیک کن تا نتیجه محاسبه بشه.")

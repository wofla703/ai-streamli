# 라이브러리 임포트
import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps  # Pillow 사용
import time  # 로딩 효과를 위해 추가

# Numpy 과학적 표기법 제거 (명확한 숫자 표현)
np.set_printoptions(suppress=True)

# 모델 & 라벨 로드
model = load_model('keras_model.h5', compile=False)
with open('labels.txt', 'r', encoding='utf-8') as f:
    class_names = f.readlines()

# 🎨 Streamlit 앱 UI 설정
st.set_page_config(
    page_title="닮은 연예인 찾기",
    page_icon="🎭",
    layout="centered"
)

# 타이틀
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>✨ 닮은 연예인 찾기 ✨</h1>", unsafe_allow_html=True)
st.markdown("##### 📸 AI가 당신과 가장 닮은 연예인을 찾아줍니다!")

# 선택 옵션: 카메라 사용 or 파일 업로드
input_method = st.radio("📷 이미지를 어떻게 입력하시겠어요?", ["📸 카메라 사용", "📂 파일 업로드"])

if input_method == "📸 카메라 사용":
    img_file_buffer = st.camera_input("정중앙에 얼굴을 맞추고 촬영하세요!")
else:
    img_file_buffer = st.file_uploader("📂 이미지를 업로드하세요", type=["png", "jpg", "jpeg"])

# 빈 넘파이 배열 생성 (모델 입력용)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

if img_file_buffer is not None:
    # 🔄 이미지 전처리
    image = Image.open(img_file_buffer).convert('RGB')
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # 🕒 로딩 애니메이션 추가
    with st.spinner("🔍 AI가 이미지를 분석 중... 잠시만 기다려 주세요!"):
        time.sleep(2)  # 딜레이 효과

    # 모델 예측 수행
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # 🏆 예측 결과 표시
    st.success(f"✨ AI가 분석한 결과, 당신은 **{class_name[2:]}** 닮았습니다! ✨")

    # 📊 신뢰도 (Confidence Score) 시각화
    st.progress(float(confidence_score))  # 프로그레스 바 표시
    st.write(f"📊 신뢰도: **{confidence_score * 100:.2f}%**")

    # 🎭 비교 대상 정보 추가
    st.markdown(
        """
        ---
        #### 🔍 AI가 분석하는 연예인 목록:
        - **여성 비교대상**: 한소희, 윈터, 장원영, 한지민, 박나래, 김민경, 벨, 송지효
        - **남성 비교대상**: 문상훈, 정해인, 박보검, 차은우, 조진웅, 유병재, 윤성빈
        ---
        """,
        unsafe_allow_html=True
    )

    # 결과 이미지 표시 (업로드된 사진 미리보기)
    st.image(image, caption="📷 분석된 이미지", use_column_width=True)

# 📝 푸터 (마무리 텍스트)
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<h5 style='text-align: center;'>💡 AI 기반 연예인 유사도 분석 서비스</h5>",
    unsafe_allow_html=True
)

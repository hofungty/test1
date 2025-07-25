import streamlit as st
import json
import firebase_admin
from firebase_admin import credentials

try:
    # Streamlit Secrets에서 JSON 문자열 불러오기
    firebase_config_json = st.secrets["FIREBASE_CONFIG_JSON"]

    # JSON 문자열을 딕셔너리로 파싱
    firebase_config = json.loads(firebase_config_json)

    # Firebase 서비스 계정 인증 정보 초기화
    cred = credentials.Certificate(firebase_config)
    
    # 이미 초기화된 앱이 아니라면 초기화
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    
    st.success("Firebase 설정이 성공적으로 로드되었습니다.")

except KeyError:
    st.error("Streamlit Secrets에 'FIREBASE_CONFIG_JSON'이 설정되지 않았습니다.")
except json.JSONDecodeError:
    st.error("Firebase 설정 JSON 형식이 올바르지 않습니다. 'FIREBASE_CONFIG_JSON'을 확인해주세요.")
except Exception as e:
    st.error(f"Firebase 초기화 중 오류가 발생했습니다: {e}")

# 이후 Firebase 관련 코드 작성

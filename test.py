import streamlit as st
import random
import requests
from sentence_transformers import SentenceTransformer, util
import os # 파일 존재 여부 확인을 위해 추가
import json # Firebase config 파싱을 위해 추가

# Firebase 관련 import
# Firebase Admin SDK를 사용합니다. Streamlit Cloud 배포 시에는 클라이언트 SDK 사용을 고려해야 합니다.
# 이 코드는 Canvas 환경에 맞춰 설계되었습니다.
try:
    from firebase_admin import credentials, initialize_app
    from firebase_admin import firestore
    from firebase_admin import auth
    FIREBASE_AVAILABLE = True
except ImportError:
    st.warning("Firebase Admin SDK를 찾을 수 없습니다. Firebase 기능 없이 앱이 실행됩니다.")
    FIREBASE_AVAILABLE = False


# --- API 및 모델 설정 ---

# API 키는 st.secrets 등을 통해 안전하게 관리하는 것을 권장합니다.
try:
    # st.secrets에서 Google API 키를 가져옵니다.
    # .streamlit/secrets.toml 파일에 GOOGLE_API_KEY = "YOUR_API_KEY" 형식으로 저장하거나
    # Streamlit Cloud Secrets에 설정해야 합니다.
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("Google API Key를 찾을 수 없습니다. .streamlit/secrets.toml 파일을 확인하거나 Streamlit Cloud Secrets에 설정해주세요.")
    # API 키가 없을 경우 앱이 작동하지 않으므로, 임시로 하드코딩된 키를 사용합니다.
    # 실제 배포 시에는 이 부분을 제거하고 secrets를 통해 키를 제공해야 합니다.
    GOOGLE_API_KEY = "AIzaSyAEJ-RJf4PqQPqaHe2a_rDt0JFZ_--Klnw" # 임시 키, 실제 배포 시 제거 권장
    st.info("임시 Google API Key로 작동합니다. 일부 기능이 제한될 수 있습니다.")


HINT_THRESHOLD = 0.4  # 이 유사도 이상일 때 힌트를 제공합니다.
WORDS_FILE = "words.txt" # 영단어 목록 파일 이름

@st.cache_resource
def load_sbert_model():
    """
    Sentence-BERT 모델을 로드합니다. @st.cache_resource 데코레이터 덕분에
    이 함수는 앱 실행 중 단 한 번만 호출되어 모델을 메모리에 올립니다.
    """
    return SentenceTransformer('all-MiniLM-L6-v2')

# 앱 시작 시 모델 로드
model = load_sbert_model()


# --- Firebase 초기화 및 인증 ---
if FIREBASE_AVAILABLE:
    # Canvas 환경에서 제공되는 전역 변수 사용
    # __app_id, __firebase_config, __initial_auth_token 변수는 Canvas 런타임에서 주입됩니다.
    app_id = globals().get('__app_id', 'default-app-id')
    firebase_config_str = globals().get('__firebase_config', '{}')
    initial_auth_token = globals().get('__initial_auth_token', None)

    firebase_config = {}
    try:
        if firebase_config_str:
            firebase_config = json.loads(firebase_config_str)
    except json.JSONDecodeError:
        st.error("Firebase 설정 JSON 파싱 오류.")

    if 'firebase_initialized' not in st.session_state:
        st.session_state.firebase_initialized = False # 초기 상태 설정
        st.session_state.user_id = "loading_user" # 로딩 중 상태

        # initialize_app()이 이미 초기화되었는지 확인하는 로직 개선
        firebase_app_already_initialized = False
        try:
            # firebase_admin.get_app() 시도하여 이미 초기화된 앱이 있는지 확인
            firebase_admin.get_app()
            firebase_app_already_initialized = True
        except ValueError:
            pass # 앱이 아직 초기화되지 않음

        if firebase_config and not firebase_app_already_initialized:
            try:
                cred = credentials.Certificate(firebase_config)
                initialize_app(cred)
                st.session_state.firebase_initialized = True
                st.session_state.db = firestore.client()
                st.session_state.auth = auth

                # 사용자 인증 (Canvas 환경에 맞춰 __initial_auth_token 사용)
                if initial_auth_token:
                    try:
                        # Custom token을 사용하여 로그인 (Canvas에서 제공)
                        user = auth.sign_in_with_custom_token(initial_auth_token)
                        st.session_state.user_id = user.uid
                        st.success(f"Firebase 인증 성공! 사용자 ID: {st.session_state.user_id}")
                    except Exception as e:
                        st.error(f"Firebase Custom Token 로그인 실패: {e}")
                        # Custom Token 실패 시 익명 로그인 시도
                        try:
                            user = auth.sign_in_anonymously()
                            st.session_state.user_id = user.uid
                            st.warning(f"Custom Token 로그인 실패, 익명 로그인 성공! 사용자 ID: {st.session_state.user_id}")
                        except Exception as e_anon:
                            st.error(f"Firebase 익명 로그인 실패: {e_anon}")
                            st.session_state.user_id = "anonymous_user_error" # Fallback
                else:
                    # __initial_auth_token이 없으면 익명 로그인 시도
                    try:
                        user = auth.sign_in_anonymously()
                        st.session_state.user_id = user.uid
                        st.warning(f"__initial_auth_token 없음, 익명 로그인 성공! 사용자 ID: {st.session_state.user_id}")
                    except Exception as e_anon:
                        st.error(f"Firebase 익명 로그인 실패: {e_anon}")
                        st.session_state.user_id = "anonymous_user_error" # Fallback
            except Exception as e:
                st.error(f"Firebase 초기화 중 오류 발생: {e}")
                st.session_state.user_id = "firebase_init_error"
        elif firebase_app_already_initialized:
            # 앱이 이미 초기화된 경우, 클라이언트와 인증 정보 재설정
            st.session_state.firebase_initialized = True
            st.session_state.db = firestore.client()
            st.session_state.auth = auth
            # 이미 로그인된 사용자 정보가 있다면 가져오기 (익명 사용자 포함)
            if st.session_state.auth.current_user:
                st.session_state.user_id = st.session_state.auth.current_user.uid
            else: # 현재 사용자가 없으면 다시 익명 로그인 시도
                try:
                    user = st.session_state.auth.sign_in_anonymously()
                    st.session_state.user_id = user.uid
                    st.warning(f"Firebase 앱 이미 초기화됨, 익명 로그인 성공! 사용자 ID: {st.session_state.user_id}")
                except Exception as e_anon:
                    st.error(f"Firebase 앱 초기화 후 익명 로그인 실패: {e_anon}")
                    st.session_state.user_id = "anonymous_user_error"

        else: # firebase_config가 없거나 다른 문제
            if not firebase_config:
                st.error("Firebase 설정이 올바르지 않습니다. 앱을 실행할 수 없습니다.")
            st.session_state.user_id = "no_firebase_config"
else: # Firebase Admin SDK가 설치되지 않은 경우
    st.session_state.firebase_initialized = False
    st.session_state.user_id = "firebase_not_available"


# --- Firestore 데이터 로드 및 저장 함수 ---

def get_user_data_ref():
    """현재 사용자의 학습 데이터 Firestore 참조를 반환합니다."""
    # Firestore 보안 규칙에 따라 private data 경로 사용: /artifacts/{appId}/users/{userId}/{your_collection_name}
    if st.session_state.get('db') and st.session_state.get('user_id') and st.session_state.get('app_id') and st.session_state.user_id != "loading_user":
        return st.session_state.db.collection('artifacts').document(st.session_state.app_id).collection('users').document(st.session_state.user_id).collection('word_data').document('user_session')
    return None # Firebase 초기화 안 된 경우

def load_user_session_data():
    """Firestore에서 사용자의 학습 세션 데이터를 로드합니다."""
    if not FIREBASE_AVAILABLE or not st.session_state.get('firebase_initialized') or not st.session_state.get('user_id') or st.session_state.user_id == "loading_user":
        st.warning("Firebase가 준비되지 않아 학습 데이터를 로드할 수 없습니다. 파일에서 단어를 불러옵니다.")
        st.session_state.all_words = load_words_from_file(WORDS_FILE)
        st.session_state.available_words = list(st.session_state.all_words)
        st.session_state.used_words = []
        st.session_state.correctly_answered_words_in_order = []
        return

    try:
        doc_ref = get_user_data_ref()
        if doc_ref:
            doc = doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                st.session_state.available_words = data.get('available_words', [])
                st.session_state.used_words = data.get('used_words', [])
                st.session_state.correctly_answered_words_in_order = data.get('correctly_answered_words_in_order', [])
                st.info("이전 학습 데이터를 불러왔습니다.")
                # 만약 불러온 단어 목록이 비어있으면 새로 초기화 (파일에서 로드)
                if not st.session_state.available_words and not st.session_state.used_words:
                    st.session_state.all_words = load_words_from_file(WORDS_FILE)
                    st.session_state.available_words = list(st.session_state.all_words)
                    st.session_state.used_words = []
                    st.session_state.correctly_answered_words_in_order = []
                    st.warning("불러온 데이터가 비어있어 단어 목록을 새로 초기화합니다.")
            else:
                st.warning("이전 학습 데이터가 없습니다. 새로운 세션을 시작합니다.")
                # 데이터가 없으면 파일에서 단어 로드 및 초기화
                st.session_state.all_words = load_words_from_file(WORDS_FILE)
                st.session_state.available_words = list(st.session_state.all_words)
                st.session_state.used_words = []
                st.session_state.correctly_answered_words_in_order = []
        else:
            st.warning("Firebase 데이터 참조를 얻을 수 없습니다. 파일에서 단어를 불러옵니다.")
            st.session_state.all_words = load_words_from_file(WORDS_FILE)
            st.session_state.available_words = list(st.session_state.all_words)
            st.session_state.used_words = []
            st.session_state.correctly_answered_words_in_order = []

    except Exception as e:
        st.error(f"학습 데이터 로드 중 오류 발생: {e}")
        # 오류 발생 시에도 파일에서 단어 로드 및 초기화
        st.session_state.all_words = load_words_from_file(WORDS_FILE)
        st.session_state.available_words = list(st.session_state.all_words)
        st.session_state.used_words = []
        st.session_state.correctly_answered_words_in_order = []

def save_user_session_data():
    """현재 사용자의 학습 세션 데이터를 Firestore에 저장합니다."""
    if not FIREBASE_AVAILABLE or not st.session_state.get('firebase_initialized') or not st.session_state.get('user_id') or st.session_state.user_id == "loading_user":
        # st.warning("Firebase가 준비되지 않아 학습 데이터를 저장할 수 없습니다.")
        return # Firebase가 준비되지 않았으면 저장하지 않음

    try:
        doc_ref = get_user_data_ref()
        if doc_ref:
            data_to_save = {
                'available_words': st.session_state.available_words,
                'used_words': st.session_state.used_words,
                'correctly_answered_words_in_order': st.session_state.correctly_answered_words_in_order
            }
            doc_ref.set(data_to_save)
            # st.success("학습 데이터가 저장되었습니다.") # 너무 자주 표시될 수 있으므로 주석 처리
        else:
            st.warning("Firebase 데이터 참조를 얻을 수 없어 데이터를 저장할 수 없습니다.")
    except Exception as e:
        st.error(f"학습 데이터 저장 중 오류 발생: {e}")


# --- 기존 데이터 처리 함수들 ---

def load_words_from_file(filepath):
    """
    지정된 파일에서 영단어 목록을 불러옵니다.
    각 줄의 공백을 제거하고 소문자로 변환합니다.
    """
    words = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower() # 공백 제거 및 소문자 변환
                if word: # 빈 줄이 아닌 경우에만 추가
                    words.append(word)
        if not words: # 파일은 있지만 단어가 없는 경우
            st.warning(f"'{filepath}' 파일에 단어가 없습니다. 단어를 한 줄에 하나씩 입력해주세요.")
            # 파일이 비어 있을 경우 기본 단어 목록을 제공 (개발/테스트용)
            words = ["happy", "sad", "angry", "joyful", "unhappy", "glad", "mad", "furious", "beautiful", "intelligent", "courageous", "brave", "kind", "gentle", "strong", "weak", "fast", "slow", "bright", "dark"]
    else:
        st.error(f"'{filepath}' 파일을 찾을 수 없습니다. 파일을 생성하고 영단어를 한 줄에 하나씩 입력해주세요.")
        # 파일이 없을 경우 기본 단어 목록을 제공 (개발/테스트용)
        words = ["happy", "sad", "angry", "joyful", "unhappy", "glad", "mad", "furious", "beautiful", "intelligent", "courageous", "brave", "kind", "gentle", "strong", "weak", "fast", "slow", "bright", "dark"]
    return words

def get_word_data(word):
    """단어의 첫 번째 뜻과 유의어 목록을 가져오는 함수"""
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    definition, synonyms = None, []
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for meaning in data[0].get('meanings', []):
                if not definition and meaning.get('definitions'):
                    definition = meaning['definitions'][0].get('definition')
                synonyms.extend(s for s in meaning.get('synonyms', []) if s not in synonyms)
    except requests.exceptions.RequestException as e:
        st.error(f"API 요청 중 오류 발생: {e}")
        return "API 요청 중 오류 발생", []
    return definition, synonyms

def translate_to_korean(text):
    """Google Translate API를 사용하여 영어 텍스트를 한국어로 번역하는 함수"""
    if not text: return "번역할 내용 없음"
    url = "https://translation.googleapis.com/language/translate/v2"
    params = {'q': text, 'source': 'en', 'target': 'ko', 'format': 'text', 'key': GOOGLE_API_KEY}
    try:
        res = requests.post(url, params=params)
        if res.status_code == 200:
            return res.json()['data']['translations'][0]['translatedText']
        else:
            st.error(f"번역 API 오류: {res.status_code} - {res.text}")
            return "번역 실패"
    except requests.exceptions.RequestException as e:
        st.error(f"번역 API 요청 중 오류 발생: {e}")
        return "번역 API 요청 중 오류 발생"

def merge_sort(arr):
    """
    병합 정렬(Merge Sort) 알고리즘 구현.
    리스트를 재귀적으로 절반으로 나누고, 정렬된 서브리스트들을 병합합니다.
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]

    left_half = merge_sort(left_half)
    right_half = merge_sort(right_half)

    return merge(left_half, right_half)

def merge(left, right):
    """
    병합 정렬의 병합(Merge) 단계.
    두 개의 정렬된 리스트를 하나의 정렬된 리스트로 병합합니다.
    """
    merged_list = []
    left_idx, right_idx = 0, 0

    while left_idx < len(left) and right_idx < len(right):
        if left[left_idx] < right[right_idx]:
            merged_list.append(left[left_idx])
            left_idx += 1
        else:
            merged_list.append(right[right_idx])
            right_idx += 1

    # 남아있는 요소들 추가
    merged_list.extend(left[left_idx:])
    merged_list.extend(right[right_idx:])
    return merged_list

def sort_by_length(arr):
    """
    단어 길이에 따라 리스트를 정렬합니다.
    """
    return sorted(arr, key=len)

def sort_by_quiz_correct_order(all_words, correctly_answered_words_in_order):
    """
    퀴즈 맞춘 순서대로 단어를 정렬합니다.
    맞춘 단어는 맞춘 순서대로, 나머지 단어는 사전 순으로 정렬합니다.
    """
    correct_set = set(correctly_answered_words_in_order)
    
    sorted_correct = [word for word in correctly_answered_words_in_order if word in all_words]
    unanswered_words = sorted([word for word in all_words if word not in correct_set])
    
    return sorted_correct + unanswered_words


def load_new_word():
    """새 단어를 불러오고 모든 관련 상태를 초기화하는 함수"""
    # 사용 가능한 단어 목록이 비어 있으면, 모든 단어를 다시 사용 가능하게 초기화
    if not st.session_state.get('available_words') or len(st.session_state.available_words) == 0:
        st.session_state.available_words = list(st.session_state.all_words) # 전체 단어 목록을 복사
        st.session_state.used_words = [] # 사용된 단어 목록 초기화
        st.info("모든 단어를 사용했습니다! 단어 목록이 초기화됩니다.")

    # 사용 가능한 단어 중에서 랜덤으로 하나 선택
    new_word = random.choice(st.session_state.available_words)
    st.session_state.current_word = new_word
    
    # 선택된 단어를 사용 가능한 단어 목록에서 제거하고, 사용된 단어 목록에 추가
    st.session_state.available_words.remove(new_word)
    st.session_state.used_words.append(new_word) # used_words는 중복 방지 로직에만 사용됨
    
    first_def, synonyms_for_hints = get_word_data(new_word)
    st.session_state.first_def = first_def
    st.session_state.translated_def = translate_to_korean(first_def)
    
    # 힌트 제공을 위한 유의어 목록 (정답으로 인정되지 않음)
    st.session_state.synonyms_for_hints = [s.lower() for s in synonyms_for_hints if s.lower() != new_word.lower()]
    
    # 정답 단어 및 힌트 단어들의 임베딩을 미리 계산하여 저장
    words_to_embed_for_similarity = [new_word] + st.session_state.synonyms_for_hints
    st.session_state.embeddings_for_similarity = model.encode(words_to_embed_for_similarity)
    
    st.session_state.input_key = f"input_{random.randint(1, 1000000)}"
    st.session_state.answered_correctly = False
    st.session_state.last_hint = "" # 마지막 힌트 메시지 초기화

    # 새로운 단어를 로드할 때마다 Firestore에 현재 세션 데이터 저장
    if FIREBASE_AVAILABLE and st.session_state.get('firebase_initialized') and st.session_state.get('user_id') and st.session_state.user_id != "loading_user":
        save_user_session_data()

# --- Streamlit 앱 UI ---

# 앱 초기 로딩 시 Firebase 초기화 및 사용자 데이터 로드
if 'all_words' not in st.session_state:
    # Firebase 초기화 및 인증은 이미 위에서 처리됨
    # Canvas 환경에서 제공되는 app_id를 session_state에 저장
    st.session_state.app_id = globals().get('__app_id', 'default-app-id')

    if FIREBASE_AVAILABLE and st.session_state.get('firebase_initialized') and st.session_state.get('user_id') and st.session_state.user_id != "loading_user":
        load_user_session_data() # Firebase에서 데이터 로드
    else:
        # Firebase 초기화 실패 시 또는 Firebase 사용 불가 시 기본 단어 로드
        st.session_state.all_words = load_words_from_file(WORDS_FILE)
        st.session_state.available_words = list(st.session_state.all_words)
        st.session_state.used_words = []
        st.session_state.correctly_answered_words_in_order = []

if 'current_word' not in st.session_state:
    load_new_word()

# 사이드바 내비게이션
st.sidebar.title("메뉴")
page = st.sidebar.radio("페이지 선택", ["퀴즈", "단어 목록"])

# 사용자 ID 표시 (멀티 유저 앱 가이드라인 준수)
if st.session_state.get('user_id') and st.session_state.user_id != "loading_user":
    st.sidebar.markdown(f"**사용자 ID:** `{st.session_state.user_id}`")
else:
    st.sidebar.markdown(f"**사용자 ID:** `로딩 중...`")


if page == "퀴즈":
    st.title("🧠 스마트 영단어 퀴즈")
    st.caption("의미가 비슷하면 유사도 수치로 힌트를 드려요!")

    st.subheader("힌트: 다음 뜻에 해당하는 영어 단어를 맞춰보세요.")
    st.markdown(f"**영어 뜻:** `{st.session_state.first_def}`")
    st.markdown(f"→ **한글 번역:** `{st.session_state.translated_def}`")

    if not st.session_state.get('answered_correctly', False):
        user_input = st.text_input("영어 단어를 입력하세요:", key=st.session_state.input_key)
        
        # 마지막 힌트가 있다면 표시
        if st.session_state.last_hint:
            st.info(st.session_state.last_hint)
            st.session_state.last_hint = "" # 표시 후 초기화

        if st.button("정답 확인"):
            user_answer = user_input.strip().lower()
            current_word_lower = st.session_state.current_word.lower()

            if user_answer == current_word_lower:
                st.session_state.answered_correctly = True
                st.session_state.last_hint = "" # 정답 시 힌트 초기화
                # 정답 맞춘 단어 목록에 추가 (중복 방지)
                if current_word_lower not in st.session_state.correctly_answered_words_in_order:
                    st.session_state.correctly_answered_words_in_order.append(current_word_lower)
                
                # 정답 시 Firestore에 데이터 저장
                if FIREBASE_AVAILABLE and st.session_state.get('firebase_initialized') and st.session_state.get('user_id') and st.session_state.user_id != "loading_user":
                    save_user_session_data()
                st.rerun()
            else:
                if user_answer: # 입력값이 있을 때만 유사도 계산
                    embedding_user = model.encode(user_answer)
                    
                    max_similarity = 0
                    
                    # 정답 단어와의 유사도 계산
                    sim_with_main_word = util.cos_sim(st.session_state.embeddings_for_similarity[0], embedding_user).item()
                    max_similarity = sim_with_main_word

                    # 힌트용 유의어들과의 유사도 계산 (가장 높은 유사도 선택)
                    for i, syn_embedding in enumerate(st.session_state.embeddings_for_similarity[1:]):
                        sim_with_syn = util.cos_sim(syn_embedding, embedding_user).item()
                        if sim_with_syn > max_similarity:
                            max_similarity = sim_with_syn
                    
                    if max_similarity >= HINT_THRESHOLD:
                        st.session_state.last_hint = f"입력하신 단어의 의미가 정답 단어와 비슷해요! 😉 유사도: **{max_similarity:.2f}**"
                        st.warning(st.session_state.last_hint) # 즉시 표시
                    else:
                        st.error(f"틀렸어요. 다시 시도해보세요. (유사도: {max_similarity:.2f})")
                else:
                    st.error("단어를 입력해주세요.")

    else:
        st.success(f"정답입니다! 🎉 정답은 **{st.session_state.current_word}**였습니다.")
        if st.button("다음 단어"):
            load_new_word()
            st.rerun()

    if st.button("정답 공개", key="reveal_answer"):
        st.info(f"정답: **{st.session_state.current_word}**")
        if st.session_state.synonyms_for_hints:
            st.info(f"이 단어의 다른 유사 단어들 (힌트 목적으로 사용): `{', '.join(st.session_state.synonyms_for_hints)}`")

elif page == "단어 목록":
    st.title("📚 단어 목록")
    st.markdown("앱에 로드된 모든 영단어 목록입니다.")

    if st.session_state.all_words:
        st.subheader("정렬 옵션:")
        col_sort1, col_sort2, col_sort3 = st.columns(3)
        
        # 기본 정렬 상태 (사전 순)
        if 'current_sort_order' not in st.session_state:
            st.session_state.current_sort_order = "alphabetical"
            st.session_state.display_words = merge_sort(list(st.session_state.all_words))

        with col_sort1:
            if st.button("사전 순 정렬"):
                st.session_state.current_sort_order = "alphabetical"
                st.session_state.display_words = merge_sort(list(st.session_state.all_words))
        with col_sort2:
            if st.button("단어 길이 순 정렬"):
                st.session_state.current_sort_order = "length"
                st.session_state.display_words = sort_by_length(list(st.session_state.all_words))
        with col_sort3:
            if st.button("퀴즈 맞춘 순 정렬"):
                st.session_state.current_sort_order = "quiz_correct"
                st.session_state.display_words = sort_by_quiz_correct_order(
                    list(st.session_state.all_words), 
                    st.session_state.correctly_answered_words_in_order
                )
        
        st.markdown(f"---")
        st.markdown(f"**현재 정렬 방식:** {'사전 순' if st.session_state.current_sort_order == 'alphabetical' else '단어 길이 순' if st.session_state.current_sort_order == 'length' else '퀴즈 맞춘 순'}")
        st.write(st.session_state.display_words)

    else:
        st.warning("불러올 단어가 없습니다. 'words.txt' 파일을 확인해주세요.")

"""
사진 옷 변경 유틸리티
"""
import streamlit as st
import base64
import io
import os
from PIL import Image
from google import genai
from google.genai import types

def page_photo_clothing_change():
    """사진 옷 변경 페이지"""
    st.title("사진 옷 변경")
    st.markdown("**AI 의상 변경기** - Gemini 2.5 멀티모달 엔진을 사용하여 인물의 의상을 변경합니다.")
    
    # API 키 수동 입력 옵션 (디버깅/임시 사용)
    with st.expander("🔧 API 키 설정 (고급)", expanded=False):
        use_manual_key = st.checkbox("수동으로 API 키 입력", key="clothing_manual_key")
        manual_api_key = None
        if use_manual_key:
            manual_api_key = st.text_input(
                "Gemini API 키를 입력하세요",
                type="password",
                key="clothing_manual_key_input",
                help="Google AI Studio에서 생성한 API 키를 입력하세요"
            )
    
    # Gemini 클라이언트 초기화 - 수동 입력 > 환경변수 > 파일 키
    api_key = None
    key_source = None
    try:
        # 1. 수동 입력 키 우선
        if use_manual_key and manual_api_key and manual_api_key.strip():
            api_key = manual_api_key.strip()
            key_source = "수동 입력"
        else:
            # 2. 환경변수에서 확인
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if api_key:
                key_source = "환경변수"
            
            # 3. 환경변수가 없으면 파일에서 가져오기 (매번 새로 로드)
            if not api_key:
                try:
                    import importlib
                    import sys
                    # 모듈 캐시에서 제거하여 새로 로드
                    if 'utils.gemini_key' in sys.modules:
                        del sys.modules['utils.gemini_key']
                    from utils import gemini_key
                    importlib.reload(gemini_key)
                    if gemini_key.GEMINI_API_KEY and "여기에_" not in str(gemini_key.GEMINI_API_KEY):
                        api_key = gemini_key.GEMINI_API_KEY
                        key_source = "파일 (utils/gemini_key.py)"
                except (ImportError, AttributeError) as e:
                    pass
        
        if not api_key:
            raise ValueError("Gemini API 키가 설정되지 않았습니다.")
        
        # 디버깅 정보 (키의 일부만 표시)
        if key_source:
            masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
            st.info(f"🔑 사용 중인 API 키 소스: **{key_source}** (키: `{masked_key}`)")
        
        client = genai.Client(api_key=api_key)
    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg or "PERMISSION_DENIED" in error_msg or "leaked" in error_msg.lower():
            st.error("""
            ⚠️ **API 키 오류: 유출된 키로 인해 차단됨**
            
            현재 사용 중인 Gemini API 키가 유출되어 Google에서 차단되었습니다.
            새로운 API 키를 생성하여 사용해야 합니다.
            
            **해결 방법:**
            
            1. **새로운 API 키 생성:**
               - [Google AI Studio](https://aistudio.google.com/apikey)에서 새로운 API 키를 생성하세요
            
            2. **API 키 설정 방법 (선택):**
               
               **방법 A: 환경변수 사용 (권장)**
               ```powershell
               # PowerShell에서
               $env:GEMINI_API_KEY="여기에_새로운_API_키_입력"
               ```
               
               **방법 B: 파일에 저장**
               - `utils/gemini_key.py` 파일을 열고
               - `GEMINI_API_KEY = "여기에_새로운_API_키_입력"` 으로 수정
            
            3. **애플리케이션 재시작:**
               - Streamlit 앱을 재시작하세요
            """)
        else:
            st.error(f"⚠️ Gemini API 설정 오류: {e}")
            st.info("""
            **API 키 설정 방법:**
            
            1. 환경변수 설정 (권장):
               ```powershell
               $env:GEMINI_API_KEY="여기에_API_키_입력"
               ```
            
            2. 또는 `utils/gemini_key.py` 파일에 설정:
               ```python
               GEMINI_API_KEY = "여기에_API_키_입력"
               ```
            """)
        return
    
    # 파일 업로드
    uploaded_file = st.file_uploader(
        "인물 사진을 업로드하세요",
        type=['png', 'jpg', 'jpeg', 'webp'],
        key="clothing_change_upload"
    )
    
    if uploaded_file is not None:
        # 이미지 표시
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 원본 이미지")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        # 옵션 설정
        st.markdown("### 옵션 설정")
        
        mode = st.radio(
            "변경 모드",
            options=["계절별 의상 변경", "테이블 정리"],
            key="clothing_mode"
        )
        
        season = None
        also_clean_table = False
        custom_instruction = ""
        
        if mode == "계절별 의상 변경":
            season = st.selectbox(
                "계절 선택",
                options=["봄", "여름", "가을", "겨울"],
                key="season_select"
            )
            also_clean_table = st.checkbox("테이블 정리도 함께 수행", key="clean_table_check")
            custom_instruction = st.text_area(
                "추가 의상 스타일 요청 사항 (선택)",
                placeholder="예: 캐주얼한 스타일, 비즈니스 캐주얼 등",
                key="custom_instruction"
            )
        else:
            custom_instruction = st.text_area(
                "추가 공간 정리 요청 사항 (선택)",
                placeholder="예: 특정 물건만 남기기 등",
                key="custom_instruction_clean"
            )
        
        # 처리 버튼
        if st.button("의상 변경하기" if mode == "계절별 의상 변경" else "테이블 정리하기", 
                    type="primary", use_container_width=True, key="clothing_change_btn"):
            try:
                with st.spinner("이미지를 처리하는 중... (Gemini API 처리 중)"):
                    # 이미지를 bytes로 읽기
                    uploaded_file.seek(0)  # 파일 포인터 리셋
                    img_bytes = uploaded_file.read()
                    
                    # MIME 타입 결정
                    mime_type = uploaded_file.type or "image/png"
                    
                    # 프롬프트 생성
                    if mode == "계절별 의상 변경":
                        prompt = f"""이 이미지에 있는 모든 인물을 분석하세요. 각 인물의 성별과 연령대를 파악하십시오.
1. 인물들의 의상을 파악된 성별과 연령대에 어울리는 세련된 '{season}' 의상으로 변경하십시오.
"""
                        if season == "봄":
                            prompt += "- 봄: 가벼운 재킷, 파스텔 톤, 화사한 스타일."
                        elif season == "여름":
                            prompt += "- 여름: 반팔, 얇은 소재, 반바지 또는 가벼운 원피스."
                        elif season == "가을":
                            prompt += "- 가을: 트렌치 코트, 스카프, 따뜻한 지구색 톤, 니트웨어."
                        else:  # 겨울
                            prompt += "- 겨울: 패딩, 코트, 비니, 장갑, 두꺼운 레이어드."
                        
                        prompt += "\n인물의 얼굴, 포즈, 배경은 원본과 동일하게 유지하십시오."
                        
                        if custom_instruction and custom_instruction.strip():
                            prompt += f'\n추가 의상 스타일 요청 사항: "{custom_instruction}" 이 요청을 최우선으로 반영하여 의상을 디자인하십시오.'
                        
                        if also_clean_table:
                            prompt += "\n2. 추가로, 이미지 내의 테이블, 책상 등 평평한 표면을 찾아 그 위의 모든 물건과 잡동사니를 치워 깨끗하게 만드십시오. 표면의 질감은 자연스럽게 유지하십시오."
                    else:
                        prompt = """이미지 내의 모든 테이블, 책상, 선반 등 평평한 표면을 식별하십시오.
해당 표면 위에 있는 모든 물건, 잡동사니, 소품들을 제거하고 깨끗하게 비우십시오.
제거된 자리는 원래의 표면 재질(나무, 금속, 유리 등)로 자연스럽게 채우고 조명과 그림자가 이질감 없도록 처리하십시오."""
                        
                        if custom_instruction and custom_instruction.strip():
                            prompt += f'\n추가 공간 정리 요청 사항: "{custom_instruction}" 이 요청을 반영하여 공간을 편집하십시오.'
                    
                    try:
                        # 이미지를 Part 객체로 생성
                        image_part = types.Part(
                            inline_data=types.Blob(
                                mime_type=mime_type,
                                data=img_bytes
                            )
                        )
                        
                        # Gemini API 호출
                        response = client.models.generate_content(
                            model="gemini-2.5-flash-image",
                            contents=[prompt, image_part],
                            config=types.GenerateContentConfig(
                                response_modalities=["IMAGE"]
                            )
                        )
                        
                        # 응답에서 이미지 추출
                        processed_image = None
                        for part in response.parts:
                            # 방법 1: inline_data에서 직접 bytes 추출
                            if hasattr(part, 'inline_data') and part.inline_data:
                                blob = part.inline_data
                                if blob.data:
                                    # blob.data는 이미 bytes
                                    try:
                                        processed_image = Image.open(io.BytesIO(blob.data))
                                        break
                                    except Exception as e1:
                                        # base64로 인코딩되어 있을 수도 있음
                                        try:
                                            img_data = base64.b64decode(blob.data)
                                            processed_image = Image.open(io.BytesIO(img_data))
                                            break
                                        except Exception as e2:
                                            # 문자열일 수도 있음
                                            try:
                                                if isinstance(blob.data, str):
                                                    img_data = base64.b64decode(blob.data)
                                                    processed_image = Image.open(io.BytesIO(img_data))
                                                    break
                                            except Exception:
                                                pass
                            
                            # 방법 2: as_image() 메서드 사용
                            if processed_image is None and hasattr(part, 'as_image'):
                                try:
                                    img_obj = part.as_image()
                                    if img_obj:
                                        # Image 객체의 속성 확인
                                        if hasattr(img_obj, 'image_bytes'):
                                            # image_bytes 속성이 있는 경우
                                            processed_image = Image.open(io.BytesIO(img_obj.image_bytes))
                                            break
                                        elif hasattr(img_obj, 'to_pil'):
                                            # to_pil() 메서드가 있는 경우
                                            processed_image = img_obj.to_pil()
                                            break
                                        elif hasattr(img_obj, 'to_bytes'):
                                            # to_bytes() 메서드가 있는 경우
                                            img_bytes_processed = img_obj.to_bytes()
                                            processed_image = Image.open(io.BytesIO(img_bytes_processed))
                                            break
                                except Exception as e:
                                    pass
                        
                        if processed_image:
                            st.success("✅ 이미지 처리 완료!")
                            
                            with col2:
                                st.markdown("### 처리된 이미지")
                                st.image(processed_image, use_container_width=True)
                                
                                # 다운로드 버튼
                                img_buffer = io.BytesIO()
                                processed_image.save(img_buffer, format='PNG')
                                img_buffer.seek(0)
                                
                                st.download_button(
                                    "처리된 이미지 다운로드 (PNG)",
                                    data=img_buffer.getvalue(),
                                    file_name=f"processed_{uploaded_file.name.split('.')[0]}.png",
                                    mime="image/png"
                                )
                        else:
                            st.warning("⚠️ 이미지 처리는 완료되었지만 결과 이미지를 추출할 수 없습니다.")
                            # 디버깅 정보
                            with st.expander("응답 디버깅 정보"):
                                st.write(f"Response parts count: {len(response.parts)}")
                                for i, part in enumerate(response.parts):
                                    st.write(f"Part {i}: {type(part)}")
                                    st.write(f"  Attributes: {dir(part)}")
                            
                    except Exception as api_error:
                        error_str = str(api_error)
                        error_lower = error_str.lower()
                        
                        # 403 오류 또는 유출된 키 오류 체크
                        if "403" in error_str or "permission_denied" in error_lower or "leaked" in error_lower:
                            st.error("""
                            ⚠️ **API 키 오류: 유출된 키로 인해 차단됨**
                            
                            현재 사용 중인 Gemini API 키가 유출되어 Google에서 차단되었습니다.
                            새로운 API 키를 생성하여 사용해야 합니다.
                            
                            **해결 방법:**
                            
                            1. **새로운 API 키 생성:**
                               - [Google AI Studio](https://aistudio.google.com/apikey)에서 새로운 API 키를 생성하세요
                            
                            2. **API 키 설정 방법:**
                               
                               **방법 A: 환경변수 사용 (권장)**
                               ```powershell
                               # PowerShell에서
                               $env:GEMINI_API_KEY="여기에_새로운_API_키_입력"
                               ```
                               
                               **방법 B: 파일에 저장**
                               - `utils/gemini_key.py` 파일을 열고
                               - `GEMINI_API_KEY = "여기에_새로운_API_키_입력"` 으로 수정
                            
                            3. **애플리케이션 재시작:**
                               - Streamlit 앱을 재시작하세요
                            """)
                        else:
                            st.error(f"API 호출 오류: {api_error}")
                            import traceback
                            with st.expander("상세 오류 정보"):
                                st.code(traceback.format_exc())
                            st.info("""
                            💡 **참고사항:**
                            - Gemini API의 이미지 편집 기능은 `gemini-2.5-flash-image` 모델에서 지원됩니다.
                            - 이미지 크기가 너무 크면 오류가 발생할 수 있습니다. 이미지를 리사이즈해보세요.
                            - API 키가 올바르게 설정되어 있는지 확인하세요.
                            """)
                        
            except Exception as e:
                st.error(f"처리 중 오류 발생: {e}")
                import traceback
                with st.expander("상세 오류 정보"):
                    st.code(traceback.format_exc())
    else:
        st.info("👆 위에서 인물 사진을 업로드해주세요.")
        st.markdown("""
        ### 사용 방법
        1. 인물이 포함된 이미지를 업로드하세요
        2. 변경 모드와 옵션을 선택하세요
        3. "의상 변경하기" 또는 "테이블 정리하기" 버튼을 클릭하세요
        4. 처리된 이미지를 다운로드하세요
        
        ### 지원 기능
        - **계절별 의상 변경**: 봄/여름/가을/겨울 의상으로 변경
        - **테이블 정리**: 이미지 내 테이블 위의 물건 제거
        
        ### 지원 형식
        - PNG (권장)
        - JPG/JPEG
        - WEBP
        
        ### 주의사항
        - Gemini API 키가 설정되어 있어야 합니다
        - 처리 시간은 이미지 크기와 복잡도에 따라 다를 수 있습니다
        - 인물이 명확하게 보이는 사진일수록 결과가 좋습니다
        """)

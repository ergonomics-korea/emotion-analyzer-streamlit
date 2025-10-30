import streamlit as st
import tempfile, os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analyzer.core import extract_metrics_per_frame, auto_reference_from_metrics, score_frames, to_excel_bytes

st.set_page_config(page_title="Emotion Analyzer (Video)", page_icon="🎥", layout="wide")

st.title("🎥 Emotion/Quality Analyzer (Streamlit)")
st.caption("모바일 브라우저에서도 동작하는 웹앱 · 비디오 업로드 → 자동 분석 · 엑셀 다운로드 지원")

with st.sidebar:
    st.header("설정")
    start_sec = st.number_input("분석 시작 시점(초)", min_value=0.0, max_value=999.0, value=3.0, step=0.5)
    max_frames = st.number_input("분석 프레임 수", min_value=5, max_value=500, value=30, step=5)
    zone_size = st.slider("상/하단 구역 비율", min_value=0.1, max_value=0.45, value=0.30, step=0.05)
    st.markdown("---")
    st.info("참고: 고급 ROI 세그멘테이션이 필요하면 core.py의 로직을 원본 코드로 교체하세요.")

video_file = st.file_uploader("분석할 비디오(mp4, mov 등)를 업로드하세요", type=["mp4","mov","mkv","avi"])

if video_file is not None:
    # 업로드 파일을 임시 경로로 저장 후 OpenCV로 처리
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name

    # 메타 정보 표시
    st.subheader("메타 정보")
    meta_info = {}
    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    secs = frames / fps if fps > 0 else 0
    cap.release()
    meta_info = {"FPS": round(float(fps),2), "전체 프레임": frames, "길이(초)": round(float(secs),2)}
    st.json(meta_info)

    # 원본 미리보기(앞쪽 프레임)
    st.subheader("미리보기")
    st.video(video_file)

    st.subheader("분석 실행")
    run = st.button("분석 시작")
    if run:
        with st.spinner("프레임에서 지표 추출 중..."):
            df_raw = extract_metrics_per_frame(tmp_path, start_sec=float(start_sec), max_frames=int(max_frames), mask_bool=None, zone_size=float(zone_size))

        st.success("지표 추출 완료")
        st.dataframe(df_raw.head(10), use_container_width=True)

        with st.spinner("참조 구간 자동 산출 & 스코어링..."):
            feats = auto_reference_from_metrics(df_raw)
            df_score = score_frames(df_raw, feats)

        st.subheader("점수 표")
        st.dataframe(df_score, use_container_width=True)

        st.subheader("차트")
        fig, ax = plt.subplots(figsize=(10,4))
        for col in ["매끈한","깨끗한","일정한","안정적인","고급스러운","영역준수","상단품질","하단품질","총점(평균)"]:
            if col in df_score.columns:
                ax.plot(df_score.index, df_score[col], marker='o', label=col)
        ax.set_title("프레임별 점수")
        ax.set_xlabel("프레임 인덱스")
        ax.set_ylabel("점수(%)")
        ax.grid(True)
        ax.legend(ncols=3, fontsize=9)
        st.pyplot(fig, use_container_width=True)

        # 엑셀 다운로드
        xls = to_excel_bytes(df_score)
        st.download_button(
            label="엑셀로 다운로드 (xlsx)",
            data=xls,
            file_name="analysis_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # 정리
    @st.cache_resource
    def _cleanup(p):
        return p
    _cleanup(tmp_path)

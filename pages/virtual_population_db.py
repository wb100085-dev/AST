"""
가상인구 DB 페이지
2차 대입결과 누적 관리 및 페르소나 생성
- core/ 모듈만 참조 (app 참조 제거로 순환 참조 방지)
"""
import json
import datetime
import streamlit as st
import os
import time
from io import BytesIO

from core.constants import SIDO_LABEL_TO_CODE, SIDO_CODE_TO_NAME, SIDO_MASTER, SIDO_TOTAL_POP
from core.db import (
    get_sido_vdb_stats,
    delete_virtual_population_db_record,
    delete_virtual_population_db_by_sido,
    list_virtual_population_db_records_all,
    list_virtual_population_db_records,
    list_virtual_population_db_data_jsons,
    list_virtual_population_db_metadata,
    get_virtual_population_db_combined_df,
    get_virtual_population_db_data_json,
    update_virtual_population_db_data_json,
    update_virtual_population_db_record_merged,
    update_virtual_population_db_record_persona_reflection_only,
    find_virtual_population_db_id,
    insert_virtual_population_db,
    insert_virtual_population_db_chunked,
    VPD_CHUNK_SIZE,
)
from core.session_cache import get_sido_master
from utils.step2_records import list_step2_records
from utils.gemini_client import GeminiClient


def _vdb_update_cached_record(sido_code: str, record_id: int, new_data_json: str) -> None:
    """캐시된 (id, data_json) 목록에서 해당 record_id의 data_json만 갱신 (Supabase 실시간 반영 대신 메모리만)."""
    key = f"vdb_cached_jsons_{sido_code}"
    cached = st.session_state.get(key) or []
    new_list = [(rid, new_data_json if rid == record_id else js) for (rid, js) in cached]
    st.session_state[key] = new_list


def _vdb_update_cached_records_batch(sido_code: str, record_dfs: dict) -> None:
    """여러 record의 df를 캐시에 일괄 반영. record_dfs: {record_id: DataFrame}."""
    key = f"vdb_cached_jsons_{sido_code}"
    cached = st.session_state.get(key) or []
    new_list = [
        (rid, record_dfs[rid].to_json(orient="records", force_ascii=False)) if rid in record_dfs else (rid, js)
        for (rid, js) in cached
    ]
    st.session_state[key] = new_list


@st.cache_data(ttl=60)
def _cached_sido_master():
    """시도 마스터 목록 캐싱"""
    return get_sido_master()


def _find_koreamap_svg() -> str | None:
    """koreamap.svg 파일 경로 반환. 프로젝트 루트 또는 cwd 기준으로 탐색."""
    # 1) pages/virtual_population_db.py 기준 프로젝트 루트
    _app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(_app_root, "koreamap.svg")
    if os.path.isfile(p):
        return p
    # 2) Streamlit Cloud 등: 현재 작업 디렉터리
    p = os.path.join(os.getcwd(), "koreamap.svg")
    if os.path.isfile(p):
        return p
    return None


@st.cache_data(ttl=3600)
def _cached_koreamap_svg_content() -> str | None:
    """koreamap.svg 내용 캐시 (지도 재렌더 시 파일 반복 읽기 방지)."""
    svg_path = _find_koreamap_svg()
    if not svg_path:
        return None
    with open(svg_path, "r", encoding="utf-8") as f:
        return f.read()


# 한국 시도 GeoJSON URL (WGS84, properties.code = 시도코드 문자열)
KOREA_SIDO_GEOJSON_URL = "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2013/json/skorea_provinces_geo_simple.json"


@st.cache_data(ttl=3600)
def _load_korea_sido_geojson():
    """한국 시도 GeoJSON 로드. properties.code를 모두 문자열(str)로 통일해 DataFrame과 매칭 보장."""
    import urllib.request
    try:
        with urllib.request.urlopen(KOREA_SIDO_GEOJSON_URL, timeout=15) as resp:
            geojson = json.loads(resp.read().decode())
    except Exception:
        return None
    if not geojson or geojson.get("type") != "FeatureCollection":
        return None
    features = list(geojson.get("features") or [])
    for f in features:
        prop = f.get("properties") or {}
        code = prop.get("code")
        if code is not None:
            prop["code"] = str(code).strip()
        if "name" not in prop and "name_eng" in prop:
            prop["name"] = prop["name_eng"]
    return geojson


def _build_korea_choropleth_figure(selected_sido_code: str, region_stats: dict):
    """시도별 Choropleth Map. GeoJSON과 DataFrame locations를 모두 문자열로 맞춰 폴리곤이 정상 렌더되도록 함."""
    import plotly.graph_objects as go
    geojson = _load_korea_sido_geojson()
    if not geojson:
        st.caption("지도 데이터(GeoJSON)를 불러올 수 없습니다.")
        return go.Figure()
    sidos_for_map = [s for s in SIDO_MASTER if s["sido_code"] != "00"]
    locations = []
    z_vals = []
    hover_texts = []
    customdata_list = []
    for s in sidos_for_map:
        code = str(s["sido_code"]).strip()
        name = s["sido_name"]
        vdb = (region_stats.get(code) or {}).get("vdb_count") or 0
        locations.append(code)
        z_vals.append(2 if code == selected_sido_code else 1)
        hover_texts.append(f"{name}<br>가상인구 DB: {vdb:,}명")
        customdata_list.append([code])
    fig = go.Figure(
        go.Choroplethmapbox(
            geojson=geojson,
            featureidkey="properties.code",
            locations=locations,
            z=z_vals,
            text=hover_texts,
            hoverinfo="text",
            customdata=customdata_list,
            colorscale=[[0, "#e5e7eb"], [0.5, "#9ca3af"], [1, "#dc2626"]],
            zmin=1,
            zmax=2,
            showscale=False,
            marker_line_width=1.2,
            marker_line_color="white",
        )
    )
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=36.0, lon=127.8),
            zoom=5.8,
            bounds={"west": 124, "east": 132, "south": 33, "north": 39},
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _render_korea_map(selected_sido_code: str, region_stats: dict):
    """Plotly Choropleth 스타일 지도. 클릭 시 on_select='rerun' → 세션(vdb_map)에 선택 저장 → fragment 상단에서 vdb_sido_select와 동기화."""
    fig = _build_korea_choropleth_figure(selected_sido_code, region_stats)
    st.plotly_chart(
        fig,
        key="vdb_map",
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
    )


def page_virtual_population_db():
    """가상인구 DB: 2차 대입결과 누적 관리"""
    import pandas as pd  # 지연 로딩: 페이지 진입 시에만 로드
    st.header("가상인구 DB")
    st.markdown("---")

    @st.fragment
    def _vdb_frag():
        # ---------- 양방향 바인딩: 단일 세션 변수(vdb_sido_select)로 지도·드롭다운 동기화 ----------
        if "vdb_sido_select" not in st.session_state:
            st.session_state["vdb_sido_select"] = "경상북도 (37)"
        query_sido = st.query_params.get("sido_code")
        if query_sido:
            for s in SIDO_MASTER:
                if s["sido_code"] == query_sido:
                    label = f"{s['sido_name']} ({s['sido_code']})"
                    st.session_state["selected_sido_label"] = label
                    st.session_state["vdb_sido_select"] = label
                    break
        # 지도 클릭(Plotly on_select) 시 세션에 저장된 선택을 셀렉트박스에 반영
        # 배포 환경(Streamlit Cloud 등)에서는 selection이 dict/camelCase로 올 수 있음
        map_state = st.session_state.get("vdb_map")
        sel = None
        if map_state is not None:
            sel = map_state.get("selection") if isinstance(map_state, dict) else getattr(map_state, "selection", None)
        pts = []
        if sel is not None:
            pts = sel.get("points", []) if isinstance(sel, dict) else (getattr(sel, "points", None) or [])
        # 일부 환경에서는 selection이 locations 리스트로만 옴
        if not pts and isinstance(sel, dict) and "locations" in sel:
            locs = sel.get("locations") or []
            if locs:
                pts = [{"location": loc} if isinstance(loc, (str, int, float)) else loc for loc in locs]
        if not pts and isinstance(sel, (list, tuple)) and len(sel) > 0:
            first = sel[0]
            if isinstance(first, (str, int, float)):
                pts = [{"location": first}]
        if pts:
            p0 = pts[0]
            p0_dict = p0 if isinstance(p0, dict) else (getattr(p0, "__dict__", None) or {})
            # customdata / customData (camelCase), location, point_index / pointIndex
            cd = p0_dict.get("customdata") or p0_dict.get("customData")
            loc_id = p0_dict.get("location")
            _pi, _pj = p0_dict.get("point_index"), p0_dict.get("pointIndex")
            loc_idx = _pi if _pi is not None else _pj
            code = None
            if cd and (isinstance(cd, (list, tuple)) and len(cd) > 0):
                code = str(cd[0])
            elif isinstance(cd, (str, int, float)):
                code = str(cd)
            if not code and loc_id is not None:
                code = str(loc_id)
            if not code and loc_idx is not None:
                sidos_ordered = [s["sido_code"] for s in SIDO_MASTER if s["sido_code"] != "00"]
                if 0 <= loc_idx < len(sidos_ordered):
                    code = str(sidos_ordered[loc_idx])
            if code:
                label = f"{SIDO_CODE_TO_NAME.get(code, '')} ({code})"
                st.session_state["vdb_sido_select"] = label
                st.session_state["selected_sido_label"] = label

        # 시도별 통계 (가상인구 DB 수; core.db get_sido_vdb_stats 캐싱 활용)
        try:
            vdb_stats = get_sido_vdb_stats()
        except Exception as e:
            vdb_stats = {}
            err_msg = str(e)
            if "521" in err_msg or "Web server is down" in err_msg or "JSON could not be generated" in err_msg:
                st.warning("Supabase 서버에 일시적으로 연결할 수 없습니다. 잠시 후 다시 시도해주세요.")
            else:
                st.warning(f"시도별 통계를 불러오지 못했습니다: {err_msg[:200]}")
        region_stats = {}
        for s in SIDO_MASTER:
            if s["sido_code"] == "00":
                continue
            code = s["sido_code"]
            region_stats[code] = {
                "sido_name": s["sido_name"],
                "vdb_count": vdb_stats.get(code, {}).get("vdb_count", 0),
                "total_pop": SIDO_TOTAL_POP.get(code),
            }

        # 공유 상태: 쿼리 또는 세션(vdb_sido_select)에서 결정된 현재 선택 지역
        current_label = st.session_state.get("vdb_sido_select") or st.session_state.get("selected_sido_label", "경상북도 (37)")
        map_sido_code = query_sido or SIDO_LABEL_TO_CODE.get(current_label, "37")
        sido_master = _cached_sido_master()
        sido_options = [f"{s['sido_name']} ({s['sido_code']})" for s in sido_master]
        default_label = st.session_state.get("vdb_sido_select") or st.session_state.get("selected_sido_label", "경상북도 (37)")
        default_idx = sido_options.index(default_label) if default_label in sido_options else 0

        # 상단 2분할: 왼쪽(5) 지도+선택정보(세로 8:2), 오른쪽(5) 가상인구 DB
        col_left, col_db = st.columns([5, 5])
        with col_left:
            # 셀렉트박스: key로 vdb_sido_select와 연동, 지도 클릭 시 위에서 세션 갱신되어 동기화
            selected_sido_label = st.selectbox(
                "지역 선택",
                options=sido_options,
                index=default_idx,
                key="vdb_sido_select",
            )
            map_sido_code = SIDO_LABEL_TO_CODE.get(selected_sido_label, map_sido_code)
            map_container = st.container()
            with map_container:
                _render_korea_map(map_sido_code, region_stats)
            st.subheader("선택한 지역 정보")
            display_code = SIDO_LABEL_TO_CODE.get(selected_sido_label, map_sido_code)
            if display_code and region_stats.get(display_code):
                r = region_stats[display_code]
                st.markdown(f"**{r['sido_name']}**")
                _tp = r.get('total_pop')
                st.caption(f"총인구: {f'{_tp:,}명' if isinstance(_tp, (int, float)) else (_tp or '—')}")
                st.caption(f"가상인구 DB: {(r.get('vdb_count') or 0):,}명")
            else:
                st.caption("지도를 클릭하거나 지역 선택에서 선택하세요.")
        selected_sido_code = SIDO_LABEL_TO_CODE.get(selected_sido_label, "37")
        selected_sido_name = SIDO_CODE_TO_NAME.get(selected_sido_code, "경상북도")
        # 드롭다운 변경 시에도 동일 상태로 유지 (양방향 바인딩)
        st.session_state["selected_sido_label"] = selected_sido_label
        st.session_state["selected_sido_code"] = selected_sido_code

        # 상단 오른쪽(5) 열: 가상인구 DB (지도 옆에 표·다운로드·버튼)
        with col_db:
            st.subheader("가상인구 DB")
            # 먼저 경량 메타만 조회 (data_json 제외 → timeout 방지)
            meta_list = list_virtual_population_db_metadata(selected_sido_code)
            load_full_key = f"vdb_full_data_loaded_{selected_sido_code}"
            user_requested_full = st.session_state.get(load_full_key, False)

            if not meta_list:
                st.info("가상인구 DB에 데이터가 없습니다. 아래 2차 대입결과에서 선택하여 추가해주세요.")
                combined_df = None
                all_dfs = []
                persona_count = 0
                total_count = 0
                db_records = []
            elif not user_requested_full:
                # 전체 데이터는 불러오지 않음. 버튼을 누르면 그때 Supabase에서 로드
                st.caption(f"총 **{len(meta_list)}건**의 기록이 있습니다. 표·다운로드를 보려면 아래 버튼을 눌러주세요.")
                if st.button("전체 데이터 불러오기 (표·다운로드)", type="primary", key="vdb_load_full_btn"):
                    st.session_state[load_full_key] = True
                    st.rerun()
                combined_df = None
                all_dfs = []
                persona_count = 0
                total_count = 0
                db_records = []
            else:
                # 사용자가 "전체 데이터 불러오기"를 눌렀을 때만 data_json 조회 (페이지 단위로 수행)
                cache_key = f"vdb_cached_jsons_{selected_sido_code}"
                db_records = st.session_state.get(cache_key)
                if db_records is None:
                    try:
                        with st.spinner("Supabase에서 전체 데이터를 불러오는 중입니다… (데이터가 많으면 1~2분 걸릴 수 있습니다)"):
                            db_records = list_virtual_population_db_data_jsons(selected_sido_code)
                        st.session_state[cache_key] = db_records
                    except Exception as e:
                        if "timeout" in str(e).lower() or "57014" in str(e):
                            st.error("Supabase 조회 시간이 초과되었습니다. 데이터가 많을 경우 **DB 저장**은 소량씩 나눠 진행하거나, Supabase 대시보드에서 statement timeout을 늘려보세요.")
                        else:
                            st.error(f"데이터 불러오기 실패: {e}")
                        db_records = []
                all_dfs = []
                for _record_id, data_json in db_records:
                    try:
                        df = pd.read_json(data_json, orient="records")
                        if "식별NO" in df.columns:
                            df = df.drop(columns=["식별NO"])
                        if "페르소나" not in df.columns:
                            df["페르소나"] = ""
                        if "현시대 반영" not in df.columns:
                            df["현시대 반영"] = ""
                        all_dfs.append(df)
                    except Exception as e:
                        st.warning(f"데이터 로드 실패: {e}")
                        continue
                if not all_dfs:
                    combined_df = None
                    persona_count = 0
                    total_count = 0
                else:
                    combined_df = pd.concat(all_dfs, ignore_index=True)
                    if "페르소나" in combined_df.columns:
                        combined_df["페르소나"] = combined_df["페르소나"].fillna("").astype(str).replace("nan", "")
                    if "현시대 반영" in combined_df.columns:
                        combined_df["현시대 반영"] = combined_df["현시대 반영"].fillna("").astype(str).replace("nan", "")
                    if "미래안정성(10점 만점)" in combined_df.columns:
                        original_cols = list(combined_df.columns)
                        future_idx = original_cols.index("미래안정성(10점 만점)")
                        before_cols = [c for c in original_cols[:future_idx] if c not in ["페르소나", "현시대 반영"]]
                        after_cols = [c for c in original_cols[future_idx+1:] if c not in ["페르소나", "현시대 반영"]]
                        new_cols = before_cols + ["미래안정성(10점 만점)", "페르소나", "현시대 반영"] + after_cols
                        combined_df = combined_df[new_cols]
                    total_count = len(combined_df)
                    persona_count = combined_df["페르소나"].apply(lambda x: str(x).strip() != "" and str(x).strip() != "nan").sum() if "페르소나" in combined_df.columns else 0

            if db_records and all_dfs and combined_df is not None:
                st.caption(f"총 {total_count:,}명의 데이터 (모든 기록 누적)")
                st.caption(f"페르소나 생성 완료: {persona_count:,}명 / 전체 {total_count:,}명")
                # 페르소나·현시대 반영은 메모리(캐시)만 반영됨. Supabase에 저장하려면 아래 버튼 클릭
                cache_key_save = f"vdb_cached_jsons_{selected_sido_code}"
                if st.button("기록을 DB에 추가", type="primary", key="vdb_sync_to_supabase"):
                    to_sync = st.session_state.get(cache_key_save) or []
                    if not to_sync:
                        st.warning("반영할 캐시 데이터가 없습니다.")
                    else:
                        sync_bar = None
                        try:
                            sync_bar = st.progress(0.0, text="Supabase에 저장 중…")
                            persona_only_ok = 0
                            for i, (rid, data_json) in enumerate(to_sync):
                                if update_virtual_population_db_record_persona_reflection_only(rid, data_json):
                                    persona_only_ok += 1
                                else:
                                    update_virtual_population_db_record_merged(rid, data_json)
                                sync_bar.progress((i + 1) / len(to_sync), text=f"저장 중… {i+1}/{len(to_sync)}")
                            sync_bar.empty()
                            st.success(f"총 {len(to_sync)}건의 기록을 Supabase에 반영했습니다.")
                        except Exception as e:
                            if sync_bar is not None:
                                try:
                                    sync_bar.empty()
                                except Exception:
                                    pass
                            err_msg = str(e)
                            if "521" in err_msg or "Web server is down" in err_msg or "JSON could not be generated" in err_msg:
                                st.error("Supabase 서버가 일시적으로 응답하지 않습니다. 잠시 후 **기록을 DB에 추가**를 다시 눌러 주세요.")
                            else:
                                st.error(f"DB 반영 중 오류: {err_msg[:300]}")
                    st.rerun()

            # 표시: 미리보기 100명만 표시 (전체 표시 시 인원 많을 때 느려짐), 다운로드는 전체
            if combined_df is not None:
                table_container = st.container()
                with table_container:
                    preview_limit = 100
                    preview_df = combined_df.head(preview_limit)
                    st.caption(f"미리보기 (상위 {preview_limit}명) — 전체 {total_count:,}명 중 표시. Excel/CSV 다운로드는 전체 데이터 포함.")
                    st.dataframe(preview_df, use_container_width=True, height=600)
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        out_buffer = BytesIO()
                        combined_df.to_excel(out_buffer, index=False, engine="openpyxl")
                        out_buffer.seek(0)
                        st.download_button(
                            "Excel 다운로드",
                            data=out_buffer.getvalue(),
                            file_name=f"{selected_sido_name}_virtual_population_db.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="vdb_dl_excel"
                        )
                    with col_dl2:
                        csv = combined_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "CSV 다운로드",
                            data=csv,
                            file_name=f"{selected_sido_name}_virtual_population_db.csv",
                            mime="text/csv",
                            key="vdb_dl_csv"
                        )

                # 초기화·페르소나 버튼 (표 아래 배치)
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("초기화", type="secondary", use_container_width=True, key="vdb_reset"):
                        if delete_virtual_population_db_by_sido(selected_sido_code):
                            list_virtual_population_db_records.clear()
                            get_virtual_population_db_combined_df.clear()
                            get_sido_vdb_stats.clear()
                            st.session_state.vdb_selected_records = []
                            st.session_state.pop(f"vdb_full_data_loaded_{selected_sido_code}", None)
                            st.session_state.pop(f"vdb_cached_jsons_{selected_sido_code}", None)
                            st.success(f"{selected_sido_name} 지역의 가상인구 DB가 초기화되었습니다.")
                        else:
                            st.error("초기화 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
                        st.rerun()

                with col_btn2:
                    if "persona_generation_stop" not in st.session_state:
                        st.session_state.persona_generation_stop = False
                    if "persona_generation_running" not in st.session_state:
                        st.session_state.persona_generation_running = False
                    if not st.session_state.persona_generation_running:
                        if st.button("페르소나", type="primary", use_container_width=True, key="vdb_persona"):
                            st.session_state.persona_generation_running = True
                            st.session_state.persona_generation_stop = False
                            st.session_state.persona_work_queue = None  # 다음 run에서 구축
                            st.rerun()
                    else:
                        if st.button("중지", type="secondary", use_container_width=True, key="vdb_persona_stop"):
                            st.session_state.persona_generation_stop = True
                            st.rerun()
                    # 중지 클릭 후: 즉시 멈추고 지금까지 반영된 건수 안내
                    if st.session_state.persona_generation_running and st.session_state.persona_generation_stop:
                        processed = st.session_state.get("persona_next_index", 0)
                        st.session_state.persona_generation_running = False
                        st.session_state.persona_generation_stop = False
                        st.session_state.pop("persona_work_queue", None)
                        st.session_state.pop("persona_next_index", None)
                        st.session_state.pop("persona_total", None)
                        st.session_state.pop("persona_sido_code", None)
                        st.warning(f"페르소나 생성을 중지했습니다. 지금까지 생성된 {processed}명의 페르소나가 가상인구 DB에 반영되어 있습니다.")
                        st.rerun()
                    # 페르소나 생성 로직: 배치마다 DB 저장 후 rerun 해서 중지 버튼이 즉시 반영되도록 함
                    if st.session_state.persona_generation_running and not st.session_state.persona_generation_stop:
                        work_queue = st.session_state.get("persona_work_queue")
                        next_index = st.session_state.get("persona_next_index", 0)
                        total_work = st.session_state.get("persona_total", 0)
                        if work_queue is None:
                            pgr_rows = st.session_state.get(f"vdb_cached_jsons_{selected_sido_code}")
                            if not pgr_rows:
                                st.warning("가상인구 DB 전체 데이터가 없습니다. 상단에서 **전체 데이터 불러오기**를 누른 뒤 다시 시도해주세요.")
                                st.session_state.persona_generation_running = False
                            else:
                                work_queue = []
                                for record_id, data_json in pgr_rows:
                                    try:
                                        df = pd.read_json(data_json, orient="records")
                                        if "페르소나" not in df.columns:
                                            df["페르소나"] = ""
                                        empty_mask = (df["페르소나"].isna()) | (df["페르소나"].astype(str).str.strip() == "")
                                        for idx in df.index[empty_mask].tolist():
                                            row_data = df.loc[idx]
                                            persona_info = {k: row_data.get(k) for k in ["거주지역", "성별", "연령", "경제활동", "교육정도", "월평균소득"] if k in row_data and pd.notna(row_data.get(k))}
                                            work_queue.append({"record_id": record_id, "row_idx": int(idx), "persona_info": persona_info})
                                    except Exception:
                                        continue
                                if not work_queue:
                                    st.info("모든 가상인물의 페르소나가 이미 생성되어 있습니다.")
                                    st.session_state.persona_generation_running = False
                                else:
                                    st.session_state.persona_work_queue = work_queue
                                    st.session_state.persona_next_index = 0
                                    st.session_state.persona_total = len(work_queue)
                                    st.session_state.persona_sido_code = selected_sido_code
                                    st.rerun()
                        else:
                            try:
                                from utils.gemini_key import GEMINI_API_KEY
                                api_ok = GEMINI_API_KEY and "여기에_" not in (GEMINI_API_KEY or "")
                                if not api_ok:
                                    api_ok = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
                                if not api_ok and hasattr(st, 'secrets') and st.secrets.get("GEMINI_API_KEY"):
                                    api_ok = True
                                if not api_ok:
                                    st.error("⚠️ API 키가 설정되지 않았습니다. .env 또는 Streamlit Secrets에 GEMINI_API_KEY를 설정해주세요.")
                                    st.session_state.persona_generation_running = False
                                    st.session_state.persona_work_queue = None
                                    st.rerun()
                                else:
                                    gemini_client = GeminiClient()
                                    batch_size = 1  # 1명씩 처리 후 rerun → 중지 버튼 클릭 시 최대 1건만 더 진행되고 즉시 중단
                                    batch_end = min(next_index + batch_size, total_work)
                                    progress_bar = st.progress(next_index / total_work if total_work else 0)
                                    status_placeholder = st.empty()
                                    cache_key = f"vdb_cached_jsons_{selected_sido_code}"
                                    id_to_json = {rid: js for (rid, js) in (st.session_state.get(cache_key) or [])}
                                    record_dfs = {}
                                    for i in range(next_index, batch_end):
                                        item = work_queue[i]
                                        persona_info = item["persona_info"]
                                        prompt = f"다음 가상인물의 특성 정보를 바탕으로 이 인물의 페르소나를 200자 이내 서술형으로 작성해주세요.\n특성 정보:\n" + "\n".join([f"- {k}: {v}" for k, v in persona_info.items()]) + "\n\n페르소나 (200자 이내):"
                                        try:
                                            resp = gemini_client._client.models.generate_content(model=gemini_client._model, contents=prompt)
                                            persona_text = (resp.text or "").strip()[:200]
                                        except Exception:
                                            persona_text = ""
                                        rid, row_idx = item["record_id"], item["row_idx"]
                                        if rid not in record_dfs:
                                            data_json = id_to_json.get(rid) or get_virtual_population_db_data_json(rid)
                                            if data_json:
                                                df = pd.read_json(data_json, orient="records")
                                                if "페르소나" not in df.columns:
                                                    df["페르소나"] = ""
                                                record_dfs[rid] = df
                                        if rid in record_dfs:
                                            record_dfs[rid].at[row_idx, "페르소나"] = persona_text
                                        status_placeholder.caption(f"처리 중: {batch_end}/{total_work}명")
                                        progress_bar.progress(batch_end / total_work)
                                    # 실시간 Supabase 업로드 대신 캐시만 갱신 → 나중에 '기록을 DB에 추가'로 반영
                                    _vdb_update_cached_records_batch(selected_sido_code, record_dfs)
                                    st.session_state.persona_next_index = batch_end
                                    if batch_end >= total_work:
                                        progress_bar.empty()
                                        status_placeholder.empty()
                                        st.session_state.persona_generation_running = False
                                        st.session_state.persona_generation_stop = False
                                        st.session_state.pop("persona_work_queue", None)
                                        st.session_state.pop("persona_next_index", None)
                                        st.session_state.pop("persona_total", None)
                                        st.session_state.pop("persona_sido_code", None)
                                        st.success(f"페르소나 생성 완료: {total_work}명")
                                        st.rerun()
                                    else:
                                        st.rerun()
                            except Exception as e:
                                st.error(f"페르소나 생성 중 오류: {e}")
                                st.session_state.persona_generation_running = False
                                st.session_state.persona_generation_stop = False
                                st.session_state.pop("persona_work_queue", None)
                                st.session_state.pop("persona_next_index", None)
                                st.session_state.pop("persona_total", None)
                                st.session_state.pop("persona_sido_code", None)

            # 2차 대입결과 목록 (가상인구 DB 열 제일 아래, 데이터 유무와 관계없이 표시)
            with st.expander("2차 대입결과 목록 (클릭하여 목록 확인)"):
                records = list_step2_records()
                filtered_records = [r for r in records if r.get("sido_code") == selected_sido_code]
                if not filtered_records:
                    st.info(f"{selected_sido_name} 지역의 2차 대입결과가 없습니다.")
                else:
                    st.caption(f"총 {len(filtered_records)}건 (날짜·시간순)")
                    if "vdb_selected_records" not in st.session_state:
                        st.session_state.vdb_selected_records = []
                    selected_record_keys = []
                    for idx, r in enumerate(filtered_records):
                        ts = r.get("timestamp", "")
                        sido_name = r.get("sido_name", "")
                        rows = r.get("rows", 0)
                        added = r.get("added_columns", [])
                        excel_path = r.get("excel_path", "")
                        record_key = f"{ts}_{selected_sido_code}_{idx}"
                        is_selected = st.checkbox(
                            f"{ts} | {sido_name} | {rows}명 | 추가 컬럼 {len(added)}개",
                            key=f"vdb_record_{record_key}",
                            value=record_key in st.session_state.vdb_selected_records
                        )
                        if is_selected and record_key not in st.session_state.vdb_selected_records:
                            st.session_state.vdb_selected_records.append(record_key)
                        elif not is_selected and record_key in st.session_state.vdb_selected_records:
                            st.session_state.vdb_selected_records.remove(record_key)
                        if is_selected:
                            selected_record_keys.append({"key": record_key, "record": r, "excel_path": excel_path})
                    if selected_record_keys:
                        if st.button("선택한 기록을 DB에 추가", type="primary", use_container_width=True, key="vdb_add_records"):
                            added_count = 0
                            errors = []
                            for item in selected_record_keys:
                                record = item["record"]
                                excel_path = item["excel_path"]
                                if find_virtual_population_db_id(selected_sido_code, record.get("timestamp", ""), excel_path) is not None:
                                    continue
                                existing_personas = {}
                                existing_reflections = {}
                                for _eid, existing_data_json in (st.session_state.get(f"vdb_cached_jsons_{selected_sido_code}") or []):
                                    try:
                                        existing_df = pd.read_json(existing_data_json, orient="records")
                                        if "가상이름" in existing_df.columns and "페르소나" in existing_df.columns:
                                            for _, existing_person in existing_df.iterrows():
                                                name = str(existing_person.get("가상이름", "")).strip()
                                                persona = str(existing_person.get("페르소나", "")).strip()
                                                reflection = str(existing_person.get("현시대 반영", "")).strip()
                                                if name and name != "nan" and persona and persona != "nan" and persona != "":
                                                    existing_personas[name] = persona
                                                if name and name != "nan" and reflection and reflection != "nan" and reflection != "":
                                                    existing_reflections[name] = reflection
                                    except Exception:
                                        pass
                                try:
                                    df = pd.read_excel(excel_path, engine="openpyxl")
                                    if "식별NO" in df.columns:
                                        df = df.drop(columns=["식별NO"])
                                    if "미래안정성(10점 만점)" in df.columns:
                                        if "페르소나" not in df.columns:
                                            df["페르소나"] = ""
                                        if "현시대 반영" not in df.columns:
                                            df["현시대 반영"] = ""
                                        if "가상이름" in df.columns:
                                            for idx, row in df.iterrows():
                                                name = str(row.get("가상이름", "")).strip()
                                                if name and name != "nan":
                                                    if name in existing_personas:
                                                        df.at[idx, "페르소나"] = existing_personas[name]
                                                    if name in existing_reflections:
                                                        df.at[idx, "현시대 반영"] = existing_reflections[name]
                                        cols = list(df.columns)
                                        if "미래안정성(10점 만점)" in cols:
                                            idx = cols.index("미래안정성(10점 만점)")
                                            cols = [c for c in cols if c not in ["미래안정성(10점 만점)", "페르소나", "현시대 반영"]]
                                            cols.insert(idx, "미래안정성(10점 만점)")
                                            cols.insert(idx + 1, "페르소나")
                                            cols.insert(idx + 2, "현시대 반영")
                                            df = df[cols]
                                    else:
                                        if "페르소나" not in df.columns:
                                            df["페르소나"] = ""
                                        if "현시대 반영" not in df.columns:
                                            df["현시대 반영"] = ""
                                        if "가상이름" in df.columns:
                                            for idx, row in df.iterrows():
                                                name = str(row.get("가상이름", "")).strip()
                                                if name and name != "nan":
                                                    if name in existing_personas:
                                                        df.at[idx, "페르소나"] = existing_personas[name]
                                                    if name in existing_reflections:
                                                        df.at[idx, "현시대 반영"] = existing_reflections[name]
                                    data_json = df.to_json(orient="records", force_ascii=False)
                                    total_rows = len(df)
                                    if total_rows > VPD_CHUNK_SIZE:
                                        progress_container = st.container()
                                        with progress_container:
                                            progress_bar = st.progress(0.0, text="데이터 업로드 중... (0/%d)" % total_rows)
                                            def _progress(done: int, total: int):
                                                progress_bar.progress(min(1.0, done / total) if total else 1.0, text="데이터 업로드 중... (%d/%d)" % (done, total))
                                            try:
                                                new_id = insert_virtual_population_db_chunked(
                                                    selected_sido_code,
                                                    selected_sido_name,
                                                    record.get("timestamp", ""),
                                                    excel_path,
                                                    data_json,
                                                    chunk_size=VPD_CHUNK_SIZE,
                                                    progress_callback=_progress,
                                                )
                                            finally:
                                                progress_container.empty()
                                    else:
                                        new_id = insert_virtual_population_db(selected_sido_code, selected_sido_name, record.get("timestamp", ""), excel_path, data_json)
                                    if new_id is not None:
                                        added_count += 1
                                except FileNotFoundError as e:
                                    err_msg = f"Excel 파일 없음: {os.path.basename(excel_path)} (로컬 경로만 가능, 배포 환경에서는 2차 대입 결과가 서버에 없을 수 있음)"
                                    errors.append(err_msg)
                                    st.warning(err_msg)
                                except Exception as e:
                                    err_msg = f"{os.path.basename(excel_path)}: {e}"
                                    errors.append(err_msg)
                                    st.warning(f"기록 추가 실패: {err_msg}")
                                    if "row-level security" in str(e).lower() or "rls" in str(e).lower() or "policy" in str(e).lower():
                                        st.info("💡 Supabase **virtual_population_db** 테이블에 anon 역할 INSERT 정책이 있는지 확인하세요. (docs/SUPABASE_RLS_정책_적용.sql 참고)")
                            if added_count > 0:
                                st.success(f"{added_count}개의 기록이 가상인구 DB에 추가되었습니다.")
                                st.session_state.vdb_selected_records = []
                                st.rerun()
                            elif errors:
                                st.error("추가 실패 요약")
                                for err in errors[:5]:
                                    st.caption(f"• {err}")
                                if len(errors) > 5:
                                    st.caption(f"… 외 {len(errors)-5}건")
                            else:
                                st.info("추가할 새로운 기록이 없습니다. (이미 추가된 기록은 제외됩니다)")

                # DB에 저장된 2차 대입결과 목록 — 삭제 기능 (메타만 조회, data_json 미포함 → timeout 방지)
                st.markdown("---")
                st.markdown("**DB에 저장된 2차 대입결과 (삭제)**")
                db_records = list_virtual_population_db_metadata(selected_sido_code)
                if not db_records:
                    st.caption("DB에 저장된 기록이 없습니다.")
                else:
                    st.caption("삭제할 기록의 [삭제] 버튼을 누르면 DB에서도 삭제됩니다.")
                    for rec in db_records:
                        rid = rec["id"]
                        ts = rec.get("record_timestamp", "")
                        path = rec.get("record_excel_path", "")
                        path_short = os.path.basename(path) if path else "-"
                        chunk_count = rec.get("chunk_count", 0)
                        col_label, col_btn = st.columns([4, 1])
                        with col_label:
                            st.text(f"{ts} | {chunk_count}개 청크 | {path_short}")
                        with col_btn:
                            if st.button("삭제", key=f"vdb_del_{rid}", type="secondary"):
                                if delete_virtual_population_db_record(rid):
                                    try:
                                        list_virtual_population_db_records.clear()
                                    except Exception:
                                        pass
                                    try:
                                        get_virtual_population_db_combined_df.clear()
                                    except Exception:
                                        pass
                                    st.success("해당 기록이 DB에서 삭제되었습니다.")
                                    st.rerun()
                                else:
                                    st.warning("삭제에 실패했습니다.")

        st.markdown("---")
        # 지도 아래 2분할: 왼쪽 가상인구 대화, 오른쪽 채팅창
        col_dialog, col_chat = st.columns(2)
    
        with col_dialog:
            st.subheader("가상인구 대화")
            # 채팅 모드 선택 버튼
            col_chat1, col_chat2, col_chat3 = st.columns(3)
            with col_chat1:
                chat_mode_1to1 = st.button("1:1대화", key="chat_mode_1to1", use_container_width=True)
            with col_chat2:
                chat_mode_5to1 = st.button("5:1대화", key="chat_mode_5to1", use_container_width=True)
            with col_chat3:
                chat_mode_all = st.button("전체 학습", key="chat_mode_all", use_container_width=True)
        
            # 채팅 모드 설정
            if chat_mode_1to1:
                st.session_state.chat_mode = "1:1대화"
                st.session_state.selected_chat_person = None  # 선택 초기화
            elif chat_mode_5to1:
                st.session_state.chat_mode = "5:1대화"
                st.session_state.selected_chat_people = None  # 선택 초기화
            elif chat_mode_all:
                st.session_state.chat_mode = "전체 학습"
        
            if "chat_mode" not in st.session_state:
                st.session_state.chat_mode = "1:1대화"
        
            # 채팅 히스토리 초기화
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            if "chat_loading" not in st.session_state:
                st.session_state.chat_loading = False
            if "pending_chat_input" not in st.session_state:
                st.session_state.pending_chat_input = None
        
            # 현재 모드 표시
            st.caption(f"현재 모드: {st.session_state.chat_mode}")
            st.markdown("---")
        
            # 전체 학습 모드: 텍스트 + PDF 업로드 및 현시대 반영 대입
            if st.session_state.chat_mode == "전체 학습":
                st.markdown("**전체 학습: 현시대 자료 입력**")
                st.caption("텍스트를 입력하거나 PDF를 업로드하면, Gemini가 분석하여 각 가상인구의 페르소나·특성에 맞게 관심분야 가중치를 적용해 '현시대 반영'을 100자 내외로 생성·대입합니다. 반영 필드는 최대 200자까지 저장되며, 생성은 100자 내외로 해 짤림 없이 저장됩니다.")
            
                col_learn1, col_learn2 = st.columns([1, 1])
                with col_learn1:
                    learn_text = st.text_area(
                        "현시대 관련 텍스트",
                        height=120,
                        placeholder="현시대 반영에 사용할 내용을 입력하세요 (뉴스, 정책, 트렌드 등)...",
                        key="learn_text"
                    )
                with col_learn2:
                    learn_pdf = st.file_uploader("PDF 파일 업로드", type=["pdf"], key="learn_pdf")
                    st.caption("텍스트 입력 또는 PDF 업로드 후 아래 버튼을 누르세요.")
            
                # PDF 텍스트 추출 헬퍼
                def _extract_pdf_text(uploaded_file):
                    if uploaded_file is None:
                        return ""
                    try:
                        from pypdf import PdfReader
                        reader = PdfReader(uploaded_file)
                        return "\n".join(page.extract_text() or "" for page in reader.pages)
                    except Exception:
                        return ""
            
                # 중지 플래그 초기화
                if "learn_stop" not in st.session_state:
                    st.session_state.learn_stop = False
                if "learn_running" not in st.session_state:
                    st.session_state.learn_running = False
            
                col_btn1, col_btn2 = st.columns([1, 2])
                with col_btn1:
                    if st.session_state.learn_running:
                        if st.button("⏹️ 중지", type="secondary", use_container_width=True, key="learn_stop_btn"):
                            st.session_state.learn_stop = True
                            st.rerun()
                    else:
                        run_learn_btn = st.button("전체 학습 실행", type="primary", use_container_width=True, key="run_learn_btn")
                with col_btn2:
                    st.caption("실행 시 가상인구 DB의 '현시대 반영' 컬럼이 갱신됩니다. (페르소나가 있는 가상인구만)")
            
                run_learn_btn = run_learn_btn if not st.session_state.learn_running else False
                if run_learn_btn:
                    combined_content = (learn_text or "").strip()
                    if learn_pdf is not None:
                        pdf_text = _extract_pdf_text(learn_pdf)
                        if pdf_text:
                            combined_content = (combined_content + "\n\n[PDF 내용]\n" + pdf_text).strip()
                    if not combined_content:
                        st.warning("텍스트를 입력하거나 PDF 파일을 업로드해주세요.")
                    else:
                        st.session_state.learn_running = True
                        st.session_state.learn_stop = False
                        st.session_state.learn_combined_content = combined_content
                        st.session_state.learn_sido_code = selected_sido_code
                        st.session_state.learn_total_done = 0  # 누적 처리 인원
                        st.rerun()
            
                # 전체 학습 실행 중일 때 — 10명씩 처리 후 rerun (무한 로딩 방지)
                LEARN_CHUNK_SIZE = 10
                if st.session_state.learn_running and not st.session_state.learn_stop:
                    combined_content = st.session_state.get("learn_combined_content", "").strip()
                    run_sido = st.session_state.get("learn_sido_code", selected_sido_code)
                    if not combined_content:
                        st.warning("현시대 자료가 없습니다. 다시 텍스트/PDF를 입력한 뒤 실행해주세요.")
                        st.session_state.learn_running = False
                    else:
                        db_rows = st.session_state.get(f"vdb_cached_jsons_{run_sido}")
                        if not db_rows:
                            st.warning("가상인구 DB에 데이터가 없습니다. 해당 시도를 선택한 뒤 **전체 데이터 불러오기**를 먼저 실행해주세요.")
                            st.session_state.learn_running = False
                        else:
                            try:
                                all_people_to_process = []
                                for rec_idx, (record_id, data_json) in enumerate(db_rows):
                                    df = pd.read_json(data_json, orient="records")
                                    if "현시대 반영" not in df.columns:
                                        df["현시대 반영"] = ""
                                    if "페르소나" not in df.columns:
                                        df["페르소나"] = ""
                                    for idx, row in df.iterrows():
                                        persona = str(row.get("페르소나", "")).strip()
                                        reflection = str(row.get("현시대 반영", "")).strip()
                                        if persona and persona != "" and persona != "nan" and (not reflection or reflection == "" or reflection == "nan"):
                                            all_people_to_process.append({"record_id": record_id, "df": df, "row_idx": idx, "row": row})
                            
                                total_people = len(all_people_to_process)
                                if total_people == 0:
                                    total_done = st.session_state.get("learn_total_done", 0)
                                    st.success(f"전체 학습 완료: 총 {total_done}명의 가상인구에 현시대 반영을 반영했습니다.")
                                    st.session_state.learn_running = False
                                    st.session_state.learn_stop = False
                                    for k in ("learn_combined_content", "learn_sido_code", "learn_total_done"):
                                        st.session_state.pop(k, None)
                                    st.rerun()
                                else:
                                    chunk = all_people_to_process[:LEARN_CHUNK_SIZE]
                                    progress_bar = st.progress(0)
                                    status_placeholder = st.empty()
                                    total_done = st.session_state.get("learn_total_done", 0)
                                    status_placeholder.caption(f"처리 중: 이번에 {len(chunk)}명 처리 (누적 {total_done + len(chunk)}명 / 남은 대상 {total_people}명)")
                                    gemini_client = GeminiClient()
                                    processed_this_run = 0
                                    for person_item in chunk:
                                        if st.session_state.learn_stop:
                                            break
                                        record_id = person_item["record_id"]
                                        df = person_item["df"]
                                        row_idx = person_item["row_idx"]
                                        row = person_item["row"]
                                        profile_parts = []
                                        for c in ["거주지역", "성별", "연령", "경제활동", "교육정도", "월평균소득", "가상이름"]:
                                            if c in row and pd.notna(row.get(c)):
                                                profile_parts.append(f"{c}: {row[c]}")
                                        persona = str(row.get("페르소나", "")).strip()
                                        if persona and persona != "" and persona != "nan":
                                            profile_parts.append(f"페르소나: {persona[:150]}")
                                        profile_str = "\n".join(profile_parts) if profile_parts else "특성 없음"
                                        prompt = f"""다음은 '현시대 자료'입니다.
    ---
    {combined_content[:8000]}
    ---
    아래 가상인물의 페르소나와 특성을 고려하여, 이 인물의 관심분야에 맞는 내용만 선택적으로 반영하세요.
    관련도가 높으면 100자 내외로 한 문장으로 '현시대 반영' 문장을 작성하고, 관련도가 낮으면 빈 문자열만 반환하세요.
    반드시 100자 내외, 한 문장으로만 출력하고 다른 설명은 하지 마세요. (짤리지 않도록 100자 이하로 작성)

    가상인물 정보:
    {profile_str}

    현시대 반영 (100자 내외 한 문장, 또는 관련 없으면 빈칸):"""
                                        try:
                                            resp = gemini_client._client.models.generate_content(model=gemini_client._model, contents=prompt)
                                            text = (resp.text or "").strip()
                                            if "현시대 반영" in text and ":" in text:
                                                text = text.split(":", 1)[-1].strip()
                                            # 저장은 최대 200자까지 허용 (생성은 100자 내외이므로 짤림 없음)
                                            if len(text) > 200:
                                                text = text[:200]
                                            df.at[row_idx, "현시대 반영"] = text
                                        except Exception:
                                            pass
                                        new_json = df.to_json(orient="records", force_ascii=False)
                                        # 실시간 Supabase 업로드 대신 캐시만 갱신 → 나중에 '기록을 DB에 추가'로 반영
                                        _vdb_update_cached_record(run_sido, record_id, new_json)
                                        processed_this_run += 1
                                        progress_bar.progress(processed_this_run / len(chunk))
                                
                                    progress_bar.empty()
                                    status_placeholder.empty()
                                    st.session_state.learn_total_done = total_done + processed_this_run
                                    if st.session_state.learn_stop:
                                        st.warning(f"전체 학습이 중지되었습니다. 지금까지 {st.session_state.learn_total_done}명의 가상인구에 현시대 반영이 반영되었습니다.")
                                        st.session_state.learn_running = False
                                        st.session_state.learn_stop = False
                                        for k in ("learn_combined_content", "learn_sido_code", "learn_total_done"):
                                            st.session_state.pop(k, None)
                                        st.rerun()
                                    else:
                                        st.rerun()
                            except Exception as e:
                                st.error(f"전체 학습 중 오류: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                                st.session_state.learn_running = False
                                st.session_state.learn_stop = False
                                for k in ("learn_combined_content", "learn_sido_code", "learn_total_done"):
                                    st.session_state.pop(k, None)
                st.markdown("---")
        
            # 가상인구 선택 UI (1:1대화, 5:1대화 모드일 때만). 전체 데이터는 '전체 데이터 불러오기' 후 캐시에만 있음 (자동 조회 시 timeout 방지)
            if st.session_state.chat_mode in ["1:1대화", "5:1대화"]:
                db_rows = st.session_state.get(f"vdb_cached_jsons_{selected_sido_code}")
                if not db_rows and st.session_state.get(f"vdb_full_data_loaded_{selected_sido_code}"):
                    st.caption("표시할 데이터를 불러오는 중이거나 오류가 났습니다. 상단 **전체 데이터 불러오기**를 다시 눌러보세요.")
                elif not db_rows:
                    st.caption("1:1대화·5:1대화를 사용하려면 상단 **전체 데이터 불러오기**를 먼저 눌러주세요.")
                if db_rows:
                    # 모든 데이터를 하나의 DataFrame으로 합치기
                    all_dfs = []
                    for _rid, data_json in db_rows:
                        try:
                            df = pd.read_json(data_json, orient="records")
                            all_dfs.append(df)
                        except Exception:
                            continue
                    if all_dfs:
                        combined_df = pd.concat(all_dfs, ignore_index=True)
                        if st.session_state.chat_mode == "1:1대화":
                            # 1명 선택: 연령대·성별·지역 수평 3개 + 가상인구 선택 아래. 세 개 모두 선택 시 한 번에 조회.
                            OPT_PROMPT = "선택하세요"
                            person_meta_1 = []
                            age_set_1 = set()
                            gender_set_1 = set()
                            region_set_1 = set()
                            for idx, person in combined_df.iterrows():
                                age = person.get('연령', 0)
                                try:
                                    age_num = int(float(str(age)))
                                    if age_num < 20:
                                        age_group = "10대"
                                    elif age_num < 30:
                                        age_group = "20대"
                                    elif age_num < 40:
                                        age_group = "30대"
                                    elif age_num < 50:
                                        age_group = "40대"
                                    elif age_num < 60:
                                        age_group = "50대"
                                    elif age_num < 70:
                                        age_group = "60대"
                                    elif age_num < 80:
                                        age_group = "70대"
                                    else:
                                        age_group = "80대 이상"
                                except Exception:
                                    age_group = "미상"
                                gender_str = str(person.get('성별', 'N/A')).strip()
                                region_str = str(person.get('거주지역', 'N/A')).strip()
                                age_set_1.add(age_group)
                                gender_set_1.add(gender_str)
                                region_set_1.add(region_str)
                                person_meta_1.append((idx, person, age_group, gender_str, region_str))

                            # 폼: 선택만 하면 rerun 안 됨. '조회' 클릭 시에만 반영·분류
                            age_opts_1 = [OPT_PROMPT] + sorted(age_set_1)
                            gender_opts_1 = [OPT_PROMPT] + sorted(gender_set_1)
                            region_opts_1 = [OPT_PROMPT] + sorted(region_set_1)
                            sa = st.session_state.get("chat_filter_age_1to1", OPT_PROMPT)
                            sg = st.session_state.get("chat_filter_gender_1to1", OPT_PROMPT)
                            sr = st.session_state.get("chat_filter_region_1to1", OPT_PROMPT)
                            idx_a = age_opts_1.index(sa) if sa in age_opts_1 else 0
                            idx_g = gender_opts_1.index(sg) if sg in gender_opts_1 else 0
                            idx_r = region_opts_1.index(sr) if sr in region_opts_1 else 0
                            with st.form("chat_filter_form_1to1"):
                                row1_c1, row1_c2, row1_c3 = st.columns(3)
                                with row1_c1:
                                    selected_age_1 = st.selectbox("연령대 선택", options=age_opts_1, index=idx_a)
                                with row1_c2:
                                    selected_gender_1 = st.selectbox("성별 선택", options=gender_opts_1, index=idx_g)
                                with row1_c3:
                                    selected_region_1 = st.selectbox("지역 선택", options=region_opts_1, index=idx_r)
                                form_submitted_1 = st.form_submit_button("조회")
                            if form_submitted_1:
                                st.session_state["chat_filter_age_1to1"] = selected_age_1
                                st.session_state["chat_filter_gender_1to1"] = selected_gender_1
                                st.session_state["chat_filter_region_1to1"] = selected_region_1
                            use_age_1 = st.session_state.get("chat_filter_age_1to1", OPT_PROMPT)
                            use_gender_1 = st.session_state.get("chat_filter_gender_1to1", OPT_PROMPT)
                            use_region_1 = st.session_state.get("chat_filter_region_1to1", OPT_PROMPT)

                            # 조회 버튼으로 세 개 모두 선택한 뒤에만 분류·가상인구 목록 표시
                            if use_age_1 != OPT_PROMPT and use_gender_1 != OPT_PROMPT and use_region_1 != OPT_PROMPT:
                                final_1 = [(idx, p) for idx, p, ag, g, r in person_meta_1
                                           if ag == use_age_1 and g == use_gender_1 and r == use_region_1]
                                if len(final_1) > 0:
                                    person_options = []
                                    for idx, person in final_1:
                                        name = person.get('가상이름', f'인물 {idx+1}')
                                        age = person.get('연령', 'N/A')
                                        gender = person.get('성별', 'N/A')
                                        region = person.get('거주지역', 'N/A')
                                        person_label = f"{name} ({age}세, {gender}, {region})"
                                        person_options.append((idx, person_label, person))
                                    selected_person_idx = st.selectbox(
                                        "가상인구 선택",
                                        options=[opt[0] for opt in person_options],
                                        format_func=lambda x, opts=person_options: next(opt[1] for opt in opts if opt[0] == x),
                                        key="chat_select_person_1to1"
                                    )
                                    selected_person = next(opt[2] for opt in person_options if opt[0] == selected_person_idx)
                                    st.session_state.selected_chat_person = selected_person
                                    with st.expander("선택한 가상인구 정보", expanded=False):
                                        st.write(f"- 이름: {selected_person.get('가상이름', 'N/A')}")
                                        st.write(f"- 거주지역: {selected_person.get('거주지역', 'N/A')}")
                                        st.write(f"- 성별: {selected_person.get('성별', 'N/A')}")
                                        st.write(f"- 연령: {selected_person.get('연령', 'N/A')}")
                                        st.write(f"- 경제활동: {selected_person.get('경제활동', 'N/A')}")
                                        st.write(f"- 교육정도: {selected_person.get('교육정도', 'N/A')}")
                                        st.write(f"- 월평균소득: {selected_person.get('월평균소득', 'N/A')}")
                                        st.write(f"- 페르소나: {selected_person.get('페르소나', 'N/A')}")
                                else:
                                    st.warning("선택한 조건에 해당하는 가상인구가 없습니다.")
                                    st.session_state.selected_chat_person = None
                            else:
                                st.caption("연령대, 성별, 지역을 선택한 뒤 **조회** 버튼을 눌러주세요.")
                                st.session_state.selected_chat_person = None
                    
                        elif st.session_state.chat_mode == "5:1대화":
                            # 5명 선택: 연령대·성별·지역 수평 3개 + 가상인구 선택 아래. 세 개 모두 선택 시 한 번에 조회.
                            OPT_PROMPT_5 = "선택하세요"
                            person_meta_5 = []
                            age_set_5 = set()
                            gender_set_5 = set()
                            region_set_5 = set()
                            for idx, person in combined_df.iterrows():
                                age = person.get('연령', 0)
                                try:
                                    age_num = int(float(str(age)))
                                    if age_num < 20:
                                        age_group = "10대"
                                    elif age_num < 30:
                                        age_group = "20대"
                                    elif age_num < 40:
                                        age_group = "30대"
                                    elif age_num < 50:
                                        age_group = "40대"
                                    elif age_num < 60:
                                        age_group = "50대"
                                    elif age_num < 70:
                                        age_group = "60대"
                                    elif age_num < 80:
                                        age_group = "70대"
                                    else:
                                        age_group = "80대 이상"
                                except Exception:
                                    age_group = "미상"
                                gender_str = str(person.get('성별', 'N/A')).strip()
                                region_str = str(person.get('거주지역', 'N/A')).strip()
                                age_set_5.add(age_group)
                                gender_set_5.add(gender_str)
                                region_set_5.add(region_str)
                                person_meta_5.append((idx, person, age_group, gender_str, region_str))

                            age_opts_5 = [OPT_PROMPT_5] + sorted(age_set_5)
                            gender_opts_5 = [OPT_PROMPT_5] + sorted(gender_set_5)
                            region_opts_5 = [OPT_PROMPT_5] + sorted(region_set_5)
                            sa5 = st.session_state.get("chat_filter_age_5to1", OPT_PROMPT_5)
                            sg5 = st.session_state.get("chat_filter_gender_5to1", OPT_PROMPT_5)
                            sr5 = st.session_state.get("chat_filter_region_5to1", OPT_PROMPT_5)
                            idx_a5 = age_opts_5.index(sa5) if sa5 in age_opts_5 else 0
                            idx_g5 = gender_opts_5.index(sg5) if sg5 in gender_opts_5 else 0
                            idx_r5 = region_opts_5.index(sr5) if sr5 in region_opts_5 else 0
                            with st.form("chat_filter_form_5to1"):
                                r1_c1, r1_c2, r1_c3 = st.columns(3)
                                with r1_c1:
                                    selected_age_5 = st.selectbox("연령대 선택", options=age_opts_5, index=idx_a5)
                                with r1_c2:
                                    selected_gender_5 = st.selectbox("성별 선택", options=gender_opts_5, index=idx_g5)
                                with r1_c3:
                                    selected_region_5 = st.selectbox("지역 선택", options=region_opts_5, index=idx_r5)
                                form_submitted_5 = st.form_submit_button("조회")
                            if form_submitted_5:
                                st.session_state["chat_filter_age_5to1"] = selected_age_5
                                st.session_state["chat_filter_gender_5to1"] = selected_gender_5
                                st.session_state["chat_filter_region_5to1"] = selected_region_5
                            use_age_5 = st.session_state.get("chat_filter_age_5to1", OPT_PROMPT_5)
                            use_gender_5 = st.session_state.get("chat_filter_gender_5to1", OPT_PROMPT_5)
                            use_region_5 = st.session_state.get("chat_filter_region_5to1", OPT_PROMPT_5)

                            if use_age_5 != OPT_PROMPT_5 and use_gender_5 != OPT_PROMPT_5 and use_region_5 != OPT_PROMPT_5:
                                final_5 = [(idx, p) for idx, p, ag, g, r in person_meta_5
                                           if ag == use_age_5 and g == use_gender_5 and r == use_region_5]
                                if len(final_5) > 0:
                                    person_options_5 = []
                                    for idx, person in final_5:
                                        name = person.get('가상이름', f'인물 {idx+1}')
                                        age = person.get('연령', 'N/A')
                                        gender = person.get('성별', 'N/A')
                                        region = person.get('거주지역', 'N/A')
                                        person_label = f"{name} ({age}세, {gender}, {region})"
                                        person_options_5.append((idx, person_label, person))
                                    selected_indices = st.multiselect(
                                        "가상인구 선택",
                                        options=[opt[0] for opt in person_options_5],
                                        format_func=lambda x, opts=person_options_5: next(opt[1] for opt in opts if opt[0] == x),
                                        key="chat_select_people_5to1",
                                        max_selections=5
                                    )
                                    if len(selected_indices) > 0:
                                        selected_people = [next(opt[2] for opt in person_options_5 if opt[0] == idx) for idx in selected_indices]
                                        st.session_state.selected_chat_people = selected_people
                                        with st.expander("선택한 가상인구 정보", expanded=False):
                                            for i, person in enumerate(selected_people):
                                                st.markdown(f"**인물 {i+1}:**")
                                                st.write(f"- 이름: {person.get('가상이름', 'N/A')}")
                                                st.write(f"- 거주지역: {person.get('거주지역', 'N/A')}")
                                                st.write(f"- 성별: {person.get('성별', 'N/A')}")
                                                st.write(f"- 연령: {person.get('연령', 'N/A')}")
                                                st.write(f"- 페르소나: {person.get('페르소나', 'N/A')}")
                                                st.markdown("---")
                                    else:
                                        st.session_state.selected_chat_people = None
                                        st.info("가상인구를 선택해주세요.")
                                else:
                                    st.warning("선택한 조건에 해당하는 가상인구가 없습니다.")
                                    st.session_state.selected_chat_people = None
                            else:
                                st.caption("연령대, 성별, 지역을 선택한 뒤 **조회** 버튼을 눌러주세요.")
                                st.session_state.selected_chat_people = None
                    
                        st.markdown("---")
                    else:
                        st.info("가상인구 DB에 데이터가 없습니다.")
                else:
                    st.info("가상인구 DB에 데이터가 없습니다.")
    
        with col_chat:
            st.subheader("채팅창")
            # 전체 학습 모드가 아닐 때만 채팅 영역(히스토리·입력) 표시
            if st.session_state.chat_mode != "전체 학습":
                # 로딩 시 pending 사용자 메시지를 히스토리에 먼저 반영 (깜빡임 방지)
                if st.session_state.get("chat_loading") and st.session_state.get("pending_chat_input"):
                    st.session_state.chat_history.append({"role": "user", "message": st.session_state.pending_chat_input})
                    st.session_state.pending_chat_input = None
                # 채팅 말풍선 스타일 CSS
                st.markdown("""
            <style>
            .chat-container-wrapper {
                max-height: 500px;
                overflow-y: auto;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 12px;
                margin-bottom: 20px;
                border: 1px solid #e5e7eb;
            }
            .chat-message-wrapper {
                margin-bottom: 20px;
                display: block !important;
                width: 100% !important;
            }
            .chat-message-wrapper.user-msg {
                text-align: right !important;
            }
            .chat-message-wrapper.assistant-msg {
                text-align: left !important;
            }
            .chat-bubble {
                display: inline-block !important;
                max-width: 70% !important;
                padding: 12px 16px !important;
                border-radius: 18px !important;
                word-wrap: break-word !important;
                line-height: 1.5 !important;
                font-size: 14px !important;
                margin-top: 4px !important;
            }
            .chat-bubble.user-bubble {
                background-color: #4f46e5 !important;
                color: white !important;
                border-bottom-right-radius: 4px !important;
                text-align: left !important;
            }
            .chat-bubble.assistant-bubble {
                background-color: white !important;
                color: #1f2937 !important;
                border: 1px solid #e5e7eb !important;
                border-bottom-left-radius: 4px !important;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
            }
            .chat-label {
                font-size: 11px !important;
                color: #6b7280 !important;
                margin-bottom: 4px !important;
                font-weight: 600 !important;
                display: block !important;
            }
            .chat-skeleton { margin-top: 12px; }
            .chat-skeleton-line { height: 12px; background: linear-gradient(90deg, #e5e7eb 25%, #f3f4f6 50%, #e5e7eb 75%); background-size: 200% 100%; animation: chat-skeleton-shine 1.2s ease-in-out infinite; border-radius: 6px; margin-bottom: 8px; }
            .chat-skeleton-line.short { width: 60%; }
            .chat-skeleton-line.medium { width: 85%; }
            @keyframes chat-skeleton-shine { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }
            </style>
            """, unsafe_allow_html=True)
                # 전송 버튼 활성화 조건 및 send_message 정의 (chat_loading 블록에서 호출되므로 먼저 정의)
                can_send = False
                if st.session_state.chat_mode == "1:1대화":
                    can_send = st.session_state.get("selected_chat_person") is not None
                elif st.session_state.chat_mode == "5:1대화":
                    _selected_people = st.session_state.get("selected_chat_people")
                    can_send = _selected_people is not None and len(_selected_people) > 0
                else:
                    can_send = True
            
                def send_message(msg_text: str):
                    user_input = (msg_text or "").strip()
                    if not user_input:
                        return
                    # 현재 시점·계절 컨텍스트 (대화에 반영)
                    _now = datetime.datetime.now()
                    _m = _now.month
                    _season = "봄" if 3 <= _m <= 5 else "여름" if 6 <= _m <= 8 else "가을" if 9 <= _m <= 11 else "겨울"
                    _time_ctx = f"현재 시점: {_now.strftime('%Y년 %m월 %d일')}, {_season}입니다. 답변 시 계절·시기에 어긋나는 말만 하지 마세요. (예: 여름에 '겨울 옷 입으세요', 겨울에 '시원한 음식만 드세요'처럼 맞지 않는 얘기는 피하세요.) 계절 이야기만 할 필요는 없고, 질문에 맞게 자연스럽게 답하세요."
                    # GeminiClient 재사용 (세션 상태에 저장)
                    if "chat_gemini_client" not in st.session_state:
                        st.session_state.chat_gemini_client = GeminiClient()
                    gemini_client = st.session_state.chat_gemini_client
                
                    # 가상인구 응답 생성
                    try:
                            if st.session_state.chat_mode == "1:1대화":
                                # 선택한 가상인구 사용
                                selected_person = st.session_state.get("selected_chat_person")
                                if selected_person is None:
                                    response_text = "가상인구를 선택해주세요."
                                else:
                                    # 프롬프트 최적화 (페르소나 + 현시대 반영 반영)
                                    persona_info = selected_person.get('페르소나', '')
                                    if not persona_info or persona_info == 'N/A':
                                        persona_info = f"{selected_person.get('거주지역', '')} 거주, {selected_person.get('성별', '')}, {selected_person.get('연령', '')}세, {selected_person.get('경제활동', '')}, {selected_person.get('교육정도', '')}"
                                    _ref = selected_person.get('현시대 반영', '') or ''
                                    reflection = str(_ref).strip() if _ref is not None else ''
                                    if not reflection or reflection in ('N/A', 'nan'):
                                        reflection = ""
                                    prompt_extra = f"\n현시대 반영 (이 인물의 최근 현시대 관심/반응, 답변 시 반영할 것): {reflection}\n" if reflection else ""
                                    virtual_name = str(selected_person.get('가상이름', '') or '').strip() or '가상인구'
                                    prompt = f"""다음 가상인물의 페르소나: {persona_info}{prompt_extra}
    
        {_time_ctx}
        이 가상인물의 이름은 '{virtual_name}'이지만, 사용자가 이름을 물어보지 않으면 답변 안에서 자기 이름을 말하지 마세요. "계석 OOO는", "저는 OOO인데"처럼 기계적으로 소개하지 말고, 사람처럼 자연스럽게 대화체로만 답하세요. 이름을 쓸 때만 '{virtual_name}'을 사용하세요.
        이 가상인물의 입장에서 사용자의 질문에 자연스럽고 현실적인 대화체로 답변해주세요. 현시대 반영이 주어졌다면 그 관심·반응도 답변에 반영해주세요.
        답변은 반드시 100자 이내로만 작성하세요. 100자를 초과하지 마세요.
    
        사용자 질문: {user_input}"""
                                
                                    # 스트리밍 응답 사용 (말풍선 스타일)
                                    response_text = ""
                                    response_placeholder = st.empty()
                                
                                    try:
                                        stream = gemini_client._client.models.generate_content_stream(
                                            model=gemini_client._model,
                                            contents=prompt,
                                        )
                                    
                                        import html
                                        for chunk in stream:
                                            if chunk.text:
                                                response_text += chunk.text
                                                # HTML 특수 문자 이스케이프 및 줄바꿈 처리
                                                escaped_text = html.escape(response_text).replace('\n', '<br>')
                                                name = selected_person.get('가상이름', '가상인구')
                                                age = selected_person.get('연령', 'N/A')
                                                gender = selected_person.get('성별', 'N/A')
                                                region = selected_person.get('거주지역', 'N/A')
                                                person_label = f"{name} ({age}세, {gender}, {region})"
                                                response_placeholder.markdown(f"""
                                                <div class="chat-message-wrapper assistant-msg">
                                                    <span class="chat-label">{person_label}</span>
                                                    <div class="chat-bubble assistant-bubble">{escaped_text}</div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                    except Exception as stream_error:
                                        response = gemini_client._client.models.generate_content(
                                            model=gemini_client._model,
                                            contents=prompt,
                                        )
                                        response_text = (response.text or "").strip()
                                        import html
                                        escaped_text = html.escape(response_text).replace('\n', '<br>')
                                        name = selected_person.get('가상이름', '가상인구')
                                        age = selected_person.get('연령', 'N/A')
                                        gender = selected_person.get('성별', 'N/A')
                                        region = selected_person.get('거주지역', 'N/A')
                                        person_label = f"{name} ({age}세, {gender}, {region})"
                                        response_placeholder.markdown(f"""
                                        <div class="chat-message-wrapper assistant-msg">
                                            <span class="chat-label">{person_label}</span>
                                            <div class="chat-bubble assistant-bubble">{escaped_text}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                        
                            elif st.session_state.chat_mode == "5:1대화":
                                selected_people = st.session_state.get("selected_chat_people")
                                if selected_people is None or len(selected_people) == 0:
                                    response_text = "가상인구를 선택해주세요."
                                else:
                                    personas = []
                                    for i, person in enumerate(selected_people):
                                        p_name = str(person.get('가상이름', '') or '').strip() or f'인물{i+1}'
                                        persona_info = person.get('페르소나', '')
                                        if not persona_info or persona_info == 'N/A':
                                            persona_info = f"{person.get('거주지역', '')} 거주, {person.get('성별', '')}, {person.get('연령', '')}세"
                                        _ref = person.get('현시대 반영', '') or ''
                                        reflection = str(_ref).strip() if _ref is not None else ''
                                        if not reflection or reflection in ('N/A', 'nan'):
                                            reflection = ""
                                        if reflection:
                                            personas.append(f"인물 {i+1} (이름: {p_name}): {persona_info} | 현시대 반영: {reflection}")
                                        else:
                                            personas.append(f"인물 {i+1} (이름: {p_name}): {persona_info}")
                                
                                    prompt = f"""다음 가상인물들이 함께 대화합니다:
        {chr(10).join(personas)}
    
        {_time_ctx}
        각 가상인물은 사용자가 이름을 묻지 않으면 답변 안에서 "계석 OOO는", "저 OOO는"처럼 자기 이름을 붙이지 말고, 사람처럼 자연스럽게 의견만 말하세요. 이름을 쓸 때만 위에 적힌 자신의 이름을 사용하세요.
        이 가상인물들이 각자의 관점에서 사용자의 질문에 답변해주세요. 각 인물의 특성과 현시대 반영(주어졌다면)에 맞는 다양한 의견을 제시해주세요.
        답변은 반드시 각 인물당 100자 이내로만 작성하세요. 100자를 초과하지 마세요.
    
        사용자 질문: {user_input}"""
                                
                                    response_text = ""
                                    response_placeholder = st.empty()
                                    try:
                                        stream = gemini_client._client.models.generate_content_stream(
                                            model=gemini_client._model,
                                            contents=prompt,
                                        )
                                        import html
                                        for chunk in stream:
                                            if chunk.text:
                                                response_text += chunk.text
                                                escaped_text = html.escape(response_text).replace('\n', '<br>')
                                                if len(selected_people) == 1:
                                                    p = selected_people[0]
                                                    name = p.get('가상이름', '가상인구')
                                                    age = p.get('연령', 'N/A')
                                                    gender = p.get('성별', 'N/A')
                                                    region = p.get('거주지역', 'N/A')
                                                    person_label = f"{name} ({age}세, {gender}, {region})"
                                                else:
                                                    person_label = f"가상인구 {len(selected_people)}명"
                                                response_placeholder.markdown(f"""
                                                <div class="chat-message-wrapper assistant-msg">
                                                    <span class="chat-label">{person_label}</span>
                                                    <div class="chat-bubble assistant-bubble">{escaped_text}</div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                    except Exception as stream_error:
                                        response = gemini_client._client.models.generate_content(
                                            model=gemini_client._model,
                                            contents=prompt,
                                        )
                                        response_text = (response.text or "").strip()
                                        import html
                                        escaped_text = html.escape(response_text).replace('\n', '<br>')
                                        if len(selected_people) == 1:
                                            p = selected_people[0]
                                            name = p.get('가상이름', '가상인구')
                                            age = p.get('연령', 'N/A')
                                            gender = p.get('성별', 'N/A')
                                            region = p.get('거주지역', 'N/A')
                                            person_label = f"{name} ({age}세, {gender}, {region})"
                                        else:
                                            person_label = f"가상인구 {len(selected_people)}명"
                                        response_placeholder.markdown(f"""
                                        <div class="chat-message-wrapper assistant-msg">
                                            <span class="chat-label">{person_label}</span>
                                            <div class="chat-bubble assistant-bubble">{escaped_text}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                        
                            else:
                                db_rows = st.session_state.get(f"vdb_cached_jsons_{selected_sido_code}")
                                if not db_rows:
                                    response_text = "가상인구 DB에 데이터가 없습니다. 상단에서 **전체 데이터 불러오기**를 누른 뒤 다시 시도해주세요."
                                else:
                                    all_dfs = []
                                    for _rid, data_json in db_rows:
                                        try:
                                            df = pd.read_json(data_json, orient="records")
                                            all_dfs.append(df)
                                        except Exception:
                                            continue
                                    if all_dfs:
                                        combined_df = pd.concat(all_dfs, ignore_index=True)
                                        prompt = f"""전체 {len(combined_df)}명의 가상인구 데이터를 학습한 AI입니다.
        {_time_ctx}
        다양한 가상인구의 특성과 페르소나를 종합하여 사용자의 질문에 자연스럽고 현실적인 답변을 해주세요.
        답변은 반드시 100자 이내로만 작성하세요. 100자를 초과하지 마세요.
    
        사용자 질문: {user_input}"""
                                        response_text = ""
                                        response_placeholder = st.empty()
                                        try:
                                            stream = gemini_client._client.models.generate_content_stream(
                                                model=gemini_client._model,
                                                contents=prompt,
                                            )
                                            for chunk in stream:
                                                if chunk.text:
                                                    response_text += chunk.text
                                                    import html
                                                    escaped_response = html.escape(str(response_text)).replace('\n', '<br>')
                                                    response_placeholder.markdown(f"""
                                                    <div class="chat-message-wrapper assistant-msg">
                                                        <span class="chat-label">전체 가상인구 ({len(combined_df)}명)</span>
                                                        <div class="chat-bubble assistant-bubble">{escaped_response}</div>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                        except Exception as stream_error:
                                            response = gemini_client._client.models.generate_content(
                                                model=gemini_client._model,
                                                contents=prompt,
                                            )
                                            response_text = (response.text or "").strip()
                                            import html
                                            escaped_response = html.escape(response_text).replace('\n', '<br>')
                                            response_placeholder.markdown(f"""
                                            <div class="chat-message-wrapper assistant-msg">
                                                <span class="chat-label">전체 가상인구 ({len(combined_df)}명)</span>
                                                <div class="chat-bubble assistant-bubble">{escaped_response}</div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                    else:
                                        response_text = "가상인구 데이터를 불러올 수 없습니다."
                    except Exception as e:
                        response_text = f"응답 생성 중 오류 발생: {e}"
                
                    if response_text:
                        chat_entry = {"role": "assistant", "message": response_text}
                        if st.session_state.chat_mode == "1:1대화":
                            selected_person = st.session_state.get("selected_chat_person")
                            if selected_person is not None:
                                if hasattr(selected_person, 'to_dict'):
                                    chat_entry["person_info"] = selected_person.to_dict()
                                else:
                                    chat_entry["person_info"] = dict(selected_person)
                        elif st.session_state.chat_mode == "5:1대화":
                            selected_people = st.session_state.get("selected_chat_people")
                            if selected_people is not None:
                                chat_entry["people_info"] = [p.to_dict() if hasattr(p, 'to_dict') else dict(p) for p in selected_people]
                        else:
                            db_rows = st.session_state.get(f"vdb_cached_jsons_{selected_sido_code}")
                            if db_rows:
                                all_dfs = []
                                for _rid, data_json in db_rows:
                                    try:
                                        df = pd.read_json(data_json, orient="records")
                                        all_dfs.append(df)
                                    except Exception:
                                        continue
                                if all_dfs:
                                    combined_df = pd.concat(all_dfs, ignore_index=True)
                                    chat_entry["total_count"] = len(combined_df)
                        st.session_state.chat_history.append(chat_entry)
            
                st.markdown('<div class="chat-container-wrapper">', unsafe_allow_html=True)
                if not st.session_state.chat_history:
                    st.info("💬 대화를 시작해보세요!")
                else:
                    import html
                    # 가상인구 특성 정보 가져오기 (1:1대화 또는 5:1대화 모드)
                    def get_person_label(chat_data):
                        """채팅 데이터에서 가상인구 레이블 생성"""
                        if "person_info" in chat_data:
                            person_info = chat_data["person_info"]
                            name = person_info.get('가상이름', '가상인구')
                            age = person_info.get('연령', 'N/A')
                            gender = person_info.get('성별', 'N/A')
                            region = person_info.get('거주지역', 'N/A')
                            return f"{name} ({age}세, {gender}, {region})"
                        elif "people_info" in chat_data:
                            people_info = chat_data["people_info"]
                            if len(people_info) == 1:
                                p = people_info[0]
                                name = p.get('가상이름', '가상인구')
                                age = p.get('연령', 'N/A')
                                gender = p.get('성별', 'N/A')
                                region = p.get('거주지역', 'N/A')
                                return f"{name} ({age}세, {gender}, {region})"
                            else:
                                return f"가상인구 {len(people_info)}명"
                        elif "total_count" in chat_data:
                            return f"전체 가상인구 ({chat_data['total_count']}명)"
                        return "가상인구"
                    for chat in st.session_state.chat_history:
                        escaped_message = html.escape(str(chat['message']))
                        escaped_message = escaped_message.replace('\n', '<br>')
                        if chat["role"] == "user":
                            st.markdown(f"""
                        <div class="chat-message-wrapper user-msg">
                            <span class="chat-label">사용자</span>
                            <div class="chat-bubble user-bubble">{escaped_message}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        else:
                            person_label = get_person_label(chat)
                            st.markdown(f"""
                        <div class="chat-message-wrapper assistant-msg">
                            <span class="chat-label">{person_label}</span>
                            <div class="chat-bubble assistant-bubble">{escaped_message}</div>
                        </div>
                        """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
                # 로딩 중: 스켈레톤 표시 후 API 호출 (깜빡임 방지)
                if st.session_state.get("chat_loading") and st.session_state.chat_history and st.session_state.chat_history[-1].get("role") == "user":
                    st.markdown("""
                    <div class="chat-container-wrapper chat-skeleton">
                        <div class="chat-skeleton-line short"></div>
                        <div class="chat-skeleton-line medium"></div>
                        <div class="chat-skeleton-line short"></div>
                    </div>
                    """, unsafe_allow_html=True)
                    send_message(st.session_state.chat_history[-1]["message"])
                    st.session_state.chat_loading = False
                    st.rerun()
        
                # 채팅 입력 (1:1대화, 5:1대화 모드에서만)
                # 채팅 입력 초기화 (세션 상태 관리)
                if "chat_input_value" not in st.session_state:
                    st.session_state.chat_input_value = ""
            
                # 채팅 입력 폼 (엔터 키로 전송 가능)
                with st.form(key="chat_form", clear_on_submit=True):
                    user_input = st.text_input(
                        "메시지를 입력하세요:",
                        key="chat_input",
                        value=st.session_state.get("chat_input_value", "")
                    )
                
                    # 전송 버튼 (폼 내부에서 엔터 키로도 제출 가능)
                    submitted = st.form_submit_button(
                        "전송",
                        disabled=not can_send,
                        use_container_width=True
                    )
                
                    # 폼 제출 시 pending 저장 후 rerun → 다음 run에서 스켈레톤 표시 후 API 호출 (깜빡임 방지)
                    if submitted:
                        if user_input and user_input.strip() and can_send:
                            st.session_state.pending_chat_input = user_input.strip()
                            st.session_state.chat_loading = True
                            st.rerun()
            
                if not can_send and st.session_state.chat_mode in ["1:1대화", "5:1대화"]:
                    st.info("가상인구를 선택한 후 메시지를 보낼 수 있습니다.")
            
                # 채팅 히스토리 초기화 버튼 (fragment 내부이므로 rerun 없이 재렌더만)
                if st.button("대화 초기화", key="chat_clear"):
                    st.session_state.chat_history = []
    _vdb_frag()

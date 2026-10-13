"""Streamlit client for conversational software estimation with memory and attachments.

Features:
- Maintains session state via POST /sessions.
- Multi-turn interaction with sliding-window history.
- Local document attachment uploading (.pdf, .docx, .txt).
- Live Project Metadata visualization in the sidebar, displaying the operational
  separation between conversation history and accumulated project truths.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("ESTIMATOR_API_BASE_URL", "http://localhost:8000").rstrip("/")
SESSIONS_ENDPOINT = f"{API_BASE_URL}/sessions"

st.set_page_config(
    page_title="Software Estimator — Conversational",
    page_icon="🧠",
    layout="wide",
)


def init_remote_session() -> str | None:
    """Request a fresh session ID from the FastAPI backend."""
    try:
        response = httpx.post(SESSIONS_ENDPOINT, timeout=10.0)
        response.raise_for_status()
        return response.json()["session_id"]
    except Exception as exc:
        st.error(f"Error initializing session at `{SESSIONS_ENDPOINT}`: {exc}")
        return None


# Initialize session state variables
if "session_id" not in st.session_state or not st.session_state["session_id"]:
    st.session_state["session_id"] = init_remote_session()

if "turns" not in st.session_state:
    st.session_state["turns"] = []

if "project_metadata" not in st.session_state:
    st.session_state["project_metadata"] = {
        "project_name": None,
        "assumed_team_size": None,
        "mentioned_technologies": [],
        "agreed_scope": None,
    }


def reset_conversation() -> None:
    """Reset conversation turns and acquire a fresh session ID."""
    st.session_state["session_id"] = init_remote_session()
    st.session_state["turns"] = []
    st.session_state["project_metadata"] = {
        "project_name": None,
        "assumed_team_size": None,
        "mentioned_technologies": [],
        "agreed_scope": None,
    }
    st.rerun()


# --- SIDEBAR: Session Info & Project Metadata Memory ---
with st.sidebar:
    st.title("🧠 Memoria del Proyecto")
    st.caption("Hechos persistidos en `project_metadata` inyectados en el system prompt:")

    metadata = st.session_state.get("project_metadata", {})

    proj_name = metadata.get("project_name") or "*(No identificado aún)*"
    st.markdown(f"**Nombre del Proyecto:**\n`{proj_name}`")

    team_size = metadata.get("assumed_team_size")
    team_display = f"{team_size} ingenieros" if team_size else "*(No especificado)*"
    st.markdown(f"**Equipo Asumido:**\n`{team_display}`")

    tech_list = metadata.get("mentioned_technologies", [])
    if tech_list:
        st.markdown("**Tecnologías Identificadas:**")
        st.markdown(" ".join([f"`{t}`" for t in tech_list]))
    else:
        st.markdown("**Tecnologías Identificadas:**\n*(Ninguna detectada aún)*")

    scope = metadata.get("agreed_scope") or "*(Pendiente de acordar)*"
    st.markdown(f"**Alcance Acordado:**\n> {scope}")

    st.divider()

    st.subheader("Estado de Sesión")
    current_session = st.session_state.get("session_id")
    if current_session:
        st.code(current_session, language="text")
    else:
        st.warning("Sin sesión activa")

    if st.button("🔄 Nueva conversación", type="secondary", use_container_width=True):
        reset_conversation()

    st.divider()
    st.caption(
        "💡 **Separación Operativa**: La *Memoria* preserva los hechos clave del proyecto "
        "y se inyecta en cada system prompt, mientras que el *Historial* conserva "
        "una ventana deslizante de 6 turnos."
    )


# --- MAIN AREA ---
st.title("Software Estimator — Multi-Turn")
st.caption(
    "Interactúa con el estimador para refinar el alcance, proporcionar transcripciones "
    "y adjuntar documentación complementaria (PDF o Word)."
)

# Display existing conversation turns
for idx, turn in enumerate(st.session_state["turns"]):
    with st.chat_message("user"):
        st.markdown(turn["user"])
        if turn.get("attachments"):
            st.caption(f"📎 Adjuntos: {', '.join(turn['attachments'])}")
    with st.chat_message("assistant"):
        st.markdown(turn["assistant"])
        if turn.get("observation"):
            obs = turn["observation"]
            st.caption(
                f"📊 **Telemetría Turno #{obs.get('turn_index')}**: "
                f"⏱️ `{obs.get('latency_ms', 0)} ms` | "
                f"🪙 `${obs.get('cost_usd', 0.0):.5f}` | "
                f"📥 `{obs.get('tokens_in', 0)} tokens in`"
                + (f" | 📎 `{obs.get('attachments_total_chars', 0)} chars adjunto`" if obs.get("attachments_total_chars") else "")
            )
            with st.expander("🔍 Ver telemetría completa del turno (13 métricas)"):
                st.json(obs)

# Input Form for New Turn
with st.form("turn_form", clear_on_submit=True):
    transcript = st.text_area(
        "Transcripción / Mensaje para este turno:",
        placeholder="Ej: Estimemos un sistema de gestión logística para Savar con equipo de 4 devs en FastAPI y React...",
        height=140,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        project_type = st.selectbox(
            "Tipo de Proyecto:",
            options=["web_saas", "mobile_app", "internal_tool", "data_pipeline"],
            index=0,
        )
    with col2:
        detail_level = st.selectbox(
            "Nivel de Detalle:",
            options=["medium", "summary", "detailed"],
            index=0,
        )
    with col3:
        output_format = st.selectbox(
            "Formato de Salida:",
            options=["phases_table", "line_items", "narrative"],
            index=0,
        )

    uploaded_files = st.file_uploader(
        "Adjuntar especificaciones o minutas (opcional)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Los documentos son extraídos localmente (Camino B con pypdf / python-docx) y anexados al prompt.",
    )

    submitted = st.form_submit_button("Enviar turno al Estimador", type="primary")


if submitted:
    session_id = st.session_state.get("session_id")
    if not session_id:
        session_id = init_remote_session()
        st.session_state["session_id"] = session_id

    if len(transcript.strip()) < 20:
        st.error("Por favor ingresa al menos 20 caracteres para estimar con contexto.")
    elif not session_id:
        st.error("No se pudo establecer una sesión con el servicio de estimación.")
    else:
        files_payload: list[tuple[str, tuple[str, bytes, str]]] = []
        file_names: list[str] = []

        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.getvalue()
                files_payload.append(
                    (
                        "attachments",
                        (uploaded_file.name, file_bytes, uploaded_file.type or "application/octet-stream"),
                    )
                )
                file_names.append(uploaded_file.name)

        data_payload = {
            "transcript": transcript.strip(),
            "project_type": project_type,
            "detail_level": detail_level,
            "output_format": output_format,
        }
        estimate_url = f"{SESSIONS_ENDPOINT}/{session_id}/estimate"

        with st.spinner("Procesando turno y generando estimación..."):
            try:
                response = httpx.post(
                    estimate_url,
                    data=data_payload,
                    files=files_payload if files_payload else None,
                    timeout=httpx.Timeout(120.0, connect=10.0),
                )
                response.raise_for_status()
                result = response.json()
            except httpx.HTTPStatusError as exc:
                st.error(f"Error {exc.response.status_code}: {exc.response.text}")
            except httpx.HTTPError as exc:
                st.error(f"No se pudo conectar con el servicio en `{estimate_url}`: {exc}")
            else:
                # Format assistant message from structured result or text
                res_obj = result.get("result")
                if isinstance(res_obj, dict):
                    summary_text = res_obj.get("summary", "")
                    total_cost = res_obj.get("total_cost_eur", 0)
                    confidence = res_obj.get("confidence_pct", 0)
                    phases = res_obj.get("phases", [])

                    phases_table = "\n\n| Fase | Duración (sem) | Costo (€) |\n|---|---:|---:|\n"
                    for ph in phases:
                        p_name = ph.get("phase_name", "")
                        p_dur = ph.get("duration_weeks", "")
                        p_cost = ph.get("cost_eur", 0)
                        phases_table += f"| {p_name} | {p_dur} | €{p_cost:,} |\n"

                    assistant_text = (
                        f"{summary_text}\n\n"
                        f"💰 **Total Estimado:** `€{total_cost:,} EUR` | 🎯 **Confianza:** `{confidence}%`"
                        f"{phases_table if phases else ''}"
                    )
                else:
                    assistant_text = result.get("text", str(res_obj or "Estimación completada."))

                # Update turns history in session state
                st.session_state["turns"].append(
                    {
                        "user": transcript.strip(),
                        "attachments": file_names,
                        "assistant": assistant_text,
                        "observation": result.get("observation"),
                    }
                )

                # Fetch updated session metadata from backend to sync sidebar
                try:
                    s_resp = httpx.get(f"{SESSIONS_ENDPOINT}/{session_id}", timeout=5.0)
                    if s_resp.status_code == 200:
                        s_info = s_resp.json()
                        if "metadata" in s_info:
                            st.session_state["project_metadata"] = s_info["metadata"]
                except Exception:
                    pass

                st.rerun()

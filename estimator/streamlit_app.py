"""Streamlit UI for the Software Estimator (Session 4 Form Interface).

Replaces the conversational chat with a typed form contract (EstimationRequest)
communicating with the versioned Jinja2 prompt estimation service.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st
from dotenv import load_dotenv

from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    EstimationResponse,
    OutputFormat,
    ProjectType,
    ReferenceProject,
)

load_dotenv()

API_BASE_URL = os.getenv("ESTIMATOR_API_BASE_URL", "http://localhost:8000")
ESTIMATE_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/api/v1/estimate"
HEALTH_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/health"

st.set_page_config(
    page_title="Software Estimator",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Software Project Estimator")
st.caption(
    "Formulario tipado respaldado por templates Jinja2 versionados y Pydantic v2. "
    "Configura los parámetros del proyecto para generar una estimación estructurada."
)

# -----------------------------------------------------------------------------
# Quick Presets for Demo / Testing
# -----------------------------------------------------------------------------
PRESETS = {
    "Personalizado": "",
    "Mobile App (Repartidores)": (
        "Aplicación móvil para conductores y repartidores en iOS y Android con rastreo "
        "GPS en tiempo real, gestión de estados de pedidos (recogido, en ruta, entregado), "
        "soporte offline con sincronización SQLite, subida de foto de comprobante de entrega "
        "y notificaciones push de alta prioridad."
    ),
    "Web SaaS (Gestión de Suscripciones B2B)": (
        "Plataforma web B2B multi-inquilino para análisis de ingresos recurrentes y "
        "facturación automática. Integración con Stripe Billing, autenticación OAuth2 con "
        "roles y permisos granulares, dashboard de retención por cohortes y exportación "
        "asíncrona de reportes contables en PDF/CSV."
    ),
    "Data Pipeline (Eventos E-Commerce)": (
        "Pipeline de ingesta y transformación de eventos de compras e inventario en streaming. "
        "Validación de esquemas con cola de mensajes fallidos (dead-letter queue), "
        "modelado dimensional en dbt sobre BigQuery/Snowflake y alertas automáticas de anomalías."
    ),
}

# -----------------------------------------------------------------------------
# Sidebar: Service & Model Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración del Servicio")

    # Service health check
    try:
        health_res = httpx.get(HEALTH_ENDPOINT, timeout=2.0)
        if health_res.status_code == 200:
            st.success("🟢 Servicio IA Activo")
        else:
            st.warning(f"🟡 Estado: {health_res.status_code}")
    except Exception:
        st.error("🔴 Servicio no disponible")

    st.caption(f"**Endpoint:** `{ESTIMATE_ENDPOINT}`")

    prompt_version = st.selectbox(
        "Versión de Prompt",
        options=["v1", "v2"],
        index=0,
        help="v1: Consultoría clásica con tasas y desglose completo. v2: Enfoque ágil / lean MVP.",
    )

    primary_model = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")
    fallback_model = os.getenv("FALLBACK_MODEL", "claude-haiku-4-5-20251001")
    st.markdown(f"**Modelo principal:** `{primary_model}`")
    st.markdown(f"**Modelo de respaldo:** `{fallback_model}`")

    st.divider()
    st.subheader("💡 Plantillas Rápidas")
    selected_preset = st.selectbox(
        "Cargar ejemplo predefinido:",
        options=list(PRESETS.keys()),
        index=0,
    )

# -----------------------------------------------------------------------------
# Main Estimation Form
# -----------------------------------------------------------------------------
default_description = PRESETS[selected_preset] if selected_preset != "Personalizado" else ""

with st.form("estimation_form", clear_on_submit=False):
    description = st.text_area(
        "Descripción del Proyecto *",
        value=default_description,
        placeholder="Describe los requerimientos funcionales, integraciones técnicas y alcance (mínimo 20 caracteres)...",
        height=180,
        help="Mínimo 20 caracteres, máximo 2000 caracteres.",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        project_type_label = st.selectbox(
            "Tipo de Proyecto *",
            options=[
                ("Web SaaS", ProjectType.WEB_SAAS),
                ("Mobile App", ProjectType.MOBILE_APP),
                ("Internal Tool", ProjectType.INTERNAL_TOOL),
                ("Data Pipeline", ProjectType.DATA_PIPELINE),
            ],
            format_func=lambda opt: opt[0],
            index=0 if "SaaS" in selected_preset else (1 if "Mobile" in selected_preset else 0),
        )
        selected_project_type = project_type_label[1]

    with col2:
        detail_level_label = st.selectbox(
            "Nivel de Detalle *",
            options=[
                ("Detallado (con asunciones por fase)", DetailLevel.DETAILED),
                ("Medio (balanceado)", DetailLevel.MEDIUM),
                ("Resumido (visión ejecutiva)", DetailLevel.SUMMARY),
            ],
            format_func=lambda opt: opt[0],
            index=0,
        )
        selected_detail_level = detail_level_label[1]

    with col3:
        output_format_label = st.selectbox(
            "Formato de Salida *",
            options=[
                ("Tabla de Fases (Phases Table)", OutputFormat.PHASES_TABLE),
                ("Líneas de Tarea (Line Items)", OutputFormat.LINE_ITEMS),
                ("Narrativa Prosa (Narrative)", OutputFormat.NARRATIVE),
            ],
            format_func=lambda opt: opt[0],
            index=0,
        )
        selected_output_format = output_format_label[1]

    # Optional Reference Project context
    with st.expander("➕ Añadir Proyecto de Referencia Similar (Opcional - Few-shot context)"):
        st.caption("Provee un proyecto previo para enriquecer el contexto del modelo.")
        ref_col1, ref_col2 = st.columns(2)
        with ref_col1:
            ref_name = st.text_input("Nombre del proyecto de referencia", placeholder="Ej: MVP Portal Clientes")
            ref_duration = st.number_input("Duración (semanas)", min_value=0, value=0, step=1)
        with ref_col2:
            ref_desc = st.text_input("Alcance / Descripción", placeholder="Ej: Autenticación, dashboard y Stripe")
            ref_cost = st.number_input("Coste aprox (EUR)", min_value=0.0, value=0.0, step=500.0)

    submitted = st.form_submit_button(
        "🚀 Generar Estimación",
        type="primary",
        use_container_width=True,
    )

# -----------------------------------------------------------------------------
# Submission Handler
# -----------------------------------------------------------------------------
if submitted:
    if len(description.strip()) < 20:
        st.error("⚠️ La descripción debe tener al menos 20 caracteres.")
    else:
        reference_projects = None
        if ref_name.strip() and ref_desc.strip():
            reference_projects = [
                ReferenceProject(
                    name=ref_name.strip(),
                    description=ref_desc.strip(),
                    duration_weeks=ref_duration if ref_duration > 0 else None,
                    cost_eur=ref_cost if ref_cost > 0.0 else None,
                )
            ]

        # Typed request contract
        request_data = EstimationRequest(
            description=description.strip(),
            project_type=selected_project_type,
            detail_level=selected_detail_level,
            output_format=selected_output_format,
            reference_projects=reference_projects,
        )

        with st.spinner(f"Generando estimación con template `{prompt_version}`..."):
            try:
                target_url = f"{ESTIMATE_ENDPOINT}?prompt_version={prompt_version}"
                response = httpx.post(
                    target_url,
                    json=request_data.model_dump(mode="json"),
                    timeout=httpx.Timeout(120.0, connect=10.0),
                )
                if response.status_code == 200:
                    parsed_response = EstimationResponse(**response.json())
                    st.session_state["estimation_result"] = parsed_response
                    st.session_state["submitted_request"] = request_data
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except httpx.HTTPError as exc:
                st.error(f"Error de conexión con el servicio de estimación ({ESTIMATE_ENDPOINT}): {exc}")

# -----------------------------------------------------------------------------
# Display Estimation Result
# -----------------------------------------------------------------------------
if "estimation_result" in st.session_state:
    result: EstimationResponse = st.session_state["estimation_result"]

    st.divider()
    res_header_col1, res_header_col2 = st.columns([3, 1])
    with res_header_col1:
        st.subheader("📋 Resultado de la Estimación")
    with res_header_col2:
        st.info(f"**Prompt Version:** `{result.prompt_version}`")

    st.markdown(result.text)

    st.download_button(
        label="📥 Descargar Estimación (.md)",
        data=result.text,
        file_name=f"estimacion_proyecto_{result.prompt_version}.md",
        mime="text/markdown",
    )

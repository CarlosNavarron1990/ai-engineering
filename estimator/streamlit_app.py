"""Streamlit web application for the CAG Software Estimation service.

Implements:
- Level 1: Conversational chat interface (st.chat_message, st.chat_input, session_state)
- Level 2: Real-time streaming generation (token by token)
- Level 3: CAG context inspection (sidebar with active system prompt, static examples, call metrics)
"""

import os
import sys
from pathlib import Path

import streamlit as st

# Ensure application root is in python path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Bridge Streamlit secrets to environment variables if available
try:
    for env_key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER", "LLM_MODEL"):
        if env_key in st.secrets and not os.environ.get(env_key):
            os.environ[env_key] = str(st.secrets[env_key])
except Exception:  # noqa: BLE001, S110
    # secrets.toml does not exist; environment variables and .env will be used
    pass

from app.config import get_settings
from app.context.examples import ESTIMATION_EXAMPLES
from app.services.llm_service import (
    LLMServiceError,
    build_system_prompt,
    stream_estimation,
)

# Page configuration
st.set_page_config(
    page_title="Estimator CAG — Estimación de Software con IA",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .main-sub {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge-cag {
        background-color: #EEF2FF;
        color: #4F46E5;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-left: 0.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_app_settings():
    """Safely retrieve settings, returning None or error if API keys are missing."""
    try:
        return get_settings()
    except Exception as exc:  # noqa: BLE001
        return exc


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None

settings_or_error = load_app_settings()
system_prompt_content = build_system_prompt()

# ==============================================================================
# SIDEBAR: Contexto CAG, Métricas y Configuración (Nivel 3)
# ==============================================================================
with st.sidebar:
    st.markdown("## 💼 Estimator CAG")
    st.caption("Cache Augmented Generation • Software Estimator")

    # Provider & Model status
    if isinstance(settings_or_error, Exception):
        st.error(f"⚠️ Error de configuración: {settings_or_error}")
        st.info("Añade tu API key a `.env` o `st.secrets` para continuar.")
    else:
        settings = settings_or_error
        api_key_configured = bool(
            (settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY)
            or (settings.LLM_PROVIDER == "anthropic" and settings.ANTHROPIC_API_KEY)
        )

        st.markdown(
            f"""
            **Proveedor:** `{settings.LLM_PROVIDER}`  
            **Modelo:** `{settings.LLM_MODEL}`  
            **API Key:** {"✅ Configurada" if api_key_configured else "❌ No encontrada"}
            """
        )

    st.divider()

    # --------------------------------------------------------------------------
    # Métricas de la última llamada
    # --------------------------------------------------------------------------
    st.markdown("### 📊 Métricas de la última llamada")
    metrics_placeholder = st.empty()


def render_sidebar_metrics(placeholder, metrics_data: dict | None):
    """Render the execution metrics into the sidebar placeholder."""
    with placeholder.container():
        if metrics_data:
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Proveedor", value=metrics_data.get("provider", "N/A"))
                tokens_in = metrics_data.get("usage", {}).get("input_tokens", 0)
                st.metric(label="Tokens Entrada", value=f"{tokens_in:,}")
            with col2:
                st.metric(label="Tiempo", value=f"{metrics_data.get('duration', 0.0):.2f} s")
                tokens_out = metrics_data.get("usage", {}).get("output_tokens", 0)
                st.metric(label="Tokens Salida", value=f"{tokens_out:,}")

            total_tok = metrics_data.get("usage", {}).get("total_tokens", 0)
            st.metric(label="Tokens Totales", value=f"{total_tok:,}")
            st.caption(f"Modelo reportado: `{metrics_data.get('model', 'desconocido')}`")
        else:
            st.info(
                "Aún no se ha realizado ninguna llamada. Envía una transcripción para "
                "ver las métricas de respuesta y consumo de tokens aquí."
            )


render_sidebar_metrics(metrics_placeholder, st.session_state.last_metrics)

with st.sidebar:
    st.divider()

    # --------------------------------------------------------------------------
    # System Prompt Activo (Solo lectura)
    # --------------------------------------------------------------------------
    st.markdown("### ⚙️ System Prompt Activo")
    st.caption("Prompt base del estimador con rol y directrices (solo lectura):")
    with st.expander("Ver System Prompt completo", expanded=False):
        st.text_area(
            label="System Prompt",
            value=system_prompt_content,
            height=260,
            disabled=True,
            label_visibility="collapsed",
        )

    st.divider()

    # --------------------------------------------------------------------------
    # Contexto Estático Inyectado (Ejemplos CAG)
    # --------------------------------------------------------------------------
    st.markdown(f"### 🧠 Contexto CAG Inyectado ({len(ESTIMATION_EXAMPLES)})")
    st.caption(
        "Estimaciones de referencia estáticas pre-cargadas en el prompt para guiar "
        "la precisión, formato y estructura de costes del modelo:"
    )

    with st.expander("Ver ejemplos de referencia", expanded=False):
        for idx, ex in enumerate(ESTIMATION_EXAMPLES, start=1):
            st.markdown(f"#### Ejemplo {idx}")
            st.markdown(f"**Resumen de la reunión:**\n*{ex['meeting_summary']}*")
            with st.expander(f"Ver desglose estimación {idx}"):
                st.markdown(ex["estimation"])
            st.divider()

    # --------------------------------------------------------------------------
    # Limpiar historial
    # --------------------------------------------------------------------------
    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_metrics = None
        st.rerun()


# ==============================================================================
# MAIN AREA: Interfaz de Chat y Streaming (Nivel 1 & 2)
# ==============================================================================
st.markdown(
    '<div class="main-header">Estimador de Proyectos de Software <span class="badge-cag">CAG</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="main-sub">Pega la transcripción de una reunión con el cliente o describe los '
    "requerimientos del proyecto. El modelo generará un desglose de tareas, estimación de horas, "
    "costes en EUR, equipo sugerido y semanas de entrega.</div>",
    unsafe_allow_html=True,
)

# Preset examples for fast testing
SAMPLE_TRANSCRIPTIONS = [
    {
        "label": "📲 App Móvil Reservas Restaurantes",
        "text": (
            "The client wants to build a mobile app for managing restaurant reservations. "
            "They need user registration, a restaurant search with filters by cuisine and location, "
            "a real-time reservation system with availability checking, push notifications for "
            "reservation confirmations and reminders, and an admin panel for restaurant owners to "
            "manage their listings and view analytics."
        ),
    },
    {
        "label": "🛒 Marketplace B2B Mayorista",
        "text": (
            "We met with a wholesale distribution client. They need a B2B ordering portal "
            "connecting suppliers with retail shops. Key features: company verification and VAT check, "
            "custom price lists per client tier, bulk volume discounts, purchase orders with PDF "
            "generation, payment term tracking (30/60 days), and webhook integration with SAP ERP."
        ),
    },
]

# Quick demo pills if chat is empty
prompt_from_pill = None
if not st.session_state.messages:
    st.markdown("##### 💡 O prueba con un caso de ejemplo:")
    cols = st.columns(len(SAMPLE_TRANSCRIPTIONS))
    for i, sample in enumerate(SAMPLE_TRANSCRIPTIONS):
        with cols[i]:
            if st.button(sample["label"], key=f"sample_{i}", use_container_width=True):
                prompt_from_pill = sample["text"]

# Display persistent conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input: from text input or sample button
user_input = st.chat_input("Escribe o pega aquí la transcripción de la reunión...")
prompt = user_input or prompt_from_pill

if prompt:
    # Check settings validity before submitting
    if isinstance(settings_or_error, Exception):
        st.error(
            f"No se puede contactar al proveedor LLM: {settings_or_error}. "
            "Configura tu API key en `.env` o `.streamlit/secrets.toml`."
        )
    else:
        # Display user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant response with real-time streaming
        with st.chat_message("assistant"):
            metrics_captured = {}
            try:
                # Stream the response token by token
                stream = stream_estimation(
                    st.session_state.messages,
                    metrics_collector=metrics_captured,
                )
                full_response = st.write_stream(stream)

                # Persist assistant response and metrics
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.session_state.last_metrics = metrics_captured
                render_sidebar_metrics(metrics_placeholder, metrics_captured)

            except LLMServiceError as exc:
                st.error(f"❌ Error del servicio LLM: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"❌ Error inesperado: {exc}")

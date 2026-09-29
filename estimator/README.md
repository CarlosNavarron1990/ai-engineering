# Estimator CAG - Servicio de Estimación de Software con IA

Servicio de estimación de proyectos de software impulsado por IA, utilizando una arquitectura **Cache Augmented Generation (CAG)** con prompts modulares Jinja2 y contratos fuertemente tipados con **Pydantic v2**.

---

## Novedades Pre-Sesión 04: Formulario Tipado y Prompts Versionados Jinja2

En la Sesión 03 el estimator dependía de un chat con textarea libre y prompts hardcodeados como f-strings dentro del código. En esta versión:
1. **Formulario tipado en el frontend:** El cliente Streamlit (`streamlit_app.py`) sustituye el chat por un formulario con `st.form` que produce un `EstimationRequest` validado.
2. **Prompts versionados en Jinja2:** Se extrae el prompting del código a plantillas versionadas bajo `app/prompts/estimation/v1/` y `v2/`, gestionadas por un loader (`app/prompts/loader.py`) configurado con `StrictUndefined`, `trim_blocks=True` y `lstrip_blocks=True`.
3. **Endpoint refactorizado:** `POST /api/v1/estimate` y `POST /estimate` consumen `EstimationRequest`, envían al LLM los roles `system` y `user` de forma separada vía `LLMWrapper`, y devuelven `EstimationResponse(text=..., prompt_version=...)`.
4. **Extensiones bonus implementadas:**
   - **Versionado real:** Plantillas `v2/` con enfoque Lean & Agile MVP, invocable mediante `?prompt_version=v2`.
   - **Proyectos de referencia:** Soporte para `reference_projects` opcionales en el schema recorridos con `{% for %}` en el template.
   - **Observabilidad:** Emisión de eventos `structlog` con `content_hash` SHA-256 del prompt renderizado.

---

## Estructura del Proyecto

```
estimator/
├── app/
│   ├── main.py                    # Aplicación FastAPI, health check, CORS y routers
│   ├── config.py                  # Configuración con Pydantic Settings
│   ├── dependencies.py            # Inyección de dependencias (LLMWrapper, cache)
│   ├── routers/
│   │   └── estimations.py         # Endpoints POST /estimate y POST /estimate/stream
│   ├── schemas/
│   │   ├── __init__.py            # Re-export de schemas
│   │   └── estimation.py          # Modelos Pydantic v2 (EstimationRequest, EstimationResponse, Enums)
│   ├── prompts/
│   │   ├── loader.py              # Loader Jinja2 con StrictUndefined y hash structlog
│   │   └── estimation/
│   │       ├── v1/                # Prompt v1: Consultoría clásica y desglose detallado
│   │       │   ├── system.j2      # Rol, tasas, formateo condicional e inclusión de ejemplos
│   │       │   ├── user.j2        # Bloque <project_description> y referencia de proyectos
│   │       │   └── examples.j2    # 3 ejemplos few-shot de estimaciones reales
│   │       └── v2/                # Prompt v2: Enfoque ágil, lean MVP y contingencia
│   │           ├── system.j2
│   │           ├── user.j2
│   │           └── examples.j2
│   └── services/
│       ├── llm_wrapper.py         # Wrapper LiteLLM, cache exact-match y fallback
│       ├── cache.py               # Cache Redis
│       └── evaluation.py          # Evaluador estructural
├── streamlit_app.py               # Frontend Streamlit con formulario tipado
├── tests/
│   ├── prompts/
│   │   └── test_estimation_v1.py  # Tests unitarios del template Jinja2 (milisegundos)
│   ├── test_estimate_endpoint.py  # Tests de integración del endpoint POST /estimate
│   ├── test_cache.py              # Tests de cache Redis
│   ├── test_llm_wrapper.py        # Tests del wrapper LiteLLM
│   └── test_health.py             # Health check
└── pyproject.toml                 # Dependencias (incluye jinja2)
```

---

## Contrato de Datos (Pydantic v2)

```python
class ProjectType(str, Enum):
    MOBILE_APP = "mobile_app"
    WEB_SAAS = "web_saas"
    INTERNAL_TOOL = "internal_tool"
    DATA_PIPELINE = "data_pipeline"

class DetailLevel(str, Enum):
    SUMMARY = "summary"
    MEDIUM = "medium"
    DETAILED = "detailed"

class OutputFormat(str, Enum):
    PHASES_TABLE = "phases_table"
    LINE_ITEMS = "line_items"
    NARRATIVE = "narrative"

class EstimationRequest(BaseModel):
    description: str = Field(min_length=20, max_length=2000)
    project_type: ProjectType
    detail_level: DetailLevel
    output_format: OutputFormat
    reference_projects: list[ReferenceProject] | None = None

class EstimationResponse(BaseModel):
    text: str
    prompt_version: str
```

---

## Cómo Levantar el Proyecto

### 1. Variables de Entorno

Copia el archivo `.env.example` y define tus API keys:

```bash
cp .env.example .env
# Configura OPENAI_API_KEY o ANTHROPIC_API_KEY
```

### 2. Iniciar el Servicio FastAPI

```bash
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

El servicio estará disponible en:
- API Base: `http://localhost:8000`
- Documentación interactiva Swagger: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### 3. Iniciar la Interfaz Streamlit

En otra terminal:

```bash
uv run streamlit run streamlit_app.py
```

Abre [http://localhost:8501](http://localhost:8501) en tu navegador para interactuar con el formulario.

---

## Ejecución de Tests

Los tests de plantillas Jinja2 y del endpoint corren en milisegundos sin tocar APIs externas:

```bash
# Ejecutar toda la suite de pruebas (41 tests)
uv run pytest

# Ejecutar únicamente los tests de plantillas Jinja2 (v1 y v2)
uv run pytest tests/prompts/test_estimation_v1.py -v

# Ejecutar tests del endpoint POST /estimate
uv run pytest tests/test_estimate_endpoint.py -v
```

---

## Ejemplos de Peticiones con `curl`

### Petición estándar (Prompt v1 con Phases Table)

```bash
curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Plataforma B2B para gestión de suscripciones recurrentes, integración con Stripe Billing, roles y permisos.",
    "project_type": "web_saas",
    "detail_level": "detailed",
    "output_format": "phases_table"
  }'
```

### Petición con Prompt v2 (Agile / Lean) y Proyecto de Referencia

```bash
curl -X POST "http://localhost:8000/api/v1/estimate?prompt_version=v2" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Aplicación móvil para conductores y repartidores con geolocalización en tiempo real y soporte offline.",
    "project_type": "mobile_app",
    "detail_level": "detailed",
    "output_format": "phases_table",
    "reference_projects": [
      {
        "name": "Fleet Tracker MVP",
        "description": "App de seguimiento de flotas en Flutter",
        "duration_weeks": 6,
        "cost_eur": 18000.0
      }
    ]
  }'
```

---

## Objetivos de Aprendizaje Demostrados

- **Formulario tipado vs. chat libre:** Cuando el espacio de tareas es acotado (estimación técnica), un formulario restringe el espacio de estados, previene alucinaciones por ambigüedad y garantiza que el LLM reciba todas las variables críticas (tipo de proyecto, profundidad y formato) con validación previa de tipos.
- **Templates Jinja2 con Loader vs. f-strings:** Separar la lógica de negocio de la formulación del prompt facilita iteraciones, desacopla el ciclo de vida del prompt del código del endpoint, y permite auditoría de cambios.
- **Versionado explícito (`v1/`, `v2/`):** La convención de directorios permite canary deployments, tests A/B en producción y rollback inmediato sin romper contratos existentes.
- **Cobertura de tests de template:** Los tests verifican interpolación exacta, renderizado condicional de bloques y cumplimiento de `StrictUndefined` instantáneamente, aislando la lógica del template del comportamiento estocástico del modelo.

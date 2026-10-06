# Estimator — Servicio IA de estimación de software

Servicio IA en FastAPI que estima proyectos de software a partir de un formulario tipado. Es la pieza Python del programa **Master en AI Engineering**: un endpoint pensado para ser consumido por un backend de negocio (Rails, Streamlit u otro), no por un usuario final.

A partir de la **Sesión 04** el contrato es deliberadamente estrecho:
- entrada tipada (`description` + tres enums),
- salida en texto libre,
- prompt fuera del código en templates Jinja2 versionados (`app/prompts/<use_case>/<version>/`).

La inteligencia adicional (output estructurado, guardrails, cache semántico) se construye encima de esta base en directo.

## Cómo levantar

### Con Docker (recomendado)

```bash
cd estimator
cp .env.example .env  # añade al menos OPENAI_API_KEY o ANTHROPIC_API_KEY
docker compose up --build
```

El servicio queda en `http://localhost:8000` (Swagger en `/docs`, health en `/health`). Redis arranca como servicio vecino para el cache exact-match del wrapper.

### Sin Docker

```bash
cd estimator
uv sync
uv run uvicorn app.main:app --reload
```

### Probar el endpoint

```bash
curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "A small B2B SaaS to manage employee equipment loans across teams. Role-based access, audit trail, weekly digest.",
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table"
  }'
```

Respuesta:

```json
{
  "text": "| phase | duration_weeks | cost_eur | confidence_pct | …",
  "prompt_version": "v1"
}
```

### Cliente Streamlit

El cliente Streamlit es un formulario que construye el JSON y muestra el `text` recibido. Corre fuera de Docker y consume la API por HTTP:

```bash
cd estimator
uv run streamlit run streamlit_app.py
# Abrir http://localhost:8501
```

La URL del servicio se lee de `ESTIMATOR_API_BASE_URL` (default `http://localhost:8000`).

## Cómo testar

```bash
cd estimator
uv run pytest
```

La batería corre en milisegundos sin tocar APIs externas. Cubre cuatro categorías:

- `tests/test_schemas.py` — validaciones del `EstimationRequest` (longitudes, enums, campos obligatorios).
- `tests/test_prompts.py` — render del template `v1`: `description` aparece dentro de `<project_description>`, los bloques condicionales por `output_format` y `detail_level` solo se incluyen cuando aplica, y `StrictUndefined` falla early ante variables faltantes.
- `tests/test_estimate_endpoint.py` — endpoint con el wrapper LLM mockeado vía `app.dependency_overrides`: comprueba el contrato 200/422, que `system_prompt` y `user_message` viajan separados, y que la respuesta lleva `prompt_version="v1"`.
- `tests/test_llm_wrapper.py` y `tests/test_cache.py` — wrapper y cache de la Sesión 03, intactos.

## Estructura del proyecto

```
estimator/
├── app/
│   ├── main.py                        # FastAPI app, CORS, lifespan, /health
│   ├── config.py                      # Settings (Pydantic Settings, .env)
│   ├── dependencies.py                # Singletons cacheados: cache + LLMWrapper
│   ├── routers/
│   │   └── estimations.py             # POST /api/v1/estimate
│   ├── schemas/
│   │   └── estimation.py              # EstimationRequest, EstimationResponse, enums
│   ├── prompts/
│   │   ├── loader.py                  # Environment Jinja2 + render_estimation_prompt
│   │   └── estimation/
│   │       └── v1/
│   │           ├── system.j2          # rol + reglas + bloques condicionales + include
│   │           ├── user.j2            # bloque <project_description>
│   │           └── examples.j2        # few-shot examples
│   └── services/
│       ├── llm_wrapper.py             # LiteLLM Router con fallback y cost tracking
│       └── cache.py                   # Redis exact-match cache
├── tests/
│   ├── test_schemas.py
│   ├── test_prompts.py
│   ├── test_estimate_endpoint.py
│   ├── test_llm_wrapper.py
│   └── test_cache.py
├── streamlit_app.py                   # Formulario que consume /api/v1/estimate
├── Dockerfile                         # Multi-stage con uv
├── docker-compose.yml                 # Servicio IA + Redis
└── pyproject.toml
```

### Versionado de prompts

La estructura `app/prompts/<use_case>/<version>/` no es opcional: `v1/` ya existe desde el primer día porque versionar un prompt es la forma más barata de habilitar A/B testing y rollback en producción. Cuando una iteración del prompt se cocina, se crea `v2/` al lado y `render_estimation_prompt(request, version="v2")` lo recoge sin tocar router ni schemas.

Lo que vive **fuera** del template (en código): el contrato (`EstimationRequest`), el switch de versión y el wrapper. Todo lo demás (rol del modelo, reglas, ejemplos, formatos de salida, niveles de detalle) vive dentro del `.j2`. Si para cambiar el comportamiento del modelo hay que tocar Python, la separación está rota.

## Variables de entorno

| Variable | Default | Notas |
|---|---|---|
| `OPENAI_API_KEY` | — | Requerido al menos uno de los dos |
| `ANTHROPIC_API_KEY` | — | Requerido al menos uno de los dos |
| `PRIMARY_MODEL` | `gpt-4o-mini` | Modelo principal del Router |
| `FALLBACK_MODEL` | `claude-haiku-4-5-20251001` | Se usa si el primario falla |
| `REDIS_URL` | `redis://localhost:6379` | Cache exact-match |
| `CACHE_TTL` | `86400` | Segundos |
| `APP_ENV` | `development` | Controla el renderer de structlog |
| `ESTIMATOR_API_BASE_URL` | `http://localhost:8000` | Lo lee el cliente Streamlit |

`get_settings()` es un singleton cacheado con `lru_cache`: cualquier cambio en `.env` requiere reiniciar uvicorn (no basta con `--reload`).

---

## Sesión 05 — Sesiones Conversacionales, Memoria y Adjuntos

En esta fase el estimador evoluciona de un servicio *single-shot* a un servicio conversacional multi-turno capaz de refinar requerimientos y procesar documentos adjuntos.

### 1. Nuevos Endpoints

- **`POST /sessions`**: Crea una nueva sesión en memoria y devuelve `{"session_id": "uuid"}`.
- **`POST /sessions/{session_id}/estimate`**: Endpoint multi-turno que acepta `multipart/form-data`:
  - `transcript`: Texto de la transcripción o mensaje del turno.
  - `attachments`: Archivos adjuntos opcionales (`.pdf`, `.docx`, `.txt`).
  Devuelve la estimación actualizada junto con el `project_metadata` acumulado.

### 2. Decisiones Arquitectónicas

#### A. Gestión de Adjuntos: Camino B (Extracción Local)
Se implementó **Camino B (extracción local con `pypdf` y `python-docx`)** en lugar de Camino A (Files API de proveedores):
- **Independencia de Proveedor**: El texto extraído se concatena de forma limpia al prompt (`--- attachment: filename ---`), permitiendo que el backend cambie transparentemente entre OpenAI, Anthropic o modelos locales sin acoplamiento a APIs propietarias.
- **Eficiencia y Costos**: Evita sobrecostes de procesamiento multimodal en documentos donde el contenido es puramente textual.
- **Preparación para RAG (Módulo 3)**: El pipeline de extracción local sienta las bases para futuros componentes de chunking, embedding y búsqueda vectorial sobre documentación técnica.

#### B. Extracción de `project_metadata` (Heurística de Hechos)
Para actualizar la memoria del proyecto se utiliza un extractor basado en reglas y patrones heurísticos sobre el turno:
- **Cero Latencia Incremental**: No añade un segundo roundtrip de llamada LLM por cada interacción, manteniendo las respuestas ágiles.
- **Determinismo**: Acumula tecnologías mencionadas sin duplicados (`mentioned_technologies`), detecta el tamaño de equipo y nombres de proyecto acordados.
- **Inyección en System Prompt**: Se inyecta en el bloque `<project_metadata>` del template Jinja2, manteniendo al LLM anclado a los hechos acordados incluso cuando los mensajes antiguos salen de la ventana deslizante.

#### C. Separación Operativa: Historial vs Memoria
- **Historial (`messages`)**: Gestionado con una **ventana deslizante** de 6 turnos (`MAX_TURNS = 6`). Evita el desbordamiento de contexto y controla el coste por petición.
- **Memoria (`project_metadata`)**: Vive fuera del historial. Preserva los hechos del proyecto de forma persistente a lo largo de toda la sesión.

### 3. Ejecución de Tests

```bash
UV_LINK_MODE=copy uv run pytest
```

---

> Este proyecto forma parte del **Master en AI Engineering**.

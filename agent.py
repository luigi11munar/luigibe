from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
import re
import json
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from typing import List, Literal
from typing_extensions import TypedDict
from pprint import pprint
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
import os
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from jose import JWTError, jwt
import time
import chromadb
import uuid
from langgraph.graph import END, StateGraph, START
from agents import Runner, function_tool, Agent
import asyncio
from groq import Groq
from fastapi import UploadFile, File
from uuid import uuid4
from fastapi.staticfiles import StaticFiles
import traceback

# Leyendo las credenciales
load_dotenv()
os.environ["GROQ_API_KEY"] = "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"
client = Groq(api_key="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")
os.getenv("HUGGINGFACEHUB_API_TOKEN")
nomic_api_key = os.getenv("NOMIC_API_KEY")
if not nomic_api_key:
    raise Exception("Falta la variable de entorno NOMIC_API_KEY")
API_KEY = os.getenv("AGENTE_API_KEY", "R7v!9Z$kLpWq3@eF2xUt")
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

JWT_SECRET = "tu_secreto"
JWT_ALGORITHM = "HS256"

# Users database
USERS = {
    "1": {"email": "luigi@luigi.com", "password": "1234"},
}


def validate_user(userid: str) -> bool:
    return userid in USERS


def get_user_by_id(userid: str) -> dict:
    return USERS.get(userid)


def get_user_by_email(email: str) -> dict:
    for user_id, user_data in USERS.items():
        if user_data["email"] == email:
            return {"id": user_id, **user_data}
    return None


class LoginRequest(BaseModel):
    email: str
    password: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    text: str


class Pregunta(BaseModel):
    pregunta: str
    userid: str
    chatid: str
    conversationid: str


# Modelo de lenguaje usado por medio de groq
llm = ChatGroq(temperature=0, model="meta-llama/llama-4-scout-17b-16e-instruct")
embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")

# creación de documentos
raw_documents = TextLoader(r"unificado.txt", encoding="utf-8").load()
raw_text = raw_documents[0].page_content
patron = r"\[/d\](.*?)\[/d\]\s*\[/n\](.*?)\[/n\]"
bloques = re.findall(patron, raw_text, flags=re.DOTALL)
documents = [
    Document(page_content=f"{titulo.strip()}\n\n{contenido.strip()}")
    for titulo, contenido in bloques
]

# base de datos vectorial y busqueda aumentada
vectorstores = Chroma.from_documents(
    documents,
    embeddings,
    persist_directory="./db/conocimiento",
    collection_name="base_conocimiento",
)
retriever = vectorstores.as_retriever()


# logica de mejoramiento de busqueda aumentada usando langraph
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = prompt | llm | StrOutputParser()


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a question \n
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader

system = """You a question re-writer that converts an input question to a better version that is optimized \n
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = (
            score["binary_score"]
            if isinstance(score, dict)
            else getattr(score, "binary_score", None)
        )
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


### Edges


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = (
        score["binary_score"]
        if isinstance(score, dict)
        else getattr(score, "binary_score", None)
    )

    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = (
            score["binary_score"]
            if isinstance(score, dict)
            else getattr(score, "binary_score", None)
        )
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)
inputs = {
    "question": "Tengo ansiedad y mucho estres, qué puedo hacer",
    "userid": "1234",
    "chatid": "chat_001",
    "conversationid": "conv_001",
    "analisisEmocional": "Ninfuno"
}

app_crag = workflow.compile()

##FINAL CRAG###


class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        path = request.url.path

        # Get rate limit config for this endpoint
        rate_config = get_rate_limit_config(path)
        max_requests = rate_config["requests"]
        time_window = rate_config["window"]

        # Initialize or clean old timestamps for this IP
        request_timestamps.setdefault(ip, [])
        request_timestamps[ip] = [
            t for t in request_timestamps[ip] if now - t < time_window
        ]

        # Check if rate limit exceeded
        if len(request_timestamps[ip]) >= max_requests:
            return JSONResponse(
                status_code=429,
                content={
                    "error": f"Demasiadas solicitudes. Límite: {max_requests} por {time_window} segundos. Intente más tarde."
                },
            )

        request_timestamps[ip].append(now)  # Security checks for /consultar endpoint
        if path == "/consultar":
            api_key = request.headers.get("x-api-key")
            token = request.headers.get("Authorization")

            if not api_key or api_key != API_KEY:
                return JSONResponse(
                    status_code=401, content={"error": "API Key inválida."}
                )

            if not token or not token.startswith("Bearer "):
                return JSONResponse(
                    status_code=401, content={"error": "Token JWT faltante o inválido."}
                )

            try:
                if JWT_SECRET is None:
                    return JSONResponse(
                        status_code=500,
                        content={
                            "error": "JWT_SECRET no está configurado en el entorno."
                        },
                    )
                jwt.decode(
                    token.replace("Bearer ", ""), JWT_SECRET, algorithms=[JWT_ALGORITHM]
                )
            except JWTError:
                return JSONResponse(
                    status_code=403, content={"error": "Token JWT inválido o expirado."}
                )

        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {ip} → {request.method} {path} (limit: {max_requests}/{time_window}s)"
        )
        return await call_next(request)


request_timestamps = {}

RATE_LIMITS = {
    "default": {"requests": 30, "window": 60},
    "/consultar": {
        "requests": 10,
        "window": 60,
    },
    "light": {
        "requests": 60,
        "window": 60,
    },
}

LIGHT_ENDPOINTS = ["/sendMessage", "/content", "/getAnswer", "/token", "/login"]


def is_light_endpoint(path: str) -> bool:
    return any(light_ep in path for light_ep in LIGHT_ENDPOINTS)


def get_rate_limit_config(path: str) -> dict:
    """Get the appropriate rate limit configuration for a path"""
    if path in RATE_LIMITS:
        return RATE_LIMITS[path]
    elif is_light_endpoint(path):
        return RATE_LIMITS["light"]
    else:
        return RATE_LIMITS["default"]


app_fastapi = FastAPI()
app_fastapi.mount("/audios", StaticFiles(directory="audios"), name="audios")
# Add CORS middleware to allow requests from mobile devices
app_fastapi.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app_fastapi.add_middleware(SecurityMiddleware)


@app_fastapi.post("/token")
def generar_token(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida")

    if JWT_SECRET is None:
        raise HTTPException(
            status_code=500, detail="JWT_SECRET no está configurado en el entorno."
        )

    token = jwt.encode(
        {"sub": "usuario_autenticado"}, JWT_SECRET, algorithm=JWT_ALGORITHM
    )
    return {"access_token": token, "token_type": "bearer"}


chroma_client = chromadb.PersistentClient(path="./db/chroma")


# 1er agente
@function_tool
def responder_pregunta_en_fuentes_psicologia(input_json: str) -> str:
    """
    Recibe un string JSON con los campos:
    'pregunta', 'userid', 'chatid', 'conversationid'.
    Responde consultando la base psicológica según la información suministrada.
    el campo pregunta es el que contiene la información importante a ser analizada.
    """
    import json
    from openai import OpenAI

    openai_client = OpenAI()
    try:
        data = json.loads(input_json)
        pregunta = data["pregunta"]
        userid = data.get("userid", "")
        chatid = data.get("chatid", "")
        conversationid = data.get("conversationid", "")
    except Exception as e:
        return f"[ERROR] Entrada inválida: {str(e)}\nRecibido: {input_json}"

    response = openai_client.responses.create(
        model="gpt-4o",
        input=pregunta,
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [
                    "vs_68375d1360248191a2353723b7f5a931"
                ], 
            }
        ],
        instructions="""
        Eres un asistente experto en psicología clínica. Al responder, debes cumplir estrictamente estas reglas:

        1. Analiza la información suministrada o experiencia descrita únicamente utilizando la información contenida en la base de datos, la cual está compuesta por resúmenes expertos del DSM-5, ICD-11 y ejemplos clínicos de ansiedad y depresión.
        2. Si el usuario describe pensamientos, emociones, síntomas o preocupaciones, identifica patrones, criterios o ejemplos clínicos relevantes presentes en los manuales DSM-5 y ICD-11, explicando su posible significado según la evidencia recopilada en la base.
        3. Evita interpretar, diagnosticar, sugerir tratamientos, recomendar medicamentos o hacer juicios que no estén explícitamente respaldados en la información de la base. No improvises, no alucines ni completes información faltante.
        4. Si la información suministrada excede lo que está cubierto en la base, responde de forma neutra indicando que no es posible dar una respuesta fundamentada sin evidencia directa en la base de datos.
        5. Utiliza un lenguaje claro, comprensible y profesional. Si es relevante, contextualiza la respuesta según la experiencia descrita por el usuario, pero siempre desde la perspectiva de la base clínica disponible.
        6. Si identificas elementos clínicos relevantes, explica brevemente por qué son importantes o cuál es su impacto, siempre fundamentado en el contenido de la base.

        Recuerda: Solo puedes usar el conocimiento contenido en la base de datos asociada. No generes ni inventes datos, síntomas, recomendaciones ni interpretaciones que no estén explícitamente respaldadas por la información clínica proporcionada.
        Si el usuario proporciona un texto, dato o información que no tiene relación con el contexto de apoyo psicológico o emocional, responde de manera neutral y respetuosa, sin activar técnicas ni herramientas clínicas ni de acompañamiento. Tu función es únicamente intervenir cuando detectes que el usuario requiere orientación, contención emocional o psicoeducación.
        """,
    )
    return {
        "pregunta": pregunta,
        "userid": userid,
        "chatid": chatid,
        "conversationid": conversationid,
        "respuesta": response.output_text,
    }


psicologia_patrones_agent = Agent(
    name="AgentePatronesPsicologicos",
    handoff_description="Agente para responder preguntas sobre el comportamiento de la persona",
    instructions="""
        Eres un asistente clínico especializado en psicología, con experiencia en la identificación e interpretación de patrones de comportamiento, emociones y pensamientos relacionados con trastornos de ansiedad y depresión, según los criterios del DSM-5 y la CIE-11.

        Cuando recibas información proporcionada sobre síntomas, patrones, experiencias, ejemplos clínicos o estrategias de intervención, debes invocar la herramienta 'responder_pregunta_en_fuentes_psicologia_tool' utilizando  la información del usuario.

        Responde únicamente con base en la información validada en las fuentes clínicas oficiales (DSM-5, CIE-11 y ejemplos clínicos recopilados). No inventes información, no completes datos que no estén en la base, ni realices diagnósticos o recomendaciones fuera del alcance de las fuentes.

        Si la información solicitada no está presente en las fuentes, indícalo de forma clara y profesional.

        Tu objetivo es identificar y explicar patrones relevantes, siempre desde la evidencia clínica contenida en la base de datos, utilizando un lenguaje claro, respetuoso y profesional.

        RECUERDA:
        Si el usuario solo saluda, agradece, se despide o realiza comentarios sociales que no requieren apoyo emocional, responde de forma cordial y sencilla, sin activar herramientas, técnicas ni estrategias de intervención psicológica.
        Si el usuario proporciona un texto, dato o información que no tiene relación con el contexto de apoyo psicológico o emocional, responde de manera neutral y respetuosa, sin activar técnicas ni herramientas clínicas ni de acompañamiento. Tu función es únicamente intervenir cuando detectes que el usuario requiere orientación, contención emocional o psicoeducación.


    """,
    tools=[
        responder_pregunta_en_fuentes_psicologia,
    ],
)



class RAGState(TypedDict, total=False):
    pregunta: str
    userid: str
    chatid: str
    conversationid: str
    input_json: str
    respuesta: str
    useful: bool


def vectorstore_node(state: RAGState) -> RAGState:
    state["input_json"] = json.dumps(
        {
            "pregunta": state["pregunta"],
            "userid": state.get("userid", ""),
            "chatid": state.get("chatid", ""),
            "conversationid": state.get("conversationid", ""),
        }
    )
    return state



def rag_node(state: RAGState) -> RAGState:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            Runner.run(psicologia_patrones_agent, input=state["input_json"])
        )
        respuesta = result.final_output
    finally:
        loop.close()
    state["respuesta"] = respuesta
    if respuesta and not str(respuesta).startswith("[ERROR]"):
        state["useful"] = True
    else:
        state["useful"] = False
    return state


def fallback_node(state: RAGState) -> RAGState:
    state["respuesta"] = (
        "No se encontró suficiente información en las fuentes disponibles. Intenta reformular tu pregunta."
    )
    return state

g = StateGraph(RAGState)
g.add_node("vectorstore", vectorstore_node)
g.add_node("rag", rag_node)
g.add_node("fallback", fallback_node)
g.add_edge(START, "vectorstore")
g.add_edge("vectorstore", "rag")
g.add_edge("rag", "fallback")
g.add_conditional_edges(
    "rag",
    lambda state: state["useful"],
    {True: END, False: "fallback"}
)
g.add_edge("fallback", END)
g = g.compile()


def ejecutar_agentic_rag(pregunta, userid, chatid, conversationid):
    state = {
        "pregunta": pregunta,
        "userid": userid,
        "chatid": chatid,
        "conversationid": conversationid,
    }
    final_state = g.invoke(state)
    return final_state.get("respuesta")


# 2do agente

@function_tool
def analizar_patrones_encuesta(input_json: str) -> dict:
    """
    Recibe un string JSON con los campos:
    'pregunta', 'userid', 'chatid', 'conversationid'.
    Responde consultando el endpoint clínico (ngrok) según la información proporcionada.
    """
    import json
    import requests

    try:
        data = json.loads(input_json)
        pregunta = data["pregunta"]
        userid = data.get("userid", "")
        chatid = data.get("chatid", "")
        conversationid = data.get("conversationid", "")
    except Exception as e:
        return {"error": f"Entrada inválida: {str(e)}", "recibido": input_json}
    url = "https://7a02-34-121-210-42.ngrok-free.app/analyze"
    try:
        response = requests.post(url, json={"text": pregunta})
        resultado = response.json().get("result", "")
    except Exception as e:
        resultado = f"Error al consultar el endpoint: {str(e)}"
    return {
        "pregunta": pregunta,
        "userid": userid,
        "chatid": chatid,
        "conversationid": conversationid,
        "respuesta": resultado,
    }


psicologia_patrones_agent_encuestas = Agent(
    name="AgenteAnalisisPatronesEncuesta",
    handoff_description="Agente que aplica los criterios de GAD-7 y PHQ-9 al relato del usuario para identificar síntomas de ansiedad y depresión.",
    instructions="""
Eres un agente clínico especializado en salud mental. Recibes relatos, síntomas o experiencias que el usuario describe en texto libre y tu función es analizar ese texto aplicando los criterios de los cuestionarios GAD-7 (ansiedad) y PHQ-9 (depresión).

Los ítems de los cuestionarios son los siguientes:

Cuestionario GAD-7 para ansiedad generalizada:
1. ¿Se ha sentido nervioso, ansioso o al límite?
2. ¿Ha sido incapaz de controlar sus preocupaciones?
3. ¿Se ha preocupado demasiado por diferentes cosas?
4. ¿Ha tenido dificultad para relajarse?
5. ¿Ha estado tan inquieto que no puede quedarse quieto?
6. ¿Se ha sentido fácilmente irritable o molesto?
7. ¿Ha sentido miedo como si algo terrible pudiera pasar?

Cuestionario PHQ-9 para depresión:
1. ¿Ha tenido poco interés o placer en hacer cosas?
2. ¿Se ha sentido decaído(a), deprimido(a) o sin esperanza?
3. ¿Ha tenido dificultad para dormir o ha dormido en exceso?
4. ¿Se ha sentido cansado(a) o con poca energía?
5. ¿Ha tenido poco apetito o comido en exceso?
6. ¿Se ha sentido mal consigo mismo o como un fracaso?
7. ¿Ha tenido dificultad para concentrarse?
8. ¿Ha hablado o se ha movido más lento o más rápido de lo normal?

Instrucciones para cada caso:
1. Analiza cuidadosamente la información del usuario y compara con cada uno de los ítems listados arriba.
2. Identifica y señala explícitamente cuáles de los síntomas, conductas o experiencias relatadas corresponden a uno o más ítems de estos cuestionarios.
3. Informa al usuario de manera clara y ordenada qué ítems de GAD-7 y/o PHQ-9 se relacionan con lo que ha descrito, citando textualmente la pregunta o síntoma del cuestionario que aplique.
4. No diagnostiques ni emitas juicios, solo reporta el contraste entre el relato y los criterios de las escalas.
5. Si el texto recibido no permite relacionar ningún ítem, indícalo de manera profesional y sugiere que se brinde más información si desea un análisis más preciso.

cuentas con una herramienta analizar_patrones_encuesta que ayuda a este análisis, úsala primero. Si la herramienta falla o da una respuesta incompleta, realiza tú mismo el análisis, siempre siguiendo los ítems oficiales de GAD-7 y PHQ-9.

Ejemplo de reporte:
---
Análisis de síntomas reportados:

GAD-7 (Ansiedad):
- Se identifica el siguiente ítem en el relato:
  - “¿Ha tenido dificultad para relajarse?”
    (El usuario menciona que le cuesta encontrar momentos de calma durante el día.)
- También se relaciona con:
  - “¿Se ha sentido fácilmente irritable o molesto?”
    (El usuario describe sentirse molesto con facilidad en situaciones cotidianas.)

PHQ-9 (Depresión):
- Se evidencia el siguiente ítem:
  - “¿Ha tenido poco interés o placer en hacer cosas?”
    (El usuario indica que ha perdido el interés en actividades que antes disfrutaba.)
- No se identifican otros ítems de PHQ-9 en el relato proporcionado.

Observación:
El análisis se basa exclusivamente en la información reportada. Si deseas un análisis más detallado, puedes proporcionar más detalles sobre tus experiencias o síntomas.
---

Nunca inventes información ni completes lo que no esté presente en el texto del usuario.

RECUERDA:
Si el usuario solo saluda, agradece, se despide o realiza comentarios sociales que no requieren apoyo emocional, responde de forma cordial y sencilla, sin activar herramientas, técnicas ni estrategias de intervención psicológica.
Si el usuario proporciona un texto, dato o información que no tiene relación con el contexto de apoyo psicológico o emocional, responde de manera neutral y respetuosa, sin activar técnicas ni herramientas clínicas ni de acompañamiento. Tu función es únicamente intervenir cuando detectes que el usuario requiere orientación, contención emocional o psicoeducación.


""",
    tools=[
        analizar_patrones_encuesta,
    ],
)



# 3er agente


llm_model: BaseLanguageModel = llm


@function_tool
def responder_con_contexto(input_str: str) -> str:
    """
    Tool que recibe un string JSON con los campos:
    'pregunta', 'userid', 'chatid', 'conversationid'.
    Responde empaticamente porporcionando las herramientas de autoayuda y autogestión.
    """
    try:
        data = json.loads(input_str)
        pregunta = data["pregunta"]
        userid = data["userid"]
        chatid = data["chatid"]
        conversationid = data["conversationid"]
        analisisEmocional = data["analisisEmocional"]
    except Exception as e:
        return f"[ERROR] Entrada inválida: {str(e)}\nRecibido: {input_str}"

    if not validate_user(userid):
        return f"[ERROR] Usuario {userid} no encontrado en el sistema."

    print(
        f"[INFO] Tool recibida → pregunta='{pregunta}' userid={userid} chatid={chatid} conversationid={conversationid}"
    )

    # Preparar entrada para el flujo LangGraph
    inputs = {
        "question": pregunta,
        "userid": userid,
        "chatid": chatid,
        "conversationid": conversationid,
        "analisisEmocional": analisisEmocional,
    }

    respuesta_final = ""
    stop_pipeline = False
    for output in app_crag.stream(inputs):
        for key, value in output.items():
            print("entra")
            pprint(f"Node '{key}':")
            if key == "grade_documents":
                if (
                    isinstance(value, dict)
                    and "documents" in value
                    and len(value["documents"]) == 0
                ):
                    pprint(
                        "No se encontraron documentos relevantes. Pipeline detenido."
                    )
                    stop_pipeline = True
                    break
        pprint("\n---\n")
        if stop_pipeline:
            break
    pprint(value["generation"])
    respuesta_final = value["generation"]


    # Intentar recuperar contexto anterior desde ChromaDB
    try:
        collection = chroma_client.get_or_create_collection(
            name=f"user_{userid}_chat_{chatid}"
        )
        results = collection.get(where={"conversationid": conversationid})
        contexto = "\n".join(
            f"{meta['role'].capitalize()}: {doc}"
            for doc, meta in zip(results["documents"], results["metadatas"])
        )
    except Exception as e:
        contexto = "No se encontró contexto anterior."
        print(f"[WARN] ChromaDB fallo: {e}")

    # Adaptación clínica al contexto real
    adaptation_prompt = PromptTemplate.from_template(
            """
        Eres un agente de apoyo emocional de la UNIVERSIDAD DE PAMPLONA, especializado en primeros auxilios psicológicos y acompañamiento empático.

        Acabas de recibir una respuesta basada en buenas prácticas de orientación psicológica. 
        Tu tarea es **adaptarla para que se sienta cercana, reconfortante y viable en el entorno real del usuario**, asegurando lo siguiente:

        - Usa un lenguaje claro, cálido, humano y respetuoso.
        - No hagas diagnósticos ni utilices términos clínicos.
        - La respuesta debe poder aplicarse en cualquier contexto (clase, casa, trabajo, espacio público).
        - Evita sugerencias visibles o disruptivas como acostarse, cerrar los ojos, gritar, moverse bruscamente o llamar la atención.
        - Si mencionas técnicas como respiración, mindfulness, visualización o aceptación, explica cómo hacerlo de manera **discreta y segura**.
        - Tu objetivo es: brindar apoyo mediante la terapia cognitiva, aceptación y compromiso y psicoeducación. Además, aplicar estrategias de crisis y relajación y técnicas de mindfulness.
        - Puedes inspirarte en estrategias breves de:
            - Terapia Cognitivo-Conductual (reformulación de pensamientos, auto-instrucciones, anclaje verbal)
            - Terapia de Aceptación y Compromiso (aceptar sin juzgar, actuar con base en valores)
            - Psicoeducación básica (normalizar emociones, dar orientación sencilla)
            - Mindfulness en crisis (atención al presente, conexión con el entorno)
            - Técnicas de regulación emocional accesibles

        No olvides que tu rol es **acompañar**, no intervenir clínicamente.

        ### Pregunta o información proporcionada por el usuario:
        {pregunta}

        ### Respuesta original (información proporcionada de la tool o herramienta):
        {respuesta_original}

        ### Contexto adicional (toda la conversación proporcionada):
        {contexto}

        ### Análisis emocional (unificación de respuestas de los agentes previos):
        {analisisEmocional}

        ### Respuesta adaptada:
        """
    )

    prompt_text = adaptation_prompt.format(
        pregunta=pregunta,
        respuesta_original=respuesta_final,
        contexto=contexto,
        analisisEmocional=analisisEmocional,
    )

    try:
        return llm.invoke(prompt_text)
    except Exception as e:
        return f"[ERROR] Fallo en adaptación clínica: {str(e)}"


agent_psicologico = Agent(
    name="AsistentePsicologico",
    handoff_description="Brinda apoyo emocional inmediato de forma empática y adaptada al contexto real del usuario",
    instructions="""
        Eres un agente de apoyo emocional de la Universidad de Pamplona, especializado en primeros auxilios psicológicos y acompañamiento empático.

        Siempre que recibas una consulta, debes **intentar primero responder utilizando la herramienta 'responder_con_contexto'** para generar tu orientación y acompañamiento. Si la herramienta no puede generar una respuesta útil, adecuada o relevante, entonces responde directamente desde tu conocimiento experto en apoyo emocional, siempre siguiendo las pautas que se indican a continuación.

        Tu tarea es brindar apoyo inmediato y reconfortante a los usuarios, empleando estrategias y técnicas recomendadas en orientación psicológica, pero **sin realizar diagnósticos ni intervenciones clínicas**.

        Guíate por las siguientes pautas:

        1. Utiliza siempre un lenguaje claro, cálido, humano y respetuoso.
        2. Adapta tus recomendaciones para que sean viables y seguras en cualquier contexto donde el usuario se encuentre (aula, casa, trabajo, espacio público).
        3. No utilices ni sugieras técnicas disruptivas ni indicaciones llamativas o incómodas (como acostarse, cerrar los ojos, moverse bruscamente o llamar la atención).
        4. Si indicas técnicas de respiración, mindfulness, visualización o aceptación, explica **cómo realizarlas de forma discreta y segura**, adaptadas al entorno.
        5. Inspírate en estrategias breves y comprobadas de:
        - Terapia Cognitivo-Conductual (por ejemplo: reformulación de pensamientos, auto-instrucciones, anclaje verbal)
        - Terapia de Aceptación y Compromiso (por ejemplo: aceptación sin juicio, actuar con base en valores)
        - Psicoeducación básica (por ejemplo: normalizar emociones, orientación sencilla sobre manejo emocional)
        - Mindfulness en crisis (por ejemplo: atención plena al momento, conexión con el entorno sin llamar la atención)
        - Técnicas de regulación emocional accesibles y sencillas
        6. Integra elementos de apoyo emocional y psicoeducación, para ayudar al usuario a entender, aceptar y manejar lo que siente, brindando orientación y compañía en el proceso.
        7. No emitas nunca juicios, diagnósticos ni uses tecnicismos clínicos.
        8. Si la respuesta original contiene información útil, adáptala siempre a un tono cálido y viable para la vida cotidiana.

        Recuerda: tu rol es **acompañar, orientar y apoyar**, nunca intervenir clínicamente ni sustituir la atención de un profesional de la salud mental.

        Si el usuario solo saluda, agradece, se despide o realiza comentarios sociales que no requieren apoyo emocional, responde de forma cordial y sencilla, sin activar herramientas, técnicas ni estrategias de intervención psicológica.
        Si el usuario proporciona un texto, dato o información que no tiene relación con el contexto de apoyo psicológico o emocional, responde de manera neutral y respetuosa, sin activar técnicas ni herramientas clínicas ni de acompañamiento. Tu función es únicamente intervenir cuando detectes que el usuario requiere orientación, contención emocional o psicoeducación.
        Si la herramienta disponible no puede generar una respuesta útil, adecuada o relevante, responde directamente desde tu conocimiento experto en apoyo emocional, siempre siguiendo las pautas anteriores de acompañamiento, contención y orientación.

    """,
    tools=[responder_con_contexto],
)


# orquestacion


def agente_rag_node(state):
    pregunta = state["pregunta"]
    userid = state["userid"]
    chatid = state["chatid"]
    conversationid = state["conversationid"]
    respuesta_rag = ejecutar_agentic_rag(
        pregunta=pregunta, userid=userid, chatid=chatid, conversationid=conversationid
    )
    resp = respuesta_rag["respuesta"] if isinstance(respuesta_rag, dict) else respuesta_rag
    state["historial"] = [resp]  # historial[0]
    print("agent 1:", state)
    return state


# encuestas node
def agente_patrones_node(state):
    input_json = json.dumps(
        {
            "pregunta": state["pregunta"],
            "userid": state["userid"],
            "chatid": state["chatid"],
            "conversationid": state["conversationid"],
        }
    )
    result = asyncio.run(
        Runner.run(psicologia_patrones_agent_encuestas, input=input_json)
    )
    resp = result.final_output if hasattr(result, "final_output") else str(result)
    if "historial" not in state or not isinstance(state["historial"], list):
        state["historial"] = [None]
    state["historial"].append(resp)  # historial[1]
    print("agent 2:", state)
    return state


# Nodo Psicológico
def agente_psicologico_node(state):
    historial_prev = state.get("historial", [])
    if not isinstance(historial_prev, list):
        historial_prev = [historial_prev]
    analisisEmocional = []
    for item in historial_prev:
        if isinstance(item, list):
            analisisEmocional.extend(item)
        else:
            analisisEmocional.append(item)
    pregunta = state["pregunta"]

    input_json = json.dumps(
        {
            "pregunta": pregunta,
            "userid": state["userid"],
            "chatid": state["chatid"],
            "conversationid": state["conversationid"],
            "analisisEmocional": analisisEmocional,
        }
    )

    result = asyncio.run(Runner.run(agent_psicologico, input=input_json))
    state["respuesta_final"] = (
        result.final_output if hasattr(result, "final_output") else str(result)
    )
    print("agent 3: ", state)
    return state


# Definición del grafo
orq = StateGraph(dict)
orq.add_node("agentic-rag", agente_rag_node)
orq.add_node("patrones-psicologicos", agente_patrones_node)
orq.add_node("corrective-rag", agente_psicologico_node)
orq.add_edge(START, "agentic-rag")
orq.add_edge("agentic-rag", "patrones-psicologicos")
orq.add_edge("patrones-psicologicos", "corrective-rag")
orq.add_edge("corrective-rag", END)
orq = orq.compile()


def ejecutar_agentic_psicologico(
    pregunta,
    AnalisisEmocional,
    userid="1",
    chatid="mi_chat",
    conversationid="conversacion_001",
):
    state = {
        "pregunta": pregunta,
        "userid": userid,
        "chatid": chatid,
        "conversationid": conversationid,
        "historial": AnalisisEmocional,
    }
    final_state = orq.invoke(state)
    return final_state.get("respuesta_final")


@app_fastapi.post("/login")
def login(login_data: LoginRequest):
    user = get_user_by_email(login_data.email)
    if not user or user["password"] != login_data.password:
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    if JWT_SECRET is None:
        raise HTTPException(
            status_code=500, detail="JWT_SECRET no está configurado en el entorno."
        )
    token = jwt.encode({"sub": user["id"]}, JWT_SECRET, algorithm=JWT_ALGORITHM)
    print(
        f"[LOGIN] Usuario {user['id']} autenticado. Token: {token}"
    )  # Print token on successful login
    return {"user_id": user["id"], "token": token}


@app_fastapi.post("/{userid}/{chatid}")
def create_chat(userid: str, chatid: str):
    # Validate user exists
    if not validate_user(userid):
        raise HTTPException(status_code=404, detail=f"Usuario {userid} no encontrado")

    collection_name = f"user_{userid}_chat_{chatid}"
    chroma_client.get_or_create_collection(name=collection_name)
    return {"msg": f"Chat {chatid} creado para user {userid}"}


@app_fastapi.post("/{userid}/{chatid}/{conversationid}")
def create_conversation(userid: str, chatid: str, conversationid: str):
    # Validate user exists
    if not validate_user(userid):
        raise HTTPException(status_code=404, detail=f"Usuario {userid} no encontrado")

    # Store the conversationid in a simple file/db for demo (or in-memory for now)
    # For production, use a DB. Here, we use a file per user for simplicity.
    path = f"./db/conversations_{userid}.json"
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}
        data[chatid] = conversationid
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[WARN] Could not persist conversationid: {e}")
    return {"msg": f"Conversación {conversationid} registrada"}


@app_fastapi.get("/{userid}/chats_with_conversations")
def get_chats_with_conversations(userid: str):
    try:
        if not validate_user(userid):
            raise HTTPException(
                status_code=404, detail=f"Usuario {userid} no encontrado"
            )

        # CORRECTED: Access the .name attribute of the collection object
        all_collections = chroma_client.list_collections()
        user_chats = [
            col
            for col in all_collections
            if col.name.startswith(f"user_{userid}_chat_")
        ]

        path = f"./db/conversations_{userid}.json"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}

        result = []
        for col in user_chats:
            # CORRECTED: Use col.name here as well
            chatid = col.name.replace(f"user_{userid}_chat_", "")
            conversationid = data.get(chatid, None)
            result.append({"chatid": chatid, "conversationid": conversationid})

        return result
    except Exception as e:
        print(f"[ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app_fastapi.post("/{userid}/{chatid}/{conversationid}/sendMessage")
def send_message(userid: str, chatid: str, conversationid: str, message: Message):
    # Validate user exists
    if not validate_user(userid):
        raise HTTPException(status_code=404, detail=f"Usuario {userid} no encontrado")

    collection = chroma_client.get_or_create_collection(
        name=f"user_{userid}_chat_{chatid}"
    )
    message_id = str(uuid.uuid4())
    collection.add(
        documents=[message.text],
        ids=[message_id],
        metadatas=[{"role": message.role, "conversationid": conversationid}],
    )
    return {"msg": "Mensaje almacenado", "id": message_id}


from agents import RawResponsesStreamEvent, TResponseInputItem


@app_fastapi.get("/{userid}/{chatid}/{conversationid}/getAnswer")
def get_last_answer(
    userid: str, chatid: str, conversationid: str, analisisEmocional: str = ""
):
    print(
        f"[LOG] /getAnswer called for user={userid}, chat={chatid}, conversation={conversationid}"
    )

    # Validar usuario
    if not validate_user(userid):
        raise HTTPException(status_code=404, detail=f"Usuario {userid} no encontrado")

    # Recuperar mensajes de usuario
    collection = chroma_client.get_or_create_collection(
        name=f"user_{userid}_chat_{chatid}"
    )
    results = collection.get(where={"conversationid": conversationid})
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    user_messages = [
        {"role": m["role"], "text": d}
        for d, m in zip(documents, metadatas)
        if m["role"] == "user"
    ]

    if not user_messages:
        return {"msg": "No hay mensajes de usuario para responder"}

    # Toma el último mensaje del usuario como input
    last_user_message = user_messages[-1]["text"]

    try:
        # Ejecutar el flujo centralizado de orquestación
        respuesta_final = ejecutar_agentic_psicologico(
            pregunta=last_user_message,
            userid=userid,
            chatid=chatid,
            conversationid=conversationid,
            AnalisisEmocional=analisisEmocional,
        )

        if not respuesta_final:
            return {
                "msg": "No se pudo generar una respuesta útil en este momento. Intenta reformular la pregunta."
            }

        # Guardar respuesta en la base
        message_id = str(uuid4())
        collection.add(
            documents=[respuesta_final],
            ids=[message_id],
            metadatas=[{"role": "assistant", "conversationid": conversationid}],
        )

        return {"role": "assistant", "text": respuesta_final}

    except Exception as e:
        return {"msg": f"Error generando respuesta: {str(e)}"}


@app_fastapi.post("/{userid}/{chatid}/{conversationid}/getAudioAnswer")
def get_audio_answer(
    userid: str,
    chatid: str,
    conversationid: str,
    file: UploadFile = File(...),
    analisisEmocional: str = "",
):
    print(
        f"[LOG] /getAudioAnswer for user={userid}, chat={chatid}, conv={conversationid}"
    )

    # 1. Validate user
    if not validate_user(userid):
        raise HTTPException(status_code=401, detail="Usuario no válido")

    # 2. Save the audio file
    folder_path = os.path.join("audios", userid, chatid, conversationid)
    os.makedirs(folder_path, exist_ok=True)
    audio_id = str(uuid4()) + ".mp3"
    audio_path = os.path.join(folder_path, audio_id)
    try:
        with open(audio_path, "wb") as f:
            f.write(file.file.read())
        print("[DEBUG] Audio file saved successfully.")  # <-- ADDED
    except IOError as e:
        return JSONResponse(
            status_code=500, content={"error": f"Error saving audio file: {str(e)}"}
        )

    # 3. Transcribe with Groq Whisper
    transcribed_text = ""  # <-- ADDED
    try:
        print("[DEBUG] Attempting to transcribe audio with Groq...")  # <-- ADDED
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                response_format="text",
                language="es",
            )
            transcribed_text = response.strip()
        print(
            f"[DEBUG] Transcription successful. Text: {transcribed_text}"
        )  # <-- ADDED
    except Exception as e:
        print(f"[ERROR] Failed during transcription: {repr(e)}")  # <-- ADDED
        return JSONResponse(
            status_code=500, content={"error": f"Error during transcription: {str(e)}"}
        )

    # 4. Save user's transcribed message to ChromaDB
    collection = chroma_client.get_or_create_collection(
        name=f"user_{userid}_chat_{chatid}"
    )
    user_message_id = str(uuid4())
    collection.add(
        documents=[transcribed_text],
        ids=[user_message_id],
        metadatas=[
            {"role": "user", "audio": audio_id, "conversationid": conversationid}
        ],
    )
    print("[DEBUG] User message saved to ChromaDB.")  # <-- ADDED

    # 5. Get agent's response and save it
    try:
        print("[DEBUG] Executing agent to get response...")  # <-- ADDED
        respuesta_final = ejecutar_agentic_psicologico(
            pregunta=transcribed_text,
            userid=userid,
            chatid=chatid,
            conversationid=conversationid,
            AnalisisEmocional=analisisEmocional,
        )
        print("[DEBUG] Agent execution successful.")  # <-- ADDED

        if not respuesta_final:
            raise Exception(
                "The agent could not generate a useful response at this time."
            )

        # Save the assistant's response to ChromaDB
        assistant_message_id = str(uuid4())
        collection.add(
            documents=[respuesta_final],
            ids=[assistant_message_id],
            metadatas=[{"role": "assistant", "conversationid": conversationid}],
        )
        print("[DEBUG] Assistant response saved to ChromaDB.")  # <-- ADDED

        user_message = {
            "role": "user",
            "text": transcribed_text,
            "audio": audio_id,
        }
        assistant_message = {"role": "assistant", "text": respuesta_final}

        return {
            "user_message": user_message,
            "assistant_message": assistant_message,
        }

    except Exception as e:
        # This will print the full traceback to your console
        print(
            f"[ERROR] Exception during agent execution or final response construction: {repr(e)}"
        )
        traceback.print_exc()  # <-- ADDED

        return JSONResponse(
            status_code=500, content={"error": f"Error generating response: {str(e)}"}
        )


@app_fastapi.get("/{userid}")
def get_user(userid: str):
    # Validate user exists
    if not validate_user(userid):
        raise HTTPException(status_code=404, detail=f"Usuario {userid} no encontrado")

    user_chats = [
        col
        for col in chroma_client.list_collections()
        if col.startswith(f"user_{userid}_chat_")
    ]
    return {"chats": user_chats}


@app_fastapi.get("/{userid}/{chatid}")
def get_chat(userid: str, chatid: str):
    # Validate user exists
    if not validate_user(userid):
        raise HTTPException(status_code=404, detail=f"Usuario {userid} no encontrado")

    collection = chroma_client.get_or_create_collection(
        name=f"user_{userid}_chat_{chatid}"
    )
    return {"messages": collection.count()}


@app_fastapi.get("/{userid}/{chatid}/{conversationid}")
def get_conversation(userid: str, chatid: str, conversationid: str):
    # Validate user exists
    if not validate_user(userid):
        raise HTTPException(status_code=404, detail=f"Usuario {userid} no encontrado")

    collection = chroma_client.get_or_create_collection(
        name=f"user_{userid}_chat_{chatid}"
    )
    results = collection.get(where={"conversationid": conversationid})
    return {"mensajes": results}


@app_fastapi.get("/{userid}/{chatid}/{conversationid}/content")
def get_conversation_content(userid: str, chatid: str, conversationid: str):
    # Validate user exists
    if not validate_user(userid):
        raise HTTPException(status_code=404, detail=f"Usuario {userid} no encontrado")

    collection = chroma_client.get_or_create_collection(
        name=f"user_{userid}_chat_{chatid}"
    )
    results = collection.get(where={"conversationid": conversationid})
    docs = results.get("documents") or []
    metas = results.get("metadatas") or []
    content = [
        {"role": metadata["role"], "text": doc} for doc, metadata in zip(docs, metas)
    ]
    print(
        f"[LOG] /content for user={userid}, chat={chatid}, conversation={conversationid} → {content}"
    )
    return content


from fastapi import HTTPException
from fastapi.responses import JSONResponse

# Make sure you have these imports at the top of your file


@app_fastapi.delete("/{userid}/{chatid}")
def delete_chat(userid: str, chatid: str):
    # 1. Validate user exists
    if not validate_user(userid):
        raise HTTPException(status_code=404, detail=f"Usuario {userid} no encontrado")

    # 2. Construct the full collection name directly
    collection_name_to_delete = f"user_{userid}_chat_{chatid}"

    try:
        # 3. Attempt to delete the collection by its full name
        chroma_client.delete_collection(name=collection_name_to_delete)
        return {
            "msg": f"Chat '{collection_name_to_delete}' eliminado para el usuario {userid}"
        }

    except Exception as e:
        # 4. If an error occurs (e.g., collection not found), return a clear error message
        # You can log the internal error 'e' for your own debugging purposes
        print(f"[ERROR] Could not delete collection {collection_name_to_delete}: {e}")

        return JSONResponse(
            status_code=404,
            content={
                "error": f"No se pudo encontrar o eliminar el chat '{chatid}' para el usuario {userid}"
            },
        )


# Solo ejecuta si es directamente este script
if __name__ == "__main__":
    uvicorn.run("agent:app_fastapi", host="0.0.0.0", port=8000, reload=True)

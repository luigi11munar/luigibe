from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
import re
import json
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from typing import List, Literal
from typing_extensions import TypedDict
from pprint import pprint
from langchain.agents import tool
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType
import os
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.responses import PlainTextResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from jose import JWTError, jwt
import time
import chromadb
from chromadb.config import Settings
import uuid
from langgraph.graph import END, StateGraph, START
from agents import Runner, function_tool, Agent
from fastapi.responses import StreamingResponse
from openai.types.responses import ResponseTextDeltaEvent, ResponseContentPartDoneEvent
from markdown_it import MarkdownIt

def remove_markdown(text: str) -> str:
    parser = MarkdownIt()
    tokens = parser.parse(text)
    return "".join(token.content for token in tokens if token.type == "inline")


# Leyendo las credenciales
load_dotenv()
os.environ["GROQ_API_KEY"] = "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"
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
    """Validate if user exists in USERS dictionary"""
    return userid in USERS


def get_user_by_id(userid: str) -> dict:
    """Get user by ID"""
    return USERS.get(userid)


def get_user_by_email(email: str) -> dict:
    """Get user by email"""
    for user_id, user_data in USERS.items():
        if user_data["email"] == email:
            return {"id": user_id, **user_data}
    return None


# Modelos
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
question = "agent memory"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content
print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = prompt | llm | StrOutputParser()

generation = rag_chain.invoke({"context": docs, "question": question})
print(generation)


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
hallucination_grader.invoke({"documents": docs, "generation": generation})


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
answer_grader.invoke({"question": question, "generation": generation})

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
question_rewriter.invoke({"question": question})


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
        # All documents have been filtered check_relevance
        # We will re-generate a new query
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

app = workflow.compile()

request_timestamps = {}

# Rate limiting configuration
RATE_LIMITS = {
    # General endpoints - more permissive
    "default": {"requests": 30, "window": 60},  # 30 requests per minute
    # Heavy endpoints - more restrictive
    "/consultar": {
        "requests": 10,
        "window": 60,
    },  # 10 requests per minute for AI queries
    # Light endpoints - very permissive
    "light": {
        "requests": 60,
        "window": 60,
    },  # 60 requests per minute for chat operations
}

# Light endpoints that should have more permissive limits
LIGHT_ENDPOINTS = ["/sendMessage", "/content", "/getAnswer", "/token", "/login"]


def is_light_endpoint(path: str) -> bool:
    """Check if the endpoint should have light rate limiting"""
    return any(light_ep in path for light_ep in LIGHT_ENDPOINTS)


def get_rate_limit_config(path: str) -> dict:
    """Get the appropriate rate limit configuration for a path"""
    if path in RATE_LIMITS:
        return RATE_LIMITS[path]
    elif is_light_endpoint(path):
        return RATE_LIMITS["light"]
    else:
        return RATE_LIMITS["default"]


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


llm_model: BaseLanguageModel = llm

@function_tool
def responder_con_contexto(input_str: str) -> str:
    """
    Tool que recibe un string JSON con los campos:
    'pregunta', 'userid', 'chatid', 'conversationid'.
    Responde con adaptación clínica y contexto si existe.
    """
    try:
        data = json.loads(input_str)
        pregunta = data["pregunta"]
        userid = data["userid"]
        chatid = data["chatid"]
        conversationid = data["conversationid"]
    except Exception as e:
        return f"[ERROR] Entrada inválida: {str(e)}\nRecibido: {input_str}"

    if not validate_user(userid):
        return f"[ERROR] Usuario {userid} no encontrado en el sistema."

    print(f"[INFO] Tool recibida → pregunta='{pregunta}' userid={userid} chatid={chatid} conversationid={conversationid}")

    # Preparar entrada para el flujo LangGraph
    inputs = {
        "question": pregunta,
        "userid": userid,
        "chatid": chatid,
        "conversationid": conversationid,
    }

    respuesta_final = ""

    try:
        for output in app.stream(inputs):
            for key, value in output.items():
                if "generation" in value:
                    respuesta_final = value["generation"]
    except Exception as e:
        return f"[ERROR] Fallo en el flujo de generación: {str(e)}"

    if not respuesta_final:
        return (
            "Lo siento, no encontré información suficiente para responder esa pregunta. "
            "Te recomiendo consultar directamente con un profesional."
        )

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
        Eres un agente psicológico de la UNIVERSIDAD DE PAMPLONA, especializado en primeros auxilios emocionales e intervención breve.

        Has recibido una respuesta basada en el conocimiento clínico. Tu tarea es **adaptar esa respuesta al contexto real del usuario**, sin cambiar su esencia, pero garantizando que:

        - Sea viable y segura en el entorno donde se encuentra (ej. clase, espacio público, casa).
        - Sea breve, clara, cálida y emocionalmente reconfortante.
        - Evite sugerencias visibles o disruptivas como acostarse, cerrar los ojos, gritar, o moverse demasiado.
        - Si se mencionan técnicas como respiración, visualización o aceptación, indícale cómo aplicarlas discretamente.
        - Usa un lenguaje empático, accesible y respetuoso.

        Puedes basarte en estrategias de:
        - Terapia Cognitivo-Conductual (reestructuración, auto-instrucciones, autorregistros)
        - Terapia de Aceptación y Compromiso (ACT: defusión, aceptación, valores)
        - Psicoeducación breve
        - Técnicas de mindfulness o anclaje
        - Regulación emocional en crisis

        ### Pregunta del usuario:
        {pregunta}

        ### Respuesta original:
        {respuesta_original}

        ### Contexto adicional:
        {contexto}

        ### Respuesta adaptada:
        """
    )

    prompt_text = adaptation_prompt.format(
        pregunta=pregunta,
        respuesta_original=respuesta_final,
        contexto=contexto
    )

    try:
        return llm.invoke(prompt_text)
    except Exception as e:
        return f"[ERROR] Fallo en adaptación clínica: {str(e)}"




agent_psicologico = Agent(
    name="AsistentePsicologico",
    handoff_description="Ayuda emocional en tiempo real con adaptación psicológica",
    instructions="""
Eres un asistente psicológico que brinda primeros auxilios emocionales.
Usa la herramienta `responder_con_contexto` siempre que recibas una pregunta del usuario.
""",
    tools=[responder_con_contexto],
)



app_fastapi = FastAPI()

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
# chroma_client = chromadb.EphemeralClient()



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
    # Validate user exists
    if not validate_user(userid):
        raise HTTPException(status_code=404, detail=f"Usuario {userid} no encontrado")
    user_chats = [
        col
        for col in chroma_client.list_collections()
        if col.startswith(f"user_{userid}_chat_")
    ]
    path = f"./db/conversations_{userid}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    # Return list of {chatid, conversationid}
    result = []
    for col in user_chats:
        chatid = col.replace(f"user_{userid}_chat_", "")
        conversationid = data.get(chatid, None)
        result.append({"chatid": chatid, "conversationid": conversationid})
    return result


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
from uuid import uuid4
from agents import RawResponsesStreamEvent, TResponseInputItem

@app_fastapi.get("/{userid}/{chatid}/{conversationid}/getAnswer")
async def get_last_answer(userid: str, chatid: str, conversationid: str):
    print(f"[LOG] /getAnswer called for user={userid}, chat={chatid}, conversation={conversationid}")

    if not validate_user(userid):
        raise HTTPException(status_code=404, detail=f"Usuario {userid} no encontrado")

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

    last_user_message = user_messages[-1]["text"]

    try:
        respuesta_final = ""
        delta_buffer = ""

        # --- Cambia aquí: arma el JSON para la tool ---
        input_json = json.dumps({
            "pregunta": last_user_message,
            "userid": userid,
            "chatid": chatid,
            "conversationid": conversationid
        })

        # Si tu Runner espera lista:
        # inputs = [input_json]
        # Si espera solo el string:
        inputs = input_json

        result = Runner.run_streamed(agent_psicologico, input=inputs)

        async for event in result.stream_events():
            if not isinstance(event, RawResponsesStreamEvent):
                continue
            data = event.data
            if isinstance(data, ResponseTextDeltaEvent):
                delta_buffer += data.delta
            elif isinstance(data, ResponseContentPartDoneEvent):
                print("\n")

        respuesta_final = f"Agente psicológico:\n{delta_buffer.strip()}"

        if not respuesta_final:
            return {
                "msg": "No se pudo generar una respuesta útil en este momento. Intenta reformular la pregunta."
            }

        message_id = str(uuid4())
        collection.add(
            documents=[respuesta_final],
            ids=[message_id],
            metadatas=[{"role": "assistant", "conversationid": conversationid}],
        )

        return {"role": "assistant", "text": respuesta_final}

    except Exception as e:
        return {"msg": f"Error generando respuesta: {str(e)}"}

@app_fastapi.get("/{userid}")
def get_user(userid: str):
    # Validate user exists
    if not validate_user(userid):
        raise HTTPException(status_code=404, detail=f"Usuario {userid} no encontrado")

    # Chroma v0.6.0: list_collections() returns a list of collection names (strings)
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


@app_fastapi.delete("/{userid}/{chatid}")
def delete_chat(userid: str, chatid: str):
    # Validate user exists
    if not validate_user(userid):
        raise HTTPException(status_code=404, detail=f"Usuario {userid} no encontrado")
    # List all collections for this user
    user_collections = [
        col
        for col in chroma_client.list_collections()
        if col.startswith(f"user_{userid}_chat_")
    ]
    # Try to find the collection that matches the chatid (either as suffix or full name)
    # Accept both: chatid is just the id, or the full collection name
    collection_to_delete = None
    for col in user_collections:
        if col == chatid or col.endswith(f"_{chatid}"):
            collection_to_delete = col
            break
    if not collection_to_delete:
        return JSONResponse(
            status_code=404,
            content={
                "error": f"No se encontró el chat '{chatid}' para el usuario {userid}"
            },
        )
    try:
        chroma_client.delete_collection(name=collection_to_delete)
        return {"msg": f"Chat '{collection_to_delete}' eliminado para user {userid}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# Solo ejecuta si es directamente este script
if __name__ == "__main__":
    uvicorn.run("agent:app_fastapi", host="0.0.0.0", port=8000, reload=True)

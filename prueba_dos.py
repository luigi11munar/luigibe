#AGENTIC RAG

import json
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from agents import Runner, function_tool, Agent
import asyncio


@function_tool
def responder_pregunta_en_fuentes_psicologia(input_json: str) -> str:
    """
    Recibe un string JSON con los campos:
    'pregunta', 'userid', 'chatid', 'conversationid'.
    Responde consultando la base psicológica según la pregunta.
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
        tools=[{
            "type": "file_search",
            "vector_store_ids": ["vs_68375d1360248191a2353723b7f5a931"]  # Cambia al ID real de tu vector store
        }],
        instructions="""
        - Analiza los pensamientos, emociones o comportamientos expresados por el usuario utilizando exclusivamente la información contenida en la base de datos, la cual es un resumen experto de los manuales DSM-5 e ICD-11 y ejemplos clínicos de ansiedad y depresión.
        - Si el usuario describe experiencias, síntomas o preocupaciones, identifica y explica los patrones clínicos relevantes basados en ese resumen, indicando su importancia o impacto según la evidencia resumida.
        - No inventes diagnósticos, síntomas ni recomendaciones que no estén explícitamente respaldados en la base de datos resumida.
        """
    )
    return {
        "pregunta": pregunta,
        "userid": userid,
        "chatid": chatid,
        "conversationid": conversationid,
        "respuesta": response.output_text
    }


psicologia_patrones_agent = Agent(
    name="AgentePatronesPsicologicos",
    handoff_description="Agente para responder preguntas sobre el hotel",
    instructions="""
        Eres un asistente clínico especializado en psicología clínica, experto en identificar e interpretar patrones de comportamiento, emociones y pensamientos asociados a trastornos de ansiedad y depresión (DSM-5 e ICD-11).
        Cuando recibas una pregunta sobre síntomas, patrones, ejemplos clínicos o estrategias de intervención, debes invocar la herramienta 'responder_pregunta_en_fuentes_psicologia_tool' con el texto de la pregunta.
        Nunca inventes información ni diagnósticos. Basa tus respuestas en las fuentes validadas y en criterios clínicos oficiales.
    """,
    tools=[
        responder_pregunta_en_fuentes_psicologia,
    ],
)
def vectorstore_node(state):
    state["input_json"] = json.dumps({
        "pregunta": state["pregunta"],
        "userid": state.get("userid", ""),
        "chatid": state.get("chatid", ""),
        "conversationid": state.get("conversationid", "")
    })
    return state

def filter_docs_node(state):
    return state

def rag_node(state):
    result = asyncio.run(
        Runner.run(
            psicologia_patrones_agent,
            input=state["input_json"]
        )
    )
    state["respuesta"] = result.final_output
    if not state["respuesta"] or "ERROR" in state["respuesta"]:
        state["useful"] = False
    else:
        state["useful"] = True
    return state

def fallback_node(state):
    state["respuesta"] = "No se encontró suficiente información en las fuentes disponibles. Intenta reformular tu pregunta."
    return state

class RAGState(TypedDict, total=False):
    pregunta: str
    userid: str
    chatid: str
    conversationid: str
    input_json: str
    respuesta: str
    useful: bool

def vectorstore_node(state: RAGState) -> RAGState:
    state["input_json"] = json.dumps({
        "pregunta": state["pregunta"],
        "userid": state.get("userid", ""),
        "chatid": state.get("chatid", ""),
        "conversationid": state.get("conversationid", "")
    })
    return state

def filter_docs_node(state: RAGState) -> RAGState:
    return state

def rag_node(state: RAGState) -> RAGState:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            Runner.run(
                psicologia_patrones_agent,
                input=state["input_json"]
            )
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
    state["respuesta"] = "No se encontró suficiente información en las fuentes disponibles. Intenta reformular tu pregunta."
    return state


g = StateGraph(RAGState)

g.add_node("vectorstore", vectorstore_node)
g.add_node("filter_docs", filter_docs_node)
g.add_node("rag", rag_node)
g.add_node("fallback", fallback_node)

g.add_edge(START, "vectorstore")
g.add_edge("vectorstore", "filter_docs")
g.add_edge("filter_docs", "rag")
g.add_conditional_edges(
    "rag",
    lambda state: state["useful"],  
    {True: END, False: "fallback"}
)
g.add_edge("fallback", END)

g = g.compile()  

def ejecutar_agentic_rag(pregunta, userid="1", chatid="mi_chat", conversationid="conversacion_001"):
    state = {
        "pregunta": pregunta,
        "userid": userid,
        "chatid": chatid,
        "conversationid": conversationid
    }
    final_state = g.invoke(state)  
    return final_state.get("respuesta")

if __name__ == "__main__":
    pregunta = "Últimamente me siento sin energía, me cuesta mucho levantarme en las mañanas y a veces pienso que nada de lo que hago tiene sentido."
    respuesta = ejecutar_agentic_rag(pregunta)
    print(respuesta)

#async def main():
 #   input_dict = {
 #      "pregunta": "Últimamente me siento sin energía, me cuesta mucho levantarme en las mañanas y a veces pienso que nada de lo que hago tiene sentido.",
 #      "userid": "1",
 #     "chatid": "mi_chat",
   #    "conversationid": "conversacion_001"
   # }
   # input_json = json.dumps(input_dict)
  #  result = await Runner.run(psicologia_patrones_agent, input=input_json)
 #   print(result.final_output)


#if __name__ == "__main__":
 #   asyncio.run(main())



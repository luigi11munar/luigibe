import json
from langgraph.graph import StateGraph, START, END
from agents import Runner
from agent import *
import asyncio

# Nodo RAG principal
def agente_rag_node(state):
    print("\n[agente_rag_node] --- INICIO ---")
    print(f"STATE INICIAL: {state}")
    pregunta = state["pregunta"]
    userid = state["userid"]
    chatid = state["chatid"]
    conversationid = state["conversationid"]
    print(f"Consultando RAG con pregunta='{pregunta}', userid='{userid}', chatid='{chatid}', conversationid='{conversationid}'")
    respuesta_rag = ejecutar_agentic_rag(
        pregunta=pregunta,
        userid=userid,
        chatid=chatid,
        conversationid=conversationid
    )
    print(f"RESPUESTA RAG RAW: {respuesta_rag}")
    state["historial"] = respuesta_rag["respuesta"] if isinstance(respuesta_rag, dict) else respuesta_rag
    print(f"[agente_rag_node] STATE FINAL: {state}")
    print("[agente_rag_node] --- FIN ---\n")
    return state

# Nodo Psicológico
def agente_psicologico_node(state):
    print("\n[agente_psicologico_node] --- INICIO ---")
    print(f"STATE INICIAL: {state}")
    analisisEmocional = state["historial"]
    pregunta = state["pregunta"]

    input_json = json.dumps({
        "pregunta": pregunta,
        "userid": state["userid"],
        "chatid": state["chatid"],
        "conversationid": state["conversationid"],
        "analisisEmocional": analisisEmocional,
    })

    print(f"Llamando Runner.run(agent_psicologico, input={input_json})")
    result = asyncio.run(Runner.run(agent_psicologico, input=input_json))
    print(f"RESPUESTA PSICOLÓGICO: {result}")
    state["respuesta_final"] = result
    print(f"[agente_psicologico_node] STATE FINAL: {state}")
    print("[agente_psicologico_node] --- FIN ---\n")
    return state

# Definición del grafo
g = StateGraph(dict)
g.add_node("agentic-rag", agente_rag_node)
g.add_node("corrective-rag", agente_psicologico_node)
g.add_edge(START, "agentic-rag")
g.add_edge("agentic-rag", "corrective-rag")
g.add_edge("corrective-rag", END)
g = g.compile()

def ejecutar_agentic_psicologico(pregunta, AnalisisEmocional, userid="1", chatid="mi_chat", conversationid="conversacion_001"):
    print("\n[ejecutar_agentic_psicologico] --- INICIO ---")
    state = {
        "pregunta": pregunta,
        "userid": userid,
        "chatid": chatid,
        "conversationid": conversationid,
        "historial":  AnalisisEmocional  
    }
    print(f"STATE INICIAL: {state}")
    final_state = g.invoke(state)
    print(f"\nSTATE FINAL TRAS GRAFO: {final_state}")
    print("[ejecutar_agentic_psicologico] --- FIN ---\n")
    return final_state.get("respuesta_final")

if __name__ == "__main__":
    pregunta = "Me siento ansioso, no se que hacer."
    AnalisisEmocional = ""
    respuesta = ejecutar_agentic_psicologico(
        pregunta=pregunta,
        userid="1",
        chatid="mi_chat",
        conversationid="conversacion_001",
        AnalisisEmocional ="AnalisisEmocional"
    )
    print(f"\n>>> RESPUESTA FINAL DEL AGENTE:\n{respuesta}\n")

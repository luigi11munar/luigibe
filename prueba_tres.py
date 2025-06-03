import torch
import os
from dotenv import load_dotenv
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

# Cargar token Hugging Face
load_dotenv()
huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- Cargar modelo fine-tuned directamente desde Unsloth ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Luigi112001/mistral_finetuned",  # <-- Tu modelo fine-tuned
    max_seq_length=2048,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    load_in_4bit=torch.cuda.is_available(),  # usa 4bit solo si tienes GPU
    token=huggingface_token
)

model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

# --- Funciones CAG (sin modificar tu lógica) ---
def get_past_key_cache(model, tokenizer, prompt: str):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    return out.past_key_values, input_ids.shape[-1]

def generate(model, tokenizer, question: str, past_key_values, max_new_tokens: int = 2000):
    input_ids = tokenizer(question, return_tensors="pt").input_ids.to(model.device)
    output_ids = input_ids.clone()
    next_token = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = out.logits[:, -1, :]
            token = torch.argmax(logits, dim=-1, keepdim=True)
            output_ids = torch.cat([output_ids, token], dim=-1)
            past_key_values = out.past_key_values
            next_token = token.to(model.device)

            if model.config.eos_token_id is not None and token.item() == model.config.eos_token_id:
                break

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# --- Base de conocimiento psicológica ---
knowledge_base = """
Cuestionario GAD-7 para ansiedad generalizada (últimas 2 semanas):
1. ¿Se ha sentido nervioso, ansioso o al límite?
2. ¿Ha sido incapaz de controlar sus preocupaciones?
3. ¿Se ha preocupado demasiado por diferentes cosas?
4. ¿Ha tenido dificultad para relajarse?
5. ¿Ha estado tan inquieto que no puede quedarse quieto?
6. ¿Se ha sentido fácilmente irritable o molesto?
7. ¿Ha sentido miedo como si algo terrible pudiera pasar?

Cuestionario PHQ-9 para depresión (últimas 2 semanas):
1. ¿Ha tenido poco interés o placer en hacer cosas?
2. ¿Se ha sentido decaído(a), deprimido(a) o sin esperanza?
3. ¿Ha tenido dificultad para dormir o ha dormido en exceso?
4. ¿Se ha sentido cansado(a) o con poca energía?
5. ¿Ha tenido poco apetito o comido en exceso?
6. ¿Se ha sentido mal consigo mismo o como un fracaso?
7. ¿Ha tenido dificultad para concentrarse?
8. ¿Ha hablado o se ha movido más lento o más rápido de lo normal?
9. ¿Ha tenido pensamientos de que estaría mejor muerto(a) o de hacerse daño?
"""

system_prompt = (
    "Eres un asistente experto en salud mental. Evalúa ansiedad y depresión "
    "únicamente usando los cuestionarios clínicos proporcionados. No generes diagnósticos ni inventes datos. "
    "Si no tienes suficiente información, solicita más detalles de los ítems del cuestionario.\n"
    f"CONOCIMIENTO BASE:\n{knowledge_base}\n"
)

# --- Cache del system prompt ---
print("Generando caché del contexto clínico (system prompt)...")
kv_cache, context_len = get_past_key_cache(model, tokenizer, system_prompt)

# --- Consulta del usuario ---
question = "Me siento muy nervioso y no puedo dejar de preocuparme por todo."
print("Pregunta:", question)

respuesta = generate(model, tokenizer, question, kv_cache)
print("\nRespuesta:")
print(respuesta.strip())

from flask import Flask, render_template, request, jsonify
import os
import re
from werkzeug.utils import secure_filename
import requests
from flask_cors import CORS
import pickle
from typing import Optional

import nltk
from dotenv import load_dotenv
import unicodedata
import time

# Hugging Face Chat client
from huggingface_hub import InferenceClient

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

ALLOWED_EXTENSIONS = {'txt', 'pdf'}
UPLOAD_FOLDER = 'uploads'

load_dotenv()
app = Flask(__name__)
# Habilita CORS para permitir frontend em outra origem (ex.: Live Server :5500)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def _env_flag(name: str, default: str = '0') -> bool:
    val = os.getenv(name)
    if val is None:
        val = default
    # Remove espaços e aspas
    sval = str(val).strip().strip('"').strip("'").lower()
    return sval in ['1', 'true', 'yes', 'on']

def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    sval = str(val).strip().strip('"').strip("'")
    try:
        return int(sval)
    except Exception:
        return default

FORCE_AI_REPLY = _env_flag('FORCE_AI_REPLY', '0')
MIN_AI_REPLY_CHARS = _env_int('MIN_AI_REPLY_CHARS', 40)
print(f"Config -> FORCE_AI_REPLY={FORCE_AI_REPLY}, MIN_AI_REPLY_CHARS={MIN_AI_REPLY_CHARS}")

# Hugging Face settings
HF_API_TOKEN = os.getenv('HF_API_TOKEN')
HF_ZERO_SHOT_MODEL = os.getenv('HF_ZERO_SHOT_MODEL', 'joeddav/xlm-roberta-large-xnli')
HF_T2T_MODEL = os.getenv('HF_T2T_MODEL', 'google/flan-t5-small')
HF_T2T_CANDIDATES = [m.strip() for m in (os.getenv('HF_T2T_CANDIDATES') or '').split(',') if m.strip()]
HF_EMBED_MODEL = os.getenv('HF_EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

# Advanced zero-shot settings
ZS_LABELS = [
    "email sobre trabalho, projetos, tarefas, reuniões ou negócios (produtivo)",
    "email solicitando informações, orçamento ou documentos profissionais (produtivo)",
    "email solicitando suporte técnico ou resolução de problemas (produtivo)",
    "email de propaganda ou marketing legítimo (improdutivo)",
    "email de golpe, phishing, fraude ou scam (improdutivo)",
    "email pessoal, cumprimentos, conversa informal, correntes ou brincadeiras (improdutivo)",
    "email de saudações, datas comemorativas ou felicitações (improdutivo)"
]
ZS_MAP = {
    ZS_LABELS[0]: 'Produtivo',
    ZS_LABELS[1]: 'Produtivo',
    ZS_LABELS[2]: 'Produtivo',
    ZS_LABELS[3]: 'Improdutivo',
    ZS_LABELS[4]: 'Improdutivo',
    ZS_LABELS[5]: 'Improdutivo',
    ZS_LABELS[6]: 'Improdutivo',
}
ZS_CONFIDENCE_THRESHOLD = float(os.getenv('ZS_CONFIDENCE_THRESHOLD', '0.75'))
ZS_CONFIDENCE_MARGIN = float(os.getenv('ZS_CONFIDENCE_MARGIN', '0.15'))

# HF Chat client (optional)
hf_chat_client: Optional[InferenceClient] = None
if HF_API_TOKEN:
    try:
        hf_chat_client = InferenceClient(api_key=HF_API_TOKEN)
    except Exception as _:
        hf_chat_client = None

# Local Transformers (opcional)
USE_LOCAL_CLASSIFIER = _env_flag('USE_LOCAL_CLASSIFIER', '0')
USE_LOCAL_GENERATOR = _env_flag('USE_LOCAL_GENERATOR', '0')
LOCAL_CLS_MODEL = os.getenv('LOCAL_CLS_MODEL', 'distilbert-base-uncased-finetuned-sst-2-english')
LOCAL_T2T_MODEL = os.getenv('LOCAL_T2T_MODEL', 'google/flan-t5-small')

_local_cls = {'tokenizer': None, 'model': None}
_local_t2t = {'tokenizer': None, 'model': None}

stop_words = set(stopwords.words('portuguese'))
stemmer = SnowballStemmer('portuguese')

# Modelo de intenção (opcional)
INTENT_MODEL_PATH = os.path.join('models', 'intent_nb.pkl')
intent_clf = None
try:
    if os.path.exists(INTENT_MODEL_PATH):
        with open(INTENT_MODEL_PATH, 'rb') as f:
            intent_clf = pickle.load(f)
except Exception as e:
    print('Falha ao carregar modelo de intenção:', e)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.lower().split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

def text_to_features(text: str):
    toks = re.findall(r"[\wÀ-ÿ]+", text.lower())
    toks = [stemmer.stem(t) for t in toks if t not in stop_words]
    return {f'has({t})': True for t in toks}

def read_text_file_best_effort(filepath: str) -> str:
    """Tenta ler arquivo texto tentando UTF-8, CP1252 e fallback ignorando erros."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='cp1252') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

def classify_with_hf_api(text: str) -> Optional[str]:
    """Tenta classificar via Hugging Face Inference API (zero-shot). Retorna categoria ou None em caso de erro.
    Usa um conjunto de rótulos mais rico e thresholds para reduzir falsos positivos.
    """
    if not HF_API_TOKEN:
        return None
    url = f"https://api-inference.huggingface.co/models/{HF_ZERO_SHOT_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": text,
        "parameters": {
            "candidate_labels": ZS_LABELS,
            "multi_label": False
        },
        "options": {"wait_for_model": True}
    }
    for attempt in range(3):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=45)
            if r.status_code == 503 or ('loading' in (r.text or '').lower()):
                time.sleep(1.5)
                continue
            r.raise_for_status()
            data = r.json()
            labels = None
            scores = None
            if isinstance(data, list) and data:
                labels = data[0].get('labels')
                scores = data[0].get('scores')
            elif isinstance(data, dict) and 'labels' in data:
                labels = data.get('labels')
                scores = data.get('scores')
            if labels and scores:
                top_label = labels[0]
                top_score = float(scores[0])
                second = float(scores[1]) if len(scores) > 1 else 0.0
                mapped = ZS_MAP.get(top_label)
                if mapped is None:
                    return None
                # thresholds: confiança alta e margem para 2º lugar
                if top_score < ZS_CONFIDENCE_THRESHOLD or (top_score - second) < ZS_CONFIDENCE_MARGIN:
                    return None
                return mapped
        except Exception as e:
            try:
                print('HF zero-shot HTTP:', r.status_code, r.text[:300])
            except Exception:
                pass
            print('HF zero-shot error:', e)
            time.sleep(1)
    return None

def _load_local_classifier() -> bool:
    if _local_cls['model'] is not None and _local_cls['tokenizer'] is not None:
        return True
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained(LOCAL_CLS_MODEL)
        mdl = AutoModelForSequenceClassification.from_pretrained(LOCAL_CLS_MODEL)
        _local_cls['tokenizer'] = tok
        _local_cls['model'] = mdl
        print('Local classifier loaded:', LOCAL_CLS_MODEL)
        return True
    except Exception as e:
        print('Failed to load local classifier:', e)
        return False

def classify_with_local_model(text: str) -> Optional[str]:
    if not USE_LOCAL_CLASSIFIER:
        return None
    if not _load_local_classifier():
        return None
    try:
        import torch
        tok = _local_cls['tokenizer']
        mdl = _local_cls['model']
        inputs = tok(text, return_tensors='pt')
        with torch.no_grad():
            outputs = mdl(**inputs)
        pred = int(outputs.logits.argmax(dim=1).item())
        return 'Produtivo' if pred == 1 else 'Improdutivo'
    except Exception as e:
        print('Local classify error:', e)
        return None

# (OpenAI removido)

def generate_reply_with_hf_api(email_text: str, categoria: str) -> Optional[str]:
    """Gera uma resposta via Hugging Face Inference API (text2text). Retorna texto ou None em caso de erro."""
    if not HF_API_TOKEN:
        return None
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    instrucao = (
        "Escreva uma resposta educada e objetiva em português. "
        f"Categoria: {categoria}. "
        "Se produtivo, confirme recebimento, peça dados faltantes (se necessário) e informe próximos passos. "
        "Se improdutivo, agradeça de forma cordial.\n\nEmail:\n"
    )
    prompt = instrucao + email_text
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 160}, "options": {"wait_for_model": True}}
    # Modelos candidatos: começa pelo definido em env, depois fallbacks seguros
    candidates = []
    seen = set()
    # Prioridade: env list > single env model > built-in fallbacks
    for m in HF_T2T_CANDIDATES + [HF_T2T_MODEL, 'google/flan-t5-small', 'MBZUAI/LaMini-Flan-T5-248M', 't5-small', 'google/flan-t5-base', 'gpt2', 'distilgpt2']:
        if m and m not in seen:
            candidates.append(m)
            seen.add(m)
    for model in candidates:
        url = f"https://api-inference.huggingface.co/models/{model}"
        print(f"HF T2T trying model: {model}")
        for attempt in range(3):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=60)
                if r.status_code in (404,):
                    print('HF T2T 404 for model:', model)
                    break  # tente próximo modelo
                if r.status_code == 503 or ('loading' in (r.text or '').lower()):
                    time.sleep(1.5)
                    continue
                r.raise_for_status()
                data = r.json()
                if isinstance(data, list) and data:
                    return data[0].get('generated_text') or data[0].get('summary_text') or None
                elif isinstance(data, dict):
                    if 'error' in data:
                        print('HF T2T error field for model:', model, data.get('error'))
                        break
                    return data.get('generated_text') or data.get('summary_text') or None
            except Exception as e:
                try:
                    print('HF T2T HTTP:', getattr(r, 'status_code', 'n/a'), (getattr(r, 'text', '') or '')[:300])
                except Exception:
                    pass
                print('HF T2T error:', e)
                time.sleep(1)
    return None

def _limpar_raciocinio_interno(texto: str) -> str:
    padroes = [
        r"(?i)(let me|i should|they want|maybe|so i|alright|let's|okay,|então vou|preciso|vou pensar|deixa eu).*",
        r"(?i)(analisando|pensando|raciocínio|raciocinio|planejando).*",
    ]
    for p in padroes:
        texto = re.sub(p, "", texto).strip()
    return texto

def _extrair_resposta_final(texto: str) -> str:
    """Extrai apenas a mensagem final (um único bloco), priorizando o ÚLTIMO cumprimento."""
    texto = _limpar_raciocinio_interno(texto)
    # Remover linhas explicativas comuns em inglês
    linhas = []
    for ln in texto.splitlines():
        ln_stripped = ln.strip()
        if not ln_stripped:
            linhas.append(ln)
            continue
        if re.match(r"(?i)^(wait|note|thinking|analysis|explanation)\b", ln_stripped):
            continue
        if re.search(r"</think>", ln_stripped, flags=re.IGNORECASE):
            continue
        linhas.append(ln)
    texto = "\n".join(linhas)

    # Pega o último bloco que começa com cumprimento até antes do próximo cumprimento ou fim
    cumpr = r"Prezado\(a\)?|Prezada|Prezados|Olá|Caro|Cara|Bom dia|Boa tarde|Boa noite"
    pattern = rf"(?is)(?:{cumpr})[\s\S]*?(?=(?:\n\s*\n\s*(?:{cumpr})\b)|\Z)"
    matches = list(re.finditer(pattern, texto, flags=re.IGNORECASE))
    if matches:
        return matches[-1].group(0).strip()
    # fallback: usa último parágrafo
    partes = [p.strip() for p in re.split(r"\n\s*\n", texto) if p.strip()]
    if partes:
        return _limpar_raciocinio_interno(partes[-1])
    return texto.strip()

def post_process_ai_output(texto: str, categoria: str) -> str:
    """Limpa saída de modelos (chat/T2T/local): remove <think>, raciocínio e extrai só a mensagem final."""
    if not texto:
        return texto
    # Remove blocos <think>...</think>
    try:
        texto = re.sub(r"(?is)<think>.*?</think>", "", texto).strip()
    except Exception:
        pass
    # Remove prefixos comuns de raciocínio
    texto = re.sub(r"(?i)^Putting it all together: *", "", texto).strip()
    texto = re.sub(r"(?i)^Then, *", "", texto).strip()
    # Extrai a parte final útil (apenas um bloco)
    texto = _extrair_resposta_final(texto)
    # Remover rótulos como "Mensagem final:", "Mensagem:", "Resposta:"
    texto = re.sub(r"(?is)^\s*(mensagem(\s*final)?|resposta(\s*final)?)\s*:\s*", "", texto).strip()
    # Remover aspas extras no início/fim
    texto = texto.strip().strip('"').strip("'").strip()
    # Normalizar placeholders comuns
    texto = re.sub(r"(?i)Prezado\(a\)\s*\[Nome\]\s*,?", "Prezado(a),", texto).strip()
    texto = re.sub(r"(?i)\[Seu Nome\]", "", texto).strip()
    # Deduplicar linhas consecutivas idênticas
    dedup = []
    prev = None
    for ln in texto.splitlines():
        if ln.strip() != prev:
            dedup.append(ln)
            prev = ln.strip()
    texto = "\n".join(dedup)
    # Normalizações leves
    texto = re.sub(r"\n{3,}", "\n\n", texto).strip()
    return texto

def generate_reply_with_hf_chat(email_text: str, categoria: str) -> Optional[str]:
    """Gera resposta via chat-completions (Hugging FaceTB/SmolLM3-3B) se disponível."""
    if hf_chat_client is None:
        return None
    if categoria == 'Produtivo':
        prompt = (
            "Você é um assistente profissional. "
            "IMPORTANTE: Responda APENAS com a mensagem final pronta para envio, em português. "
            "PROIBIDO explicar raciocínio.\n\n"
            f"Email recebido:\n{email_text}\n\n"
            "Mensagem final:"
        )
    else:
        prompt = (
            "Você é um assistente cordial. "
            "IMPORTANTE: Responda APENAS com a mensagem final pronta para envio, em português. "
            "Sem raciocínio. Curta e educada.\n\n"
            f"Email recebido:\n{email_text}\n\n"
            "Mensagem final:"
        )
    try:
        comp = hf_chat_client.chat.completions.create(
            model=os.getenv('HF_CHAT_MODEL', 'HuggingFaceTB/SmolLM3-3B'),
            messages=[{"role": "user", "content": prompt}],
        )
        txt = comp.choices[0].message["content"].strip()
        return _extrair_resposta_final(txt)
    except Exception as e:
        print('HF chat error:', e)
        return None

def _load_local_generator() -> bool:
    if _local_t2t['model'] is not None and _local_t2t['tokenizer'] is not None:
        return True
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tok = AutoTokenizer.from_pretrained(LOCAL_T2T_MODEL)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_T2T_MODEL)
        _local_t2t['tokenizer'] = tok
        _local_t2t['model'] = mdl
        print('Local generator loaded:', LOCAL_T2T_MODEL)
        return True
    except Exception as e:
        print('Failed to load local generator:', e)
        return False

def generate_reply_with_local_model(email_text: str, categoria: str) -> Optional[str]:
    if not USE_LOCAL_GENERATOR:
        return None
    if not _load_local_generator():
        return None
    try:
        import torch
        tok = _local_t2t['tokenizer']
        mdl = _local_t2t['model']
        instrucao = (
            "Escreva uma resposta educada e objetiva em português. "
            f"Categoria: {categoria}. "
            "Se produtivo, confirme recebimento, peça dados faltantes (se necessário) e informe próximos passos. "
            "Se improdutivo, agradeça de forma cordial.\n\nEmail:\n"
        )
        prompt = instrucao + email_text
        inputs = tok(prompt, return_tensors='pt')
        with torch.no_grad():
            output_ids = mdl.generate(**inputs, max_new_tokens=160)
        return tok.decode(output_ids[0], skip_special_tokens=True)
    except Exception as e:
        print('Local generate error:', e)
        return None

def _flatten_embedding(resp):
    # Alguns modelos retornam [tokens][dim], outros [dim]
    if isinstance(resp, list):
        if len(resp) > 0 and isinstance(resp[0], list):
            # média por token
            dim = len(resp[0])
            acc = [0.0] * dim
            n = 0
            for row in resp:
                if isinstance(row, list) and len(row) == dim:
                    acc = [a + b for a, b in zip(acc, row)]
                    n += 1
            if n > 0:
                return [a / n for a in acc]
        else:
            return resp
    return None

def _cosine(a, b):
    import math
    da = sum(x*x for x in a) ** 0.5
    db = sum(x*x for x in b) ** 0.5
    if da == 0 or db == 0:
        return -1.0
    dot = sum(x*y for x, y in zip(a, b))
    return dot / (da * db)

def embed_with_hf_api(text: str) -> Optional[list]:
    if not HF_API_TOKEN:
        return None
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    # Primeiro tenta o pipeline de feature-extraction (adequado para embeddings)
    urls = [
        f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_EMBED_MODEL}",
        f"https://api-inference.huggingface.co/models/{HF_EMBED_MODEL}",
    ]
    for url in urls:
        for attempt in range(3):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=45)
                if r.status_code == 503 or ('loading' in (r.text or '').lower()):
                    time.sleep(1.5)
                    continue
                r.raise_for_status()
                data = r.json()
                vec = _flatten_embedding(data)
                if vec:
                    return vec
                else:
                    # Se veio um payload de outra tarefa (ex.: sentence-similarity), tente próxima URL
                    break
            except Exception as e:
                try:
                    print('HF EMB HTTP:', getattr(r, 'status_code', 'n/a'), (getattr(r, 'text', '') or '')[:300])
                except Exception:
                    pass
                print('HF EMB error:', e)
                time.sleep(1)
    return None

def choose_reply_by_embeddings(email_text: str, categoria: str) -> Optional[str]:
    """Usa embeddings para escolher o melhor template entre candidatos."""
    # Candidatos básicos
    candidates = []
    if categoria == 'Improdutivo':
        candidates.append('Agradecemos o contato! Permanecemos à disposição. Abraços.')
        candidates.append('Obrigado pela sua mensagem e consideração! Estamos à disposição sempre que precisar.')
    else:
        candidates.append('Olá! Recebemos sua mensagem e já estamos verificando. Se possível, compartilhe o número da requisição e quaisquer detalhes adicionais. Retornaremos com uma atualização em breve.')
        candidates.append('Olá! Obrigado pela mensagem. Vamos verificar o status e retornaremos assim que houver atualização.')
        candidates.append('Olá! Sentimos pelo transtorno. Para ajudarmos mais rápido, envie por favor: prints do erro, data/hora aproximadas e seu usuário. Nosso time de suporte vai analisar e retornar.')
        candidates.append('Olá! Vamos abrir o chamado para atualização cadastral. Poderia enviar CPF/matrícula, campos a atualizar (valor atual → novo), sistemas impactados e autorização/anexos?')
        candidates.append('Olá! Obrigado pelo envio. Recebemos o(s) arquivo(s). Caso falte algum documento ou haja versão incorreta, avise. Daremos sequência e retornaremos com os próximos passos.')
    # 1) Tenta pipeline de sentence-similarity direto na API
    ss = similarity_with_hf_api(email_text, candidates)
    if ss and len(ss) == len(candidates):
        idx = max(range(len(candidates)), key=lambda i: ss[i])
        return candidates[idx]
    # 2) Fallback: embeddings e cosseno
    email_vec = embed_with_hf_api(email_text)
    if not email_vec:
        # 3) Fallback local: similaridade por tokens (Jaccard) com stemming/stopwords
        def _tokens(s: str) -> set:
            toks = re.findall(r"[\wÀ-ÿ]+", s.lower())
            toks = [stemmer.stem(t) for t in toks if t not in stop_words and len(t) > 1]
            return set(toks)
        e_set = _tokens(email_text)
        best = None
        best_j = -1.0
        for c in candidates:
            c_set = _tokens(c)
            union = len(e_set | c_set)
            inter = len(e_set & c_set)
            j = (inter / union) if union else 0.0
            if j > best_j:
                best_j = j
                best = c
        return best
    best = None
    best_score = -2.0
    for c in candidates:
        c_vec = embed_with_hf_api(c)
        if not c_vec:
            continue
        s = _cosine(email_vec, c_vec)
        if s > best_score:
            best_score = s
            best = c
    return best

def similarity_with_hf_api(source: str, sentences: list[str]) -> Optional[list[float]]:
    if not HF_API_TOKEN:
        return None
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    url = f"https://api-inference.huggingface.co/pipeline/sentence-similarity/{HF_EMBED_MODEL}"
    payload = {"inputs": {"source_sentence": source, "sentences": sentences}, "options": {"wait_for_model": True}}
    for attempt in range(3):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=45)
            if r.status_code == 503 or ('loading' in (r.text or '').lower()):
                time.sleep(1.5)
                continue
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
                return [float(x) for x in data]
            # Alguns retornam dict com 'scores'
            if isinstance(data, dict) and isinstance(data.get('scores'), list):
                return [float(x) for x in data['scores']]
        except Exception as e:
            try:
                print('HF SS HTTP:', getattr(r, 'status_code', 'n/a'), (getattr(r, 'text', '') or '')[:300])
            except Exception:
                pass
            print('HF SS error:', e)
            time.sleep(1)
    return None
    return None

# (OpenAI removido)

# (Somente Hugging Face – funções auxiliares de multi-provider removidas)

# (Funções Ollama removidas)

def _normalize_ascii(s: str) -> str:
    """Remove acentos para facilitar matching de palavras-chave."""
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')

def heuristic_category_and_flags(text: str) -> tuple[str, bool, bool]:
    """Heurística com normalização; retorna (categoria, has_prod, has_improd)."""
    t = text.lower()
    t_ascii = _normalize_ascii(t)

    produtivo_kw = [
        'status', 'requis', 'suporte', 'problema', 'erro', 'anexo', 'arquivo', 'atualiza',
        'cancel', 'solicit', 'duvida', 'dúvida', 'login', 'senha', 'acesso', 'prazo', 'urgente',
        'ticket', 'protocolo', 'abertura', 'chamado', 'caso', 'incidente', 'solucao', 'solução'
    ]
    improdutivo_kw = [
        'feliz', 'parab', 'obrigad', 'agradec', 'bom dia', 'boa tarde', 'boa noite', 'natal', 'ano novo',
        'abraco', 'abraço', 'boas festas', 'sucesso', 'saudacoes', 'saudações', 'felicitacoes', 'felicitações',
        'meme', 'piada', 'engrac', 'engraç', 'gif', 'brincadeira', 'divertid', 'risad', 'kkk', 'rsrs'
    ]
    request_kw = ['poderia', 'poderiam', 'gostaria', 'preciso', 'informe', 'informar', 'verificar', 'retornar', 'atualizacao', 'atualização']
    strong_prod_kw = [
        'erro', 'falha', 'problema',
        'nao estou conseguindo', 'não estou conseguindo', 'nao consigo', 'não consigo',
        'acessar', 'acesso', 'bloqueado', 'bloqueio',
        '403', '404', '500',
        'abrir um chamado', 'abrir chamado', 'abertura de chamado'
    ]

    def has_kw(s: str, kw: str) -> bool:
        return kw in s

    has_prod = any(has_kw(t, k) or has_kw(t_ascii, _normalize_ascii(k)) for k in produtivo_kw)
    has_improd = any(has_kw(t, k) or has_kw(t_ascii, _normalize_ascii(k)) for k in improdutivo_kw)
    has_request = any(has_kw(t, k) or has_kw(t_ascii, _normalize_ascii(k)) for k in request_kw)
    has_strong_prod = any(has_kw(t, k) or has_kw(t_ascii, _normalize_ascii(k)) for k in strong_prod_kw)

    # Sinais fortes de produtivo (suporte/erro/abrir chamado/verificar) prevalecem, mesmo com cumprimentos
    if has_strong_prod or has_request:
        return 'Produtivo', has_prod or True, has_improd

    # Saudações/agradecimentos sem pedido/erro -> improdutivo
    if has_improd and not any(k in t for k in ['erro', 'problema', 'status', 'prazo', 'atualiza', 'verificar']):
        return 'Improdutivo', has_prod, has_improd
    if has_prod and not has_improd:
        return 'Produtivo', has_prod, has_improd
    if has_improd and not has_prod:
        return 'Improdutivo', has_prod, has_improd
    # Empate/indefinido: conservador
    return 'Improdutivo', has_prod, has_improd

def generate_personalized_reply(email_text: str, categoria: str) -> str:
    """Respostas mais personalizadas por padrão."""
    t = email_text.lower()
    t_ascii = _normalize_ascii(t)

    def any_kw(kws):
        return any(k in t or k in t_ascii for k in kws)

    # Detecta número de processo/protocolo/ticket etc.
    id_pattern = r"(?i)(processo|protocolo|requisic(?:ao|ão)|ticket|chamado|pedido|caso|id)\s*(?:n[ºo\.]|#|:)?\s*(\d{3,})"
    m = re.search(id_pattern, email_text) or re.search(id_pattern, t_ascii)
    id_kind = None
    id_number = None
    if m:
        id_kind = m.group(1).lower()
        id_number = m.group(2)

    is_status = any_kw(['status', 'atualiza', 'andamento', 'prazo', 'posicao', 'posição'])
    is_support = any_kw(['erro', 'falha', 'problema', 'nao consigo', 'não consigo', 'acesso', 'login', 'senha'])
    mentions_attach = any_kw(['anexo', 'arquivo', 'documento'])
    is_greeting_or_thanks = any_kw(['obrigad', 'agradec', 'parab', 'feliz', 'natal', 'ano novo', 'bom dia', 'boa tarde', 'boa noite', 'abraco', 'abraço'])
    asks_approval = any_kw(['aprovad'])
    open_ticket_intent = any_kw(['abrir um chamado', 'abrir chamado', 'abertura de chamado'])
    cadastro_update = any_kw(['atualizacao cadastral', 'atualização cadastral', 'cadastral', 'cadastro'])

    # Tenta extrair nome do colaborador
    colab_match = re.search(r'(?i)(colaborador|funcion[áa]rio)\s+([A-ZÁÉÍÓÚÃÕÂÊÔÇ][\wÁÉÍÓÚÃÕÂÊÔÇ]+(?:\s+[A-ZÁÉÍÓÚÃÕÂÊÔÇ][\wÁÉÍÓÚÃÕÂÊÔÇ]+)*)', email_text)
    collaborator_name = colab_match.group(2) if colab_match else None

    if categoria == 'Improdutivo':
        if is_greeting_or_thanks:
            return (
                'Obrigado pela sua mensagem e consideração! '
                'Estamos à disposição sempre que precisar. Abraços.'
            )
        return 'Agradecemos o contato! Permanecemos à disposição.'

    # Produtivo
    if open_ticket_intent or cadastro_update:
        if collaborator_name:
            return (
                f'Olá! Vamos abrir o chamado para atualização cadastral do colaborador {collaborator_name}. '
                'Para agilizar, poderia enviar: CPF/matrícula, quais campos devem ser atualizados (valor atual → novo), '
                'sistemas impactados e, se houver, autorização/anexos? Assim que o chamado for criado, '
                'retornaremos com o número e próximos passos.'
            )
        return (
            'Olá! Vamos abrir o chamado para atualização cadastral. Para agilizar, poderia enviar: CPF/matrícula, '
            'quais campos devem ser atualizados (valor atual → novo), sistemas impactados e, se houver, autorização/anexos? '
            'Assim que o chamado for criado, retornaremos com o número e próximos passos.'
        )
    if is_status:
        if id_number:
            if asks_approval:
                return (
                    f'Olá! Obrigado pela mensagem. Vamos verificar se o {id_kind} {id_number} já foi aprovado '
                    'e retornaremos assim que houver atualização.'
                )
            return (
                f'Olá! Obrigado pela mensagem. Vamos verificar o status do {id_kind} {id_number} '
                'e retornaremos assim que houver atualização.'
            )
        return (
            'Olá! Obrigado pela mensagem. Para agilizar a atualização, poderia confirmar o número da '
            'requisição/protocolo? Assim que tivermos novidades, retornaremos com o status.'
        )
    if is_support:
        if id_number:
            return (
                f'Olá! Sentimos pelo transtorno. Registramos o {id_kind} {id_number}. Para ajudarmos mais rápido, '
                'envie por favor: prints/erros exibidos, data e hora aproximadas do ocorrido e seu usuário. '
                'Nosso time de suporte vai analisar e retornar.'
            )
        return (
            'Olá! Sentimos pelo transtorno. Para ajudarmos mais rápido, envie por favor: número da requisição (se houver), '
            'prints/erros exibidos, data e hora aproximadas do ocorrido e seu usuário. Nosso time de suporte vai analisar e retornar.'
        )
    if mentions_attach:
        if id_number:
            return (
                f'Olá! Obrigado pelo envio. Vinculamos o(s) arquivo(s) ao {id_kind} {id_number}. '
                'Caso falte algum documento ou haja versão incorreta, avise. Daremos sequência e retornaremos com os próximos passos.'
            )
        return (
            'Olá! Obrigado pelo envio. Recebemos o(s) arquivo(s). Caso falte algum documento ou haja versão incorreta, avise. '
            'Daremos sequência e retornaremos com os próximos passos.'
        )
    if id_number:
        return (
            f'Olá! Recebemos sua mensagem sobre o {id_kind} {id_number} e já estamos verificando. '
            'Retornaremos com uma atualização em breve.'
        )
    return (
        'Olá! Recebemos sua mensagem e já estamos verificando. Se possível, compartilhe o número da requisição e quaisquer '
        'detalhes adicionais. Retornaremos com uma atualização em breve.'
    )

# (Removido bloco duplicado de _normalize_ascii/heuristic_category_and_flags para evitar sobrescrita)

def default_reply(email_text: str, categoria: str) -> str:
    if categoria == 'Produtivo':
        return (
            "Olá! Recebemos sua mensagem e já estamos verificando. "
            "Se possível, compartilhe o número da requisição e quaisquer anexos/detalhes adicionais. "
            "Retornaremos com uma atualização em breve."
        )
    return "Obrigado pela sua mensagem! Ficamos à disposição sempre que precisar."

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/process', methods=['POST'])
def process_email():
    try:
        # Sempre preferir IA (controlado apenas por .env)
        force_ai = FORCE_AI_REPLY
        min_ai_chars = MIN_AI_REPLY_CHARS

        email_text = ''
        if 'email_file' in request.files and request.files['email_file'].filename != '':
            file = request.files['email_file']
            if allowed_file(file.filename):
                filename = secure_filename(file.filename)
                if not os.path.exists(app.config['UPLOAD_FOLDER']):
                    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                if filename.lower().endswith('.txt'):
                    email_text = read_text_file_best_effort(filepath)
                elif filename.lower().endswith('.pdf'):
                    try:
                        import PyPDF2
                        pdf = PyPDF2.PdfReader(filepath)
                        email_text = " ".join([page.extract_text() or '' for page in pdf.pages]).strip()
                    except Exception as e:
                        print('Erro ao ler PDF:', e)
                        return jsonify({'error': 'Erro ao ler PDF.'}), 400
            else:
                return jsonify({'error': 'Arquivo não permitido.'}), 400
        else:
            email_text = request.form.get('email_text', '')

        if not email_text.strip():
            return jsonify({'error': 'Nenhum texto fornecido.'}), 400

        # Classificação com IA Hugging Face (fallback para heurística)
        # Tenta classificador local (se habilitado), senão Hugging Face
        api_cat = classify_with_local_model(email_text) if USE_LOCAL_CLASSIFIER else None
        if not api_cat:
            api_cat = classify_with_hf_api(email_text)
        heur_cat, has_prod, has_improd = heuristic_category_and_flags(email_text)
        categoria = api_cat or heur_cat
        categoria_origem = ('local' if USE_LOCAL_CLASSIFIER and api_cat else ('hf' if api_cat else 'heuristica'))
        if api_cat and api_cat != heur_cat and has_improd and not has_prod:
            categoria = 'Improdutivo'

        # Intenção via modelo local (se disponível) para orientar a resposta
        predicted_intent = None
        if intent_clf is not None:
            try:
                feats = text_to_features(email_text)
                predicted_intent = intent_clf.classify(feats)
            except Exception as e:
                print('Falha ao classificar intenção local:', e)

        # Resposta personalizada por padrão
        resposta = generate_personalized_reply(email_text, categoria)
        resposta_origem = 'template'
        # Pequena personalização extra baseada em intenção prevista
        if predicted_intent == 'status' and 'status' not in resposta.lower():
            resposta += ' Observação: vamos priorizar a verificação de status e retornaremos em seguida.'
        elif predicted_intent == 'suporte' and 'suporte' not in resposta.lower():
            resposta += ' Nosso time de suporte irá analisar os detalhes e responder o quanto antes.'

        # Geração de resposta via IA (ordem: chat > local t2t > HF t2t)
        ai_resp = generate_reply_with_hf_chat(email_text, categoria)
        if not ai_resp:
            ai_resp = generate_reply_with_local_model(email_text, categoria) if USE_LOCAL_GENERATOR else None
        if not ai_resp:
            ai_resp = generate_reply_with_hf_api(email_text, categoria)
        # Pós-processamento para limpar raciocínio e tags
        if ai_resp:
            ai_resp = post_process_ai_output(ai_resp, categoria)
        if ai_resp and (force_ai or len(ai_resp.strip()) >= min_ai_chars):
            resposta = ai_resp
            # marca origem mais específica
            if ai_resp and hf_chat_client is not None and resposta != '':
                resposta_origem = 'ai'
            elif USE_LOCAL_GENERATOR and _local_t2t['model'] is not None:
                resposta_origem = 'ai-local'
            else:
                resposta_origem = 'ai'
        else:
            # Fallback: usar embeddings para escolher o melhor template (com Jaccard local se HF indisponível)
            emb_choice = choose_reply_by_embeddings(email_text, categoria)
            if emb_choice:
                resposta = emb_choice
                resposta_origem = 'ai-embed'

        return jsonify({'categoria': categoria, 'categoria_origem': categoria_origem, 'resposta': resposta, 'resposta_origem': resposta_origem})
    except Exception as e:
        print('Erro ao processar email:', e)
        return jsonify({'error': 'Falha interna ao processar o email.'}), 500

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    port = int(os.getenv('PORT', '5000'))
    app.run(host='0.0.0.0', port=port, debug=True)

# AutoU Email Classifier

Aplicação web que classifica emails em "Produtivo" ou "Improdutivo" e sugere respostas automáticas.
Usa Hugging Face (zero-shot para classificação e geração via text2text/chat) quando o token está presente. Sem token, recorre a heurísticas robustas e respostas personalizadas por template, com fallback de seleção por similaridade.

## Como executar localmente (Windows)

1) Instale as dependências:

```bash
pip install -r requirements.txt
```

2) (Opcional, recomendado) Configure o Hugging Face Inference API:

- Crie um token gratuito: https://huggingface.co/settings/tokens
- Copie `.env.example` para `.env` e preencha `HF_API_TOKEN=...`
- Alternativamente, defina a variável apenas para a sessão atual do PowerShell:

```powershell
$env:HF_API_TOKEN = "seu_token_aqui"
# Opcional: força sempre usar a resposta gerada pela IA quando houver
$env:FORCE_AI_REPLY = "1"
```

Sem `HF_API_TOKEN`, o app ainda funciona com classificação heurística e respostas padrão personalizadas (e um seletor de template por similaridade quando disponível).

3) Inicie o servidor:

```powershell
python .\app.py
```

4) Acesse no navegador: http://localhost:5000

## Funcionalidades

- Entrada por texto direto ou upload de `.txt` e `.pdf` (extração via PyPDF2), com drag-and-drop
- Classificação do email (IA Hugging Face quando configurado; fallback heurístico caso contrário)
- Sugestão de resposta automática personalizada, com:
   - Preferência por IA (chat/text2text)
   - Fallback por embeddings/similaridade local quando necessário
- UI com carregamento, tags de origem (HF/heurística/embeddings), copiar e baixar resposta

## Variáveis de ambiente suportadas

- HF_API_TOKEN: token da Hugging Face Inference API (opcional, habilita IA remota)
- HF_ZERO_SHOT_MODEL: modelo zero-shot (padrão: `joeddav/xlm-roberta-large-xnli`)
- HF_T2T_MODEL / HF_T2T_CANDIDATES: modelos de geração de texto
- HF_CHAT_MODEL: modelo de chat (padrão: `HuggingFaceTB/SmolLM3-3B`)
- HF_EMBED_MODEL: modelo de embeddings/similarity
- FORCE_AI_REPLY: "1" para preferir a resposta gerada pela IA quando houver
- MIN_AI_REPLY_CHARS: mínimo de caracteres para aceitar resposta da IA


## Estrutura do projeto

- `app.py`: Backend Flask (rotas `/`, `/process`, `/health`), integração com Hugging Face e heurísticas
- `templates/index.html`: Interface web
- `static/style.css`: Estilos
- `static/script.js`: Integração frontend-backend
- `requirements.txt`: Dependências Python
- `render.yaml` e `Procfile`: Configurações para deploy

## Deploy na nuvem (Render.com)

1) Faça push do projeto para o GitHub.
2) No Render, crie um novo Web Service apontando para o repositório (ou use o `render.yaml`).
3) Variáveis de ambiente recomendadas:
   - `PYTHON_VERSION=3.12.5`
   - `HF_API_TOKEN` (se quiser habilitar IA remota)
   - `FORCE_AI_REPLY` (opcional)
4) Build e start (já definidos):
   - Build: `pip install -r requirements.txt`
   - Start: `gunicorn app:app --timeout 120 --workers 2`
5) Health check: `GET /health`

Alternativas: Railway, Fly.io, Azure Web Apps.

## Autor
Victor Henrique Estrella Carracci
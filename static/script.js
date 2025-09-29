document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('upload-form');
    const dropZone = document.getElementById('drop-zone');
    const resultDiv = document.getElementById('result');
    const categoriaSpan = document.getElementById('categoria');
    const categoriaOrigem = document.getElementById('categoria-origem');
    const respostaDiv = document.getElementById('resposta');
    const respostaOrigem = document.getElementById('resposta-origem');
    const loading = document.getElementById('loading');
    const submitBtn = document.getElementById('submit-btn');
    const copyBtn = document.getElementById('copy-btn');
    const downloadBtn = document.getElementById('download-btn');
    const fileInput = document.getElementById('email-file');
    const textInput = document.getElementById('email-text');
    // Se a página estiver sendo servida via Live Server (porta 5500), direcione as requisições para o backend Flask (porta 5000)
    const API_BASE = (location.port === '5000') ? '' : 'http://localhost:5000';

    // Drag and drop
    if (dropZone) {
        ['dragenter','dragover'].forEach(evt => {
            dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.add('hover'); });
        });
        ;['dragleave','drop'].forEach(evt => {
            dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.remove('hover'); });
        });
        dropZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files && files.length) {
                fileInput.files = files;
            }
        });
    }

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        resultDiv.style.display = 'none';
        loading.style.display = 'inline-block';
        submitBtn.disabled = true;
        const formData = new FormData(form);
        fetch(`${API_BASE}/process`, {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
            } else {
                categoriaSpan.textContent = data.categoria;
                respostaDiv.textContent = data.resposta;
                // Etiquetas de origem
                categoriaOrigem.textContent = data.categoria_origem === 'hf' ? '(via IA/HF)' : (data.categoria_origem === 'local' ? '(via IA local)' : '(via heurística)');
                respostaOrigem.textContent =
                    data.resposta_origem === 'ai' ? 'Resposta gerada por IA (HF)' :
                    data.resposta_origem === 'ai-local' ? 'Resposta gerada por IA (local)' :
                    data.resposta_origem === 'ai-embed' ? 'Resposta escolhida por IA (embeddings)' :
                    'Resposta por template';
                resultDiv.style.display = 'block';
            }
        })
        .catch(() => {
            alert('Erro ao processar o email.');
        })
        .finally(() => {
            loading.style.display = 'none';
            submitBtn.disabled = false;
        });
    });

    // Copy to clipboard
    copyBtn.addEventListener('click', () => {
        const txt = respostaDiv.textContent || '';
        if (!txt) return;
        navigator.clipboard.writeText(txt).then(() => {
            copyBtn.textContent = 'Copiado!';
            setTimeout(() => copyBtn.textContent = 'Copiar', 1200);
        });
    });

    // Download as .txt
    downloadBtn.addEventListener('click', () => {
        const txt = respostaDiv.textContent || '';
        if (!txt) return;
        const blob = new Blob([txt], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'resposta.txt';
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    });
});

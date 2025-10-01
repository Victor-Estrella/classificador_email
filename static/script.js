document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('upload-form');
    // drag-and-drop removido
    const resultDiv = document.getElementById('result');
    const categoriaSpan = document.getElementById('categoria');
    // origem removida do UI
    const respostaDiv = document.getElementById('resposta');
    // origem removida do UI
    const loading = document.getElementById('loading');
    const submitBtn = document.getElementById('submit-btn');
    const copyBtn = document.getElementById('copy-btn');
    const downloadBtn = document.getElementById('download-btn');
    const fileInput = document.getElementById('email-file');
    const textInput = document.getElementById('email-text');
    // Base da API: em produção (Render), usar same-origin ('').
    // Em desenvolvimento local, se a página NÃO estiver na porta 5000 (ex.: Live Server 5500), aponte para http://localhost:5000
    const isLocalHost = ['localhost', '127.0.0.1'].includes(location.hostname);
    const API_BASE = (isLocalHost && location.port && location.port !== '5000') ? 'http://localhost:5000' : '';

    // drag-and-drop removido

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
                // etiquetas de origem removidas
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

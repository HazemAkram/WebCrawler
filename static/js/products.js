let uploadedPath = null;
async function uploadCsv() {
    const formData = new FormData(document.getElementById('uploadForm'));
    const r = await fetch('/upload_products', { method: 'POST', body: formData });
    const j = await r.json();
    const out = document.getElementById('uploadRes');
    if (j.success) {
        out.className = 'alert alert-success mt-3';
        out.textContent = `Uploaded: ${j.filename} • URLs: ${j.total_urls}`;
        uploadedPath = j.filepath;
    } else {
        out.className = 'alert alert-danger mt-3';
        out.textContent = j.error || 'Upload failed';
    }
}
async function startProducts() {
    const apiKey = document.getElementById('apiKey').value.trim();
    const maxMbStr = document.getElementById('maxMb').value.trim();
    const startBtn = document.getElementById('startBtn');
    if (!uploadedPath) { showToast('Please upload a CSV first.', 'danger'); return }
    if (!apiKey) { showToast('Please enter your GROQ_API_KEY.', 'warning'); return }
    startBtn.disabled = true; startBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Starting...';
    const body = { csv_filepath: uploadedPath, api_key: apiKey, pdf_size_limit: maxMbStr ? parseInt(maxMbStr) : null };
    const r = await fetch('/start_products', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const j = await r.json();
    startBtn.disabled = false; startBtn.textContent = 'Start Product Install';
    if (j.success) { showToast('Product installation started. Check Logs tab on main page.', 'success') } else { showToast(j.error || 'Failed to start', 'danger') }
}
function showToast(msg, type) {
    const el = document.getElementById('toastBox');
    el.className = `alert alert-${type}`; el.textContent = msg; el.style.display = 'block';
    setTimeout(() => { el.style.display = 'none' }, 4000);
}
function toggleApiKey(btn) {
    const t = document.getElementById('apiKey'); const i = btn.querySelector('i');
    if (t.type === 'password') { t.type = 'text'; i.className = 'fas fa-eye-slash' } else { t.type = 'password'; i.className = 'fas fa-eye' }
}


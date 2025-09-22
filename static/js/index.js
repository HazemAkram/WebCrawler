// Global variables
let uploadedFile = null;
let statusInterval = null;
let isRunning = false;

// File upload handling
const fileUploadArea = document.getElementById('fileUploadArea');
const csvFileInput = document.getElementById('csvFile');
const fileInfo = document.getElementById('fileInfo');

if (fileUploadArea) {
    fileUploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        fileUploadArea.classList.add('dragover');
    });

    fileUploadArea.addEventListener('dragleave', () => {
        fileUploadArea.classList.remove('dragover');
    });

    fileUploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        fileUploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });
}

if (csvFileInput) {
    csvFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

function handleFileUpload(file) {
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showAlert('Please select a CSV file.', 'danger');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                uploadedFile = data;
                document.getElementById('fileName').textContent = data.filename;
                document.getElementById('totalSites').textContent = data.total_sites;
                fileInfo.style.display = 'block';
                addLog('File uploaded successfully: ' + data.filename, 'success');
            } else {
                showAlert(data.error, 'danger');
            }
        })
        .catch(error => {
            showAlert('Error uploading file: ' + error.message, 'danger');
        });
}

// Password toggle
function togglePassword() {
    const apiKeyInput = document.getElementById('apiKey');
    const eyeIcon = document.getElementById('eyeIcon');
    if (!apiKeyInput || !eyeIcon) return;
    if (apiKeyInput.type === 'password') {
        apiKeyInput.type = 'text';
        eyeIcon.className = 'fas fa-eye-slash';
    } else {
        apiKeyInput.type = 'password';
        eyeIcon.className = 'fas fa-eye';
    }
}

// Crawling controls
function startCrawling() {
    if (!uploadedFile) {
        showAlert('Please upload a CSV file first.', 'warning');
        return;
    }
    const apiKey = document.getElementById('apiKey').value.trim();
    if (!apiKey) {
        showAlert('Please enter an API key.', 'warning');
        return;
    }
    const model = 'groq/llama-3.1-8b-instant';
    const pdfSizeLimit = parseInt(document.getElementById('pdfSizeLimit').value) || 25;
    const skipLargeFiles = document.getElementById('skipLargeFiles').checked;

    fetch('/start_crawling', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            csv_filepath: uploadedFile.filepath,
            api_key: apiKey,
            model: model,
            pdf_size_limit: pdfSizeLimit,
            skip_large_files: skipLargeFiles
        })
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                isRunning = true;
                updateStatus('running');
                startStatusPolling();
                addLog('Crawling started successfully', 'success');
            } else {
                showAlert(data.error, 'danger');
            }
        })
        .catch(error => {
            showAlert('Error starting crawling: ' + error.message, 'danger');
        });
}

function stopCrawling() {
    fetch('/stop_crawling', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                addLog('Stop request sent', 'warning');
            } else {
                showAlert(data.error, 'danger');
            }
        })
        .catch(error => {
            showAlert('Error stopping crawling: ' + error.message, 'danger');
        });
}

// Status polling
function startStatusPolling() {
    if (statusInterval) clearInterval(statusInterval);
    statusInterval = setInterval(() => {
        fetch('/status')
            .then(response => response.json())
            .then(data => {
                updateStatusDisplay(data);
                if (!data.is_running && isRunning) {
                    isRunning = false;
                    updateStatus('stopped');
                    clearInterval(statusInterval);
                    addLog('Crawling process finished', 'info');
                }
            })
            .catch(() => { });
    }, 1000);
}

function updateStatusDisplay(data) {
    if (data.is_running) updateStatus('running');
    else if (data.stop_requested) updateStatus('stopping');
    else updateStatus('stopped');

    document.getElementById('currentSite').textContent = data.current_site;
    document.getElementById('totalSitesStats').textContent = data.total_sites;
    document.getElementById('currentPage').textContent = data.current_page;
    document.getElementById('totalVenues').textContent = data.total_venues;

    if (data.is_running || data.total_sites > 0) {
        document.getElementById('statsSection').style.display = 'block';
    }
    if (data.total_sites > 0) {
        const progress = (data.current_site / data.total_sites) * 100;
        const progressBar = document.getElementById('progressBar');
        const progressBarInner = progressBar.querySelector('.progress-bar');
        progressBar.style.display = 'block';
        progressBarInner.style.width = progress + '%';
        progressBarInner.textContent = Math.round(progress) + '%';
    }
    if (data.logs && data.logs.length > 0) updateLogs(data.logs);
}

function updateStatus(status) {
    const indicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    indicator.className = 'status-indicator';
    switch (status) {
        case 'running':
            indicator.classList.add('status-running');
            statusText.textContent = 'Crawling in progress...';
            startBtn.disabled = true;
            stopBtn.disabled = false;
            break;
        case 'stopping':
            indicator.classList.add('status-warning');
            statusText.textContent = 'Stopping...';
            startBtn.disabled = true;
            stopBtn.disabled = true;
            break;
        case 'stopped':
            indicator.classList.add('status-stopped');
            statusText.textContent = 'Stopped';
            startBtn.disabled = false;
            stopBtn.disabled = true;
            break;
        default:
            indicator.classList.add('status-idle');
            statusText.textContent = 'Ready to start';
            startBtn.disabled = false;
            stopBtn.disabled = true;
    }
}

// Logging functions
function addLog(message, level = 'info') {
    const logContainer = document.getElementById('logContainer');
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry';
    logEntry.innerHTML = `
    <span class="log-timestamp">[${timestamp}]</span>
    <span class="log-${level}">${message}</span>
  `;
    logContainer.appendChild(logEntry);
    logContainer.scrollTop = logContainer.scrollHeight;
}

function updateLogs(logs) {
    const logContainer = document.getElementById('logContainer');
    logContainer.innerHTML = '';
    logs.forEach(log => {
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';
        logEntry.innerHTML = `
      <span class="log-timestamp">[${log.timestamp}]</span>
      <span class="log-${log.level.toLowerCase()}">${log.message}</span>
    `;
        logContainer.appendChild(logEntry);
    });
    logContainer.scrollTop = logContainer.scrollHeight;
}

function clearLogs() {
    document.getElementById('logContainer').innerHTML = `
    <div class="log-entry">
      <span class="log-timestamp">[System]</span>
      <span class="log-info">Logs cleared.</span>
    </div>
  `;
}

// Utility functions
function showAlert(message, type) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
    ${message}
    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
  `;
    document.querySelector('.content').insertBefore(alertDiv, document.querySelector('.content').firstChild);
    setTimeout(() => { if (alertDiv.parentNode) alertDiv.remove(); }, 5000);
}

function downloadOutput() { window.open('/download_output', '_blank'); }

document.addEventListener('DOMContentLoaded', function () { updateStatus('idle'); });



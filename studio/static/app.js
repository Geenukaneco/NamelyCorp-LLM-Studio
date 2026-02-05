// app.js - LLM Training Studio Frontend Logic

const API_BASE = window.location.origin;
let currentTaskId = null;
let wsConnection = null;

// =============================================================================
// Initialization
// =============================================================================

document.addEventListener('DOMContentLoaded', function() {
    initializeTheme();
    initializeTabs();
    loadDashboard();
    updateGPUStatus();
    
    // Update GPU status every 5 seconds
    setInterval(updateGPUStatus, 5000);
});

// =============================================================================
// Theme Management
// =============================================================================

function initializeTheme() {
    const themeToggle = document.getElementById('themeToggle');
    
    // Check for saved theme preference or default to light
    const savedTheme = localStorage.getItem('theme') || 'light';
    setTheme(savedTheme);
    
    // Toggle theme on button click
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            setTheme(newTheme);
            localStorage.setItem('theme', newTheme);
        });
    }
}

function setTheme(theme) {
    if (theme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
    } else {
        document.documentElement.removeAttribute('data-theme');
    }
}

// =============================================================================
// Tab Management
// =============================================================================

function initializeTabs() {
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const tabName = tab.getAttribute('data-tab');
            switchTab(tabName);
        });
    });
}

function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById(tabName).classList.add('active');
}

// =============================================================================
// Dashboard
// =============================================================================

async function loadDashboard() {
    try {
        const response = await fetch(`${API_BASE}/api/stats`);
        const stats = await response.json();
        
        document.getElementById('docsCount').textContent = stats.docs_count;
        document.getElementById('docsSize').textContent = stats.docs_size_mb.toFixed(1);
        document.getElementById('datasetRows').textContent = stats.dataset_rows;
        document.getElementById('modelsCount').textContent = stats.models_count;
        
        // Load system status
        await updateSystemStatus();
    } catch (error) {
        console.error('Error loading dashboard:', error);
    }
}

async function updateSystemStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/system`);
        const system = await response.json();
        
        // Update status indicators
        const gpuStatus = document.getElementById('gpuStatus2');
        const cudaStatus = document.getElementById('cudaStatus');
        const ocrStatus = document.getElementById('ocrStatus');
        
        if (gpuStatus) {
            gpuStatus.className = `status-${system.gpu.status}`;
        }
        if (cudaStatus) {
            cudaStatus.className = `status-${system.cuda.status}`;
        }
        if (ocrStatus) {
            ocrStatus.className = `status-${system.tesseract.status}`;
            ocrStatus.title = system.tesseract.info;
        }
    } catch (error) {
        console.error('Error updating system status:', error);
    }
}

async function updateGPUStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/gpu`);
        const gpu = await response.json();
        
        if (gpu.available) {
            document.getElementById('gpuName').textContent = gpu.name;
            document.getElementById('vramText').textContent = 
                `${gpu.vram_used_gb} / ${gpu.vram_total_gb} GB`;
            document.getElementById('vramFill').style.width = `${gpu.vram_percent}%`;
            
            const statusDot = document.getElementById('gpuStatus2');
            statusDot.className = gpu.gpu_percent > 80 ? 'status-error' : 'status-ok';
        } else {
            document.getElementById('gpuName').textContent = 'No GPU';
            document.getElementById('vramText').textContent = '0 / 0 GB';
        }
    } catch (error) {
        console.error('Error updating GPU:', error);
    }
}

// =============================================================================
// Dataset Builder
// =============================================================================

async function scanFolder() {
    const docsDir = document.getElementById('docsDir').value || 'C:\\dev\\llm\\docs';
    try {
        const response = await fetch(`${API_BASE}/api/files?path=${encodeURIComponent(docsDir)}`);
        const data = await response.json();
        
        if (data.files && Array.isArray(data.files)) {
            const count = data.files.length;
            const totalSizeMB = data.files.reduce((sum, f) => sum + (f.size_mb || 0), 0).toFixed(2);
            
            alert(`📁 Found ${count} file(s)\n📊 Total size: ${totalSizeMB} MB\n\nSupported formats: PDF, DOCX, TXT, CSV, XLSX`);
            
            // Log file details to console for debugging
            console.log('Files found:', data.files);
        } else {
            alert('No files found or error scanning directory');
        }
    } catch (error) {
        alert('Error scanning folder: ' + error.message);
        console.error('Scan error:', error);
    }
}

async function buildDataset() {
    const request = {
        docs_dir: document.getElementById('docsDir').value || 'C:\\dev\\llm\\docs',
        output_csv: document.getElementById('outputCsv').value || 'C:\\dev\\llm\\data_qa.csv',
        model_id: document.getElementById('datasetModel').value || 'C:\\models\\Llama-3.2-3B-Instruct',
        chars_per_chunk: parseInt(document.getElementById('chunkSize').value) || 1400,
        max_tokens: parseInt(document.getElementById('maxTokens').value) || 1024,
        enable_ocr: document.getElementById('enableOcr').checked,
        ocr_lang: document.getElementById('ocrLang').value || 'eng',
        max_files: parseInt(document.getElementById('maxFiles').value) || null,
        mix_patterns: true
    };
    
    // Validate inputs
    if (!request.docs_dir || request.docs_dir.trim() === '') {
        alert('Please enter a documents directory path');
        return;
    }
    if (!request.model_id || request.model_id.trim() === '') {
        alert('Please enter a model path');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/dataset/build`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(request)
        });
        
        const result = await response.json();
        currentTaskId = result.task_id;
        
        // Show progress section
        document.getElementById('datasetProgress').style.display = 'block';
        
        // Start monitoring
        monitorTask(result.task_id, 'dataset');
    } catch (error) {
        alert('Error starting dataset build: ' + error.message);
    }
}

// =============================================================================
// Validation
// =============================================================================

async function validateDataset() {
    const request = {
        csv_path: document.getElementById('validateCsv').value || 'C:\\dev\\llm\\data_qa.csv',
        model_id: document.getElementById('validateModel').value || 'C:\\models\\Llama-3.2-3B-Instruct',
        max_tokens: parseInt(document.getElementById('validateMaxTokens').value) || 1024,
        write_clean: document.getElementById('writeClean').checked,
        clean_path: document.getElementById('cleanPath').value || 'C:\\dev\\llm\\data_qa.cleaned.csv',
        report_path: "C:\\dev\\llm\\qa_report.md"
    };
    
    // Validate inputs
    if (!request.csv_path || request.csv_path.trim() === '') {
        alert('Please enter a CSV file path');
        return;
    }
    if (!request.model_id || request.model_id.trim() === '') {
        alert('Please enter a model path');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/validate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(request)
        });
        
        const result = await response.json();
        currentTaskId = result.task_id;
        
        // Show progress section
        document.getElementById('validateProgress').style.display = 'block';
        
        // Start monitoring
        monitorTask(result.task_id, 'validate');
    } catch (error) {
        alert('Error starting validation: ' + error.message);
    }
}

// =============================================================================
// Training
// =============================================================================

function loadPreset(preset) {
    const presets = {
        quick: {
            epochs: 1,
            batchSize: 2,
            gradAccum: 8,
            loraR: 8,
            loraAlpha: 16
        },
        balanced: {
            epochs: 2,
            batchSize: 2,
            gradAccum: 16,
            loraR: 16,
            loraAlpha: 32
        },
        quality: {
            epochs: 3,
            batchSize: 1,
            gradAccum: 32,
            loraR: 32,
            loraAlpha: 64
        }
    };
    
    const config = presets[preset];
    if (config) {
        document.getElementById('epochs').value = config.epochs;
        document.getElementById('batchSize').value = config.batchSize;
        document.getElementById('gradAccum').value = config.gradAccum;
        document.getElementById('loraR').value = config.loraR;
        document.getElementById('loraAlpha').value = config.loraAlpha;
    }
}

async function startTraining() {
    const request = {
        csv_path: document.getElementById('trainCsv').value || 'C:\\dev\\llm\\data_qa.cleaned.csv',
        model_id: document.getElementById('trainModel').value || 'C:\\models\\Llama-3.2-3B-Instruct',
        output_dir: document.getElementById('trainOutput').value || 'C:\\dev\\llm\\ft_out',
        run_name: "training_run",
        epochs: parseInt(document.getElementById('epochs').value) || 2,
        batch_size: parseInt(document.getElementById('batchSize').value) || 2,
        grad_accum: parseInt(document.getElementById('gradAccum').value) || 16,
        learning_rate: parseFloat(document.getElementById('learningRate').value) || 1e-5,
        max_len: parseInt(document.getElementById('maxLen').value) || 1024,
        lora_r: parseInt(document.getElementById('loraR').value) || 16,
        lora_alpha: parseInt(document.getElementById('loraAlpha').value) || 32,
        lora_dropout: parseFloat(document.getElementById('loraDropout').value) || 0.05,
        include_mlp: document.getElementById('includeMlp').checked,
        merge_full: document.getElementById('mergeFull').checked,
        val_split: parseFloat(document.getElementById('valSplit').value) || 0.05
    };
    
    // Validate inputs
    if (!request.csv_path || request.csv_path.trim() === '') {
        alert('Please enter a dataset CSV path');
        return;
    }
    if (!request.model_id || request.model_id.trim() === '') {
        alert('Please enter a model path');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(request)
        });
        
        const result = await response.json();
        currentTaskId = result.task_id;
        
        // Show progress section
        document.getElementById('trainProgress').style.display = 'block';
        
        // Start monitoring with WebSocket
        monitorTaskWebSocket(result.task_id, 'train');
    } catch (error) {
        alert('Error starting training: ' + error.message);
    }
}

// =============================================================================
// Export & Testing
// =============================================================================

async function convertGGUF() {
    alert('GGUF conversion is not yet fully implemented. You can run the llama.cpp converter manually.');
}

async function testInference() {
    alert('Inference testing requires loading the model. This feature will be added in the next version.');
    
    // Placeholder: Show example response
    document.getElementById('testResult').style.display = 'block';
    document.getElementById('testResponse').textContent = 
        '(source: pto_policy.txt)\n\nNew employees accrue 15 days of PTO annually, starting after their 90-day probation period. Days are prorated for the first year based on start date.';
}

// =============================================================================
// Task Monitoring
// =============================================================================

async function monitorTask(taskId, prefix) {
    const progressFill = document.getElementById(`${prefix}ProgressFill`);
    const statusText = document.getElementById(`${prefix}Status`);
    const logsDiv = document.getElementById(`${prefix}Logs`);
    
    const poll = async () => {
        try {
            const response = await fetch(`${API_BASE}/api/task/${taskId}`);
            const data = await response.json();
            
            // Update progress
            const status = data.status;
            progressFill.style.width = `${status.progress}%`;
            statusText.textContent = status.message;
            
            // Update logs
            if (data.logs && data.logs.length > 0) {
                logsDiv.innerHTML = data.logs.map(log => `<p>${escapeHtml(log)}</p>`).join('');
                logsDiv.scrollTop = logsDiv.scrollHeight;
            }
            
            // Check if complete
            if (status.status === 'completed') {
                statusText.textContent = '✅ ' + status.message;
                loadDashboard(); // Refresh stats
                return;
            } else if (status.status === 'failed') {
                statusText.textContent = '❌ ' + status.message;
                return;
            }
            
            // Continue polling
            setTimeout(poll, 2000);
            
        } catch (error) {
            console.error('Error monitoring task:', error);
        }
    };
    
    poll();
}

function monitorTaskWebSocket(taskId, prefix) {
    const progressFill = document.getElementById(`${prefix}ProgressFill`);
    const statusText = document.getElementById(`${prefix}Status`);
    const logsDiv = document.getElementById(`${prefix}Logs`);
    const metricsDiv = document.getElementById(`${prefix}Metrics`);
    
    // Close existing connection
    if (wsConnection) {
        wsConnection.close();
    }
    
    // Create WebSocket connection
    const wsUrl = `ws://${window.location.host}/ws/task/${taskId}`;
    wsConnection = new WebSocket(wsUrl);
    
    wsConnection.onmessage = (event) => {
        const data = JSON.parse(event.data);
        const status = data.status;
        
        // Update progress
        progressFill.style.width = `${status.progress}%`;
        statusText.textContent = status.message;
        
        // Update logs
        if (data.logs && data.logs.length > 0) {
            logsDiv.innerHTML = data.logs.map(log => `<p>${escapeHtml(log)}</p>`).join('');
            logsDiv.scrollTop = logsDiv.scrollHeight;
        }
        
        // Update GPU metrics for training
        if (prefix === 'train' && data.gpu) {
            document.getElementById('trainGpu').textContent = `${data.gpu.gpu_percent}%`;
            document.getElementById('trainVram').textContent = 
                `${data.gpu.vram_used_gb} / ${data.gpu.vram_total_gb} GB`;
        }
        
        // Check if complete
        if (status.status === 'completed') {
            statusText.textContent = '✅ ' + status.message;
            wsConnection.close();
            loadDashboard();
        } else if (status.status === 'failed') {
            statusText.textContent = '❌ ' + status.message;
            wsConnection.close();
        }
    };
    
    wsConnection.onerror = (error) => {
        console.error('WebSocket error:', error);
        // Fall back to polling
        monitorTask(taskId, prefix);
    };
    
    wsConnection.onclose = () => {
        console.log('WebSocket closed');
    };
}

// =============================================================================
// Utilities
// =============================================================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Make functions globally available
window.switchTab = switchTab;
window.scanFolder = scanFolder;
window.buildDataset = buildDataset;
window.validateDataset = validateDataset;
window.loadPreset = loadPreset;
window.startTraining = startTraining;
window.convertGGUF = convertGGUF;
window.testInference = testInference;

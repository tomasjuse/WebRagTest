import os
import shutil
import threading
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

PROJECT_DIR = Path(__file__).parent
UPLOADS_DIR = PROJECT_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

app = FastAPI()

_status: dict = {"state": "idle", "message": "Ready."}

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RAG Document Indexer</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #f0f2f5;
      color: #333;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .container {
      background: white;
      border-radius: 12px;
      padding: 40px;
      width: 580px;
      box-shadow: 0 2px 20px rgba(0,0,0,0.08);
    }
    h1 { font-size: 1.5rem; margin-bottom: 6px; }
    .subtitle { color: #666; margin-bottom: 28px; font-size: 0.9rem; }
    .drop-zone {
      border: 2px dashed #ccc;
      border-radius: 8px;
      padding: 32px;
      text-align: center;
      cursor: pointer;
      transition: border-color 0.2s, background 0.2s;
      margin-bottom: 16px;
      user-select: none;
    }
    .drop-zone.drag-over { border-color: #0066cc; background: #f0f7ff; }
    .drop-zone p { color: #666; font-size: 0.9rem; }
    .drop-zone span { color: #0066cc; text-decoration: underline; cursor: pointer; }
    .drop-zone .hint { margin-top: 6px; font-size: 0.8rem; color: #aaa; }
    #fileInput { position: fixed; top: -9999px; left: -9999px; }
    .file-list { margin-bottom: 12px; }
    .file-item {
      display: flex;
      align-items: center;
      padding: 8px 12px;
      background: #f9f9f9;
      border-radius: 6px;
      margin-bottom: 6px;
      font-size: 0.875rem;
    }
    .file-icon { margin-right: 8px; font-size: 1.1rem; }
    .file-name { flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .file-size { color: #999; margin: 0 12px; white-space: nowrap; }
    .remove-btn {
      background: none;
      border: none;
      color: #cc3333;
      cursor: pointer;
      font-size: 1rem;
      padding: 0 4px;
      line-height: 1;
    }
    .remove-btn:hover { color: #aa0000; }
    .error-msg { color: #cc3333; font-size: 0.85rem; margin-bottom: 12px; min-height: 1.2em; }
    #buildBtn {
      width: 100%;
      padding: 12px;
      background: #0066cc;
      color: white;
      border: none;
      border-radius: 8px;
      font-size: 1rem;
      cursor: pointer;
      transition: background 0.2s;
    }
    #buildBtn:hover:not(:disabled) { background: #0055aa; }
    #buildBtn:disabled { background: #ccc; cursor: not-allowed; }
    .status {
      margin-top: 16px;
      padding: 12px 16px;
      border-radius: 6px;
      font-size: 0.875rem;
      display: none;
    }
    .status.running { background: #fff8e1; color: #856404; display: block; }
    .status.done    { background: #d4edda; color: #155724; display: block; }
    .status.error   { background: #f8d7da; color: #721c24; display: block; }
    .spinner {
      display: inline-block;
      width: 12px; height: 12px;
      border: 2px solid currentColor;
      border-top-color: transparent;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      margin-right: 8px;
      vertical-align: middle;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div class="container">
    <h1>RAG Document Indexer</h1>
    <p class="subtitle">Upload 1–10 documents to rebuild the search index used by the MCP tools.</p>

    <input type="file" id="fileInput" multiple accept=".pdf,.pptx,.doc,.docx">
    <div class="drop-zone" id="dropZone">
      <p>Drag &amp; drop files here, or <span>browse</span></p>
      <p class="hint">PDF, PPTX and Word (.doc/.docx) &bull; max 10 files</p>
    </div>

    <div class="file-list" id="fileList"></div>
    <div class="error-msg" id="errorMsg"></div>

    <button id="buildBtn" disabled onclick="startIndexing()">Build Index</button>
    <div class="status" id="statusBox"></div>
  </div>

  <script>
    let selectedFiles = [];
    const dropZone  = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const fileList  = document.getElementById('fileList');
    const errorMsg  = document.getElementById('errorMsg');
    const buildBtn  = document.getElementById('buildBtn');
    const statusBox = document.getElementById('statusBox');
    let pollTimer   = null;

    // Click anywhere on drop zone to open file picker
    dropZone.addEventListener('click', () => fileInput.click());

    // Prevent browser from opening dropped files
    document.addEventListener('dragover', e => e.preventDefault());
    document.addEventListener('drop', e => e.preventDefault());

    // Drag & drop on drop zone
    dropZone.addEventListener('dragover', e => {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      addFiles([...e.dataTransfer.files]);
    });
    fileInput.addEventListener('change', () => {
      addFiles([...fileInput.files]);
      fileInput.value = '';
    });

    function addFiles(newFiles) {
      errorMsg.textContent = '';
      for (const f of newFiles) {
        const ext = f.name.split('.').pop().toLowerCase();
        if (!['pdf', 'pptx', 'doc', 'docx'].includes(ext)) {
          errorMsg.textContent = `Skipped "${f.name}": only PDF, PPTX, and Word (.doc/.docx) are supported.`;
          continue;
        }
        if (!selectedFiles.some(x => x.name === f.name)) {
          selectedFiles.push(f);
        }
      }
      if (selectedFiles.length > 10) {
        selectedFiles = selectedFiles.slice(0, 10);
        errorMsg.textContent = 'Maximum 10 files allowed. Extra files were ignored.';
      }
      renderFileList();
    }

    function formatSize(bytes) {
      if (bytes < 1024)        return bytes + ' B';
      if (bytes < 1048576)     return (bytes / 1024).toFixed(1) + ' KB';
      return (bytes / 1048576).toFixed(1) + ' MB';
    }

    function renderFileList() {
      fileList.innerHTML = '';
      selectedFiles.forEach((f, idx) => {
        const item = document.createElement('div');
        item.className = 'file-item';
        const ext = f.name.split('.').pop().toLowerCase();
        const icon = ext === 'pdf' ? 'PDF' : ext === 'pptx' ? 'PPT' : 'DOC';
        item.innerHTML =
          '<span class="file-icon">' + icon + '</span>' +
          '<span class="file-name"></span>' +
          '<span class="file-size">' + formatSize(f.size) + '</span>' +
          '<button class="remove-btn" data-idx="' + idx + '" title="Remove">&#x2715;</button>';
        item.querySelector('.file-name').textContent = f.name;
        fileList.appendChild(item);
      });
      buildBtn.disabled = selectedFiles.length === 0;
    }

    fileList.addEventListener('click', e => {
      const btn = e.target.closest('.remove-btn');
      if (!btn) return;
      selectedFiles.splice(parseInt(btn.dataset.idx), 1);
      errorMsg.textContent = '';
      renderFileList();
    });

    async function startIndexing() {
      if (selectedFiles.length === 0) return;
      buildBtn.disabled = true;
      showStatus('running', 'Uploading files...');

      const formData = new FormData();
      for (const f of selectedFiles) formData.append('files', f);

      try {
        const res = await fetch('/index', { method: 'POST', body: formData });
        if (!res.ok) {
          const data = await res.json();
          showStatus('error', data.detail || 'Upload failed.');
          buildBtn.disabled = false;
          return;
        }
        showStatus('running', 'Indexing documents...');
        startPolling();
      } catch (e) {
        showStatus('error', 'Network error: ' + e.message);
        buildBtn.disabled = false;
      }
    }

    function startPolling() {
      if (pollTimer) clearInterval(pollTimer);
      pollTimer = setInterval(async () => {
        try {
          const data = await fetch('/status').then(r => r.json());
          if (data.state === 'running') {
            showStatus('running', data.message);
          } else if (data.state === 'done') {
            showStatus('done', data.message);
            clearInterval(pollTimer);
            buildBtn.disabled = selectedFiles.length === 0;
          } else if (data.state === 'error') {
            showStatus('error', data.message);
            clearInterval(pollTimer);
            buildBtn.disabled = selectedFiles.length === 0;
          }
        } catch (_) {}
      }, 1500);
    }

    function showStatus(state, message) {
      statusBox.className = 'status ' + state;
      statusBox.innerHTML = state === 'running'
        ? `<span class="spinner"></span>${message}`
        : message;
    }

    fetch('/status').then(r => r.json()).then(data => {
      if (data.state !== 'idle') showStatus(data.state, data.message);
    });
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


@app.post("/index")
async def build_index(files: List[UploadFile] = File(...)):
    if _status["state"] == "running":
        raise HTTPException(status_code=409, detail="Indexing already in progress.")
    if not 1 <= len(files) <= 10:
        raise HTTPException(status_code=400, detail="Upload between 1 and 10 files.")

    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in (".pdf", ".pptx", ".doc", ".docx"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: '{f.filename}'. Only PDF, PPTX, and Word (.doc/.docx) are accepted.",
            )

    # Save uploaded files to uploads/
    shutil.rmtree(UPLOADS_DIR, ignore_errors=True)
    UPLOADS_DIR.mkdir()
    saved_paths = []
    for f in files:
        dest = UPLOADS_DIR / f.filename
        with open(dest, "wb") as out:
            out.write(await f.read())
        saved_paths.append(str(dest))

    def run():
        _status["state"] = "running"
        _status["message"] = f"Indexing {len(saved_paths)} document(s)..."
        try:
            from build_db import build_database
            n = build_database(saved_paths)
            names = ", ".join(Path(p).name for p in saved_paths)
            _status["state"] = "done"
            _status["message"] = f"Done — {n} chunks indexed from: {names}"
        except Exception as e:
            _status["state"] = "error"
            _status["message"] = f"Error: {e}"

    threading.Thread(target=run, daemon=True).start()
    return {"message": f"Indexing started for {len(saved_paths)} file(s)."}


@app.get("/status")
async def get_status():
    return _status


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

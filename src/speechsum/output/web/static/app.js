document.addEventListener('DOMContentLoaded', () => {
  const uploadZone = document.getElementById('uploadZone');
  const fileInput = document.getElementById('fileInput');
  const selectBtn = document.getElementById('selectBtn');
  const processBtn = document.getElementById('processBtn');
  const fileInfo = document.getElementById('fileInfo');
  const fileName = document.getElementById('fileName');
  const fileSize = document.getElementById('fileSize');
  const summarizeToggle = document.getElementById('summarizeToggle');
  const resultsSection = document.getElementById('resultsSection');
  const transcriptionText = document.getElementById('transcriptionText');
  const transcriptionSpinner = document.getElementById('transcriptionSpinner');
  const summaryText = document.getElementById('summaryText');
  const summarySpinner = document.getElementById('summarySpinner');
  const summaryOuter = document.getElementById('summaryOuter');
  const resultMeta = document.getElementById('resultMeta');
  const errorSection = document.getElementById('errorSection');
  const errorText = document.getElementById('errorText');
  const toast = document.getElementById('toast');

  let selectedFile = null;

  function showToast(msg, isError = false) {
    toast.textContent = msg;
    toast.classList.remove('hidden', 'error');
    if (isError) toast.classList.add('error');
    setTimeout(() => toast.classList.add('hidden'), 3000);
  }

  function hide(el) { el.classList.add('hidden'); }
  function show(el) { el.classList.remove('hidden'); }

  selectBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  uploadZone.addEventListener('click', () => fileInput.click());

  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
  });

  uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
  });

  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) {
      handleFile(fileInput.files[0]);
    }
  });

  function handleFile(file) {
    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = (file.size / (1024 * 1024)).toFixed(1) + ' MB';
    show(fileInfo);
    show(processBtn);
    hide(resultsSection);
    hide(errorSection);
  }

  processBtn.addEventListener('click', processFile);

  async function processFile() {
    if (!selectedFile) return;

    hide(errorSection);
    show(resultsSection);
    show(transcriptionSpinner);
    hide(transcriptionText);
    hide(summaryOuter);
    transcriptionText.textContent = '';
    summaryText.textContent = '';
    resultMeta.textContent = '';

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('summarize', summarizeToggle.checked);

    try {
      const res = await fetch('/transcribe', { method: 'POST', body: formData });
      const data = await res.json();

      if (!res.ok || data.error) {
        throw new Error(data.error || 'Processing failed');
      }

      hide(transcriptionSpinner);
      transcriptionText.textContent = data.transcription;
      show(transcriptionText);
      resultMeta.textContent = `${data.duration_seconds.toFixed(1)}s · ${data.source.split('/').pop() || data.source}`;

      if (data.summary) {
        hide(summarySpinner);
        summaryText.textContent = data.summary;
        show(summaryOuter);
      } else {
        hide(summaryOuter);
      }

      showToast('Processing complete');
    } catch (err) {
      hide(transcriptionSpinner);
      hide(resultsSection);
      errorText.textContent = err.message;
      show(errorSection);
      showToast(err.message, true);
    }
  }
});

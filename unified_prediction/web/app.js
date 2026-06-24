const $ = (s) => document.querySelector(s);
const form = $('#settings');
let currentState = 'idle';
let streamAttached = false;
let streamGeneration = 0;

function updateConditionalFields() {
  const model = new FormData(form).get('model');
  document.querySelectorAll('.bus-only').forEach(el => el.style.display = model === 'bus' ? '' : 'none');
  document.querySelectorAll('.intersection-only').forEach(el => el.style.display = model === 'intersection' ? 'grid' : 'none');
  $('#webcam-field').style.display = $('#camera').value === 'webcam' ? 'grid' : 'none';
  const fd = new FormData(form);
  $('#source-label').textContent = `${fd.get('camera').toUpperCase()} / ${fd.get('width')} x ${fd.get('height')}`;
}

function payload() {
  const fd = new FormData(form);
  return {
    model: fd.get('model'), pose: fd.get('pose'), camera: fd.get('camera'),
    webcam_index: Number(fd.get('webcam_index')), width: Number(fd.get('width')),
    height: Number(fd.get('height')), fps: Number(fd.get('fps')),
    intersection_fold: Number(fd.get('intersection_fold')), device: fd.get('device'),
    bus_swap_upper_labels: fd.get('bus_swap_upper_labels') === 'on'
  };
}

async function post(path, data = {}) {
  const response = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Request failed');
  return result;
}

$('#start').addEventListener('click', async () => {
  $('#message').textContent = '';
  try { await post('/api/start', payload()); await refresh(); }
  catch (error) { $('#message').textContent = error.message; }
});
$('#stop').addEventListener('click', async () => {
  try { await post('/api/stop'); await refresh(); }
  catch (error) { $('#message').textContent = error.message; }
});
form.addEventListener('change', updateConditionalFields);

function setState(state, error) {
  currentState = state;
  const active = ['starting','running','stopping'].includes(state);
  const labels = {idle:'System idle',starting:'Loading models',running:'Prediction active',stopping:'Stopping pipeline',error:'Pipeline error'};
  const pill = $('#status-pill');
  pill.className = `status-pill ${state}`;
  pill.querySelector('span').textContent = labels[state] || state;
  $('#start').disabled = active;
  $('#stop').disabled = !['starting','running'].includes(state);
  [...form.elements].forEach(el => el.disabled = active);
  if (error) $('#message').textContent = error;
  if (['starting','running'].includes(state) && !streamAttached) {
    streamAttached = true;
    pollVideoFrames(++streamGeneration);
  }
  $('.live-dot').classList.toggle('active', state === 'running');
  if (state === 'idle' || state === 'error') {
    streamAttached = false;
    streamGeneration++;
    $('#viewport').classList.remove('streaming');
    $('#stream').removeAttribute('src');
  }
}

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function pollVideoFrames(generation) {
  while (streamAttached && generation === streamGeneration) {
    try {
      const response = await fetch(`/api/frame?t=${Date.now()}`, {cache:'no-store'});
      if (response.status === 204) {
        await delay(100);
        continue;
      }
      if (!response.ok) throw new Error(`Frame request failed: ${response.status}`);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const image = $('#stream');
      const previous = image.dataset.objectUrl;
      image.src = url;
      image.dataset.objectUrl = url;
      $('#viewport').classList.add('streaming');
      if (previous) URL.revokeObjectURL(previous);
      await delay(60);
    } catch (_) {
      await delay(150);
    }
  }
}

function setMetrics(data) {
  const m = data.metrics || {};
  if (m.frame_size && m.frame_size[0] && m.frame_size[1]) {
    $('#source-label').textContent = `${(data.config?.camera || 'camera').toUpperCase()} / ${m.frame_size[0]} x ${m.frame_size[1]}`;
  }
  $('#fps').textContent = `${Number(m.fps || 0).toFixed(1)} FPS`;
  $('#pose-state').textContent = m.pose_ok ? 'Cyclist locked' : (currentState === 'running' ? 'No cyclist detected' : 'Awaiting detection');
  $('#pose-indicator').classList.toggle('active', Boolean(m.pose_ok));
  const lines = m.predictions || [];
  $('#prediction').textContent = lines.length ? lines.slice(0, 3).join(' / ') : 'Prediction not started';
  const progress = m.progress || [0, 0];
  $('#progress').textContent = `${progress[0] || 0} / ${progress[1] || 0}`;
  $('#progress-bar').style.width = progress[1] ? `${Math.min(100, progress[0] / progress[1] * 100)}%` : '0%';
  if (data.started_at) $('#clock').textContent = new Date(data.started_at * 1000).toLocaleTimeString('en-GB', {hour12:false});
  else $('#clock').textContent = new Date().toLocaleTimeString('en-GB', {hour12:false});
}

async function refresh() {
  try {
    const response = await fetch('/api/status', {cache:'no-store'});
    const data = await response.json();
    setState(data.state, data.error);
    setMetrics(data);
  } catch (_) { setState('error', 'Unable to reach the local service'); }
}

updateConditionalFields();
refresh();
setInterval(refresh, 750);

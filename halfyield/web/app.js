'use strict';
const $ = id => document.getElementById(id);
let ending = false;
let calibrationLoaded = false;
let frameUrl;
const fields = [
  ['half_confidence', 'Half confidence', false],
  ['piece_confidence', 'Piece confidence', false],
  ['obscured_confidence', 'Obscured confidence', false],
  ['shell_confidence', 'Shell confidence', false],
  ['half_mass_per_area_g_per_model_pixel2', 'Half mass / area', true],
  ['piece_mass_per_area_g_per_model_pixel2', 'Piece mass / area', true],
];
for (const [name, title, mass] of fields) {
  const label = document.createElement('label');
  label.textContent = title;
  const output = document.createElement('output');
  const input = document.createElement('input');
  input.type = 'range'; input.id = name;
  input.min = mass ? -6 : 0; input.max = mass ? 0 : 1;
  input.step = mass ? .01 : .01; input.value = mass ? -3 : .3;
  input.addEventListener('input', () => {
    output.textContent = mass ? `${(10 ** Number(input.value)).toPrecision(4)} g/model px²` : `${Math.round(input.value * 100)}%`;
  });
  label.append(output, input); $('sliders').append(label);
  input.dispatchEvent(new Event('input'));
}
async function post(route, data) {
  const response = await fetch(route, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'The request failed.');
  return result;
}
function errorMessage(error) { $('message').textContent = error.message; }
$('start-form').addEventListener('submit', async event => {
  event.preventDefault(); $('start').disabled = true; $('message').textContent = '';
  try {
    let source = $('source').value.trim();
    if (!source && $('video').files.length) {
      $('status').textContent = 'Copying video to the runner…';
      const file = $('video').files[0];
      const response = await fetch('upload', {method: 'POST', headers: {'X-Filename': encodeURIComponent(file.name)}, body: file});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error);
      source = result.source;
    }
    if (!source) throw new Error('Choose a video file or enter its local path.');
    await post('start', {source, report: $('report').value, window: Number($('window').value)});
    $('status').textContent = 'Loading model and video…';
  } catch (error) { errorMessage(error); $('start').disabled = false; $('status').textContent = 'Choose a video'; }
});
async function action(value) {
  if (ending) return;
  try {
    await post('action', {action: value});
    if (value === 'quit') { ending = true; $('status').textContent = 'Saving report…'; }
  } catch (error) { errorMessage(error); }
}
document.querySelectorAll('[data-action]').forEach(button => button.addEventListener('click', () => action(button.dataset.action)));
$('calibration-form').addEventListener('submit', async event => {
  event.preventDefault();
  const values = {};
  fields.forEach(([name, , mass]) => { values[name] = mass ? 10 ** Number($(name).value) : Number($(name).value); });
  try { await post('calibration', values); } catch (error) { errorMessage(error); }
});
document.addEventListener('keydown', event => {
  if (['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON', 'SUMMARY'].includes(event.target.tagName) || $('review').hidden) return;
  const map = {' ': 'toggle_pause', ArrowLeft: 'left', ArrowRight: 'right', ',': 'prev_frame', '.': 'next_frame', s: 'toggle_slow', q: 'quit', Escape: 'quit'};
  const value = map[event.key] || map[event.key.toLowerCase()];
  if (value) { event.preventDefault(); action(value); }
});
function percent(value) { return value == null ? '—' : `${(value * 100).toFixed(1)}%`; }
function clock(value) { const seconds = Math.floor(value || 0); return `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`; }
function trend(series) {
  const valid = series.filter(([time, ratio]) => ratio != null && Number.isFinite(ratio));
  if (valid.length < 2) { $('trend').setAttribute('d', ''); return; }
  const start = valid[0][0], span = valid.at(-1)[0] - start;
  if (span <= 0) return;
  let path = '', gap = true;
  for (const [time, ratio] of series) {
    if (ratio == null || !Number.isFinite(ratio)) { gap = true; continue; }
    path += `${gap ? 'M' : 'L'}${((time - start) / span * 350).toFixed(2)} ${(140 - Math.max(0, Math.min(1, ratio)) * 130).toFixed(2)} `;
    gap = false;
  }
  $('trend').setAttribute('d', path);
}
async function poll() {
  try {
    const response = await fetch('state');
    if (!response.ok) throw new Error('The runner is unavailable.');
    const state = await response.json();
    if (state.status === 'error') { $('status').textContent = 'Review failed'; $('message').textContent = state.error; ending = true; return; }
    if (state.status === 'finished') {
      $('status').textContent = 'Review finished';
      $('notice').textContent = state.report ? `Report saved: ${state.report}` : 'Review finished. Check the runner output for the report location or insufficient samples.';
      ending = true; document.querySelectorAll('button').forEach(button => { button.disabled = true; }); return;
    }
    if (state.status === 'loading') { $('status').textContent = 'Loading model and video…'; $('setup').hidden = true; }
    if (state.status === 'reviewing') {
      $('setup').hidden = true; $('review').hidden = false;
      if (!ending) $('status').textContent = state.complete ? 'Review complete' : (state.paused ? 'Paused' : 'Analyzing');
      $('filename').textContent = state.source;
      $('half').textContent = percent(state.ratio); $('piece').textContent = percent(state.ratio == null ? null : 1 - state.ratio);
      $('mass').textContent = percent(state.mass_share);
      $('window-label').textContent = `Rolling ${state.window_seconds}-second window`;
      $('time').textContent = `${clock(state.position)} / ${clock(state.duration)}`;
      $('progress').value = state.duration > 0 ? state.position / state.duration : 0;
      $('pause').textContent = state.paused ? 'Resume' : 'Pause'; $('pause').disabled = state.complete;
      $('slow').textContent = `Slow: ${state.slow ? 'on' : 'off'}`;
      document.querySelector('[data-action="seek_back"]').textContent = `Back ${state.skip_seconds}s`;
      document.querySelector('[data-action="seek_forward"]').textContent = `Ahead ${state.skip_seconds}s`;
      $('notice').textContent = state.notice;
      trend(state.series);
      if (!calibrationLoaded) {
        fields.forEach(([name, , mass]) => { $(name).value = mass ? Math.log10(state.calibration[name]) : state.calibration[name]; $(name).dispatchEvent(new Event('input')); });
        calibrationLoaded = true;
      }
      const image = await fetch('frame');
      if (image.ok) {
        const blob = await image.blob();
        if (blob.size) { const old = frameUrl; frameUrl = URL.createObjectURL(blob); $('frame').src = frameUrl; if (old) URL.revokeObjectURL(old); }
      }
    }
  } catch (error) {
    if (ending) return;
    $('message').textContent = 'Connection to the runner was lost. Check its terminal output.';
  }
  setTimeout(poll, 150);
}
$('spacing').addEventListener('change', event => document.body.classList.toggle('compact', event.target.value === 'compact'));
$('chart-view').addEventListener('change', event => { $('chart').hidden = event.target.value === 'hide'; });
window.addEventListener('pagehide', () => { if (!ending) navigator.sendBeacon('quit', '{}'); });
poll();

const $ = id => document.getElementById(id);
async function api(path, options = {}) {
  const headers = {...options.headers};
  if ($('token').value) headers.Authorization = `Bearer ${$('token').value}`;
  const response = await fetch(path, {...options, headers});
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
  return data;
}
function status(message) { $('status').textContent = message; }
async function refresh() {
  const docs = await api('/api/documents');
  const current = $('source').value;
  $('documents').replaceChildren();
  $('source').replaceChildren(new Option('All documents', ''));
  for (const doc of docs) {
    const li = document.createElement('li');
    li.textContent = `${doc.source} · ${doc.chunks} chunks`;
    $('documents').append(li);
    $('source').add(new Option(doc.source, doc.source));
  }
  if (docs.some(d => d.source === current)) $('source').value = current;
}
$('refresh').onclick = () => refresh().catch(e => status(e.message));
$('upload').onsubmit = async event => {
  event.preventDefault();
  const button = event.target.querySelector('button'); button.disabled = true;
  try {
    const file = $('file').files[0];
    if (file.size > 10 * 1024 * 1024) throw new Error('Maximum file size is 10 MiB');
    const body = new FormData(); body.append('file', file);
    status('Extracting, embedding, and indexing…');
    const result = await api('/api/documents', {method:'POST', body});
    status(`${result.source}: ${result.status}, ${result.chunks} chunks.`);
    await refresh();
  } catch(e) { status(e.message); } finally { button.disabled = false; }
};
$('ask').onsubmit = async event => {
  event.preventDefault();
  const button = event.target.querySelector('button'); button.disabled = true;
  $('answer').textContent = ''; $('sources').replaceChildren();
  try {
    status('Retrieving evidence and writing an answer…');
    const data = await api('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:$('question').value, top_k:Number($('topk').value), source:$('source').value || null})});
    $('answer').textContent = data.answer;
    status(`${data.latency_ms} ms · ${data.sources.length} passages. ${data.warnings.join(' ')}`);
    for (const source of data.sources) {
      const details = document.createElement('details'), summary = document.createElement('summary'), passage = document.createElement('pre');
      summary.textContent = `[${source.label}] ${source.source}${source.page ? ' · page ' + source.page : ''} · distance ${source.distance.toFixed(3)}`;
      passage.textContent = source.text; // Render as text, never executable document HTML.
      details.append(summary, passage); $('sources').append(details);
    }
  } catch(e) { status(e.message); } finally { button.disabled = false; }
};
refresh().catch(e => status(e.message));

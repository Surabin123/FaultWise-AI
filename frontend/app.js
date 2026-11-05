// @ts-check
import {ApiError, createClient} from './api.js';
import {gateTone, numberText, percentText, timestampText} from './format.js';

const api = createClient({base: window.FAULTWISE_CONFIG?.apiBase || ''});
const q = (id) => document.getElementById(id);
const state = {datasets:[], metadata:null, engineIds:[], variant:null, engineId:null, trajectory:null,
  model:null, provider:null, evaluations:[], reports:[], latestRun:null, selectedRun:null, selectedReport:null};
let trajectoryAbort = null;
const pages = new Set(['overview','explorer','prediction','reports','quality','history']);

function element(tag, className = '', value = '') {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (value !== '') item.textContent = String(value);
  return item;
}
function clear(target) { target.replaceChildren(); }
function text(id, value) { q(id).textContent = value == null ? 'Unavailable' : String(value); }
function status(id, value, tone = '') { const target = q(id); target.textContent = value; target.className = `inline-status ${tone}`.trim(); }
function errorText(error) { return error instanceof ApiError ? `${error.code}: ${error.message}` : 'The request could not be completed.'; }
function valueText(value) { return value == null ? 'Unavailable' : typeof value === 'object' ? JSON.stringify(value) : String(value); }
function paragraph(value) { return element('p', '', value); }
function addDetail(parent, label, value) {
  const row = element('div'); row.append(element('dt','',label), element('dd','',value)); parent.append(row);
}
function addParagraphs(target, values) { clear(target); for (const value of values) target.append(paragraph(value)); }
function createList(items) { const list = element('ul'); for (const item of items) list.append(element('li','',item)); return list; }
function setBadge(target, label, tone) { target.className = `badge ${tone}`; target.textContent = label; }
function selectOptions(select, values, label) {
  clear(select);
  if (!values.length) { select.append(new Option(label, '')); select.disabled = true; return; }
  select.disabled = false;
  for (const entry of values) select.append(new Option(String(entry.label ?? entry), String(entry.value ?? entry)));
}
function navigate(page, updateHash = true) {
  if (!pages.has(page)) page = 'overview';
  for (const name of pages) {
    q(`page-${name}`).hidden = name !== page;
    const button = document.querySelector(`[data-page="${name}"]`);
    button.classList.toggle('active', name === page);
    if (name === page) button.setAttribute('aria-current','page'); else button.removeAttribute('aria-current');
  }
  text('crumb-page', page.replace('-', ' ').toUpperCase());
  if (updateHash) history.replaceState(null, '', `#${page}`);
  window.scrollTo({top:0,behavior:'instant'});
}

async function refresh() {
  text('overview-alert','Refreshing backend state…');
  status('quality-status','Loading evaluations…');
  const keys = ['health','ready','datasets','model','metrics','provider','reports','evaluations'];
  const tasks = [api.health(),api.readiness(),api.datasets(),api.model(),api.modelEvaluation(),api.provider(),api.reports(),api.evaluations()];
  const results = await Promise.allSettled(tasks);
  const got = (name) => { const result = results[keys.indexOf(name)]; return result.status === 'fulfilled' ? result.value : null; };
  const health = got('health');
  q('connection').className = `connection ${health ? 'ok' : 'error'}`;
  q('connection').lastChild.textContent = health ? ' API online' : ' API offline';
  state.model = got('model'); state.provider = got('provider');
  state.datasets = got('datasets')?.datasets || [];
  state.reports = got('reports')?.reports || [];
  state.evaluations = got('evaluations')?.evaluations || [];
  const available = state.datasets.filter((item) => item.status === 'available');
  const current = available.find((item) => item.variant === state.variant) || available[0] || null;
  state.variant = current?.variant || null;
  text('stat-dataset', current?.variant || 'Unavailable');
  text('stat-dataset-note', current ? `${current.status} · ${current.row_count ?? 'unknown'} rows` : 'No available dataset');
  text('stat-engines', current?.engine_count ?? '—');
  text('stat-model', state.model?.status ? state.model.status[0].toUpperCase() + state.model.status.slice(1) : 'Unavailable');
  text('stat-model-note', state.model?.model_id || 'No loaded model');
  const ready = got('ready')?.status === 'ready';
  setBadge(q('ready-badge'), ready ? 'READY' : 'NOT READY', ready ? 'pass' : 'inconclusive');
  const details = q('overview-details'); clear(details);
  addDetail(details,'API',health ? 'Online' : 'Offline');
  addDetail(details,'Dataset',current ? `${current.variant} · ${current.engine_count ?? 'unknown'} engines` : 'Unavailable');
  addDetail(details,'Model',state.model?.status || 'Unavailable');
  addDetail(details,'Provider',state.provider?.configured ? `${state.provider.provider} · ${state.provider.model}` : 'Not configured');
  const metrics = got('metrics');
  renderModelMetrics(metrics);
  const reportInfo = q('overview-reports'); clear(reportInfo);
  reportInfo.append(paragraph(state.provider?.configured ? `Provider: ${state.provider.provider} (${state.provider.model})` : 'LLM provider not configured. Sufficient requests return provider_unavailable; insufficient evidence can still produce an abstained report.'));
  reportInfo.append(paragraph(`Saved reports: ${got('reports') ? state.reports.length : 'Unavailable'}.`));
  const qualityInfo = q('overview-quality'); clear(qualityInfo);
  renderReportList(); renderHistoryList();
  const last = state.evaluations[0];
  text('stat-gate', last?.overall_status?.toUpperCase() || 'Unavailable');
  q('stat-gate').className = `status-${gateTone(last?.overall_status)}`;
  text('stat-gate-note', last ? timestampText(last.timestamp_utc) : 'No saved evaluation');
  if (last) {
    qualityInfo.append(paragraph(`${last.passed_count}/${last.case_count} synthetic golden cases passed; ${last.failed_count} failed.`));
    qualityInfo.append(paragraph(`Gate: ${last.overall_status || 'unavailable'}. Live provider: ${last.live_provider_status || 'unavailable'}.`));
    await loadEvaluation(last.run_id, true);
  } else {
    qualityInfo.append(paragraph('No saved evaluation is available. Run the local evaluation CLI to create one.'));
    renderQuality(null);
  }
  q('overview-alert').textContent = !health ? 'API unavailable. Check that the FastAPI server is running.' : !ready ? 'The API is online, but the configured dataset or model is not ready.' : !state.provider?.configured ? 'Dataset and model are ready. Diagnostic generation needs a configured provider when evidence is sufficient.' : 'Dataset and model are ready. Report output remains subject to evidence and grounding checks.';
  const variants = state.datasets.map((item) => ({label:`${item.variant} · ${item.status}`, value:item.variant}));
  selectOptions(q('dataset-select'), variants, 'No datasets');
  if (state.variant) { q('dataset-select').value = state.variant; await loadVariant(state.variant); }
  else { status('explorer-status','No dataset available.'); resetEngine(); }
}

function renderModelMetrics(metrics) {
  const target = q('model-metrics'); clear(target);
  if (!metrics || !metrics.integrity_verified) { target.append(paragraph('Verified saved ML evaluation unavailable.')); return; }
  for (const [label,value] of [['MAE',metrics.mae],['RMSE',metrics.rmse]]) {
    const box = element('div'); box.append(element('span','',label),element('strong','',numberText(value,2)),element('span','','cycles')); target.append(box);
  }
  const note = element('p','',`${metrics.dataset_kind} · ${metrics.held_out_engines} held-out engines · ${metrics.n_observations} cycle rows`);
  note.style.width = '100%'; target.append(note);
}

async function loadVariant(variant) {
  state.variant = variant; status('explorer-status',`Loading ${variant} metadata and engines…`);
  try {
    const [metadata, engines] = await Promise.all([api.metadata(variant), api.engines(variant)]);
    if (state.variant !== variant) return;
    state.metadata = metadata; state.engineIds = engines.engine_ids;
    selectOptions(q('sensor-select'), metadata.sensors, 'No sensors');
    selectOptions(q('engine-select'), state.engineIds, 'No engines');
    const meta = q('dataset-metadata'); clear(meta);
    meta.append(paragraph(`${metadata.variant}: ${metadata.engine_count} engines, ${metadata.row_count} observations.`));
    meta.append(paragraph(`Cycle range: ${metadata.cycle_range?.join('–') || 'Unavailable'}. ${metadata.label_source}`));
    meta.append(paragraph(`Sensors: ${metadata.sensors.length}; operating settings: ${metadata.operating_settings.join(', ')}.`));
    if (!state.engineIds.length) { status('explorer-status','This dataset has no engines.'); resetEngine(); return; }
    const next = state.engineIds.includes(state.engineId) ? state.engineId : state.engineIds[0];
    q('engine-select').value = String(next);
    await selectEngine(next);
  } catch (error) { state.metadata = null; state.engineIds = []; resetEngine(); status('explorer-status',errorText(error),'error'); }
}
function resetEngine() {
  state.engineId = null; state.trajectory = null;
  text('analysis-context','Select an engine in the explorer first.'); text('report-context','Select an engine to request a report.');
  q('predict-button').disabled = true; q('anomaly-button').disabled = true; q('generate-report').disabled = true;
  text('trajectory-chart','No trajectory loaded.'); text('cycle-details','No cycle selected.');
}
async function selectEngine(engineId) {
  state.engineId = engineId; state.trajectory = null;
  const modelMatches = state.model?.status === 'loaded' && state.model?.variant === state.variant;
  text('analysis-context',`${state.variant} · engine ${engineId}`);
  text('report-context',modelMatches ? `${state.variant} · engine ${engineId}` : `No loaded model for ${state.variant}; reports require a matching model.`);
  q('predict-button').disabled = !modelMatches;
  q('anomaly-button').disabled = false; q('generate-report').disabled = !modelMatches;
  status('prediction-status',modelMatches ? 'No prediction requested.' : `No loaded model for ${state.variant}.`); clear(q('prediction-result'));
  status('anomaly-status','No analysis requested.'); clear(q('anomaly-result'));
  if (trajectoryAbort) trajectoryAbort.abort();
  trajectoryAbort = new AbortController(); const signal = trajectoryAbort.signal;
  status('explorer-status',`Loading ${state.variant} engine ${engineId} trajectory…`);
  try {
    const trajectory = await api.trajectory(state.variant, engineId, signal);
    if (signal.aborted || state.engineId !== engineId) return;
    state.trajectory = trajectory;
    status('explorer-status',`${trajectory.total_cycles} observed cycles loaded.`,'success');
    renderTrajectory();
  } catch (error) { if (error?.name === 'AbortError') return; status('explorer-status',errorText(error),'error'); text('trajectory-chart','Trajectory unavailable.'); }
}
function renderTrajectory() {
  const trajectory = state.trajectory; if (!trajectory) return;
  const sensor = q('sensor-select').value;
  text('chart-title',`${sensor} · engine ${trajectory.engine_id}`);
  text('chart-range',`${trajectory.total_cycles} cycles`);
  const chart = q('trajectory-chart'); clear(chart);
  const rows = trajectory.cycles;
  const valid = rows.filter((row) => typeof row.sensors[sensor] === 'number' && Number.isFinite(row.sensors[sensor]));
  if (!valid.length) { chart.textContent = 'No measurements for this sensor.'; return; }
  const minCycle = rows[0].cycle, maxCycle = rows[rows.length-1].cycle;
  let low = Math.min(...valid.map((row) => row.sensors[sensor])), high = Math.max(...valid.map((row) => row.sensors[sensor]));
  if (low === high) { low -= 1; high += 1; }
  const svgNS = 'http://www.w3.org/2000/svg'; const svg = document.createElementNS(svgNS,'svg');
  svg.setAttribute('viewBox','0 0 720 270'); svg.setAttribute('role','img'); svg.setAttribute('aria-label',`${sensor} values from cycle ${minCycle} to ${maxCycle}`);
  const xpos = (cycle) => 50 + (cycle-minCycle)/(maxCycle-minCycle || 1)*635;
  const ypos = (value) => 230 - (value-low)/(high-low)*185;
  for (let index=0; index<4; index++) {
    const y = 45 + index*61.7;
    const line = document.createElementNS(svgNS,'line');
    for (const [key,val] of Object.entries({x1:'50',x2:'690',y1:String(y),y2:String(y),stroke:'#dce9e3','stroke-width':'1'})) line.setAttribute(key,val);
    svg.append(line);
    const label = document.createElementNS(svgNS,'text'); label.setAttribute('x','5'); label.setAttribute('y',String(y+4)); label.setAttribute('fill','#6c847d'); label.setAttribute('font-size','10'); label.textContent = numberText(high-(high-low)*index/3,1); svg.append(label);
  }
  let path = '', drawing = false;
  for (const row of rows) {
    const value = row.sensors[sensor];
    if (typeof value !== 'number' || !Number.isFinite(value)) { drawing = false; continue; }
    path += `${drawing ? 'L' : 'M'}${xpos(row.cycle).toFixed(1)},${ypos(value).toFixed(1)} `; drawing = true;
  }
  const line = document.createElementNS(svgNS,'path'); line.setAttribute('d',path); line.setAttribute('fill','none'); line.setAttribute('stroke','#13816b'); line.setAttribute('stroke-width','2.5'); line.setAttribute('stroke-linejoin','round'); svg.append(line);
  for (const [label,x] of [[String(minCycle),50],[String(maxCycle),680]]) { const tick = document.createElementNS(svgNS,'text'); tick.setAttribute('x',String(x)); tick.setAttribute('y','255'); tick.setAttribute('fill','#6c847d'); tick.setAttribute('font-size','11'); tick.textContent = `cycle ${label}`; svg.append(tick); }
  chart.append(svg);
  selectOptions(q('cycle-select'), rows.map((row) => row.cycle), 'No cycles');
  q('cycle-select').value = String(maxCycle); renderCycle();
}
function renderCycle() {
  const cycle = Number(q('cycle-select').value); const row = state.trajectory?.cycles.find((item) => item.cycle === cycle);
  const target = q('cycle-details'); clear(target);
  if (!row) { target.append(paragraph('No selected observation.')); return; }
  target.append(paragraph(`Cycle ${cycle} · measured values; missing fields are shown as unavailable.`));
  const grid = element('div','readings-grid');
  for (const [name,value] of Object.entries({...row.operating_settings,...row.sensors})) { const item = element('div','reading'); item.append(element('span','',name),element('strong','',numberText(value,3))); grid.append(item); }
  target.append(grid);
}

async function runPrediction() {
  if (!state.variant || !state.engineId) return;
  status('prediction-status','Running the saved RUL model…'); q('predict-button').disabled = true;
  try { renderPrediction(await api.prediction(state.variant,state.engineId)); status('prediction-status','Prediction returned from the saved model.','success'); }
  catch (error) { status('prediction-status',errorText(error),'error'); clear(q('prediction-result')); }
  finally { q('predict-button').disabled = state.model?.status !== 'loaded' || state.model?.variant !== state.variant; }
}
function renderPrediction(result) {
  const target = q('prediction-result'); clear(target);
  const value = element('div','estimate'); value.append(element('strong','',numberText(result.predicted_rul,1)),element('span','',result.unit)); target.append(value);
  target.append(paragraph('Estimated remaining useful life. This is not a confirmed failure time or a calibrated confidence interval.'));
  const info = element('dl','detail-list');
  addDetail(info,'Engine',`${result.dataset_variant} / ${result.engine_id}`); addDetail(info,'Sensor cycle',result.evidence.cycle); addDetail(info,'Model',result.model_id); addDetail(info,'Generated',timestampText(result.generated_at_utc)); target.append(info);
  const readings = element('details'); readings.append(element('summary','','Observed sensor evidence'));
  const grid = element('div','readings-grid'); for (const [name,value] of Object.entries(result.evidence.readings)) { const item = element('div','reading'); item.append(element('span','',name),element('strong','',numberText(value,3))); grid.append(item); } readings.append(grid); target.append(readings);
  target.append(element('h3','section-title','Limitations'),createList(result.limitations));
}
async function runAnomaly() {
  if (!state.variant || !state.engineId) return;
  status('anomaly-status','Analyzing observed trajectory…'); q('anomaly-button').disabled = true;
  try { renderAnomaly(await api.anomaly(state.variant,state.engineId)); status('anomaly-status','Analysis complete.','success'); }
  catch (error) { status('anomaly-status',errorText(error),'error'); clear(q('anomaly-result')); }
  finally { q('anomaly-button').disabled = !state.engineId; }
}
function renderAnomaly(result) {
  const target = q('anomaly-result'); clear(target);
  target.append(paragraph(`${result.status === 'analyzed' ? 'Analyzed' : 'Insufficient evidence'} · ${result.method} ${result.method_version}.`));
  target.append(paragraph(result.score_definition));
  if (!result.unusual_sensor_behavior.length) target.append(paragraph(result.status === 'insufficient_evidence' ? 'Not enough trajectory history to assess deviation.' : 'No sensor channels crossed the method threshold.'));
  for (const item of result.unusual_sensor_behavior) {
    const box = element('div','evidence-item');
    box.append(element('strong','',`${item.sensor} · score ${numberText(item.robust_z_score,2)}`),element('span','',item.explanation));
    box.append(paragraph(`Earlier median ${numberText(item.baseline_median,3)} · recent median ${numberText(item.recent_median,3)}`)); target.append(box);
  }
  target.append(element('h3','section-title','Limitations'),createList(result.limitations));
}

async function generateReport() {
  if (!state.variant || !state.engineId) return;
  status('report-status','Collecting evidence and requesting a report…'); q('generate-report').disabled = true;
  try { const report = await api.generateReport(state.variant,state.engineId); renderReport(report); status('report-status',report.assessment === 'insufficient_evidence' ? 'The service abstained: evidence is insufficient.' : 'Validated report returned. Hypotheses require engineering review.','success'); await reloadReports(); }
  catch (error) { status('report-status',errorText(error),'error'); }
  finally { q('generate-report').disabled = !state.engineId || state.model?.status !== 'loaded' || state.model?.variant !== state.variant; }
}
async function reloadReports() { try { state.reports = (await api.reports()).reports; renderReportList(); } catch { q('report-list').textContent = 'Saved report history unavailable.'; } }
function renderReportList() {
  const target = q('report-list'); clear(target);
  if (!state.reports.length) { target.className = 'list-body empty'; target.textContent = 'No saved reports. A provider may be unavailable; the UI does not supply examples as real reports.'; return; }
  target.className = 'list-body';
  for (const item of state.reports) { const button = element('button','list-button'); button.type = 'button'; button.append(element('strong','',`${item.dataset_variant} · engine ${item.engine_id} · ${item.assessment}`),element('small','',`${timestampText(item.generated_at_utc)} · ${item.provider || 'abstained'}`)); button.addEventListener('click',async () => { try { renderReport(await api.report(item.report_id)); status('report-status','Loaded saved report.','success'); } catch(error) { status('report-status',errorText(error),'error'); } }); target.append(button); }
}
function renderReport(report) {
  state.selectedReport = report;
  text('report-detail-title',`${report.dataset_variant} · engine ${report.engine_id}`);
  const target = q('report-detail'); clear(target); target.className = '';
  const badge = element('span',`badge ${report.assessment === 'insufficient_evidence' ? 'inconclusive' : 'neutral'}`,report.assessment.replaceAll('_',' ').toUpperCase()); target.append(badge);
  target.append(paragraph(report.executive_summary));
  const info = element('dl','detail-list'); addDetail(info,'Generated',timestampText(report.generated_at_utc)); addDetail(info,'Provider',report.provider || 'None · abstained'); addDetail(info,'Model',report.model_id || 'Unavailable'); addDetail(info,'Dataset hash',report.dataset_sha256 || 'Unavailable'); target.append(info);
  if (report.missing_evidence.length) { target.append(element('h3','section-title','Missing evidence / coverage')); target.append(createList(report.missing_evidence)); }
  target.append(element('h3','section-title','Potential concerns · hypotheses'));
  if (!report.potential_concerns.length) target.append(paragraph('No supported diagnostic concern was generated.'));
  for (const claim of report.potential_concerns) target.append(renderClaim(claim));
  target.append(element('h3','section-title','Recommended investigation'));
  if (!report.recommended_investigation.length) target.append(paragraph('No investigation was generated.'));
  for (const claim of report.recommended_investigation) target.append(renderClaim(claim));
  target.append(element('h3','section-title','Observed evidence'));
  for (const item of report.observed_evidence) target.append(renderEvidence(item));
  target.append(element('h3','section-title','ML predictions'));
  for (const item of report.ml_predictions) target.append(renderEvidence(item));
  target.append(element('h3','section-title','Uncertainty'),paragraph(report.uncertainty));
  target.append(element('h3','section-title','Limitations'),createList(report.limitations));
}
function renderClaim(claim) {
  const box = element('div','claim'); box.append(element('span','hypothesis',`${claim.kind.toUpperCase()} · ${claim.scope}`),paragraph(claim.text));
  const ids = state.selectedReport ? [...state.selectedReport.observed_evidence,...state.selectedReport.ml_predictions].map((item) => item.id) : [];
  for (const id of claim.evidence_ids) {
    const button = element('button','',id); button.type = 'button'; button.disabled = !ids.includes(id); button.title = button.disabled ? 'Evidence unavailable in this report' : 'Jump to supporting evidence';
    button.addEventListener('click',() => { const target = document.getElementById(`evidence-${id}`); if (!target) return; target.scrollIntoView({behavior:'smooth',block:'center'}); target.classList.add('evidence-target'); setTimeout(() => target.classList.remove('evidence-target'),1500); }); box.append(button);
  }
  return box;
}
function renderEvidence(item) { const box = element('div','evidence-item'); box.id = `evidence-${item.id}`; box.append(element('strong','',`${item.id} · ${item.type}`),element('span','',`${valueText(item.value)}${item.unit ? ` ${item.unit}` : ''} · cycle ${item.cycle ?? 'unavailable'}`),paragraph(item.interpretation)); return box; }

async function loadEvaluation(runId, latest = false) {
  try { const run = await api.evaluation(runId); if (latest) { state.latestRun = run; renderQuality(run); } else { state.selectedRun = run; renderHistoryDetail(run); } }
  catch (error) { if (latest) status('quality-status',errorText(error),'error'); else q('history-detail').textContent = errorText(error); }
}
function renderQuality(run) {
  if (!run) { status('quality-status','No saved evaluation available.'); text('gate-banner','Release gate unavailable.'); return; }
  status('quality-status',`${run.evaluation_dataset_version} · ${timestampText(run.timestamp_utc)} · ${run.evaluation_dataset_provenance}`,'success');
  const gate = run.release_gate || {status:'unavailable',missing:['Gate result unavailable']}; const tone = gateTone(gate.status);
  const banner = q('gate-banner'); banner.className = `gate-banner ${tone}`; clear(banner); banner.append(element('strong','',`RELEASE GATE · ${gate.status.toUpperCase()}`));
  banner.append(element('span','',gate.failures?.length ? gate.failures.join(' · ') : gate.missing?.length ? gate.missing.join(' · ') : 'All configured checks reported a result. This is not safety certification.'));
  text('quality-cases',`${run.cases.length-run.failed_cases.length}/${run.cases.length}`);
  text('quality-unsupported',percentText(run.metrics?.unsupported_claim_rate_rule)); text('quality-grounded',percentText(run.metrics?.grounded_claim_rate_structural)); text('quality-schema',percentText(run.metrics?.schema_validity));
  const abstention = q('abstention-metrics'); clear(abstention);
  const values = run.metrics?.abstention;
  if (values) { const details = element('dl','detail-list'); for (const [key,value] of Object.entries(values)) addDetail(details,key.replaceAll('_',' '),percentText(value)); abstention.append(details); } else abstention.append(paragraph('Abstention metrics unavailable.'));
  const limits = q('quality-limitations'); clear(limits);
  limits.append(paragraph(`Evaluation input: ${run.evaluation_dataset_provenance}. Live provider: ${run.live_provider?.status || 'unavailable'}.`));
  limits.append(paragraph(`Saved NASA ML report: ${run.ml_metrics?.integrity_verified ? 'source hash and engine split verified' : 'unavailable'}. These metrics are row-level holdout values, not an official NASA endpoint benchmark.`));
  limits.append(createList(run.limitations || []));
  const cases = q('quality-cases-list'); clear(cases);
  const selected = run.cases.filter((item) => !item.passed || item.category === 'safety');
  if (!selected.length) { cases.textContent = 'No failed or safety cases in this run.'; return; }
  for (const item of selected) cases.append(renderCase(item));
}
function renderCase(item) {
  const details = element('details',`case-row ${item.passed ? 'pass' : 'fail'}`);
  details.append(element('summary','',`${item.passed ? 'PASS' : 'FAIL'} · ${item.case_id} · ${item.severity}`));
  details.append(paragraph(`Expected: ${item.expected_behavior}`));
  details.append(paragraph(`Actual: ${item.actual_error || item.actual_assessment || 'Unavailable'}`));
  if (item.attack) details.append(paragraph(`Adversarial input: ${item.attack}`));
  if (item.failure_explanation) details.append(paragraph(item.failure_explanation));
  return details;
}
function renderHistoryList() {
  const target = q('history-list'); clear(target);
  if (!state.evaluations.length) { target.className = 'list-body empty'; target.textContent = 'No saved evaluation runs are available.'; selectOptions(q('compare-previous'),[],'No runs'); selectOptions(q('compare-current'),[],'No runs'); return; }
  target.className = 'list-body';
  for (const run of state.evaluations) {
    const button = element('button','list-button'); button.type = 'button'; button.append(element('strong','',`${run.overall_status?.toUpperCase() || 'UNAVAILABLE'} · ${run.passed_count}/${run.case_count} cases`),element('small','',`${timestampText(run.timestamp_utc)} · ${run.prompt_version || 'unknown prompt'} · ${run.run_id}`));
    button.addEventListener('click',() => loadEvaluation(run.run_id)); target.append(button);
  }
  const options = state.evaluations.map((run) => ({label:`${timestampText(run.timestamp_utc)} · ${run.run_id.slice(0,8)}`,value:run.run_id}));
  selectOptions(q('compare-previous'),options,'No runs'); selectOptions(q('compare-current'),options,'No runs');
  q('compare-current').value = state.evaluations[0].run_id;
  q('compare-previous').value = state.evaluations[1]?.run_id || state.evaluations[0].run_id;
}
function renderHistoryDetail(run) {
  text('history-detail-title',`${run.evaluation_dataset_version} · ${run.run_id.slice(0,8)}`);
  const target = q('history-detail'); clear(target); target.className = 'body-copy';
  target.append(paragraph(`${timestampText(run.timestamp_utc)} · gate ${run.overall_status} · prompt ${run.prompt_version} · model ${run.model_version}.`));
  target.append(paragraph(`${run.cases.length-run.failed_cases.length}/${run.cases.length} cases passed. Dataset provenance: ${run.evaluation_dataset_provenance}.`));
  for (const item of run.cases) target.append(renderCase(item));
}
async function compareRuns() {
  const previous = q('compare-previous').value, current = q('compare-current').value; const target = q('compare-result'); clear(target);
  if (!previous || !current || previous === current) { target.textContent = 'Choose two different runs.'; return; }
  target.textContent = 'Comparing compatible runs…';
  try {
    const result = await api.compare(previous,current); clear(target);
    target.append(paragraph(`Regression findings: ${result.findings.length}. Newly failing cases: ${result.new_failed_cases?.length || 0}. Resolved cases: ${result.resolved_cases?.length || 0}.`));
    if (result.new_failed_cases?.length) target.append(element('h3','section-title','Newly failing'),createList(result.new_failed_cases));
    if (result.resolved_cases?.length) target.append(element('h3','section-title','Resolved'),createList(result.resolved_cases));
    for (const finding of result.findings) target.append(paragraph(`${finding.metric}: ${valueText(finding.previous)} → ${valueText(finding.current)} (Δ ${valueText(finding.difference)}, threshold ${valueText(finding.threshold)}; ${finding.severity}).`));
    if (!result.findings.length) target.append(paragraph('No worsening metric or newly failed case met the configured regression rules.'));
  } catch (error) { target.textContent = errorText(error); }
}

document.querySelectorAll('[data-page]').forEach((button) => button.addEventListener('click',() => navigate(button.dataset.page)));
document.querySelectorAll('[data-go]').forEach((button) => button.addEventListener('click',() => navigate(button.dataset.go)));
window.addEventListener('hashchange',() => navigate(location.hash.slice(1),false));
q('refresh').addEventListener('click',refresh);
q('dataset-select').addEventListener('change',(event) => loadVariant(event.target.value));
q('engine-select').addEventListener('change',(event) => selectEngine(Number(event.target.value)));
q('engine-search').addEventListener('keydown',(event) => { if (event.key !== 'Enter') return; const id = Number(event.target.value); if (state.engineIds.includes(id)) { q('engine-select').value = String(id); selectEngine(id); } else status('explorer-status',`Engine ${event.target.value} is not in ${state.variant || 'the selected dataset'}.`,'error'); });
q('sensor-select').addEventListener('change',renderTrajectory); q('cycle-select').addEventListener('change',renderCycle);
q('predict-button').addEventListener('click',runPrediction); q('anomaly-button').addEventListener('click',runAnomaly); q('generate-report').addEventListener('click',generateReport); q('compare-button').addEventListener('click',compareRuns);
navigate(location.hash.slice(1),false); resetEngine(); refresh();

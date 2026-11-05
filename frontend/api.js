// @ts-check
/** Typed client for the current FastAPI contract. All requests use one configured base URL. */

/** @typedef {{variant:string,status:'available'|'unavailable'|'invalid',engine_count:number|null,row_count:number|null}} DatasetSummary */
/** @typedef {DatasetSummary & {sensors:string[],operating_settings:string[],cycle_range:number[]|null,label_source:string}} DatasetMetadata */
/** @typedef {{cycle:number,sensors:Record<string,number|null>,operating_settings:Record<string,number|null>}} EngineCycle */
/** @typedef {{variant:string,engine_id:number,total_cycles:number,offset:number,cycles:EngineCycle[]}} TrajectoryPage */
/** @typedef {{engine_id:number,dataset_variant:string,predicted_rul:number,unit:string,model_id:string,model_dataset_kind:string,generated_at_utc:string,evidence:{cycle:number,readings:Record<string,number|null>,missing_sensors:string[]},limitations:string[]}} Prediction */
/** @typedef {{engine_id:number,dataset_variant:string,status:string,method:string,method_version:string,score_definition:string,unusual_sensor_behavior:Array<{sensor:string,robust_z_score:number|null,baseline_median:number,recent_median:number,explanation:string}>,skipped_sensors:string[],limitations:string[]}} Anomaly */
/** @typedef {{id:string,type:string,source:string,value:unknown,unit:string|null,cycle:number|null,interpretation:string,limitations:string[]}} Evidence */
/** @typedef {{text:string,kind:'hypothesis',scope:string,evidence_ids:string[]}} Claim */
/** @typedef {{report_id:string,engine_id:number,dataset_variant:string,assessment:string,executive_summary:string,observed_evidence:Evidence[],ml_predictions:Evidence[],potential_concerns:Claim[],recommended_investigation:Claim[],uncertainty:string,limitations:string[],missing_evidence:string[],provider:string|null,generated_at_utc:string,model_id:string,dataset_sha256:string}} DiagnosticReport */

export class ApiError extends Error {
  /** @param {number} status @param {string} code @param {string} message */
  constructor(status, code, message) { super(message); this.name = 'ApiError'; this.status = status; this.code = code; }
}

/** @param {{base?:string,fetchImpl?:typeof fetch}} options */
export function createClient({base = '', fetchImpl = fetch} = {}) {
  const origin = base.replace(/\/$/, '');
  /** @template T @param {string} path @param {{method?:string,body?:unknown,signal?:AbortSignal}} [options] @returns {Promise<T>} */
  async function request(path, options = {}) {
    let response;
    try {
      response = await fetchImpl(`${origin}${path}`, {
        method: options.method || 'GET',
        headers: options.body === undefined ? {} : {'Content-Type': 'application/json'},
        body: options.body === undefined ? undefined : JSON.stringify(options.body),
        signal: options.signal,
      });
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') throw error;
      throw new ApiError(0, 'network_unavailable', 'Could not reach the FaultWise API.');
    }
    let payload;
    try { payload = await response.json(); }
    catch { throw new ApiError(response.status, 'invalid_response', 'The API returned an unreadable response.'); }
    if (!response.ok) {
      const code = payload?.error?.code || `http_${response.status}`;
      const message = payload?.error?.message || payload?.detail || 'The request failed.';
      throw new ApiError(response.status, code, String(message));
    }
    return /** @type {T} */ (payload);
  }
  const enginePath = (variant, id) => `/api/v1/datasets/${encodeURIComponent(variant)}/engines/${encodeURIComponent(id)}`;
  return {
    request,
    health: () => request('/health'),
    readiness: () => request('/ready'),
    datasets: () => request('/api/v1/datasets'),
    metadata: (variant) => /** @type {Promise<DatasetMetadata>} */ (request(`/api/v1/datasets/${encodeURIComponent(variant)}`)),
    engines: (variant) => /** @type {Promise<{engine_ids:number[],count:number}>} */ (request(`/api/v1/datasets/${encodeURIComponent(variant)}/engines`)),
    model: () => request('/api/v1/model'),
    modelEvaluation: () => request('/api/v1/model/evaluation'),
    provider: () => request('/api/v1/provider'),
    reports: () => request('/api/v1/reports'),
    report: (id) => /** @type {Promise<DiagnosticReport>} */ (request(`/api/v1/reports/${encodeURIComponent(id)}`)),
    evaluations: () => request('/api/v1/evaluations'),
    evaluation: (id) => request(`/api/v1/evaluations/${encodeURIComponent(id)}`),
    compare: (previous, current) => request(`/api/v1/evaluations/compare?previous=${encodeURIComponent(previous)}&current=${encodeURIComponent(current)}`),
    prediction: (variant, id) => /** @type {Promise<Prediction>} */ (request(`${enginePath(variant, id)}/prediction`, {method:'POST'})),
    anomaly: (variant, id) => /** @type {Promise<Anomaly>} */ (request(`${enginePath(variant, id)}/anomaly-analysis`, {method:'POST'})),
    generateReport: (variant, id) => /** @type {Promise<DiagnosticReport>} */ (request(`${enginePath(variant, id)}/diagnostic-report`, {method:'POST',body:{persist:true}})),
    /** @param {string} variant @param {number} id @param {AbortSignal} [signal] @returns {Promise<TrajectoryPage>} */
    async trajectory(variant, id, signal) {
      /** @type {EngineCycle[]} */ const cycles = [];
      let total = 0;
      do {
        const page = /** @type {TrajectoryPage} */ (await request(`${enginePath(variant, id)}/trajectory?offset=${cycles.length}&limit=1000`, {signal}));
        total = page.total_cycles;
        if (!page.cycles.length && cycles.length < total) throw new ApiError(502, 'incomplete_trajectory', 'The API stopped returning trajectory rows.');
        cycles.push(...page.cycles);
      } while (cycles.length < total);
      return {variant, engine_id:id, total_cycles:total, offset:0, cycles};
    },
  };
}

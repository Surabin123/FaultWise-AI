import {test, expect} from '@playwright/test';

// Every response below is an explicitly synthetic browser fixture, never NASA output.
const id = '11111111-1111-4111-8111-111111111111';
const oldId = '33333333-3333-4333-8333-333333333333';
const evaluation = {run_id:id,timestamp_utc:'2026-11-05T00:00:00Z',evaluation_dataset_version:'synthetic-browser-v1',evaluation_dataset_provenance:'synthetic_fixture',model_version:'fixture-model',prompt_version:'diagnostic-v3',overall_status:'inconclusive',cases:[{case_id:'fixture_safety',category:'safety',severity:'high',passed:false,expected_behavior:'Reject unsupported claim',actual_error:'report_validation_failed',failure_explanation:'Synthetic fixture failure.'}],failed_cases:['fixture_safety'],metrics:{unsupported_claim_rate_rule:null,grounded_claim_rate_structural:null,schema_validity:null,abstention:{precision:null,recall:null,false_abstention_rate:null,missed_abstention_rate:null}},release_gate:{status:'inconclusive',failures:[],missing:['live-provider evaluation unavailable']},ml_metrics:null,live_provider:{status:'not_run'},limitations:['Synthetic browser fixture only.']};
const diagnostic = {report_id:'22222222-2222-4222-8222-222222222222',engine_id:1,dataset_variant:'FD001',dataset_sha256:'synthetic-fixture-hash',model_id:'fixture-model',prompt_version:'diagnostic-v3',provider:null,generated_at_utc:'2026-11-05T00:00:00Z',assessment:'insufficient_evidence',executive_summary:'Evidence is insufficient for a diagnostic concern.',observed_evidence:[{id:'sensor:sensor_1:cycle:3',type:'observation',source:'fixture',value:12,unit:null,cycle:3,interpretation:'Observed synthetic sensor value.',limitations:[]}],ml_predictions:[{id:'model:rul',type:'prediction',source:'fixture-model',value:12.4,unit:'cycles',cycle:3,interpretation:'Estimate only.',limitations:[]}],potential_concerns:[],recommended_investigation:[],uncertainty:'A diagnostic concern cannot be supported.',limitations:['Synthetic fixture only.'],missing_evidence:['supported operating conditions']};
const json = (body, status=200) => ({status,contentType:'application/json',body:JSON.stringify(body)});

async function mockApi(page, {available=true, model=true, reportError=false, reportWithClaims=false, apiDown=false}={}) {
  await page.route('**/health', route => route.fulfill(apiDown ? json({error:{code:'service_unavailable',message:'Offline'}},503) : json({status:'ok',service:'faultwise-api'})));
  await page.route('**/ready', route => route.fulfill(json({status:available && model && !apiDown ? 'ready' : 'not_ready'},available && model && !apiDown ? 200 : 503)));
  await page.route('**/api/v1/**', route => {
    const url = new URL(route.request().url()); const path = url.pathname;
    if (apiDown) return route.fulfill(json({error:{code:'service_unavailable',message:'Offline'}},503));
    if (path === '/api/v1/datasets') return route.fulfill(json({datasets:[{variant:'FD001',status:available?'available':'unavailable',engine_count:available?2:null,row_count:available?6:null}]}));
    if (path === '/api/v1/datasets/FD001') return route.fulfill(json({variant:'FD001',status:'available',engine_count:2,row_count:6,sensors:['sensor_1'],operating_settings:['setting_1'],cycle_range:[1,3],label_source:'synthetic browser fixture'}));
    if (path === '/api/v1/datasets/FD001/engines') return route.fulfill(json({variant:'FD001',engine_ids:[1,2],count:2}));
    if (path.endsWith('/trajectory')) return route.fulfill(json({variant:'FD001',engine_id:Number(path.split('/')[6]),total_cycles:3,offset:0,cycles:[{cycle:1,sensors:{sensor_1:10},operating_settings:{setting_1:1}},{cycle:2,sensors:{sensor_1:11},operating_settings:{setting_1:1}},{cycle:3,sensors:{sensor_1:12},operating_settings:{setting_1:1}}]}));
    if (path.endsWith('/prediction')) return route.fulfill(json({engine_id:1,dataset_variant:'FD001',predicted_rul:12.4,unit:'cycles',model_id:'fixture-model',model_dataset_kind:'synthetic_fixture',generated_at_utc:'2026-11-05T00:00:00Z',evidence:{cycle:3,readings:{sensor_1:12},missing_sensors:[]},limitations:['Estimate only.']}));
    if (path.endsWith('/anomaly-analysis')) return route.fulfill(json({engine_id:1,dataset_variant:'FD001',status:'analyzed',method:'median-MAD',method_version:'1',score_definition:'Robust deviation, not fault probability.',unusual_sensor_behavior:[{sensor:'sensor_1',robust_z_score:4.2,baseline_median:10,recent_median:12,explanation:'Synthetic unusual behavior.'}],skipped_sensors:[],limitations:['No mechanical fault established.']}));
    if (path.endsWith('/diagnostic-report')) {
      const claim = {text:'The unusual sensor pattern may indicate degradation.',kind:'hypothesis',scope:'sensor_behavior',evidence_ids:['anomaly:sensor_1']};
      const claimed = {...diagnostic,assessment:'investigate',provider:'synthetic-fake',missing_evidence:[],observed_evidence:[...diagnostic.observed_evidence,{id:'anomaly:sensor_1',type:'anomaly',source:'fixture-method',value:{robust_z_score:4.2},unit:null,cycle:3,interpretation:'Synthetic sensor deviation, not a fault.',limitations:[]}],potential_concerns:[claim],recommended_investigation:[{...claim,text:'Inspect the unusual sensor pattern.'}]};
      return route.fulfill(reportError ? json({error:{code:'provider_unavailable',message:'Provider not configured.'}},503) : json(reportWithClaims ? claimed : diagnostic));
    }
    if (path === '/api/v1/model') return route.fulfill(json({status:model?'loaded':'unavailable',model_id:model?'fixture-model':null,variant:'FD001'}));
    if (path === '/api/v1/model/evaluation') return route.fulfill(json({integrity_verified:true,mae:1.2,rmse:1.8,dataset_kind:'synthetic_fixture',held_out_engines:1,n_observations:2}));
    if (path === '/api/v1/provider') return route.fulfill(json({configured:false,provider:null,model:null}));
    if (path === '/api/v1/reports') return route.fulfill(json({reports:[]}));
    if (path === '/api/v1/evaluations') return route.fulfill(json({evaluations:[{run_id:id,timestamp_utc:evaluation.timestamp_utc,evaluation_dataset_version:evaluation.evaluation_dataset_version,prompt_version:evaluation.prompt_version,overall_status:evaluation.overall_status,case_count:1,passed_count:0,failed_count:1,live_provider_status:'not_run'},{run_id:oldId,timestamp_utc:'2026-11-04T00:00:00Z',evaluation_dataset_version:evaluation.evaluation_dataset_version,prompt_version:'diagnostic-v2',overall_status:'fail',case_count:1,passed_count:0,failed_count:1,live_provider_status:'not_run'}]}));
    if (path === '/api/v1/evaluations/compare') return route.fulfill(json({previous_run_id:oldId,current_run_id:id,findings:[],new_failed_cases:['fixture_safety'],resolved_cases:['old_issue']}));
    if (path === `/api/v1/evaluations/${id}`) return route.fulfill(json(evaluation));
    if (path === `/api/v1/evaluations/${oldId}`) return route.fulfill(json({...evaluation,run_id:oldId,failed_cases:['old_issue'],cases:[{...evaluation.cases[0],case_id:'old_issue'}]}));
    return route.continue();
  });
}

test('main workflow renders actual API fields and explicit limitations', async ({page}) => {
  await mockApi(page);
  await page.goto('/');
  await expect(page.locator('#connection')).toContainText('API online');
  await page.getByRole('button',{name:'Engine explorer'}).click();
  await expect(page.locator('#engine-select')).toHaveValue('1');
  await expect(page.locator('#trajectory-chart svg')).toBeVisible();
  await expect(page.locator('#cycle-details')).toContainText('sensor_1');
  await page.locator('#engine-select').selectOption('2');
  await expect(page.locator('#chart-title')).toContainText('engine 2');
  await page.locator('#engine-search').fill('1');
  await page.locator('#engine-search').press('Enter');
  await expect(page.locator('#engine-select')).toHaveValue('1');
  await page.getByRole('button',{name:'Prediction & anomaly'}).click();
  await page.getByRole('button',{name:'Run RUL prediction'}).click();
  await expect(page.locator('#prediction-result')).toContainText('12.4');
  await expect(page.locator('#prediction-result')).toContainText('not a confirmed failure time');
  await page.getByRole('button',{name:'Analyze anomalies'}).click();
  await expect(page.locator('#anomaly-result')).toContainText('sensor_1');
  await page.locator('[data-page="reports"]').click();
  await page.getByRole('button',{name:'Generate report'}).click();
  await expect(page.locator('#report-detail')).toContainText('supported operating conditions');
  await expect(page.locator('#report-detail')).toContainText('sensor:sensor_1:cycle:3');
  await page.getByRole('button',{name:'AI quality'}).click();
  await expect(page.locator('#gate-banner')).toContainText('INCONCLUSIVE');
  await expect(page.locator('#quality-unsupported')).toHaveText('Unavailable');
  await expect(page.locator('#quality-cases-list')).toContainText('fixture_safety');
});

test('unavailable dataset and model remain explicit on mobile', async ({page}) => {
  await page.setViewportSize({width:390,height:844});
  await mockApi(page,{available:false,model:false,reportError:true});
  await page.goto('/');
  await expect(page.locator('#stat-dataset')).toHaveText('Unavailable');
  await expect(page.locator('#stat-model')).toHaveText('Unavailable');
  await expect(page.locator('#stat-gate')).toHaveText('INCONCLUSIVE');
  await page.getByRole('button',{name:'Engine explorer'}).click();
  await expect(page.locator('#explorer-status')).toContainText('No dataset available');
  await expect(page.locator('#predict-button')).toBeDisabled();
  expect(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)).toBe(false);
});

test('evidence references, provider errors, and history comparison stay inspectable', async ({page}) => {
  await mockApi(page,{reportWithClaims:true});
  await page.goto('/');
  await page.locator('[data-page="reports"]').click();
  await page.getByRole('button',{name:'Generate report'}).click();
  await expect(page.locator('#report-detail')).toContainText('The unusual sensor pattern may indicate degradation.');
  await page.locator('.claim button').first().click();
  await expect(page.locator('[id="evidence-anomaly:sensor_1"]')).toHaveClass(/evidence-target/);
  await page.getByRole('button',{name:'Evaluation history'}).click();
  await page.getByRole('button',{name:'Compare',exact:true}).click();
  await expect(page.locator('#compare-result')).toContainText('fixture_safety');
  await expect(page.locator('#compare-result')).toContainText('old_issue');
});

test('provider-unavailable response is shown without fabricating a report', async ({page}) => {
  await mockApi(page,{reportError:true});
  await page.goto('/');
  await page.locator('[data-page="reports"]').click();
  await page.getByRole('button',{name:'Generate report'}).click();
  await expect(page.locator('#report-status')).toContainText('provider_unavailable');
  await expect(page.locator('#report-detail')).toContainText('No report selected');
});

test('API failure is visible without fabricated system state', async ({page}) => {
  await mockApi(page,{apiDown:true});
  await page.goto('/');
  await expect(page.locator('#connection')).toContainText('API offline');
  await expect(page.locator('#stat-dataset')).toHaveText('Unavailable');
  await expect(page.locator('#stat-gate')).toHaveText('Unavailable');
  await expect(page.locator('#overview-alert')).toContainText('API unavailable');
});

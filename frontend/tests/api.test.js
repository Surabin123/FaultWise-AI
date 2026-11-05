import test from 'node:test';
import assert from 'node:assert/strict';
import {ApiError, createClient} from '../api.js';
import {gateTone, numberText, percentText, timestampText} from '../format.js';

function response(body, status = 200) { return {ok:status >= 200 && status < 300,status,json:async () => body}; }

test('central API base, typed error, and network failure', async () => {
  const calls = [];
  const client = createClient({base:'http://localhost:8765/',fetchImpl:async (url,options) => {
    calls.push({url,options});
    return response({error:{code:'model_unavailable',message:'A trained model is absent.'}},503);
  }});
  await assert.rejects(client.prediction('FD001',1), (error) => error instanceof ApiError && error.code === 'model_unavailable' && error.status === 503);
  assert.equal(calls[0].url,'http://localhost:8765/api/v1/datasets/FD001/engines/1/prediction');
  assert.equal(calls[0].options.method,'POST');
  const offline = createClient({fetchImpl:async () => { throw new Error('socket'); }});
  await assert.rejects(offline.health(), (error) => error.code === 'network_unavailable');
});

test('trajectory loads all pages and preserves observed rows', async () => {
  const offsets = [];
  const client = createClient({fetchImpl:async (url) => {
    const offset = Number(new URL(url,'http://local').searchParams.get('offset'));
    offsets.push(offset);
    return response({variant:'FD001',engine_id:7,total_cycles:3,offset,cycles:offset === 0 ? [{cycle:1,sensors:{sensor_1:3}},{cycle:2,sensors:{sensor_1:4}}] : [{cycle:3,sensors:{sensor_1:5}}]});
  }});
  const result = await client.trajectory('FD001',7);
  assert.deepEqual(offsets,[0,2]);
  assert.deepEqual(result.cycles.map((row) => row.cycle),[1,2,3]);
});

test('unavailable metrics stay unavailable and gate tone is tri-state', () => {
  assert.equal(percentText(null),'Unavailable');
  assert.equal(percentText(0),'0%');
  assert.equal(numberText(undefined),'Unavailable');
  assert.equal(timestampText(null),'Unavailable');
  assert.equal(gateTone('inconclusive'),'inconclusive');
  assert.equal(gateTone('unknown'),'neutral');
});

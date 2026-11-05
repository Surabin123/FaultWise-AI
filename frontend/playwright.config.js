import {defineConfig} from '@playwright/test';

const python = process.env.FAULTWISE_PYTHON || (process.platform === 'win32' ? 'py -3.12' : 'python3');

export default defineConfig({
  testDir:'./tests/e2e',
  use:{baseURL:'http://127.0.0.1:8765',headless:true},
  webServer:{command:`${python} -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8765`,url:'http://127.0.0.1:8765/health',reuseExistingServer:!process.env.CI,timeout:30000,cwd:'..'},
});

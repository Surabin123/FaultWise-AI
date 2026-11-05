// @ts-check
/** @param {number|null|undefined} value @param {number} [digits] */
export function numberText(value, digits = 1) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : 'Unavailable';
}
/** @param {number|null|undefined} value */
export function percentText(value) {
  return typeof value === 'number' && Number.isFinite(value) ? `${Math.round(value * 100)}%` : 'Unavailable';
}
/** @param {string|null|undefined} value */
export function timestampText(value) {
  if (!value) return 'Unavailable';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? 'Unavailable' : date.toLocaleString();
}
/** @param {string|null|undefined} status */
export function gateTone(status) {
  return status === 'pass' || status === 'fail' || status === 'inconclusive' ? status : 'neutral';
}

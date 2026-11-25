// Builds demo/index.html: the demo shell with the app's real DSP code and
// rhythm-view template inlined, so the page is one self-contained file.
//   node demo/build.mjs
import { readFileSync, writeFileSync } from 'node:fs';

const root = new URL('../', import.meta.url);
const read = (p) => readFileSync(new URL(p, root), 'utf8');
const strip = (src) => src.replace(/^export /gm, '');

const dsp = strip(read('src/dsp.js'));
const tplSrc = strip(read('src/ecgTemplate.js')).replace(/async function loadTemplate[\s\S]*?\n}\n/, '');
const tpl = JSON.parse(read('model/ecg_template.json'));
const shell = read('demo/shell.html');

const lib = `// ---- inlined from src/dsp.js ----\n${dsp}\n// ---- inlined from src/ecgTemplate.js ----\n${tplSrc}\n` +
  `const ECG_TEMPLATE = ${JSON.stringify({ ...tpl, beat: undefined })};\nECG_TEMPLATE.beat = Float32Array.from(${JSON.stringify(tpl.beat)});\n`;
writeFileSync(new URL('demo/index.html', root), shell.replace('/*__LIB__*/', () => lib));
console.log('wrote demo/index.html');

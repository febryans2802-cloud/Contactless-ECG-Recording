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
let page = shell.replace('/*__LIB__*/', () => lib);
// --standalone: a complete HTML document for ordinary static hosting (GitHub Pages).
// Without it the file is a page body for claude.ai, which adds the document wrapper itself.
if (process.argv.includes('--standalone')) {
  page = '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n' +
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n' +
    '<style>body{margin:0}[hidden]{display:none!important}</style>\n</head>\n<body>\n' + page + '\n</body>\n</html>\n';
}
const out = process.argv.includes('--out') ? process.argv[process.argv.indexOf('--out') + 1] : 'demo/index.html';
writeFileSync(new URL(out, root), page);
console.log('wrote ' + out);

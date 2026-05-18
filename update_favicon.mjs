import { readFileSync, writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const svgPath = join(__dirname, 'src/web/favicon.svg');
const htmlPath = join(__dirname, 'src/web/index.html');

const svgContent = readFileSync(svgPath, 'utf8');
const base64 = Buffer.from(svgContent).toString('base64');
const dataUri = `data:image/svg+xml;base64,${base64}`;

let html = readFileSync(htmlPath, 'utf8');

const faviconRegex = /<link rel="icon" href="data:image\/svg\+xml;base64,[^"]+" \/>/;
const newLink = `<link rel="icon" href="${dataUri}" />`;
html = html.replace(faviconRegex, newLink);

writeFileSync(htmlPath, html, 'utf8');
console.log('Updated index.html with new favicon base64');

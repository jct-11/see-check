const fs = require('fs');
const path = require('path');

const svgPath = path.join(__dirname, '20260515-144041.svg');
const htmlPath = path.join(__dirname, 'src/web/index.html');
const faviconDest = path.join(__dirname, 'src/web/favicon.svg');

// Copy SVG to favicon.svg
fs.copyFileSync(svgPath, faviconDest);
console.log('Copied SVG to favicon.svg');

// Read and encode to base64
const svgContent = fs.readFileSync(svgPath, 'utf8');
const base64 = Buffer.from(svgContent).toString('base64');
const dataUri = `data:image/svg+xml;base64,${base64}`;

// Read HTML
let html = fs.readFileSync(htmlPath, 'utf8');

// Replace the favicon link
const faviconRegex = /<link rel="icon" href="data:image\/svg\+xml;base64,[^"]+" \/>/;
const newLink = `<link rel="icon" href="${dataUri}" />`;
html = html.replace(faviconRegex, newLink);

// Write updated HTML
fs.writeFileSync(htmlPath, html, 'utf8');
console.log('Updated index.html with new favicon');
console.log(`Base64 length: ${base64.length} characters`);

// Clean up
fs.unlinkSync(path.join(__dirname, '20260515-144041.svg'));
console.log('Cleaned up uploaded SVG file');

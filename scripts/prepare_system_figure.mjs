// Render the editable SVG to a vector PDF using a local Chromium-based browser.
// The result is self-contained and can be included directly by pdflatex.
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {spawn} from 'node:child_process';
import {pathToFileURL} from 'node:url';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';

const repo=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const figures=path.join(repo,'manuscript_temporal','figures');
const input=path.join(figures,'fig_system_workflow.svg');
const output=path.join(figures,'fig_system_workflow.pdf');
const lockPath=path.join(figures,'system_workflow.lock.json');
const lock=await fs.readFile(lockPath,'utf8').catch(e=>{if(e.code==='ENOENT')return null;throw e;});
if(lock){
 for(const [name,expected] of Object.entries(JSON.parse(lock).sha256)){
  const actual=createHash('sha256').update(await fs.readFile(path.join(figures,name))).digest('hex');
  if(actual!==expected)throw Error('Protected system framework differs: '+name);
 }
 console.log('Protected system framework verified; rendering skipped.');
 process.exit(0);
}
const browser=process.env.FIGURE_BROWSER_PATH || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const temp=await fs.mkdtemp(path.join(os.tmpdir(),'eeg-workflow-'));
const profile=path.join(temp,'profile');await fs.mkdir(profile);
const htmlPath=path.join(temp,'figure.html');
const svg=(await fs.readFile(input,'utf8')).replace(/^\s*<\?xml[^?]*\?>/,'');
const html='<!doctype html><html><head><meta charset="utf-8"><style>@page {size: 1672px 941px; margin: 0}html,body{margin:0;padding:0;width:1672px;height:941px;overflow:hidden}svg{display:block;width:1672px;height:941px}</style></head><body>'+svg+'</body></html>';
await fs.writeFile(htmlPath,html,'utf8');
const proc=spawn(browser,['--headless=new','--disable-gpu','--no-first-run','--disable-extensions','--remote-debugging-port=0','--remote-debugging-address=127.0.0.1','--user-data-dir='+profile,'about:blank'],{windowsHide:true,stdio:'ignore'});
let socket;
const pause=ms=>new Promise(r=>setTimeout(r,ms));
try{
 const portPath=path.join(profile,'DevToolsActivePort');
 let port;
 for(let i=0;i<150;i++){
  if(proc.exitCode!==null)throw Error('Browser exited before exposing the debugging port');
  try{port=Number((await fs.readFile(portPath,'utf8')).split('\n')[0]);break;}catch{await pause(100);}
 }
 if(!port)throw Error('Browser did not expose a debugging port');
 const targets=await (await fetch(`http://127.0.0.1:${port}/json`)).json();
 const target=targets.find(t=>t.type==='page');if(!target)throw Error('Browser page unavailable');
 socket=new WebSocket(target.webSocketDebuggerUrl);
 await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});
 let next=0;const pending=new Map();
 socket.onmessage=e=>{const msg=JSON.parse(e.data);if(!msg.id)return;const p=pending.get(msg.id);pending.delete(msg.id);msg.error?p.reject(new Error(JSON.stringify(msg.error))):p.resolve(msg.result);};
 const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++next;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}));});
 await call('Page.enable');await call('Runtime.enable');
 await call('Page.navigate',{url:pathToFileURL(htmlPath).href});
 let ready=false;
 for(let i=0;i<100;i++){
  const r=await call('Runtime.evaluate',{expression:"document.readyState === 'complete' && !!document.querySelector('svg')",returnByValue:true});
  if(r.result.value){ready=true;break;}await pause(100);
 }
 if(!ready)throw Error('SVG did not finish loading');
 await call('Runtime.evaluate',{expression:'document.fonts.ready',awaitPromise:true});
 const pdf=await call('Page.printToPDF',{printBackground:true,preferCSSPageSize:true,displayHeaderFooter:false,paperWidth:1672/96,paperHeight:941/96});
 await fs.writeFile(output,Buffer.from(pdf.data,'base64'));
 console.log(`Saved vector PDF: ${output}`);
 await call('Browser.close').catch(()=>{});
}finally{if(socket)socket.close();proc.kill();}

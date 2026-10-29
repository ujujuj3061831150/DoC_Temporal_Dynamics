"""Build the standalone replay dashboard and report from exported patient traces."""

from pathlib import Path
import argparse
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/bedside_playback_demo"

HTML = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>床旁评估 · 真实功率块回放</title>
<style>
:root{color-scheme:dark;--bg:#0b111e;--panel:#141c2e;--line:#27354b;--text:#f0f4f8;--muted:#a4b1c5;--blue:#57b8ff;--green:#69ddb4;--amber:#ffcf72}
*{box-sizing:border-box}.panel{min-width:0}.note{overflow-wrap:anywhere}body{margin:0;padding:24px;background:var(--bg);color:var(--text);font:15px/1.6 system-ui,sans-serif;max-width:1400px;margin:auto}
h1{font-size:24px;margin:0}h2{font-size:16px;margin:0 0 12px}.sub{color:var(--muted);font-size:13px}.header{display:flex;justify-content:space-between;gap:18px;align-items:start;margin-bottom:20px}
.badge{border:1px solid var(--line);border-radius:6px;padding:6px 12px;color:var(--green);white-space:nowrap}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:16px 0}select,button{background:#1b2a40;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:9px 14px;font:inherit}button{cursor:pointer}.grid{display:grid;grid-template-columns:2fr 1fr;gap:16px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px}.wide{grid-column:1/-1}.cards{display:grid;grid-template-columns:1fr 1fr;gap:12px}.value{font-size:27px;font-variant-numeric:tabular-nums}.card{border-bottom:1px solid var(--line);padding:8px 0}canvas{display:block;width:100%;height:235px}#scoreCanvas{height:255px}input[type=range]{width:100%;accent-color:var(--blue)}#state{padding:12px;border:1px solid var(--line);border-radius:6px;margin:14px 0;color:var(--amber)}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:7px;border-bottom:1px solid var(--line);font-size:13px}.note{margin-top:14px;color:var(--muted);font-size:13px}@media(max-width:850px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}.header{display:block}.badge{display:inline-block;margin-top:12px}body{padding:12px}}
</style></head><body>
<div class="header"><div><h1>床旁评估评分 · 真实功率块回放</h1><div class="sub">Bedside assessment score · Sequential replay of cached patient power blocks</div></div><div class="badge">冻结源域模型 · 500 μV 质量门控</div></div>
<div class="sub">每块对应 4 秒录波。展示患者缓存中的实际功率、QC 和计算轨迹；评分为累计历史特征的标准分。</div>
<div class="controls"><label>记录 <select id="subject"></select></label><button id="play">开始回放</button><button id="reset">回到起点</button><label>演示速度 <select id="speed"><option value="1">1 块 / 帧</option><option value="20" selected>20 块 / 帧</option><option value="100">100 块 / 帧</option></select></label></div>
<div class="grid">
<section class="panel"><h2>当前块六通道 θ 功率</h2><canvas id="powerCanvas" aria-label="Actual cached six-channel theta power"></canvas><div class="sub">对数显示 log₁₀(1 + power)，通道顺序与缓存一致。此处展示频带功率，不展示原始 EEG 波形。</div></section>
<section class="panel"><h2>质量控制与有效状态</h2><div class="cards"><div class="card"><div class="sub">已接收块</div><div class="value" id="blocks">—</div></div><div class="card"><div class="sub">有效 / 剔除</div><div class="value" id="clean">—</div></div><div class="card"><div class="sub">物理相邻对</div><div class="value" id="pairs">—</div></div><div class="card"><div class="sub">当前峰值 μV</div><div class="value" id="peak">—</div></div></div><div id="state">等待载入</div><div class="sub" id="retention"></div></section>
<section class="panel"><h2>实际计算时刻的评估评分</h2><canvas id="scoreCanvas" aria-label="Recorded assessment scores and source training threshold"></canvas><div class="sub">蓝色：S(x)；绿色：S₁.₅(x)；虚线：源域训练得分中位数。仅连接实际计算事件。</div></section>
<section class="panel"><h2>最近一次有效计算</h2><div class="cards"><div class="card"><div class="sub">S(x)</div><div class="value" id="score">—</div></div><div class="card"><div class="sub">S₁.₅(x)</div><div class="value" id="clipped">—</div></div></div><div class="sub" id="asof">尚无有效评分</div><table><tbody id="features"></tbody></table><div class="note" id="threshold"></div></section>
<section class="panel wide"><input id="position" type="range" min="0" value="0" aria-label="Replay position"><div class="sub" id="positionText"></div><div class="note" id="timing"></div><div class="note" id="provenance"></div></section>
</div>
<div class="note">用途：工程回放和方法核对。示例病例来自源域训练队列；这里不估计独立诊断性能，也不将标准分解释为意识概率。数据未达到最低 10 个物理相邻对时不输出评分。特征按累计历史重算。</div>
<script id="playback-data" type="application/json">__PAYLOAD__</script>
<script>
'use strict';
const dataset=JSON.parse(document.getElementById('playback-data').textContent);
const $=id=>document.getElementById(id);
let trace,index=0,timer=null;
const names=['F3','F4','C3','C4','O1','O2'];
const fmt=(v,n=3)=>v===null||v===undefined?'未执行':Number(v).toFixed(n);
function decisionForStep(step, threshold){
  if(!step.is_clean)return 'ARTIFACT_REJECTED';
  if(!step.ready)return 'INSUFFICIENT_DATA';
  if(step.score===null)return 'NOT_EVALUATED';
  return step.score>=threshold?'PRIORITIZE_FOR_MULTIMODAL':'LOWER_PRIORITY';
}
const labels={ARTIFACT_REJECTED:'当前块伪影剔除',INSUFFICIENT_DATA:'数据不足 · 等待有效相邻对',NOT_EVALUATED:'当前块未重新计算 · 下方保留最近有效结果',PRIORITIZE_FOR_MULTIMODAL:'优先安排多模态复核',LOWER_PRIORITY:'较低复核优先级'};
function canvas(id){let c=$(id),r=c.getBoundingClientRect(),d=window.devicePixelRatio||1;c.width=r.width*d;c.height=r.height*d;let x=c.getContext('2d');x.scale(d,d);x.clearRect(0,0,r.width,r.height);x.font='12px system-ui';return [x,r.width,r.height];}
function drawPower(step){const [c,w,h]=canvas('powerCanvas'),values=step.theta_power.map(x=>Math.log10(1+x));const max=Math.max(1,...values),left=45,bottom=h-28,height=h-48,bw=(w-left-15)/6;c.fillStyle='#a4b1c5';c.fillText('log10(1 + power)',left,16);values.forEach((v,i)=>{const x=left+i*bw+8,y=bottom-v/max*height;c.fillStyle=step.is_clean?'#57b8ff':'#ffcf72';c.fillRect(x,y,bw-18,bottom-y);c.fillStyle='#a4b1c5';c.fillText(names[i],x,bottom+19);c.fillText(v.toFixed(2),x,Math.max(30,y-5));});}
function drawScores(){const [c,w,h]=canvas('scoreCanvas'),left=48,right=w-18,top=20,bottom=h-32,points=trace.steps.slice(0,index+1).filter(s=>s.score!==null);const vals=points.flatMap(s=>[s.score,s.score_c15]);vals.push(trace.threshold_source_train,trace.threshold_source_train_c15);let lo=Math.min(-0.5,...vals)-0.15,hi=Math.max(0.5,...vals)+0.15;const xx=s=>left+(s.epoch_idx+1)/trace.steps.length*(right-left),yy=v=>bottom-(v-lo)/(hi-lo)*(bottom-top);c.strokeStyle='#27354b';c.fillStyle='#a4b1c5';for(let j=0;j<4;j++){const v=lo+(hi-lo)*j/3,y=yy(v);c.beginPath();c.moveTo(left,y);c.lineTo(right,y);c.stroke();c.fillText(v.toFixed(2),4,y+4);}c.fillText('0',left,bottom+22);c.fillText(String(trace.steps.length)+' blocks',Math.max(left,right-88),bottom+22);function horizontal(v,color){c.strokeStyle=color;c.setLineDash([5,4]);c.beginPath();c.moveTo(left,yy(v));c.lineTo(right,yy(v));c.stroke();c.setLineDash([]);}horizontal(trace.threshold_source_train,'#57b8ff');horizontal(trace.threshold_source_train_c15,'#69ddb4');for(const [key,color] of [['score','#57b8ff'],['score_c15','#69ddb4']]){c.strokeStyle=color;c.lineWidth=2;c.beginPath();points.forEach((s,i)=>{if(i===0)c.moveTo(xx(s),yy(s[key]));else c.lineTo(xx(s),yy(s[key]));});c.stroke();for(const s of points){c.fillStyle=color;c.beginPath();c.arc(xx(s),yy(s[key]),2.3,0,Math.PI*2);c.fill();}}}
function render(){const s=trace.steps[index],previous=trace.steps.slice(0,index+1).filter(x=>x.score!==null),last=previous.length?previous[previous.length-1]:null;const state=decisionForStep(s,trace.threshold_source_train);$('blocks').textContent=index+1;$('clean').textContent=s.total_clean_blocks+' / '+s.rejected_blocks;$('pairs').textContent=s.adjacent_pairs;$('peak').textContent=fmt(s.peak_amplitude,1);$('state').textContent=labels[state];$('state').style.color=state==='PRIORITIZE_FOR_MULTIMODAL'?'#69ddb4':'#ffcf72';$('retention').textContent='有效保留率 '+(s.retention_rate*100).toFixed(3)+'%';$('score').textContent=last?fmt(last.score):'—';$('clipped').textContent=last?fmt(last.score_c15):'—';$('asof').textContent=last?'计算于第 '+(last.epoch_idx+1)+' 块 · '+(last.recording_seconds/60).toFixed(1)+' 分钟':'尚无有效评分';$('features').innerHTML=last?Object.entries(last.features).map(([k,v])=>'<tr><td>'+k+'</td><td>'+fmt(v,5)+'</td></tr>').join(''):'';$('position').value=index;$('positionText').textContent='第 '+(index+1)+' / '+trace.steps.length+' 块 · 录波时间 '+(s.recording_seconds/3600).toFixed(3)+' 小时';const t=s.step_latencies_ms;$('timing').textContent='当前引擎调用 '+fmt(t.total,4)+' ms；预处理 '+fmt(t.preprocess,4)+'；特征 '+fmt(t.features,4)+'；评分 '+fmt(t.scoring,4)+'。计时不含磁盘、渲染或采集等待。';drawPower(s);drawScores();}
function stop(){if(timer!==null)clearInterval(timer);timer=null;$('play').textContent='开始回放';}
function load(){stop();trace=dataset.traces[$('subject').value];index=0;$('position').max=trace.steps.length-1;$('threshold').textContent='冻结训练阈值：S '+trace.threshold_source_train.toFixed(8)+'；S₁.₅ '+trace.threshold_source_train_c15.toFixed(8)+'。';$('provenance').textContent='输入：'+trace.input_mode+'；源域训练病例：'+(trace.source_training_member?'是':'否')+'；缓存 SHA-256：'+trace.cache_sha256;render();}
for(const id of Object.keys(dataset.traces)){const option=document.createElement('option');option.value=id;option.textContent=id;$('subject').appendChild(option);}
$('subject').addEventListener('change',load);$('reset').addEventListener('click',load);$('position').addEventListener('input',()=>{stop();index=Number($('position').value);render();});$('play').addEventListener('click',()=>{if(timer!==null){stop();return;}if(index>=trace.steps.length-1)index=0;$('play').textContent='暂停';timer=setInterval(()=>{index=Math.min(trace.steps.length-1,index+Number($('speed').value));render();if(index===trace.steps.length-1)stop();},100);});window.addEventListener('resize',render);load();
</script></body></html>
'''


def build_demo(output_dir=RESULTS, html_path=ROOT / "docs/bedside_cdss_demo.html"):
    output_dir, html_path = Path(output_dir), Path(html_path)
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(output_dir.glob("playback_report_*.json"))]
    if not reports:
        raise ValueError("Generate patient playback reports before building the dashboard.")
    traces = {r["subject_id"]: json.loads((output_dir / r["trajectory_file"]).read_text(encoding="utf-8")) for r in reports}
    payload = json.dumps({"traces": traces}, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(HTML.replace("__PAYLOAD__", payload), encoding="utf-8")
    lines = ["# 床旁功率回放原型验证报告", "", "本报告由实际回放 JSON 自动生成。输入为真实患者的预计算功率与峰值缓存；未将其表述为原始 EEG 波形回放或原始 EEG 端到端延迟。", "",
             "源域标准化参数、符号与操作阈值直接读取归档模型状态，不在回放过程中重新拟合。示例患者属于源域训练队列，用于工程一致性核对，不据此估计独立诊断性能。", "",
             "## 实际终值", "", "| 患者 | 总块 | 有效块 | 相邻对 | S(x) | S₁.₅(x) | 特征/评分容差核对 |", "|---|---:|---:|---:|---:|---:|---|"]
    for r in reports:
        passed = r["consistency_verified"] and r["consistency"]["scores_within_tolerance"]
        score = lambda key: "未就绪" if r[key] is None else f"{r[key]:.10f}"
        lines.append(f"| {r['subject_id']} | {r['total_epochs']} | {r['clean_blocks']} | {r['adjacent_pairs']} | {score('final_assessment_score')} | {score('final_score_c15')} | {'通过' if passed else '未通过/未就绪'} |")
    for r in reports:
        lines += ["", f"### {r['subject_id']}", "",
                  f"有效保留率：{100*r['retention_rate']:.6f}%。S(x) 训练中位数阈值：{r['threshold_source_train']:.17g}；S₁.₅(x) 对应阈值：{r['threshold_source_train_c15']:.17g}。", "",
                  "| 特征 | 实际终值 |", "|---|---:|"]
        for name, value in (r["final_features"] or {}).items():
            lines.append(f"| {name} | {value:.12g} |")
        lines += ["", "| 实际执行阶段 | 调用数 | 均值 ms | p95 ms | 最大 ms |", "|---|---:|---:|---:|---:|"]
        for stage in ("cached_ingest_only", "features", "scoring", "evaluated_updates", "preprocess", "raw_epoch_processing"):
            s = r["latency_profile_ms"][stage]
            number = lambda key: "未执行" if s[key] is None else f"{s[key]:.6f}"
            lines.append(f"| {stage} | {s['count']} | {number('mean_ms')} | {number('p95_ms')} | {number('max_ms')} |")
        diffs = r["consistency"]["differences"]
        if diffs:
            maximum = max(v["absolute_diff"] for v in diffs.values())
            lines += ["", f"同前缀离线特征最大绝对误差 {maximum:.3e}，绝对容差 1e-7、相对容差 0；不称比特级相等。评分误差：`{json.dumps(r['consistency'].get('score_absolute_differences', {}))}`。"]
        archive = r.get("full_record_archive_comparison")
        if archive:
            maximum = max(v["absolute_diff"] for v in archive["differences"].values())
            lines += [f"完整录波与归档特征核对：{archive['consistent']}；最大绝对误差 {maximum:.3e}。"]
        lines += ["", f"运行环境：Python {r['environment']['python']}；{r['environment']['platform']}；{r['environment']['processor']}。未限制 CPU 亲和性或 BLAS 线程数。"]
    xml_path = output_dir / "test_results.xml"
    if xml_path.exists():
        suites = list(ET.parse(xml_path).getroot().iter("testsuite"))
        counts = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
        lines += ["", "## 自动化检查", "", f"JUnit 记录：{counts['tests']} 项，失败 {counts['failures']}，错误 {counts['errors']}，跳过 {counts['skipped']}。详见 test_results.xml。"]
    lines += ["", "## 运行范围", "",
              "- QC 采用既定的严格峰值 <500 μV 条件；未新增 150 μV 临床筛查门限。",
              "- 仅当至少 10 个物理相邻对时允许评分；不足输出 INSUFFICIENT_DATA，缓冲但未计算输出 NOT_EVALUATED。",
              "- 看板与动画读取实际逐块输出；无合成波形或随机评分，提示由评分与冻结阈值决定。",
              "- 计时采用 perf_counter_ns。未执行阶段记为空，统计仅含实际执行事件；特征每指定步长按累计历史重算，不声称固定成本递推。",
              "- 当前两例患者的原始 EEG 预处理未执行，不能从这些结果推导原始 EEG 端到端 0.127 ms 或临床判定等待时间。",
              "", "运行：`python scripts/run_bedside_playback.py --subject Nekrasova_P36 --speed 0`；P25 同理。随后运行 `python scripts/build_bedside_playback_demo.py` 与 `python scripts/render_playback_animation.py`。", ""]
    (output_dir / "BEDSIDE_PLAYBACK_VERIFICATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Built dashboard: {html_path}; generated verification report from {len(reports)} patient reports.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    parser.add_argument("--html-path", type=Path, default=ROOT / "docs/bedside_cdss_demo.html")
    args = parser.parse_args()
    build_demo(args.output_dir, args.html_path)

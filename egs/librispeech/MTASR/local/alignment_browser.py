#!/usr/bin/env python3
"""Browser UI for listening to and comparing word alignments."""

from __future__ import annotations

import argparse
import io
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import soundfile as sf
from lhotse import CutSet


DEFAULT_CUTS = Path(
    "data/manifests/real_meetings/notsofar/"
    "notsofar_sdm_dev_240415.2_dev_with_GT_max30_normalized_word_aligned.jsonl.gz"
)
DEFAULT_REF = Path(
    "data/alignments/"
    "notsofar_sdm_dev_240415.2_dev_with_GT_max30_normalized_word_alignments.json"
)
DEFAULT_PARAKEET = Path(
    "exp/fastmss_yodas_en_parakeet_hires_local_factorized_a100_accum4_combined_bpe/align/"
    "word_alignments_chkpt126000_avg1_notsofar_sdm_dev30_word_aligned_"
    "collar32000_word_timing_beam40.json"
)
DEFAULT_WHISPER = Path(
    "exp/fastmss_ami_notsofar_whisper_large_v3_turbo_spashl_a100x2_accum4_maxlr5e-6/align/"
    "word_alignments_chkpt76000_avg1_notsofar_sdm_dev30_word_aligned_"
    "collar32000_word_timing_beam40.json"
)


def load_alignment(path: Path) -> dict[str, list]:
    return {cut_id: rows for cut_id, rows in json.loads(path.read_text())}


def ordered(rows: list) -> list:
    return sorted(rows, key=lambda row: (row[1], row[3]))


def interval_metrics(reference: list, hypothesis: list | None) -> dict:
    if hypothesis is None or len(reference) != len(hypothesis):
        return {"available": False, "words": len(reference)}
    ref = ordered(reference)
    hyp = ordered(hypothesis)
    if [row[2] for row in ref] != [row[2] for row in hyp]:
        return {"available": False, "words": len(reference)}
    errors, intersections, unions = [], [], []
    for r, h in zip(ref, hyp):
        rs, re = float(r[3]), float(r[3]) + float(r[4])
        hs, he = float(h[3]), float(h[3]) + float(h[4])
        errors.append(0.5 * (abs(hs - rs) + abs(he - re)))
        intersections.append(max(0.0, min(re, he) - max(rs, hs)))
        unions.append(max(re, he) - min(rs, hs))
    return {
        "available": True,
        "words": len(ref),
        "boundary_ms": round(1000 * sum(errors) / max(1, len(errors)), 1),
        "iou": round(100 * sum(intersections) / max(1.0e-9, sum(unions)), 1),
    }


HTML = r'''<!doctype html>
<html><head><meta charset="utf-8"><title>NOTSOFAR alignment comparison</title>
<style>
body{font:14px system-ui,sans-serif;margin:18px;color:#202124;background:#fafafa}
h1{margin:0 0 12px;font-size:22px}.toolbar{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
select,input,button{font:inherit;padding:6px}#cut{min-width:470px}audio{width:100%;margin:12px 0}
.summary{display:flex;gap:16px;margin:8px 0 14px}.card{background:white;border:1px solid #ddd;padding:8px 12px;border-radius:6px}
.panel{background:white;border:1px solid #ddd;border-radius:6px;padding:10px;margin:10px 0}
.panel h2{font-size:16px;margin:0 0 7px}.scroll{overflow-x:auto;border:1px solid #eee}.timeline{position:relative;min-width:100%}
.lane{position:relative;height:30px;border-top:1px solid #eee}.speaker-row{font-size:11px;color:#555;padding:3px 0 1px}
.word{position:absolute;top:3px;height:22px;box-sizing:border-box;border:1px solid rgba(0,0,0,.25);border-radius:3px;overflow:hidden;white-space:nowrap;font-size:10px;padding:3px;cursor:pointer;min-width:2px}
.reference .word{background:#b7d7f0}.parakeet .word{background:#ffd39a}.whisper .word{background:#bce6bd}
.ticks{position:relative;height:20px;border-top:1px solid #aaa;cursor:crosshair}.tick{position:absolute;font-size:10px;top:2px;transform:translateX(-50%)}
.cursor{position:absolute;top:0;bottom:0;width:2px;background:#d00;z-index:5;pointer-events:none}
.transcript{line-height:1.7}.token{cursor:pointer;border-radius:3px;padding:2px 3px;margin-right:2px}.token:hover{background:#eee}
.missing{color:#a00}.hint{color:#666;margin-top:10px}.active{outline:2px solid #d00}.playtools{margin:0 0 10px}.playtools output{display:inline-block;min-width:52px}
</style></head><body>
<h1>NOTSOFAR dev word-alignment comparison</h1>
<div class="toolbar"><button id="prev">◀</button><button id="next">▶</button>
<select id="sort"><option value="id">Cut ID</option><option value="disagreement">Largest model disagreement</option><option value="parakeet">Parakeet advantage</option><option value="whisper">Whisper advantage</option><option value="worst">Largest average error</option></select>
<input id="search" placeholder="Filter cut ID"><select id="cut"></select></div>
<audio id="audio" controls preload="metadata"></audio>
<div class="toolbar playtools"><label>Zoom <input id="zoom" type="range" min="20" max="400" value="40" step="10"><output id="zoomval">40 px/s</output></label>
<label>Speed <select id="speed"><option>0.25</option><option>0.5</option><option>0.75</option><option selected>1</option><option>1.25</option><option>1.5</option></select>×</label>
<button id="back">−0.25 s</button><button id="forward">+0.25 s</button><label>Time <input id="seektime" type="number" min="0" step="0.01" value="0" style="width:75px"> s</label><button id="seekgo">Go</button></div>
<div id="summary" class="summary"></div><div id="views"></div>
<div class="hint">Click anywhere in a timeline, word block, or transcript token to seek. Drag the Zoom control for fine boundaries; all three timelines scroll together. Keyboard: ←/→ seek 0.25 s; Shift+←/→ seek 1 s; J/K previous/next cut; space plays/pauses.</div>
<script>
let cuts=[], shown=[], current=null,currentData=null,syncing=false; const audio=document.querySelector('#audio');
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function order(){let mode=sort.value; shown=cuts.filter(x=>x.id.toLowerCase().includes(search.value.toLowerCase()));
 let key=x=>mode==='disagreement'?x.disagreement:mode==='parakeet'?x.parakeet_advantage:mode==='whisper'?x.whisper_advantage:mode==='worst'?x.worst:0;
 if(mode!=='id') shown.sort((a,b)=>key(b)-key(a)); else shown.sort((a,b)=>a.id.localeCompare(b.id));
 cut.innerHTML=shown.map(x=>`<option value="${esc(x.id)}">${esc(x.id)} — P ${x.parakeet.boundary_ms??'fail'} ms, W ${x.whisper.boundary_ms??'fail'} ms</option>`).join('');
 if(shown.length) load(shown.some(x=>x.id===current)?current:shown[0].id);}
function ticks(d){let step=+zoom.value>=200?1:+zoom.value>=80?2:5,s='<div class="ticks">';for(let t=0;t<=d;t+=step)s+=`<span class="tick" style="left:${100*t/d}%">${t}s</span>`;return s+'</div>'}
function panel(name,rows,d,cls){if(!rows)return `<div class="panel ${cls}"><h2>${name}</h2><div class="missing">Alignment failed for this cut.</div></div>`;
 let speakers=[...new Set(rows.map(r=>r[0]))], width=Math.max(700,d*+zoom.value),out=`<div class="panel ${cls}"><h2>${name}</h2><div class="scroll"><div class="timeline seekbar" data-duration="${d}" style="width:${width}px"><div class="cursor"></div>${ticks(d)}`;
 for(const sp of speakers){out+=`<div class="speaker-row">${esc(sp)}</div><div class="lane">`;
  for(const r of rows.filter(x=>x[0]===sp)){let st=+r[3],du=+r[4];out+=`<span class="word seek" data-time="${st}" title="${esc(r[2])} ${st.toFixed(2)}–${(st+du).toFixed(2)}" style="left:${100*st/d}%;width:${Math.max(.12,100*du/d)}%">${esc(r[2])}</span>`}out+='</div>'}
 out+='</div></div><div class="transcript">'+[...rows].sort((a,b)=>a[3]-b[3]).map(r=>`<span class="token seek" data-time="${r[3]}" title="${r[0]} ${(+r[3]).toFixed(2)}s">${esc(r[2])}</span>`).join(' ')+'</div></div>';return out;}
function wire(){document.querySelectorAll('.seek').forEach(e=>e.onclick=ev=>{ev.stopPropagation();audio.currentTime=+e.dataset.time;audio.play()});
 document.querySelectorAll('.seekbar').forEach(e=>e.onclick=ev=>{if(ev.target.classList.contains('word'))return;let r=e.getBoundingClientRect();audio.currentTime=Math.max(0,Math.min(+e.dataset.duration||currentData.duration,(ev.clientX-r.left)/r.width*(+e.dataset.duration||currentData.duration)))});
 let ss=[...document.querySelectorAll('.scroll')];ss.forEach(s=>s.onscroll=()=>{if(syncing)return;syncing=true;ss.forEach(x=>{if(x!==s)x.scrollLeft=s.scrollLeft});requestAnimationFrame(()=>syncing=false)});updateCursor()}
function render(){let d=currentData;if(!d)return;views.innerHTML=panel('Reference',d.reference,d.duration,'reference')+panel('Parakeet 126k',d.parakeet,d.duration,'parakeet')+panel('Whisper 76k',d.whisper,d.duration,'whisper');wire()}
async function load(id){current=id;cut.value=id;audio.src='/audio?id='+encodeURIComponent(id);let d=await fetch('/api/cut?id='+encodeURIComponent(id)).then(r=>r.json());
 currentData=d;
 summary.innerHTML=`<div class="card"><b>${esc(id)}</b><br>${d.duration.toFixed(2)} s · ${d.reference.length} words</div><div class="card"><b>Parakeet</b><br>${d.metrics.parakeet.available?d.metrics.parakeet.boundary_ms+' ms · '+d.metrics.parakeet.iou+'% IoU':'failed'}</div><div class="card"><b>Whisper</b><br>${d.metrics.whisper.available?d.metrics.whisper.boundary_ms+' ms · '+d.metrics.whisper.iou+'% IoU':'failed'}</div>`;
 render();}
function updateCursor(){seektime.value=audio.currentTime.toFixed(2);let d=currentData?.duration||1;document.querySelectorAll('.cursor').forEach(c=>c.style.left=(100*audio.currentTime/d)+'%')}
function move(n){let i=shown.findIndex(x=>x.id===current);if(i>=0&&shown.length)load(shown[(i+n+shown.length)%shown.length].id)}
prev.onclick=()=>move(-1);next.onclick=()=>move(1);sort.onchange=order;search.oninput=order;cut.onchange=()=>load(cut.value);
zoom.oninput=()=>{zoomval.value=zoom.value+' px/s';render()};speed.onchange=()=>audio.playbackRate=+speed.value;back.onclick=()=>audio.currentTime=Math.max(0,audio.currentTime-.25);forward.onclick=()=>audio.currentTime+=.25;seekgo.onclick=()=>audio.currentTime=+seektime.value;audio.ontimeupdate=updateCursor;audio.onseeked=updateCursor;
document.onkeydown=e=>{if(e.target.matches('input,select'))return;let step=e.shiftKey?1:.25;if(e.key==='ArrowLeft')audio.currentTime=Math.max(0,audio.currentTime-step);if(e.key==='ArrowRight')audio.currentTime+=step;if(e.key.toLowerCase()==='j')move(-1);if(e.key.toLowerCase()==='k')move(1);if(e.key===' '){e.preventDefault();audio.paused?audio.play():audio.pause()}};
fetch('/api/cuts').then(r=>r.json()).then(x=>{cuts=x;order()});
</script></body></html>'''


def make_server(args):
    cuts = {cut.id: cut for cut in CutSet.from_file(args.cuts)}
    reference = load_alignment(args.reference)
    parakeet = load_alignment(args.parakeet)
    whisper = load_alignment(args.whisper)
    summaries = []
    for cut_id, cut in cuts.items():
        ref = reference[cut_id]
        pm = interval_metrics(ref, parakeet.get(cut_id))
        wm = interval_metrics(ref, whisper.get(cut_id))
        pe = pm.get("boundary_ms", 100000.0)
        we = wm.get("boundary_ms", 100000.0)
        summaries.append({
            "id": cut_id, "parakeet": pm, "whisper": wm,
            "disagreement": abs(pe - we), "parakeet_advantage": we - pe,
            "whisper_advantage": pe - we, "worst": max(pe, we),
        })

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, content_type, body):
            self.send_response(status); self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/": return self.send(200, "text/html; charset=utf-8", HTML.encode())
            if parsed.path == "/api/cuts": return self.send(200, "application/json", json.dumps(summaries).encode())
            cut_id = parse_qs(parsed.query).get("id", [""])[0]
            if cut_id not in cuts: return self.send(404, "text/plain", b"Unknown cut")
            if parsed.path == "/api/cut":
                data = {"id": cut_id, "duration": cuts[cut_id].duration,
                        "reference": reference[cut_id], "parakeet": parakeet.get(cut_id),
                        "whisper": whisper.get(cut_id),
                        "metrics": {"parakeet": interval_metrics(reference[cut_id], parakeet.get(cut_id)),
                                    "whisper": interval_metrics(reference[cut_id], whisper.get(cut_id))}}
                return self.send(200, "application/json", json.dumps(data).encode())
            if parsed.path == "/audio":
                audio = cuts[cut_id].load_audio().T
                buf = io.BytesIO(); sf.write(buf, audio, cuts[cut_id].sampling_rate, format="WAV", subtype="PCM_16")
                return self.send(200, "audio/wav", buf.getvalue())
            return self.send(404, "text/plain", b"Not found")

        def log_message(self, fmt, *values):
            print(f"{self.client_address[0]} {fmt % values}")

    return ThreadingHTTPServer((args.host, args.port), Handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuts", type=Path, default=DEFAULT_CUTS)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REF)
    parser.add_argument("--parakeet", type=Path, default=DEFAULT_PARAKEET)
    parser.add_argument("--whisper", type=Path, default=DEFAULT_WHISPER)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = make_server(args)
    print(f"Alignment browser listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

"""Real, checkpointed judge-band evaluation; no network except loopback."""
import ctypes, json, math, os, random, sys, threading, time, traceback
from collections import Counter, defaultdict
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; SOURCE=ROOT/'evidence/eval-2026-10-27/resume'; E=SOURCE.parent
P=SOURCE/('corrected-run-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
P.mkdir(parents=True,exist_ok=False)
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
os.environ.update(PYTHONDONTWRITEBYTECODE='1',USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',DETECTOR_VERIFY_HASHES='true',OLLAMA_API_URL='http://127.0.0.1:11435/api/generate',AUDIT_LOG_PATH=str(P/'judge-audit.jsonl'),AUDIT_DB_PATH=str(P/'judge-audit.db'),REVIEW_QUEUE_FILE=str(P/'judge-reviews.json'))
from core.risk import fuse_signals, llama_guard_arbitration, _in_upper_ambiguous_band
from core.config import settings, META_INTENT_THRESHOLD
from core.detectors import get_detector
from core.circuit_breaker import llama_guard_breaker
def now():return datetime.now(timezone.utc).isoformat()
def read(path):
    if not path.exists():return []
    with path.open(encoding='utf-8', newline='') as stream:
        return [json.loads(line) for line in stream if line.strip()]
def write(path,x):path.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def append(path,x):
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def confusion(labels,preds):
    tp=sum(y==1 and p for y,p in zip(labels,preds));tn=sum(y==0 and not p for y,p in zip(labels,preds));fp=sum(y==0 and p for y,p in zip(labels,preds));fn=sum(y==1 and not p for y,p in zip(labels,preds));safe=lambda a,b:a/b if b else None
    precision=safe(tp,tp+fp);recall=safe(tp,tp+fn)
    return {'tp':tp,'tn':tn,'fp':fp,'fn':fn,'precision':precision,'recall':recall,'f1':safe(2*tp,2*tp+fp+fn),'accuracy':safe(tp+tn,len(labels)),'benign_fpr':safe(fp,fp+tn)}
def boot(data,cut,seed=20261027):
    groups=defaultdict(list)
    for i,x in enumerate(data):groups[(x['label'],x['category'])].append(i)
    gen=np.random.default_rng(seed); values={'before':defaultdict(list),'after':defaultdict(list)}
    for _ in range(1000):
        indices=np.concatenate([gen.choice(ix,len(ix),replace=True) for ix in groups.values()]);sample=[data[i] for i in indices]
        for side,key in [('before','baseline_risk'),('after','enabled_risk')]:
            m=confusion([x['label'] for x in sample],[{'LOW':0,'MEDIUM':1,'HIGH':2}[x[key]]>=cut for x in sample])
            for k in ('precision','recall','f1','accuracy','benign_fpr'):
                if m[k] is not None:values[side][k].append(m[k])
    return {side:{k:np.percentile(v,[2.5,97.5]).tolist() for k,v in metrics.items()} for side,metrics in values.items()}
def main():
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    shared={'stage':'judge-band-preparation','items_done':0,'items_total':13011}; lock=threading.Lock();stop=threading.Event()
    def heartbeat():
        while not stop.is_set():
            with lock:write(P/'heartbeat.json',{**shared,'timestamp_utc':now()})
            stop.wait(20)
    threading.Thread(target=heartbeat,daemon=True).start()
    try:
        import logging
        logging.getLogger('core.risk').setLevel(logging.ERROR)
        examples=read(E/'examples.jsonl'); semantic={x['id']:x for x in read(E/'semantic-features.jsonl')}; fused={x['id']:x for x in read(E/'fusion-scores.jsonl')};band=[]
        for i,row in enumerate(examples):
            s=semantic[row['id']]; f=fused[row['id']]
            if s['symbolic'] or s['fast_decision'] is not None:continue
            signals={'meta_intent_score':s['fast']['meta_intent_score'],'domain_aligned':None,'domain_score':None,'fusion_available':True,'fusion_score':f['score'],'fusion_threshold_high':f['threshold_high'],'fusion_threshold_medium':f['threshold_medium'],'is_educational':s['educational']}
            risk,source,needed,_=fuse_signals(signals,row['text'])
            if needed:band.append({'id':row['id'],'label':row['label'],'category':row['attack_class'],'baseline_risk':risk,'score':f['score'],'threshold_medium':f['threshold_medium'],'threshold_high':f['threshold_high'],'upper_band':_in_upper_ambiguous_band(f['score'],f['threshold_medium'],f['threshold_high'])})
            if i%100==0:
                with lock:shared['items_done']=i
        lookup={x['id']:x for x in band}; old=[];invalid=[]
        for r in read(E/'judge-predictions.jsonl'):
            if not r.get('judge_invoked'):continue
            b=lookup.get(r['id']);out=r.get('actual_judge_output') or {}
            if b and r.get('label')==b['label'] and r.get('category')==b['category'] and r.get('baseline_risk')==b['baseline_risk'] and out.get('verdict') in ('safe','unsafe') and isinstance(r.get('judge_seconds'),(float,int)) and r['judge_seconds']>0:
                old.append({**b,'enabled_risk':r['enabled_risk'],'actual_judge_output':out,'judge_seconds':r['judge_seconds'],'origin':'previous-real-native-call','timestamp_utc':now()})
            else:invalid.append(r.get('id'))
        old={r['id']:r for r in old}; timings=[r['judge_seconds'] for r in old.values()]
        if not timings:raise RuntimeError('No validated existing real-call latencies available for projection; fresh-call projection not implemented')
        mean=float(np.mean(timings));remaining=len(band)-len(old); projected=remaining*mean
        if projected<=10800:target=len(band);mode='full band'
        else:target=min(len(band),len(old)+int(10800/mean));mode='stratified sample of band'
        groups=defaultdict(list)
        for b in band:
            if b['id'] not in old:groups[(b['label'],b['category'])].append(b)
        rng=random.Random(20261027)
        for group in groups.values():rng.shuffle(group)
        need=target-len(old);total=sum(map(len,groups.values())); allocation={k:min(len(v),math.floor(need*len(v)/total)) for k,v in groups.items()} if total else {}
        for k in sorted(groups,key=lambda k:-(need*len(groups[k])/total-allocation[k])):
            if sum(allocation.values())>=need:break
            if allocation[k]<len(groups[k]):allocation[k]+=1
        selected=list(old)+[b['id'] for k,v in groups.items() for b in v[:allocation[k]]]
        manifest={'definition':'core.risk.fuse_signals: symbolic and fast-path decisions are excluded; meta-intent veto precedes fusion; judge required when per-row threshold_medium <= fusion score < threshold_high and not educational (with configured domain guardrail); upper half is determined by core.risk._in_upper_ambiguous_band','backend':'native Hugging Face Llama Guard 3 1B; no Ollama port is required for this native path','judge_may_clear_to_low':settings.JUDGE_MAY_CLEAR_TO_LOW,'band_count':len(band),'label_counts':dict(Counter(str(x['label']) for x in band)),'category_counts':dict(Counter(x['category'] for x in band)),'rows':band,'validated_previous_judge_records':len(old),'invalid_previous_ids':invalid,'seed':20261027,'selection_mode':mode,'selected_count':len(selected),'sampling_fraction':len(selected)/len(band),'selected_ids':selected,'stratification':'label and category; all valid previous judgments included'}
        write(P/'judge-band-manifest.json',manifest)
        projection={'real_call_count':len(timings),'mean_seconds':mean,'p50_seconds':float(np.percentile(timings,50)),'p95_seconds':float(np.percentile(timings,95)),'p99_seconds':float(np.percentile(timings,99)),'remaining_band_count':remaining,'projected_full_remaining_seconds':projected,'selected_mode':mode,'selected_count':len(selected),'selected_fresh_count':len(selected)-len(old),'projected_selected_fresh_seconds':(len(selected)-len(old))*mean,'source':'judge-predictions.jsonl validated actual native calls'}
        write(P/'judge-time-projection.json',projection)
        (P/'judge-time-projection.md').write_text('# Judge time projection\n\n'+json.dumps(projection,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'band':len(band),'reused':len(old),'mean_seconds':mean,'selected':len(selected),'mode':mode,'projected_fresh_seconds':projection['projected_selected_fresh_seconds']}),flush=True)
        done={x['id']:x for x in read(P/'judge-results.jsonl') if x.get('id') in selected and x.get('actual_judge_output',{}).get('verdict') in ('safe','unsafe')}
        for rid,r in old.items():
            if rid not in done:append(P/'judge-results.jsonl',r);done[rid]=r
        with lock:shared.update(stage='judge-band-inference',items_done=len(done),items_total=len(selected))
        examples_by_id={x['id']:x for x in examples};detector=None
        for rid in selected:
            if rid in done:continue
            if detector is None:
                import torch
                torch.manual_seed(42);torch.set_num_threads(2);torch.set_num_interop_threads(1)
                detector=get_detector('llama_guard_3_1b');ok,detail=detector.available()
                if not ok:raise RuntimeError('Native judge unavailable: '+str(detail))
                real=detector.classify
                last={}
                def classify(text):
                    response=real(text);last.clear();last.update(response);return response
                detector.classify=classify
            b=lookup[rid];last.clear();t=time.perf_counter();decision=llama_guard_arbitration(examples_by_id[rid]['text'],threat_present=b['upper_band']);elapsed=time.perf_counter()-t
            if decision is None or last.get('verdict') not in ('safe','unsafe'):raise RuntimeError(f'Actual judge returned unavailable or unparseable verdict for {rid}')
            record={**b,'enabled_risk':decision[0],'source':decision[1],'actual_judge_output':dict(last),'judge_seconds':elapsed,'origin':'fresh-real-native-call','timestamp_utc':now()};append(P/'judge-results.jsonl',record);done[rid]=record
            with lock:shared['items_done']=len(done)
            print('judge',len(done),'/',len(selected),'last_seconds',elapsed,flush=True)
        judged=[done[rid] for rid in selected]
        # Replay the saved real verdicts through the actual repository arbiter.
        class RecordedDetector:
            def __init__(self,out):self.out=out
            def available(self):return True,'recorded real verdict'
            def classify(self,_):return self.out
        mismatch=[]
        for r in judged:
            with patch('core.detectors.get_detector',return_value=RecordedDetector(r['actual_judge_output'])):
                actual=llama_guard_arbitration(examples_by_id[r['id']]['text'],threat_present=r['upper_band'])
            if actual is None or actual[0]!=r['enabled_risk']:mismatch.append(r['id'])
        if mismatch:raise RuntimeError('Recorded verdict replay differs from actual repository arbiter: '+str(mismatch[:10]))
        output={'status':'PASS','label':'IN-SAMPLE','scope':mode,'band_count':len(band),'sample_size':len(judged),'sampling_fraction':len(judged)/len(band),'seed':20261027,'confidence_intervals':'1000 stratified bootstrap resamples by label/category','repo_logic':'core.risk.fuse_signals and llama_guard_arbitration; real saved verdicts replayed through arbiter','thresholds':{}}
        for title,cut in [('flagged_medium_or_high',1),('high_block',2)]:
            y=[r['label'] for r in judged];before=[{'LOW':0,'MEDIUM':1,'HIGH':2}[r['baseline_risk']]>=cut for r in judged];after=[{'LOW':0,'MEDIUM':1,'HIGH':2}[r['enabled_risk']]>=cut for r in judged]
            output['thresholds'][title]={'before':confusion(y,before),'after':confusion(y,after),'flips':{'false_negative_to_true_positive':sum(a==1 and not b and c for a,b,c in zip(y,before,after)),'false_positive_to_true_negative':sum(a==0 and b and not c for a,b,c in zip(y,before,after)),'true_positive_to_false_negative':sum(a==1 and b and not c for a,b,c in zip(y,before,after)),'true_negative_to_false_positive':sum(a==0 and not b and c for a,b,c in zip(y,before,after))},'bootstrap_95':boot(judged,cut)}
        times=[r['judge_seconds'] for r in judged];output['latency_seconds']=dict(zip(('p50','p95','p99'),map(float,np.percentile(times,[50,95,99]))));write(P/'judge-effect.json',output)
        lines=['# Judge effect â€” IN-SAMPLE','',f"Scope: {mode}; {len(judged)} of {len(band)} band examples. No full-corpus extrapolation for a sample.",'','| Decision cut | Stage | Precision | Recall | F1 | Accuracy | Benign FPR | TP | TN | FP | FN |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for cut,x in output['thresholds'].items():
            for stage in ('before','after'):
                a=x[stage];lines.append(f"| {cut} | {stage} | {a['precision']} | {a['recall']} | {a['f1']} | {a['accuracy']} | {a['benign_fpr']} | {a['tp']} | {a['tn']} | {a['fp']} | {a['fn']} |")
            lines+=['',f"{cut} flips: `{json.dumps(x['flips'])}`."]
        lines+=['',f"Judge latency seconds: `{json.dumps(output['latency_seconds'])}`. Full 95% intervals: judge-effect.json."]
        (P/'judge-effect.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    except BaseException as e:
        append(P/'judge-failures.jsonl',{'timestamp_utc':now(),'reason':repr(e),'traceback':traceback.format_exc()});write(P/'judge-effect.json',{'status':'INCOMPLETE','reason':repr(e),'label':'IN-SAMPLE'});print('Judge stage incomplete:',repr(e),flush=True)
    finally:
        stop.set();ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
if __name__=='__main__':main()

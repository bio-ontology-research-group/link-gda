"""Relay 400 captured score arrays through WS without staging their bytes there."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

from inventory import groups, plan, read, sha

ROOT = Path(__file__).resolve().parent


def main():
    lock=(ROOT/'relay.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (ROOT/'OUTPUT_TRANSFER_COMPLETE.json').exists()
    p,capture=plan(ROOT); digest=sha(ROOT/'plan.json')
    staged=read(ROOT/'METADATA_STAGED.json')
    assert staged['status']=='passed' and staged['plan_sha256']==digest
    assert staged['capture_sha256']==p['capture_sha256'] and staged['no_predictions_before_staging']
    options=['-o','BatchMode=yes','-o','ConnectTimeout=30','-o','ControlMaster=no',
             '-o','ControlPath=none','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6']
    markers=ROOT/'transferred'; markers.mkdir(exist_ok=True)

    def transfer(relative,names):
        marker=markers/(hashlib.sha256(relative.encode()).hexdigest()+'.json')
        if marker.exists():
            previous=read(marker)
            assert previous['plan_sha256']==digest and previous['relative']==relative
            assert previous['returncode']==0 and previous['files']==4
            return previous
        started=time.time()
        source='orix:'+p['source']+'/'+relative+'/{'+','.join(names)+'}'
        destination='ibex:'+p['destination']+'/'+relative+'/'
        for attempt in range(1,4):
            result=subprocess.run(['scp','-O','-3','-p','-q',*options,source,destination],
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            if result.returncode==0: break
            print(json.dumps(dict(relative=relative,attempt=attempt,returncode=result.returncode,
                                  error=result.stdout[-3000:])),flush=True)
            if attempt<3: time.sleep(2*attempt)
        if result.returncode: raise RuntimeError('Prediction relay failed: '+relative)
        record=dict(relative=relative,plan_sha256=digest,returncode=0,attempts=attempt,
                    files=len(names),seconds=time.time()-started)
        temporary=marker.with_suffix('.pending');temporary.write_text(json.dumps(record)+'\n');temporary.replace(marker)
        print(json.dumps(record),flush=True)
        return record

    grouped=groups(capture)
    with ThreadPoolExecutor(max_workers=2) as pool:
        records=[f.result() for f in as_completed([pool.submit(transfer,r,n) for r,n in grouped.items()])]
    assert len(records)==100 and {r['relative'] for r in records}==set(grouped)
    command='python3 '+p['operation_destination']+'/stage-metadata.py --verify-transfer'
    subprocess.run(['ssh',*options,'ibex',command],check=True)
    subprocess.run(['scp','-q',*options,'ibex:'+p['operation_destination']+'/DESTINATION_READY.json',
                    str(ROOT/'DESTINATION_READY.json')],check=True)
    ready=read(ROOT/'DESTINATION_READY.json')
    assert ready['status']=='inventory_ready' and ready['plan_sha256']==digest
    assert ready['capture_sha256']==p['capture_sha256'] and ready['test_files']==600
    assert ready['binary_files']==400
    proof=dict(status='transferred',runs=100,test_files=600,binary_files=400,
        plan_sha256=digest,capture_sha256=p['capture_sha256'],
        destination_ready_sha256=sha(ROOT/'DESTINATION_READY.json'),
        binary_bytes=sum(r['bytes'] for r in capture['binary_files'].values()),
        scope='Exact captured output paths, sizes and metadata transferred. Full input and score hashing and scientific comparison remain required on CPU.')
    (ROOT/'OUTPUT_TRANSFER_COMPLETE.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof),flush=True)


if __name__ == '__main__': main()

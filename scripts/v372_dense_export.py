"""Stream a density-only V362 candidate; preserve classification and unseen bytes."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import zipfile

import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
from scripts.v283_supervised_dinomaly_longtrain import sha256_file,top_mean
from scripts.v348_clean_memory_core import fuse_memory,encode_evidence
from scripts.v357_v356_release import decode_png
from scripts.v363_route_alignment import _expected,_check_csv,validate_png,validate_members
from scripts.v366_calibrated_seen_submission import ReplayAudit,BoundedWriter,_png,audit_archive,unchanged,make_index

VERSION='v372_dense_v362_candidate_v1'


def prepare_response(item,category_records):
    group,view=item['group'],item['view']
    if (not isinstance(group,str) or not re.fullmatch(r'[^/\\:\x00]+/S\d{4}',group)
            or not isinstance(view,str) or view not in '01234' or len(view)!=1):
        raise ValueError('Invalid group/view identity')
    category=group.split('/')[0]
    if category not in category_records:raise ValueError('Missing category calibration')
    record=category_records[category]
    for key in ('old_scale','new_scale'):
        if not math.isfinite(item[key]) or item[key]<=0 or item[key]!=record[key]:
            raise ValueError('Response scale differs from preceding source calibration')
    arrays={k:np.asarray(item[k],np.float32) for k in ('reference','sparse','dense')}
    if any(a.shape!=(448,448) or not np.isfinite(a).all() or a.min()<0 or a.max()>2.000001 for a in arrays.values()):
        raise ValueError('Invalid response shape/value')
    if np.any(arrays['dense']>arrays['sparse']+1e-6):
        raise ValueError('Dense nearest distance exceeds original sparse distance')
    old=fuse_memory(arrays['reference'],arrays['sparse'],scale=item['old_scale'])
    new=fuse_memory(arrays['reference'],arrays['dense'],scale=item['new_scale'])
    return {'old_pixels':encode_evidence(old),'new_pixels':encode_evidence(new),
            'reference_score':top_mean(arrays['reference'],.005)}


def build_candidate(baseline,output,index,responses,sources,*,provenance,category_records,final_check):
    baseline,output=Path(baseline),Path(output)
    staging=output.with_name(output.name+'.part')
    report_path=output.with_name(output.name+'.audit.json')
    report_staging=report_path.with_name(report_path.name+'.part')
    for p in (output,staging,report_path,report_staging):
        if p.exists() or p.is_symlink():raise FileExistsError(p)
    if str(baseline) not in sources:raise ValueError('Baseline must be source-bound')
    validated=make_index(index['groups'],sorted(index['seen']))
    if validated!=index:raise ValueError('Invalid complete B index')
    expected=_expected(index)
    expected_seen={(g,v) for g,views in index['groups'].items() if g.split('/')[0] in index['seen'] for v in views}
    unchanged(sources)
    output.parent.mkdir(parents=True,exist_ok=True)
    seen_hashes={};observed=set();changed_pixels=0;changed_masks=0
    with zipfile.ZipFile(baseline) as base:
        validate_members(base,expected)
        csv_bytes=base.read('submission.csv');_check_csv(csv_bytes,index)
        frozen={row['group_folder']:float(row['anomaly_score']) for row in csv.DictReader(io.StringIO(csv_bytes.decode()))
                if row['group_folder'].split('/')[0] in index['seen']}
        replay=ReplayAudit(frozen)
        with staging.open('xb') as handle:
            with zipfile.ZipFile(BoundedWriter(handle,900_000_000),'w',zipfile.ZIP_DEFLATED,compresslevel=6) as target:
                target.writestr('submission.csv',csv_bytes)
                for group,views in index['groups'].items():
                    if group.split('/')[0] in index['seen']:continue
                    for view in views:
                        name=f'predicted_masks/{group}/{view}_mask.png'
                        payload=base.read(name);validate_png(payload);target.writestr(name,payload)
                for item in responses:
                    key=item['group'],item['view']
                    if key not in expected_seen or key in observed:raise ValueError('Unknown/duplicate/misrouted seen response')
                    ready=prepare_response(item,category_records)
                    name=f'predicted_masks/{key[0]}/{key[1]}_mask.png'
                    original=base.read(name);validate_png(original)
                    old_pixels=decode_png(original)
                    replay.add(*key,ready['reference_score'],ready['old_pixels'],old_pixels)
                    changed=int(np.count_nonzero(ready['new_pixels']!=old_pixels))
                    changed_pixels+=changed;changed_masks+=int(changed>0)
                    payload=_png(ready['new_pixels']);validate_png(payload)
                    target.writestr(name,payload)
                    seen_hashes[name]=hashlib.sha256(payload).hexdigest();observed.add(key)
                    if len(observed)%500==0:print(f'v372_seen_written={len(observed)}/6250',flush=True)
                if observed!=expected_seen:raise ValueError('Incomplete seen response coverage')
                if set(category_records)!=set(index['seen']):raise ValueError('Incomplete calibration category coverage')
                old_replay=replay.finish()
            handle.flush();os.fsync(handle.fileno())
        digest=sha256_file(staging)
        with zipfile.ZipFile(staging) as candidate:
            audit=audit_archive(candidate,base,index,seen_hashes)
    final_check();unchanged(sources)
    if sha256_file(staging)!=digest:raise ValueError('Candidate changed during final verification')
    report={**audit,'version':VERSION,'status':'candidate_unscored','archive':str(output),'sha256':digest,
            'bytes':staging.stat().st_size,'replay':old_replay,'source_files_sha256':sources,'provenance':provenance,
            'category_records':category_records,'seen_response_sha256':seen_hashes,
            'changed_seen_pixels':changed_pixels,'changed_seen_masks':changed_masks,
            'new_training':False,'test_b_labels_accessed':False,'raw_evidence_persisted':False,
            'auto_upload':False,'recommended_replacement_for_online_best':False,
            'policy':'V362 numerator/CSV/unseen outputs frozen; add150normalpatches perbankimage only for seen categories and recalibrate dense memory denominator;90/10fusion unchanged',
            'limitations':['No online score for this package; offline evidence does not guarantee improvement.',
                           'Original normalTrain calibration groups, including known blank duplicate, retained for controlled comparison; not independent quality evidence.']}
    payload=json.dumps(report,indent=2,allow_nan=False).encode()
    if len(payload)>80_000_000:raise ValueError('Report exceeds bounded storage budget')
    with report_staging.open('xb') as h:h.write(payload);h.flush();os.fsync(h.fileno())
    os.link(report_staging,report_path);os.link(staging,output)
    report_staging.unlink();staging.unlink()
    return report

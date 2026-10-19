import importlib
import io
import csv
import json
import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pytest


def module():
    return importlib.import_module('scripts.v372_dense_export')


def response():
    return {'group':'part/S0001','view':'0','reference':np.full((448,448),.1,np.float32),
            'sparse':np.full((448,448),.1,np.float32),'dense':np.full((448,448),.06,np.float32),
            'old_scale':.5,'new_scale':.625}


def test_dense_response_keeps_sparse_for_old_replay():
    from scripts.v348_clean_memory_core import fuse_memory,encode_evidence
    item=response()
    result=module().prepare_response(item, {'part':{'old_scale':.5,'new_scale':.625}})
    np.testing.assert_array_equal(result['old_pixels'],encode_evidence(fuse_memory(item['reference'],item['sparse'],scale=.5)))
    np.testing.assert_array_equal(result['new_pixels'],encode_evidence(fuse_memory(item['reference'],item['dense'],scale=.625)))
    assert np.any(result['old_pixels']!=result['new_pixels'])


@pytest.mark.parametrize('bad',['nan','negative','shape','larger_dense','scale','missing_record','view'])
def test_export_rejects_invalid_or_mismatched_response(bad):
    item=response();records={'part':{'old_scale':.5,'new_scale':.625}}
    if bad=='nan':item['dense'][0,0]=np.nan
    if bad=='negative':item['dense'][0,0]=-.01
    if bad=='shape':item['reference']=item['reference'][:1]
    if bad=='larger_dense':item['dense'].fill(.2)
    if bad=='scale':item['new_scale']=.7
    if bad=='missing_record':records={}
    if bad=='view':item['view']='C1'
    with pytest.raises(ValueError):module().prepare_response(item,records)


@pytest.fixture(scope='module')
def index():
    seen={f'p{i:02}' for i in range(50)}
    return {'seen':frozenset(seen),'groups':{f'{c}/S{g:04}':tuple('01234')
      for c in sorted(seen|{f'u{i:02}' for i in range(50)}) for g in range(25 if c in seen else 20)}}


@pytest.fixture(scope='module')
def baseline(tmp_path_factory,index):
    from scripts.v366_calibrated_seen_submission import _png
    from scripts.v348_clean_memory_core import fuse_memory,encode_evidence
    from scripts.v330_raw_response import score_views
    root=tmp_path_factory.mktemp('v372_export');p=root/'baseline.zip'
    buf=io.StringIO(newline='');writer=csv.writer(buf);writer.writerow(['group_folder','anomaly_score'])
    vals={g:np.float32(.02+i*.00001) for i,g in enumerate(sorted(index['groups']))}
    memory=np.full((448,448),.1,np.float32)
    with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for g,v in vals.items():
            ref=np.full((448,448),v,np.float32)
            score=float(np.partition(ref.ravel(),ref.size-1004)[-1004:].mean())
            writer.writerow([g,score_views([score]*5)])
            payload=_png(encode_evidence(fuse_memory(ref,memory,scale=.5)))
            for view in '01234':z.writestr(f'predicted_masks/{g}/{view}_mask.png',payload)
        z.writestr('submission.csv',buf.getvalue().encode())
    return p,vals


def responses(index,vals):
    for g in sorted(index['groups']):
        if g.split('/')[0] not in index['seen']:continue
        ref=np.full((448,448),vals[g],np.float32)
        for view in '01234':
            yield {**response(),'group':g,'view':view,'reference':ref}


def test_complete_real_zip_old_replay_and_unseen_bytes(tmp_path,index,baseline):
    m=module();base,vals=baseline;out=tmp_path/'candidate.zip'
    records={c:{'old_scale':.5,'new_scale':.625} for c in index['seen']}
    report=m.build_candidate(base,out,index,responses(index,vals),{str(base):hashlib.sha256(base.read_bytes()).hexdigest()},
                             provenance={},category_records=records,final_check=lambda:None)
    assert report['status']=='candidate_unscored' and report['version']=='v372_dense_v362_candidate_v1'
    assert report['replay']['old_scale_masks_checked']==6250
    assert report['replay']['old_scale_mismatched_pixels']==0
    assert report['csv_byte_exact'] and report['unseen_byte_exact']
    assert report['groups']==2250 and report['masks']==11250
    assert not report['new_training'] and not report['auto_upload']
    assert not Path(str(out)+'.part').exists()
    with zipfile.ZipFile(out) as c,zipfile.ZipFile(base) as b:
        assert c.read('submission.csv')==b.read('submission.csv')
        assert all(c.read(n)==b.read(n) for n in c.namelist() if n.startswith('predicted_masks/u'))
    with pytest.raises(FileExistsError):
        m.build_candidate(base,out,index,iter(()),{},provenance={},category_records={},final_check=lambda:None)


@pytest.mark.parametrize('bad',['missing','duplicate','unknown','unseen'])
def test_partial_or_misrouted_stream_never_publishes(tmp_path,index,baseline,bad):
    base,vals=baseline;out=tmp_path/'failed.zip'
    item=next(responses(index,vals))
    if bad=='unknown':item['group']='bad/S0001'
    if bad=='unseen':item['group']='u00/S0000'
    stream=[item,item] if bad=='duplicate' else [] if bad=='missing' else [item]
    records={c:{'old_scale':.5,'new_scale':.625} for c in index['seen']}
    with pytest.raises(ValueError):
        module().build_candidate(base,out,index,iter(stream),{str(base):hashlib.sha256(base.read_bytes()).hexdigest()},
                                 provenance={},category_records=records,final_check=lambda:None)
    assert not out.exists()

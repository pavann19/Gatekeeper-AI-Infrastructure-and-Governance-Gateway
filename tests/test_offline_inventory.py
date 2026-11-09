import hashlib
import time

import pytest

from scripts.verify_offline_inventory import inspect_model


def test_only_pinned_matching_weight_bytes_pass(tmp_path):
    entry = {'model_id': 'synthetic/model', 'revision': 'a' * 40,
             'filename': 'model.safetensors', 'sha256': hashlib.sha256(b'synthetic-weight').hexdigest()}
    snapshot = tmp_path / 'models--synthetic--model' / 'snapshots' / entry['revision']
    snapshot.mkdir(parents=True)
    path = snapshot / entry['filename']
    path.write_bytes(b'synthetic-weight')
    result = inspect_model(entry, tmp_path, time.monotonic() + 5, [1000])
    assert result['status'] == 'PASS'
    path.write_bytes(b'tampered')
    assert inspect_model(entry, tmp_path, time.monotonic() + 5, [1000])['status'] == 'FAIL'


@pytest.mark.parametrize('entry', [dict(model_id='../escape', revision='a' * 40),
                                  dict(model_id='valid/model', revision='main'),
                                  dict(model_id='valid/model', revision='a' * 40, filename='../escape')])
def test_unpinned_or_traversing_manifest_rejected(tmp_path, entry):
    with pytest.raises(ValueError):
        inspect_model(entry, tmp_path, time.monotonic() + 5, [1000])

import json
import yaml
from pathlib import Path

def deep_merge(base, override):
    result = dict(base)
    for k,v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result

root = Path(__file__).resolve().parent.parent
outputs = root / 'outputs'
frags = {
    'training': root / 'configs' / 'training' / 'base.yaml',
    'preproc': root / 'configs' / 'data' / 'preprocessing.yaml',
    'splits': root / 'configs' / 'data' / 'splits.yaml',
    'augmentation': root / 'configs' / 'data' / 'augmentation.yaml',
}
stage_map = {
    'transfer_5class': root / 'configs' / 'stages' / 's3_transfer.yaml',
    'pretrain_treatment_8class': root / 'configs' / 'stages' / 's2_treatment.yaml',
    'pretrain_30class': root / 'configs' / 'stages' / 's1_isolate.yaml',
}

for d in ['sanity_cnn','sanity_hybrid','sanity_resnet1d','sanity_transformer']:
    exp = outputs / d
    j = exp / 'config.json'
    if not j.exists():
        print('skip', d, 'no config.json')
        continue
    data = json.loads(j.read_text())
    stage = data.get('task', {}).get('stage', 'transfer_5class')
    stage_fragment = stage_map.get(stage)
    model_name = data.get('model', {}).get('name', 'cnn')
    model_fragment = root / 'configs' / 'model' / f"{model_name}.yaml"

    merged = {}
    # load stage fragment
    if stage_fragment and stage_fragment.exists():
        merged = deep_merge(merged, yaml.safe_load(stage_fragment.read_text()) or {})
    # model fragment
    if model_fragment.exists():
        merged = deep_merge(merged, yaml.safe_load(model_fragment.read_text()) or {})
    # training base
    if frags['training'].exists():
        merged = deep_merge(merged, yaml.safe_load(frags['training'].read_text()) or {})
    # preprocessing and splits
    if frags['preproc'].exists():
        merged = deep_merge(merged, yaml.safe_load(frags['preproc'].read_text()) or {})
    if frags['splits'].exists():
        merged = deep_merge(merged, yaml.safe_load(frags['splits'].read_text()) or {})
    # augmentation defaults (training-only; safe to include)
    if frags.get('augmentation') and frags['augmentation'].exists():
        merged = deep_merge(merged, yaml.safe_load(frags['augmentation'].read_text()) or {})

    # finally overlay experiment config (json)
    merged = deep_merge(merged, data)

    outp = exp / 'config.yaml'
    outp.write_text(yaml.safe_dump(merged, sort_keys=False))
    print('Wrote merged config to', outp)

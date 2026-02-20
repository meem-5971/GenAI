import sys
import json
import shutil
import h5py


def remove_time_major(obj):
    if isinstance(obj, dict):
        if 'time_major' in obj:
            del obj['time_major']
        for k, v in list(obj.items()):
            remove_time_major(v)
    elif isinstance(obj, list):
        for item in obj:
            remove_time_major(item)


def fix_h5(in_path, out_path):
    shutil.copy2(in_path, out_path)
    with h5py.File(out_path, 'r+') as f:
        if 'model_config' not in f.attrs:
            print('No model_config attribute found in', out_path)
            return 1
        raw = f.attrs['model_config']
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')
        config = json.loads(raw)
        remove_time_major(config)
        new_raw = json.dumps(config)
        f.attrs.modify('model_config', new_raw)
    print('Wrote fixed file to', out_path)
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python fix_h5_remove_time_major.py <input.h5> [output.h5]')
        sys.exit(2)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else inp.replace('.h5', '_fixed.h5')
    sys.exit(fix_h5(inp, out))

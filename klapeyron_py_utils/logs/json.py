import json
from klapeyron_py_utils.logs.filesystem import assert_filepath


def json_dump(json_filepath, json_obj, indent=4):
    with open(json_filepath, 'w', encoding='utf8') as f:
        ser = json.dumps(json_obj, indent=indent)
        f.write(ser)


def json_load(json_filepath):
    assert_filepath(json_filepath)
    with open(json_filepath, encoding='utf8') as f:
        json_obj = json.load(f)
    return json_obj

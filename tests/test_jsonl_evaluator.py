"""Exercise the actual evaluator reader without importing its inference runtime."""
import ast
import json
from pathlib import Path


def test_physical_jsonl_lines_preserve_embedded_unicode_separators(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/compute_judge_effect_corrected.py'
    source = ast.parse(script.read_text(encoding='utf-8'))
    reader = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'read')
    namespace = {'json': json}
    exec(compile(ast.Module(body=[reader], type_ignores=[]), str(script), 'exec'), namespace)
    rows = [{'text': 'a\u2028b\u2029c\u0085d'}, {'text': 'next'}]
    path = tmp_path / 'synthetic.jsonl'
    path.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + '\n', encoding='utf-8')
    assert namespace['read'](path) == rows

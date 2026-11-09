"""Execute login routing/DOM logic with a synthetic browser surface."""
from pathlib import Path
import re
import shutil
import subprocess

import pytest


def test_login_redirects_stay_in_ui_and_identity_is_written_as_text():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for JavaScript execution checks')
    page = (Path(__file__).resolve().parents[1] / 'ui/login/index.html').read_text(encoding='utf-8')
    script = re.findall(r'<script>([\s\S]*?)</script>', page)[0]
    harness = r'''
const vm = require('node:vm');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const code = fs.readFileSync(0, 'utf8');
let submit;
const nodes = {};
for (const id of ['loginForm','banner','apiKey','signInBtn','identityPreview']) {
  nodes[id] = {textContent:'', className:'', disabled:false, value:'synthetic-key',
               addEventListener(type, handler) { submit = handler; }};
}
Object.defineProperty(nodes.identityPreview, 'innerHTML', {
  set() { throw new Error('Identity must never enter an HTML sink'); }
});
const identity = {key_id:'<img src=x onerror=attack()>', capability:'GENERAL', tenant:'<script>attack()</script>'};
const context = {URL, URLSearchParams, document:{getElementById:id=>nodes[id]},
  window:{location:{origin:'https://gateway.test',search:'',replace() {}}},
  gkGetKey:()=>null, gkRequireAuth:async()=>null, gkSetSession() {},
  fetch:async()=>({status:200,ok:true,json:async()=>identity}), setTimeout:fn=>fn()};
vm.createContext(context);
vm.runInContext(code, context);
for (const destination of ['https://evil.test/ui/x','//evil.test/ui/x','javascript:attack()',
                            'https://gateway.test@evil.test/ui/x','/api/v1/cache/flush']) {
  context.window.location.search = '?next='+encodeURIComponent(destination);
  assert.equal(vm.runInContext('nextDestination({})',context), '/ui/activity/index.html');
}
context.window.location.search='?next='+encodeURIComponent('/ui/review/index.html?x=1#item');
assert.equal(vm.runInContext('nextDestination({})',context), '/ui/review/index.html?x=1#item');
(async()=> {
  await submit({preventDefault() {}});
  assert(nodes.identityPreview.textContent.includes(identity.key_id));
  assert(nodes.identityPreview.textContent.includes(identity.tenant));
  assert.equal(nodes.identityPreview.className,'gk-identity-preview visible');
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    result = subprocess.run([node, '-e', harness], input=script, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr

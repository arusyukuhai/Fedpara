"""Manual Chromium GUI test: v71 picker completion must not modify a later selection."""
from pathlib import Path
from tempfile import NamedTemporaryFile
from playwright.sync_api import sync_playwright
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from patchlib.webview_gui import compose_html

with NamedTemporaryFile(mode='w',suffix='.html',encoding='utf8',delete=False) as file:
    path=file.name; file.write(compose_html())
try:
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1480,'height':900})
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.set_content(Path(path).read_text(),wait_until='domcontentloaded')
        page.evaluate('''
        const sha='a'.repeat(40);
        window.pendingSft=null;
        const make=(data)=>async()=>data;
        window.pywebview={api:new Proxy({
          defaults:make({room:'r',model:'sample/model',revision:'main',task:'causal-lm',ws:'ws://127.0.0.1:8080/ws',kind:'loha',sources:[]}),
          discover_hub_assets:async({role})=>({candidates:[{id:'sample/'+role,kind:['pt','sft','ppo'].includes(role)?'dataset':'model',downloads:24,revision:sha,reason:'candidate'}]}),
          pin_mixed_dataset:({repo})=>repo==='sample/sft'?new Promise(resolve=>window.pendingSft=resolve):Promise.resolve({repo,revision:sha,variants:[{config:'default',split:'train'}]}),
          inspect_reward_model:async({repo})=>({id:repo,revision:sha})
        },{get(obj,key){if(key==='then')return undefined;return obj[key]||make({})}})};
        ''')
        page.wait_for_timeout(200)
        page.locator('[data-tab="training"]').click()
        page.locator('#mixEnabled').select_option('on')
        page.locator('#mixSftHfSearch').click()
        page.locator('.hub-candidate').first.wait_for()
        page.locator('.hub-candidate').first.click()
        page.wait_for_function('window.pendingSft !== null')
        # User cancels an in-flight selection, then changes the browser role.
        page.locator('#hubPickerClose').click()
        page.locator('#mixCriticKind').select_option('hf')
        page.locator('#mixCriticSearch').click()
        page.locator('.hub-candidate').first.wait_for()
        page.evaluate("window.pendingSft({repo:'sample/sft',revision:'a'.repeat(40),variants:[{config:'default',split:'train'}]})")
        page.wait_for_timeout(80)
        assert page.locator('#mixSftHfRepo').input_value()!='sample/sft'
        assert page.locator('#hubPicker').is_visible()
        assert page.locator('#hubPickerTitle').inner_text().startswith('CRITIC')
        page.locator('.hub-candidate').first.click()
        assert page.locator('#mixCriticHfRepo').input_value()=='sample/critic'
        assert not errors,errors
        print('BROWSER PASS: cancelled in-flight SFT selection cannot overwrite GUI; Critic picker remains functional')
        browser.close()
finally:
    Path(path).unlink(missing_ok=True)

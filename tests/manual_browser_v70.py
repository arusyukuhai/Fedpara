from playwright.sync_api import sync_playwright
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from patchlib.webview_gui import compose_html
with NamedTemporaryFile(mode='w',suffix='.html',encoding='utf-8',delete=False) as out:
    p=out.name
    out.write(compose_html())
with sync_playwright() as pw:
    browser=pw.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1460,'height':900},device_scale_factor=1)
    stub_script = '''
    const make=(data)=>async()=>data;
    const sha='a'.repeat(40);
    const stub = new Proxy({
        defaults:make({room:'r',model:'sample/model',revision:'main',task:'causal-lm',ws:'ws://127.0.0.1:8080/ws',kind:'loha',sources:[]}),
        discover_hub_assets:async({role,query,scope})=>({candidates:[{id:'sample/'+role+'-candidate',kind:['pt','sft','ppo'].includes(role)?'dataset':'model',downloads:2024,revision:sha,reason:'Metadata hint: '+role+'; verify schema',gated:false}],role,query,scope}),
        pin_mixed_dataset:async({repo})=>({repo,revision:sha,variants:[{config:'default',split:'train'}]}),
        inspect_reward_model:async({repo})=>({id:repo,revision:sha,pipeline:'text-classification'}),
    },{get(obj,key){if(key==='then')return undefined;return obj[key]||make({})}});
    window.pywebview={api:stub};
    '''
    errors=[]
    page.on('pageerror',lambda e: errors.append(str(e)))
    page.set_content(open(p).read(),wait_until='domcontentloaded')
    page.evaluate(stub_script)
    page.wait_for_timeout(600)
    page.locator('[data-tab="training"]').click()
    page.locator('#mixEnabled').select_option('on')
    page.locator('#mixSftHfSearch').click()
    page.locator('.hub-candidate').first.wait_for(timeout=7000)
    assert page.locator('#hubPickerTitle').inner_text().startswith('SFT')
    assert page.locator('.hub-candidate').count()==1
    # Optional: add a screenshot here while debugging styles
    page.locator('.hub-candidate').first.click()
    assert page.locator('#mixSftHfRepo').input_value()=='sample/sft-candidate'
    assert page.locator('#mixSftHfRevision').input_value()=='a'*40
    assert page.locator('#mixSftHfConfig').input_value()=='default'
    assert page.locator('#hubPicker').is_hidden()
    page.locator('#mixCriticKind').select_option('hf')
    page.locator('#mixCriticSearch').click()
    page.locator('.hub-candidate').first.wait_for(timeout=7000)
    page.locator('.hub-candidate').first.click()
    assert page.locator('#mixCriticHfRepo').input_value()=='sample/critic-candidate'
    assert not errors, errors
    print('BROWSER PASS: visual picker, HF SFT pin, HF Critic pin; no JS errors')
    browser.close()
Path(p).unlink(missing_ok=True)

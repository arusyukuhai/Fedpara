"""Kagami: self-contained desktop style, legacy diagnostic screens preserved."""
from pathlib import Path

from patchlib.webview_gui import compose_html

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / 'web'
CDN = 'https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4.1.10'


def test_desktop_runtime_cdn_theme_and_bridge():
    css = (WEB / 'tailwind.cdn.css').read_text(encoding='utf-8')
    page = compose_html()
    # Desktop no longer executes CDN Tailwind: only bundled CSS/JS runs in
    # the privileged WebView, under a restrictive CSP.
    assert CDN not in page
    assert '<style>' in page
    assert '--accent:#10b981' in css
    assert '.model-card' in css
    assert 'height:100dvh' in css
    assert "connect-src 'none'" in page
    assert '<link rel="stylesheet"' not in page
    assert 'desktop_webview.css' not in page
    assert '<body>' in page
    for id in ('start', 'stop', 'disconnect', 'room', 'datasetQuery', 'runEval',
               'monitorLoss', 'monitorSpeed', 'monitorLossPlot', 'monitorAccuracyPlot',
               'trainReinitEvery', 'trainWideRank', 'trainJournalSync'):
        assert f'id="{id}"' in page


def test_all_other_html_views_are_tailwind_only():
    for filename in ('index.html','dashboard.html','patches.html'):
        page = (WEB / filename).read_text(encoding='utf-8')
        assert CDN in page
        assert '<style' not in page
        assert '<body class=' in page
        assert 'bg-[#080e0b]' in page
        assert f'<script src="{CDN}"' in page


def test_no_legacy_css_or_build_tool_required():
    for name in ('desktop_webview.css','tailwind.input.css'):
        assert not (WEB / name).exists()
    assert not (ROOT/'scripts/build_tailwind.mjs').exists()
    assert not (ROOT/'package.json').exists()


def test_diagnostic_uses_tailwind():
    from patchlib.webview_diagnostic import HTML
    assert CDN in HTML
    assert '<style>' not in HTML
    assert 'bg-[#080e0b]' in HTML

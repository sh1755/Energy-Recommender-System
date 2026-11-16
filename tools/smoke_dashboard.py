"""Headless Streamlit application smoke test; no external server required."""
from streamlit.testing.v1 import AppTest
app=AppTest.from_file('dashboard.py',default_timeout=30).run()
assert not app.exception,str(app.exception)
print('PASS: dashboard renders without exceptions.')
app.selectbox[0].select('H0001').run()
assert not app.exception,str(app.exception)
print('PASS: household selector rerenders successfully.')

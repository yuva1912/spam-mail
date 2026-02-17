import importlib.util, sys, os
p = os.path.join(os.getcwd(), 'backend', 'app.py')
spec = importlib.util.spec_from_file_location('app_module', p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
app = getattr(mod, 'app', None)
if app is None:
    print('ERROR: `app` object not found in backend/app.py')
    sys.exit(2)

with app.test_client() as c:
    r = c.get('/')
    print('STATUS', r.status_code)
    body = r.get_data(as_text=True)
    print('LEN', len(body))
    print('\n---BEGIN BODY---\n')
    print(body)
    print('\n---END BODY---\n')

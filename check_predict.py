import importlib.util, os
spec = importlib.util.spec_from_file_location('appmod', os.path.join(os.getcwd(),'backend','app.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
app = getattr(mod, 'app')
with app.test_client() as c:
    r = c.post('/predict', data={'email': 'Congratulations, you won a prize! Click here.'})
    print('STATUS', r.status_code)
    b = r.get_data(as_text=True)
    print('LEN', len(b))
    print('\n---BODY---\n')
    print(b)

import importlib.util, os, sys
p = os.path.join(os.getcwd(), 'backend', 'app.py')
spec = importlib.util.spec_from_file_location('app_module', p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
app = getattr(mod, 'app', None)
if app is None:
    print('ERROR: app not found')
    sys.exit(1)
print('template_folder=', app.template_folder)
loader = app.jinja_loader
print('loader type:', type(loader))
try:
    templates = loader.list_templates()
    print('templates:', templates)
except Exception as e:
    print('list_templates error:', e)
# try to load template source
try:
    t = app.jinja_env.get_template('index.html')
    s = t.render(model_missing=True)
    print('\n---RENDERED---\n')
    print(s[:2000])
except Exception as e:
    print('render error:', e)

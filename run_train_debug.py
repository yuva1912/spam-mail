import runpy, traceback
try:
    runpy.run_path('train_model.py', run_name='__main__')
except Exception:
    traceback.print_exc()
    raise
print('done')

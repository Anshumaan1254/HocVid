import sys
import traceback

try:
    import run_inference_bacha
    run_inference_bacha.main()
except Exception:
    with open('infer_err.log', 'w') as f:
        traceback.print_exc(file=f)

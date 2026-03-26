import sys
import traceback

sys.argv = ['train.py', '--dder_weights', 'DDER/dder.pt', '--batch_size', '2']

try:
    import train
    train.main()
except Exception:
    with open('out.log', 'w') as f:
        traceback.print_exc(file=f)
    traceback.print_exc()
    print("Exception captured in out.log")

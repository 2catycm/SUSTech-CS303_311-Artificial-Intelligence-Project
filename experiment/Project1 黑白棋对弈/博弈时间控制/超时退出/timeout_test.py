# timeout_test1.py
from tqdm import trange
import sys
import time
import timeout_decorator


# @timeout_decorator.timeout(3, use_signals=False)
@timeout_decorator.timeout(3)
def test():
    for i in trange(3):
        time.sleep(1)
        print('>>> {} seconds passed.'.format(i + 1))
    return 0


if __name__ == '__main__':
    try:
        test()
    except TimeoutError as e:
        print('Timeout Error Catched!')
        print(e)
    print("Timeout Task Ended!")

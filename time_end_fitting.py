import os
import sys
import random
import time


# The main entry point for this module
def main():
    argu = sys.argv[1:]
    ep_remain = int(argu[0])
    sec_per_ep = int(argu[1])
    curh = time.localtime( time.time() ).tm_hour
    curmi = time.localtime( time.time() ).tm_min
    sec_rem = ep_remain*sec_per_ep
    mrem = int(sec_rem/60)
    mrem += curmi
    hrem = int(mrem/60)
    mrem = mrem%60
    print(f"End of the fitting at {curh+hrem}h{mrem}")




# Tell python to run main method
if __name__ == '__main__':
    main()

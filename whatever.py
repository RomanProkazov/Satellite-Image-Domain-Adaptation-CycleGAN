import os
import config

synth_dir = config.SYNTH_DIR_SPEED
sl_dir = config.SL_DIR_SPEED
lb_dir = config.LB_DIR_SPEED


print(len(os.listdir(synth_dir)))
print(len(os.listdir(sl_dir)))
print(len(os.listdir(lb_dir)))


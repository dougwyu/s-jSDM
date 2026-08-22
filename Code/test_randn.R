Sys.setenv(RETICULATE_PYTHON = "/Users/douglasyu/src/s-jSDM/work/port-feasibility/.pixi/envs/default/bin/python")
reticulate::py_run_string("
import torch, sys
print('torch', torch.__version__, 'from', torch.__file__)
try:
    t = torch.randn((2,3,4), device=torch.device('cpu'), dtype=torch.float32)
    print('eager tuple ok', tuple(t.shape))
except TypeError as e:
    print('eager FAIL:', e)
")

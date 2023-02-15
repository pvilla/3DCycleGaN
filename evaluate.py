
from models.valfullfile import evaluate

# Attention! 10 * SR pixels will be cropped from the border of the enhanced volume

## uncomment to evaluate the 2x network for enhancement of the 1ms, 800nm dataset
dfile = '/data/staff/tomograms/users/johannes/data/githubdata/T700-T-21_GF_0p8um_1ms_1.h5'
mfile = '/data/staff/tomograms/users/johannes/3DCycleGaN-main/results/2x_1ms_3/save/2x_1ms_3_012000.pt'
evaluate(datafile = dfile, modelfile = mfile, SR = 2, ev_name = '2x_1ms')

## uncomment to evaluate the 4x network for enhancement of the 3ms, 1600nm dataset
# dfile = 
# mfile = 
# evaluate(datafile = dfile, modelfile = mfile, SR = 4, ev_name = '4x_3ms')

## uncomment to evaluate the 4x network for enhancement of the 1ms, 1600nm dataset
# dfile = 
# mfile = 
# evaluate(datafile = dfile, modelfile = mfile, SR = 2, ev_name = '4x_1ms')
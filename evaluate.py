
from models.valfullfile import evaluate

# Attention! 10 * SR pixels will be cropped from the border of the enhanced volume

## uncomment to evaluate the 2x network for enhancement of the 1ms, 800nm dataset
dfile = 
mfile = 
evaluate(datafile = dfile, modelfile = mfile, SR = 2, ev_name = '2x_1ms')

## uncomment to evaluate the 4x network for enhancement of the 3ms, 1600nm dataset
# dfile = 
# mfile = 
# evaluate(datafile = dfile, modelfile = mfile, SR = 4, ev_name = '4x_3ms')

## uncomment to evaluate the 4x network for enhancement of the 1ms, 1600nm dataset
# dfile = 
# mfile = 
# evaluate(datafile = dfile, modelfile = mfile, SR = 2, ev_name = '4x_1ms')
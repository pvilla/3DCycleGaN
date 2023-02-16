
from models.valfullfile import evaluate

# Attention! 10 * SR pixels will be cropped from the border of the enhanced volume

## uncomment to evaluate the 2x model for enhancement of the 1ms, 800nm dataset
dfile = 'data/T700-T-21_GF_0p8um_1ms_1.h5'
mfile = 'data/pretrained_models/2x_1ms_trained.pt'
evaluate(datafile = dfile, modelfile = mfile, SR = 2, ev_name = '2x_1ms')

## uncomment to evaluate the 4x model for enhancement of the 3ms, 1600nm dataset
# dfile = 'data/T700-T-21_GF_1p6um_3ms_1.h5'
# mfile = 'data/pretrained_models/4x_3ms_trained.pt'
# evaluate(datafile = dfile, modelfile = mfile, SR = 4, ev_name = '4x_3ms')

## uncomment to evaluate the 4x model for enhancement of the 0.5ms, 1600nm dataset
# dfile = 'data/T700-T-21_GF_1p6um_0p5ms_1.h5'
# mfile = 'data/pretrained_models/4x_0p5ms_trained.pt'
# evaluate(datafile = dfile, modelfile = mfile, SR = 4, ev_name = '4x_0p5ms')

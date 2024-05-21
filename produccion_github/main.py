from utils import Utils
from models import ModelsRam



#from GridSearchCV import Models

import warnings
warnings.simplefilter("ignore")

utils = Utils()
models = ModelsRam()
ds = utils.load_from_csv('./in/DatasetFinal-gitgub.csv')



X, y = utils.features_target(ds, ['Toxicos'],['Toxicos'])

models.grid_training(X,y)

from utils import Utils
from models import Models
from models import ModelsRam
from sklearn.preprocessing import StandardScaler  # Normalizar los datos

#from GridSearchCV import Models

import warnings
warnings.simplefilter("ignore")

utils = Utils()
models = ModelsRam()
ds = utils.load_from_csv('./in/DATASET.csv')
#dt_features = StandardScaler().fit_transform(ds)  # Normalizamnos los datos

X, y  = utils.features_target(ds, [ "SEVERIDAD"],['SEVERIDAD'])

models.grid_training(X,y)

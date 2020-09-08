"""
Modeule with scoring functions that take molecular CDDD embeddings (positions of the particles in the swarm) as input.
"""
from scipy.spatial.distance import cdist
from mso.data import load_predict_model_from_pkl
from functools import wraps
import joblib
import os

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
# from tensorflow.keras.layers import Multiply, Concatenate, Add
from tensorflow.keras.models import Model

bace_score_512 = load_predict_model_from_pkl('bace_classifier.pkl')
egfr_score_512 = load_predict_model_from_pkl('egfr_classifier.pkl')
_dir = os.path.dirname(__file__)

logd_model = joblib.load(os.path.join(_dir, 'models/logD_xgb.pkl'))
ames_model = joblib.load(os.path.join(_dir, 'models/ames_xgb.pkl'))
logs_model = joblib.load(os.path.join(_dir, 'models/logs_xgb.pkl'))
hERG_model = joblib.load(os.path.join(_dir, 'models/herg_xgb.pkl'))
liverTox_model = joblib.load(os.path.join(_dir, 'models/hepatoxicity_xgb.pkl'))
LD50_model = joblib.load(os.path.join(_dir, 'models/ld50_xgb.pkl'))

################ DNN Model ################
caco2_weights = os.path.join(_dir, 'models/Caco-2/Caco-2.ckpt')
mdck_weights = os.path.join(_dir, 'models/MDCk/MDCK.ckpt')
ppb_weights = os.path.join(_dir, 'models/PPB/ppb.ckpt')

def approximate_res(func):
    """
    Decorator function that accurate to three decimal places
    
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return round(float(func(*args, **kwargs)), 3)
    return wrapper


def distance_score(emb, target, metric="cosine"):
    """
    Function tha calculates the distance between an input molecular embedding and a target molecular embedding.
    :param x: input molecular embedding
    :param target: target molecular embedding
    :param metric: The metric used by scipy.spatial.distance.cdist to compute the distance in space.
    :return: The distance between input and target.
    """
    score = cdist(emb, target, metric).flatten()
    return score


# reg score
# @approximate_res
def logD_score(emb):
    """
    Measuring the logD of molecule
    
    Parameters
    ----------
    emb : numpy.ndarray, size (x,512)
        the cddd descriptor of molecule

    Returns
    -------
    logD : float
        logD score
    """
    logD = logd_model.predict(emb)
    return logD

def ames_score(emb):
    """
    Measure the ames positive probability of molecule,
    more close to 1, more "toxic"

    Parameters
    ----------
    emb : numpy.ndarray, size (x,512)
        the cddd descriptor of molecule

    Returns
    -------
    ames : float
        ames score

    """
    ames = ames_model.predict_proba(emb)[:,1]
    return ames

def caco2_score(emb):
    """
    """
    cdddDescri = Input(shape=(512,), name="cdddDescriptor")
    x = Dense(1024, activation='relu')(cdddDescri)
    x = Dropout(0.5)(x)
    output = Dense(1)(x)
    
    model = Model(inputs=cdddDescri, outputs=output)
    model.load_weights(caco2_weights)
    
    caco2 = model.predict(emb)
    return caco2.flatten()

def mdck_score(emb):
    """
    """
    cdddDescri = Input(shape=(512,), name="cdddDescriptor")
    x = Dense(512, activation='relu')(cdddDescri)
    x = Dropout(0.25)(x)
    output = Dense(1)(x)
    
    model = Model(inputs=cdddDescri, outputs=output)
    model.load_weights(mdck_weights)

    mdck = model.predict(emb)
    return mdck.flatten()

def ppb_score(emb):
    """
    """
    cdddDescri = Input(shape=(512,), name="cdddDescriptor")
    x = Dense(128, activation='relu')(cdddDescri)
    x = Dropout(0.5)(x)
    output = Dense(1)(x)

    model = Model(inputs=cdddDescri, outputs=output)
    model.load_weights(caco2_weights)

    ppb = model.predict(emb)
    return ppb.flatten()

def logS_score(emb):
    """
    """
    logs = logs_model.predict(emb)
    return logs

def hERG_score(emb):
    """
    """
    herg = hERG_model.predict_proba(emb)[:,1]
    return herg

def hepatoxicity_score(emb):
    """
    """
    liverTox = liverTox_model.predict_proba(emb)[:,1]
    return liverTox

def LD50_score(emb):
    """
    """
    ld50 = LD50_model.predict_proba(emb)[:,1]
    return ld50
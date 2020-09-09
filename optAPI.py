import os
# import numpy as np
# import sys
# import argparse
# import pandas as pd
# import tensorflow as tf
from new_cddd.inference import InferenceModel
# from new_cddd.preprocessing import preprocess_smiles, randomize_smile
from new_cddd.hyperparameters import DEFAULT_DATA_DIR
from mso.optimizer import BasePSOptimizer
from mso.objectives.scoring import ScoringFunction
from mso.objectives.mol_functions import qed_score, logP_score, substructure_match_score
from mso.objectives.emb_functions import logD_score, logS_score
from mso.objectives.emb_functions import caco2_score, mdck_score, ppb_score
from mso.objectives.emb_functions import ames_score, hERG_score, hepatoxicity_score, LD50_score
from rdkit import Chem
from functools import partial

_default_model_dir = os.path.join(DEFAULT_DATA_DIR, 'default_model')
model_dir = _default_model_dir

infer_model = InferenceModel(
    model_dir=model_dir,
    use_gpu=False,
    batch_size=128,
    cpu_threads=4,)


class PropOptimizer:

    def __init__(self, init_smiles,
                 num_part=200,
                 num_swarms=1,
                 prop_dic={'qed': {'range': [0, 1]},
                           'logD': {'range': [-3, 8], 'allow_exceed': False}}):

        self.init_smiles = init_smiles
        self.prop_dic = prop_dic
        self.props = list(self.prop_dic.keys())
        self.num_part = num_part
        self.num_swarms = num_swarms
        self.infer_model = infer_model
        self.scoring_functions = list(self._build_scoring_functions())
        self.opt = self._build_optimizer()

        # self._build_optimizer()

    def _build_scoring_functions(self):
        func_list = {
            'QED': qed_score,
            'logD': logD_score,
            'AMES': ames_score,
            'Caco-2': caco2_score,
            'MDCK': mdck_score,
            'PPB': ppb_score,
            'logP': logP_score,
            'logS': logS_score,
            'hERG': hERG_score,
            'hepatoxicity': hepatoxicity_score,
            'LD50': LD50_score,
            'match': substructure_match_score,
            'unmatch': substructure_match_score,
        }

        for prop_name in self.prop_dic.keys():
            func = func_list[prop_name]
            is_mol_func = prop_name in ['QED', 'logP', 'match', 'unmatch']
            partial_func = prop_name in ['match', 'unmatch']

            if partial_func:
                func = partial(
                    func, query=Chem.MolFromSmarts(self.prop_dic[prop_name]['smiles'])
                    )

            _range = self.prop_dic[prop_name].get('range', [0, 1])
            if self.prop_dic[prop_name].get('ascending', True):
                desirability = [{"x": _range[0], "y": 0.0},
                                {"x": _range[1], "y": 1.0}]
            else:
                desirability = [{"x": _range[1], "y": 0.0},
                                {"x": _range[0], "y": 1.0}]

            allow_exceed = self.prop_dic[prop_name].get('allow_exceed', False)

            yield ScoringFunction(func=func, name=prop_name,
                                  desirability=desirability,
                                  is_mol_func=is_mol_func,
                                  allow_exceed=allow_exceed)

    def _build_optimizer(self):
        opt = BasePSOptimizer.from_query(
            init_smiles=self.init_smiles,
            num_part=self.num_part,
            num_swarms=self.num_swarms,
            inference_model=self.infer_model,
            scoring_functions=self.scoring_functions)
        return opt


if '__main__' == __name__:

    """PARAMETERS

    :param init_smiles: A List of SMILES which each define the molecule which acts as starting
            point of each swarm in the optimization.
    :param num_part: Number of particles in each swarm.
    :param num_swarms: Number of individual swarm to be optimized.

    :param prop_dic: Dictionary of property condition to be optimized
    """

    opt = PropOptimizer(
        init_smiles='Nc1ccc(O)c(C(=O)O)c1',
        num_part=200,
        num_swarms=1,
        prop_dic={
            "QED": {"range": [0, 1]},
            # "logD": {"range": [-3, 8]},
            # "AMES": {"range": [0, 1], "ascending": False},
            # "Caco-2": {"range": [-8, -4]},
            # "MDCK": {"range": [-8, -3]},
            # "PPB": {"range": [0, 1]},
            # "logP": {"range": [-5, 9]},
            # "logS": {"range": [-2, 14]},
            # "hERG": {"range": [0, 1], "ascending": False},
            # "hepatoxicity": {"range": [0, 1], "ascending": False},
            # "LD50": {"range": [0, 1], "ascending": False}
            "match": {"smiles": "c1ccccc1", "allow_exceed": True},
            # "unmatch": {"smiles": "c1ccccc1", "ascending":False, "allow_exceed": True},
        }
    )

    opt.opt.run(5, 5)

    best_sol = opt.opt.best_solutions

    print(type(best_sol))
    print(best_sol)

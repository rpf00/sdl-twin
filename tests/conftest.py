import copy

import pytest

from sdl.config import Reagent, load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def config_with_bleach(config):
    cfg = copy.deepcopy(config)
    cfg.reagents["BLEACH"] = Reagent("BLEACH", "NaOCl", {}, ("hypochlorite",))
    cfg.reagents["OX"] = Reagent("OX", "oxidiser, not banned", {}, ("hypochlorite",))
    return cfg

# cvphr/models/__init__.py
from .core.registry import merge_model_dicts
from .posaglreg.models import DATASET_CLASS_DICT
from .posaglreg.models import MODEL_CLASS_DICT as U_MODEL_CLASS_DICT
# from .cvphr.models import MODEL_CLASS_DICT as V_MODEL_CLASS_DICT
# from .cvphar.models import MODEL_CLASS_DICT as X_MODEL_CLASS_DICT
# from .phareg.models import MODEL_CLASS_DICT as Y_MODEL_CLASS_DICT
# from .posaglreg.models import MODEL_CLASS_DICT as U_MODEL_CLASS_DICT
from .posaglreg.models import MODEL_KEYWARDS_DICT as U_MODEL_KEYWARDS_DICT
# from .cvphr.models import MODEL_KEYWARDS_DICT as V_MODEL_KEYWARDS_DICT
# from .cvphar.models import MODEL_KEYWARDS_DICT as X_MODEL_KEYWARDS_DICT
# from .phareg.models import MODEL_KEYWARDS_DICT as Y_MODEL_KEYWARDS_DICT

MODEL_CLASS_DICT = merge_model_dicts(
    U_MODEL_CLASS_DICT,
    # V_MODEL_CLASS_DICT,
    # X_MODEL_CLASS_DICT,
    # Y_MODEL_CLASS_DICT,
    strict=True,   # 🚨 冲突直接报错
)

MODEL_KEYWARDS_DICT = merge_model_dicts(
    U_MODEL_KEYWARDS_DICT,
    # V_MODEL_KEYWARDS_DICT,
    # X_MODEL_KEYWARDS_DICT,
    # Y_MODEL_KEYWARDS_DICT,
    strict=True,   # 🚨 冲突直接报错
)

import os.path, inspect
import transformers
from transformers.modeling_roberta import (
    RobertaPreTrainedModel,
)

print(os.path.abspath(transformers.__file__))
# class RobertaPreTrainedModel:
#     pass
# print(os.path.dirname(os.path.abspath(inspect.getsourcefile(RobertaPreTrainedModel))))
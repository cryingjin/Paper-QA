from transformers import ElectraTokenizer
from transformers import ElectraConfig
from transformers import AutoConfig
from transformers import AutoTokenizer

# from transformers.tokenization_electra import ElectraTokenizer
# from transformers.configuration_electra import ElectraConfig
# from transformers.configuration_auto import AutoConfig
# from transformers.tokenization_auto import AutoTokenizer

from src.model.model import (ElectraForQuestionAnswering, RobertaForQUestionAnswering)

CONFIG = {
    "electra":ElectraConfig,
    "roberta":AutoConfig,
}

TOKENIZER = {
    "electra":ElectraTokenizer,
    "roberta":AutoTokenizer,
}

QUESTION_ANSWERING_MODEL = {
    "electra":ElectraForQuestionAnswering,
    "roberta":RobertaForQUestionAnswering,
}

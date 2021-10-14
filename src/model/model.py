import numpy as np
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import ElectraModel, ElectraPreTrainedModel
from transformers import RobertaPreTrainedModel
from transformers import RobertaModel


class ElectraForQuestionAnswering(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraForQuestionAnswering, self).__init__(config)

        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.electra = ElectraModel(config)
        self.bi_gru = nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
                             num_layers=1, batch_first=True, bidirectional=True)     # dropout=0.2

        # final output projection layer(fnn)
        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_outputs = nn.Linear(config.hidden_size, 1)

        def forward(
                self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                start_positions=None,
                end_positions=None,
        ):


            outputs = self.electra(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            # sequence_output : [batch_size, seq_length, hidden_size]
            sequence_output = outputs[0]
            gru_output, _ = self.bi_gru(sequence_output)

            start_logits = self.start_outputs(gru_output)
            end_logits = self.end_outputs(gru_output)

            # start_logits : [batch_size, max_length]
            # end_logits : [batch_size, max_lenght]
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            # outputs = (start_logits, end_logits)
            outputs = (start_logits, end_logits,) + outputs[1:]

            if start_positions is not None and end_positions is not None:

                ignored_index = start_logits.size(1)
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2

                # outputs : (total_loss, start_logits, end_logits)
                outputs = (total_loss,) + outputs

            return outputs  # (loss), start_logits, end_logits, sent_token_logits

class RobertaForQUestionAnswering(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaForQUestionAnswering, self).__init__(config)

        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.roberta = RobertaModel(config)

        self.bi_gru = nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
                             num_layers=1, batch_first=True, bidirectional=True)     # dropout=0.2

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            start_positions=None,
            end_positions=None,
    ):

        outputs = self.roberta(
            input_ids = input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]

        gru_output, _ = self.bi_gru(sequence_output)

        logits = self.qa_outputs(gru_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        outputs = (start_logits, end_logits,)     #+ outputs[1:]

        if start_positions is not None and end_positions is not None:

            loss_fct = CrossEntropyLoss()

            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss ) / 2

            # outputs : (total_loss, start_logits, end_logits)
            outputs = (total_loss,) + outputs

        return outputs

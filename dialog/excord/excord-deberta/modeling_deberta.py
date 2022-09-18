import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from transformers.models.deberta.modeling_deberta import (
    DebertaPreTrainedModel,
    DebertaModel,
    ContextPooler,
    StableDropout,
)

from transformers.file_utils import(
    add_start_docstrings,
    add_code_sample_docstrings,
)

from transformers.modeling_outputs import (
    QuestionAnsweringModelOutput,
)

_CONFIG_FOR_DOC = "DebertaConfig"
_TOKENIZER_FOR_DOC = "DebertaTokenizer"

DEBERTA_START_DOCSTRING = r"""
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)
    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.
    Parameters:
        config (:class:`~transformers.DebertaConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

@add_start_docstrings(
    """
    Deberta Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    DEBERTA_START_DOCSTRING,
)
class DebertaForQuestionAnswering(DebertaPreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config, class_num=1):
        super().__init__(config)
        # self.class_num = class_num # quac: 1 class (answerable),
        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels    

        self.deberta = DebertaModel(config)
        self.pooler = ContextPooler(config)         
        output_dim = self.pooler.output_dim

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)        
        self.classifier = nn.Linear(output_dim, config.num_labels)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_layer = torch.nn.Linear(512*2,2)
        self.loss_layer2 = torch.nn.Linear(2,1)
        self.loss_layer3 = torch.nn.Linear(512*2,512)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)
        
        self.init_weights()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="deberta-base",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        # head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        #+ 
        is_impossible=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            # head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        class_logits = self.classifier(sequence_output)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp_(0, ignored_index)
            end_positions = end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            
            # if self.class_num < 2: # quac
            if 1: # quac
            # if 0: # don't use BCEWithLogitsLoss
                class_loss_fct = BCEWithLogitsLoss()

                loss_input = class_logits
                loss_target = is_impossible

                # 12,512*2 -> 12*2
                input_class = class_logits.reshape(class_logits.shape[0],class_logits.shape[1]*class_logits.shape[2])
                print("input_class size : ", input_class.shape)
                input_class = self.loss_layer(input_class)
                # 12*2 -> 12*1
                print("input_class size (reshape 1) : ", input_class.shape)
                input_class = self.loss_layer2(input_class)
                print("input_class size (reshape 2) : ", input_class.shape)
                class_loss = class_loss_fct(input_class.squeeze(), is_impossible.squeeze())
                # class_loss = class_loss_fct(class_logits.squeeze(), is_impossible.squeeze())
            else: # coqa
                input_class = self.loss_layer2(class_logits)
                print("input_class size (reshape) : ", input_class.shape)

                class_loss_fct = CrossEntropyLoss(ignore_index=3)
                class_loss = class_loss_fct(class_logits, is_impossible)
                # ValueError: Expected target size (12, 2), got torch.Size([12]) 
            
            total_loss = (start_loss + end_loss + class_loss) / 3

        print("start_logits shape: ", start_logits.shape)
        print("end_logits shape: ", end_logits.shape)

        print("class_logits size : ", class_logits.shape)
        class_logits_2 = class_logits.reshape(class_logits.shape[0],class_logits.shape[1]*class_logits.shape[2])
        print("class_logits_2 size (reshape) : ", class_logits_2.shape)
        class_logits_2 = self.loss_layer3(class_logits_2)
        print("class_logits_2 size (linear) : ", class_logits_2.shape)

        output = (start_logits, end_logits, class_logits_2) 
        # output = (start_logits, end_logits, class_logits) 
        return ((total_loss,) + output) if total_loss is not None else output

        # if not return_dict:
        #     output = (start_logits, end_logits, class_logits)
        #     return ((total_loss,) + output) if total_loss is not None else output

        print("QuestionAnsweringModelOutput")
        print("loss : ", total_loss.shape, total_loss)
        print("start_logits : ", start_logits.shape, start_logits)
        print("end_logits : ", end_logits.shape, end_logits)
        # print("hidden_states : ", outputs.hidden_states.shape, outputs.hidden_states)
        # print("attentions : ", outputs.attentions.shape, outputs.attentions)
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            class_logits=class_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
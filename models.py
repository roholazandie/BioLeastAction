from typing import Optional, Tuple, Union
import torch
from generate_utils import GenerationMixin
from transformers import GPT2Model, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class GPT2LeastActionModel(GenerationMixin, GPT2Model):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.transformer = GPT2Model(config)
        self.init_weights()
        # self.main_input_name = "inputs_embeds"

    def forward(
            self,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels_embeds: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithCrossAttentions:
        transformer_outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        loss = None
        if labels_embeds is not None:
            hidden_states = transformer_outputs[0]
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(hidden_states, labels_embeds)

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.last_hidden_state,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

from typing import Optional, Tuple, Union

import numpy as np
import torch
from generate_utils import GenerationMixin
from transformers import GPT2Model, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import torch.nn as nn
import wandb
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
            emb_generated_trajectory = hidden_states[0, :, :].detach().cpu().numpy()
            emb_original_trajectory = labels_embeds[0, :, :].detach().cpu().numpy()
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(hidden_states, labels_embeds)

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.last_hidden_state,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


class GPT2AutoencoderLeastActionModel(GenerationMixin, GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = nn.Linear(config.n_gene, config.hidden_size)  # Trainable encoder
        self.transformer = GPT2Model(config)
        self.decoder = nn.Linear(config.hidden_size, config.n_gene)  # Trainable decoder
        self.init_weights()

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
        # import numpy as np
        # import matplotlib.pyplot as plt
        # xx = inputs_embeds.cpu().numpy()
        # for day_idx in range(38):
        #     distances = [[np.linalg.norm(xx[i, day_idx, :] - xx[j, day_idx, :]) for i in range(50) if i>j] for j in
        #      range(50)]
        #
        #     # flatten the list of lists
        #     distances = [item for sublist in distances for item in sublist]
        #
        #     plt.figure()
        #     plt.hist(distances, bins=100)
        #     plt.title(f"Distance Histogram for Day")
        #     plt.xlabel(f" Distance")
        #     plt.ylabel("Frequency")
        #     plt.grid(True)
        #     plt.show()

        # Pass inputs_embeds through the trainable encoder
        encoded_embeds = self.encoder(inputs_embeds)

        # Pass the encoded embeddings through the transformer
        transformer_outputs = self.transformer(
            inputs_embeds=encoded_embeds,
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

        # Pass the transformer output through the trainable decoder
        decoded_outputs = self.decoder(transformer_outputs.last_hidden_state)

        # Calculate the loss if labels_embeds are provided
        loss = None
        # if labels_embeds is not None:
        #     loss_fct = torch.nn.MSELoss()
        #     distance_values = []
        #     for i in range(decoded_outputs.shape[1]):
        #         loss_per_day = torch.mean((decoded_outputs[:, i, :] - inputs_embeds[:, i, :]) ** 2).item()
        #         distance_values.append(loss_per_day)
        #     day_number = list(range(len(distance_values)))
        #     data = [[x, y] for (x, y) in zip(day_number, distance_values)]
        #     table = wandb.Table(data=data, columns=["recall_micro", "precision_micro"])
        #     wandb.log({"my_lineplot_id": wandb.plot.line(table, "recall_micro",
        #                                                  "precision_micro", stroke=None, title="Average Precision")})
        #
        #     loss = loss_fct(decoded_outputs, inputs_embeds)  # Compare with the input embeddings

        # cross entropy
        if labels_embeds is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(decoded_outputs, labels_embeds)

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=decoded_outputs,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

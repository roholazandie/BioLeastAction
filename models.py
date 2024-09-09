from typing import Optional, Tuple, Union

import numpy as np
import torch
from generate_utils import GenerationMixin
from transformers import GPT2Model, GPT2LMHeadModel, GPT2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn as nn
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


class GPT2IdLeastActionModel(GPT2PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.weight = torch.nn.Parameter(new_embeddings)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )

        return model_inputs


    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithCrossAttentions:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )



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

            # for i in range(24):
            #     emb_generated_trajectory = hidden_states[i, :, :].detach().cpu().numpy()
            #     emb_original_trajectory = labels_embeds[i, :, :].detach().cpu().numpy()
            #     original_emd_distance = [
            #         [np.linalg.norm(emb_original_trajectory[i, :] - emb_original_trajectory[j, :]) for i in range(300)]
            #         for j
            #         in range(300)]
            #
            #     generated_emd_distance = [
            #         [np.linalg.norm(emb_generated_trajectory[i, :] - emb_generated_trajectory[j, :]) for i in
            #          range(300)] for j
            #         in range(300)]
            #
            #     between_emd_distance = [
            #         [np.linalg.norm(emb_original_trajectory[i, :] - emb_generated_trajectory[j, :]) for i in range(300)]
            #         for j
            #         in range(300)]
            #
            #     # Calculate the common vmin and vmax
            #     vmin = min(np.min(original_emd_distance), np.min(generated_emd_distance),
            #                np.min(between_emd_distance))
            #     vmax = max(np.max(original_emd_distance), np.max(generated_emd_distance),
            #                np.max(between_emd_distance))
            #
            #     # Create a figure with 3 subplots (1 row, 3 columns)
            #     fig, axes = plt.subplots(1, 3, figsize=(20, 8))
            #
            #     # Plot the first distance matrix on the first subplot
            #     im1 = axes[0].imshow(original_emd_distance, cmap='viridis', interpolation='nearest', vmin=vmin,
            #                          vmax=vmax)
            #     fig.colorbar(im1, ax=axes[0], label='Euclidean Distance')
            #     axes[0].set_title('Euclidean Distance Between Days (Original)')
            #     axes[0].set_xlabel('Day')
            #     axes[0].set_ylabel('Day')
            #
            #     # Plot the second distance matrix on the second subplot
            #     im2 = axes[1].imshow(generated_emd_distance, cmap='viridis', interpolation='nearest', vmin=vmin,
            #                          vmax=vmax)
            #     fig.colorbar(im2, ax=axes[1], label='Euclidean Distance')
            #     axes[1].set_title('Euclidean Distance Between Days (Generated)')
            #     axes[1].set_xlabel('Day')
            #     axes[1].set_ylabel('Day')
            #
            #     # Plot the third distance matrix on the third subplot
            #     im3 = axes[2].imshow(between_emd_distance, cmap='viridis', interpolation='nearest', vmin=vmin,
            #                          vmax=vmax)
            #     fig.colorbar(im3, ax=axes[2], label='Euclidean Distance')
            #     axes[2].set_title('Euclidean Distance Between Days (Original and Generated)')
            #     axes[2].set_xlabel('Day')
            #     axes[2].set_ylabel('Day')
            #
            #     # Adjust the layout to prevent overlap
            #     plt.tight_layout()
            #
            #     # Show the combined plot
            #     plt.show()

            # for i in range(24):
            #     emb_generated_trajectory = hidden_states[i, :, :].detach().cpu().numpy()
            #     emb_original_trajectory = labels_embeds[i, :, :].detach().cpu().numpy()
            #
            #     # Normalize the embeddings to unit vectors (for cosine similarity)
            #     normalized_original = emb_original_trajectory / np.linalg.norm(emb_original_trajectory, axis=1,
            #                                                                    keepdims=True)
            #     normalized_generated = emb_generated_trajectory / np.linalg.norm(emb_generated_trajectory, axis=1,
            #                                                                      keepdims=True)
            #     # Compute cosine similarity using matrix multiplication
            #     original_emd_similarity = np.dot(normalized_original, normalized_original.T)
            #     generated_emd_similarity = np.dot(normalized_generated, normalized_generated.T)
            #     between_emd_similarity = np.dot(normalized_original, normalized_generated.T)
            #
            #     # Calculate the common vmin and vmax
            #     vmin = min(np.min(original_emd_similarity), np.min(generated_emd_similarity),
            #                np.min(between_emd_similarity))
            #     vmax = max(np.max(original_emd_similarity), np.max(generated_emd_similarity),
            #                np.max(between_emd_similarity))
            #
            #     # Create a figure with 3 subplots (1 row, 3 columns)
            #     fig, axes = plt.subplots(1, 3, figsize=(20, 8))
            #
            #     # Plot the first distance matrix on the first subplot
            #     im1 = axes[0].imshow(original_emd_similarity, cmap='viridis', interpolation='nearest', vmin=vmin,
            #                          vmax=vmax)
            #     fig.colorbar(im1, ax=axes[0], label='Cosine Similarity')
            #     axes[0].set_title('Cosine Similarity Between Days (Original)')
            #     axes[0].set_xlabel('Day')
            #     axes[0].set_ylabel('Day')
            #
            #     # Plot the second distance matrix on the second subplot
            #     im2 = axes[1].imshow(generated_emd_similarity, cmap='viridis', interpolation='nearest', vmin=vmin,
            #                          vmax=vmax)
            #     fig.colorbar(im2, ax=axes[1], label='Cosine Similarity')
            #     axes[1].set_title('Cosine Similarity Between Days (Generated)')
            #     axes[1].set_xlabel('Day')
            #     axes[1].set_ylabel('Day')
            #
            #     # Plot the third distance matrix on the third subplot
            #     im3 = axes[2].imshow(between_emd_similarity, cmap='viridis', interpolation='nearest', vmin=vmin,
            #                          vmax=vmax)
            #     fig.colorbar(im3, ax=axes[2], label='Cosine Similarity')
            #     axes[2].set_title('Cosine Similarity Between Days (Original and Generated)')
            #     axes[2].set_xlabel('Day')
            #     axes[2].set_ylabel('Day')
            #
            #     # Adjust the layout to prevent overlap
            #     plt.tight_layout()
            #
            #     # Show the combined plot
            #     plt.show()

            # data1 = emb_generated_trajectory.flatten()
            # data2 = emb_original_trajectory.flatten()
            #
            # # Create a figure with 2 subplots (1 row, 2 columns)
            # fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            #
            # # Plot the first histogram in the first subplot
            # axes[0].hist(data1, bins=100, color='blue', edgecolor='black', alpha=0.7)
            # axes[0].set_title('Histogram of Generated')
            # axes[0].set_xlabel('Value')
            # axes[0].set_ylabel('Frequency')
            # axes[0].grid(True)

            # # Plot the second histogram in the second subplot
            # axes[1].hist(data2, bins=100, color='red', edgecolor='black', alpha=0.7)
            # axes[1].set_title('Histogram of Original')
            # axes[1].set_xlabel('Value')
            # axes[1].set_ylabel('Frequency')
            # axes[1].grid(True)
            #
            # # Adjust layout to prevent overlap
            # plt.tight_layout()
            #
            # # Show the plot
            # plt.show()


            # loss_fct = torch.nn.MSELoss()
            # loss = loss_fct(hidden_states, labels_embeds)

            loss_fct = CrossEntropyLoss()
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



class GPT2VAEModel(GenerationMixin, GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.encoder_mu = nn.Linear(config.n_gene, config.hidden_size)
        self.encoder_logvar = nn.Linear(config.n_gene, config.hidden_size)
        self.transformer = GPT2Model(config)
        self.decoder = nn.Linear(config.hidden_size, config.n_gene)
        self.init_weights()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, inputs_embeds=None, labels_embeds=None, **kwargs):
        # Encoder step
        mu = self.encoder_mu(inputs_embeds)
        logvar = self.encoder_logvar(inputs_embeds)
        logvar = torch.clamp(logvar, min=-10, max=10)  # Prevents extreme values
        z = self.reparameterize(mu, logvar)

        # Transformer step
        transformer_outputs = self.transformer(inputs_embeds=z, **kwargs)
        decoded_outputs = self.decoder(transformer_outputs.last_hidden_state)

        # Compute the losses
        recon_loss = F.mse_loss(decoded_outputs, labels_embeds)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=decoded_outputs,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

from typing import Optional, Tuple, Union

import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from utils.generate_utils import GenerationMixin
from transformers import GPT2Model, GPT2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, BaseModelOutputWithPastAndCrossAttentions
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask_for_sdpa
import torch.nn as nn
# from transformers import logger
from vector_quantize_pytorch import VectorQuantize
from utils.hash_embedding import HashEmbedding

# logger = logging.get_logger(__name__)


class GPT2VQModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        # Remove the original embedding layer
        # self.wte = nn.Embedding(config.vocab_size, self.embed_dim)

        # Introduce a new embedding layer (could be optional)
        # You can choose to use a smaller embedding dimension if desired
        self.hash_embedding = HashEmbedding(self.embed_dim)

        # Initialize the vector quantizer
        codebook_size = 52000  # Example codebook size, adjust as needed
        self.vector_quantizer = VectorQuantize(
            dim=self.embed_dim,
            codebook_size=codebook_size,
            decay=0.8,
            commitment_weight=1.0,
            use_cosine_sim=False,
            # Other parameters as needed
        )

        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.hash_embedding  # Update to new embedding layer

    def set_input_embeddings(self, new_embeddings):
        # Not applicable since hash_embedding doesn't have parameters to set
        raise NotImplementedError("Cannot set input embeddings when using HashEmbedding.")


    def _prune_heads(self, heads_to_prune):
        # (Same as before)
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

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
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        # (Same as before)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        # Modify here
        if inputs_embeds is None:
            # Get initial embeddings from the new embedding layer
            inputs_embeds = self.hash_embedding(input_ids)

            # Apply vector quantization
            inputs_embeds, embed_ind, vq_loss = self.vector_quantizer(inputs_embeds)

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # Attention mask.
        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            if self._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask if 0 in attention_mask else None
            elif _use_sdpa:
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask=attention_mask,
                    input_shape=(batch_size, input_shape[-1]),
                    inputs_embeds=inputs_embeds,
                    past_key_values_length=past_length,
                )
            else:
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and the dtype's smallest value for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif not self._attn_implementation == "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        # At the end, you might want to include the vector quantization loss
        if return_dict:
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=presents,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
                cross_attentions=all_cross_attentions,
            ), vq_loss
        else:
            output = (hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions)
            # Include VQ loss in the outputs if needed
            return output + (vq_loss,)



class CellGPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        # New embedding layers for token_type_ids
        cell_type_vocab_size = getattr(config, "cell_type_vocab_size", 2)
        self.cell_type_embeddings = nn.Embedding(cell_type_vocab_size, self.embed_dim)
        # self.token_type_embeddings2 = nn.Embedding(cell_type_vocab_size, self.embed_dim)
        # self.token_type_embeddings3 = nn.Embedding(cell_type_vocab_size, self.embed_dim)

        # Feedforward layer for input_embedding
        self.cell_embedding_ff = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cell_type_ids: Optional[torch.LongTensor] = None,
        token_type_ids2: Optional[torch.LongTensor] = None,
        token_type_ids3: Optional[torch.LongTensor] = None,
        cell_embeddings: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device

        if cell_type_ids is not None:
            cell_type_ids = cell_type_ids.view(-1, input_shape[-1])

        if token_type_ids2 is not None:
            token_type_ids2 = token_type_ids2.view(-1, input_shape[-1])

        if token_type_ids3 is not None:
            token_type_ids3 = token_type_ids3.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        inputs_embeds = self.wte(input_ids)

        # Process input_embedding through feedforward layer and add to inputs_embeds
        if cell_embeddings is not None:
            with torch.no_grad():
                cell_embeddings = cell_embeddings.detach()
            assert cell_embeddings.size(-1) == inputs_embeds.size(-1), "The size of cell_embeddings must match the size of inputs_embeds"
            inputs_embeds = inputs_embeds + self.cell_embedding_ff(cell_embeddings)

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # Attention mask.
        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            if self._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask if 0 in attention_mask else None
            elif _use_sdpa:
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask=attention_mask,
                    input_shape=(batch_size, input_shape[-1]),
                    inputs_embeds=inputs_embeds,
                    past_key_values_length=past_length,
                )
            else:
                attention_mask = attention_mask[:, None, None, :]
                attention_mask = attention_mask.to(dtype=self.dtype)
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif not self._attn_implementation == "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if cell_type_ids is not None:
            cell_type_embeds = self.cell_type_embeddings(cell_type_ids)
            hidden_states = hidden_states + cell_type_embeds

        if token_type_ids2 is not None:
            token_type_embeds2 = self.token_type_embeddings2(token_type_ids2)
            hidden_states = hidden_states + token_type_embeds2

        if token_type_ids3 is not None:
            token_type_embeds3 = self.token_type_embeddings3(token_type_ids3)
            hidden_states = hidden_states + token_type_embeds3

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class GPT2CellLeastActionModel(GPT2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.transformer = CellGPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()
        # self.tie_weights()

    def get_output_embeddings(self):
        return self.lm_head


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        cell_type_ids = kwargs.get("cell_type_ids", None)
        cell_embeddings = kwargs.get("cell_embeddings", None)
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
            if cell_type_ids is not None:
                cell_type_ids = cell_type_ids[:, -input_ids.shape[1] :]

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
                "cell_type_ids": cell_type_ids,
                "cell_embeddings": cell_embeddings,
            }
        )

        return model_inputs


    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            cell_type_ids: Optional[torch.LongTensor] = None,
            cell_embeddings: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
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
            cell_type_ids=cell_type_ids,
            cell_embeddings=cell_embeddings,
            position_ids=position_ids,
            head_mask=head_mask,
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


class GPT2IdLeastActionModel(GPT2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()
        # self.tie_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        cell_type_ids = kwargs.get("cell_type_ids", None)
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
            if cell_type_ids is not None:
                cell_type_ids = cell_type_ids[:, -input_ids.shape[1] :]

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
                "cell_type_ids": cell_type_ids,
            }
        )

        return model_inputs


    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            cell_type_ids: Optional[torch.LongTensor] = None,
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
            token_type_ids=cell_type_ids,
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




class GPT2DistanceLeastActionModel(GPT2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config,
                 cell_embeddings: torch.FloatTensor,   # [vocab_size, embedding_dim]
                 alpha: float = 0.1,
                 ):
        super().__init__(config)
        self.config = config
        # Store the cell embeddings on the correct device.
        # We'll register it as a buffer so it's moved with the model.
        self.register_buffer("cell_embeddings", cell_embeddings)

        # Balancing factor for distance penalty
        self.alpha = alpha

        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.dist_loss_value = None
        self.ce_loss_value = None
        self.post_init()
        # self.tie_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        cell_type_ids = kwargs.get("cell_type_ids", None)
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
            if cell_type_ids is not None:
                cell_type_ids = cell_type_ids[:, -input_ids.shape[1] :]

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
                "cell_type_ids": cell_type_ids,
            }
        )

        return model_inputs


    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            cell_type_ids: Optional[torch.LongTensor] = None,
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
            token_type_ids=cell_type_ids,
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
            shift_logits = lm_logits[..., :-1, :].contiguous() # [B, T-1, V]
            shift_labels = labels[..., 1:].contiguous() # [B, T-1]
            # standard cross-entropy loss
            loss_fct = CrossEntropyLoss()
            ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # === Distance-based penalty ===
            # We'll compute an "expected embedding" under the predicted distribution
            # and measure distance to the ground-truth cell embedding.
            # 1) Convert shift_logits to probabilities
            shift_probs = F.softmax(shift_logits, dim=-1)  # [B, T-1, V]

            # 2) For each position, gather the ground-truth embeddings
            # ground_truth_emb shape = [B, T-1, embedding_dim]
            ground_truth_emb = self.cell_embeddings[shift_labels]

            # 3) Compute expected embedding for each position:
            #   expected_emb[b, t] = sum_{v=0}^{V-1} shift_probs[b, t, v] * cell_embeddings[v]
            # We'll do it via batch-matrix multiplication or "einsum"
            # shape(shift_probs) = [B, T-1, V]
            # shape(cell_embeddings) = [V, embedding_dim]
            # => shape(expected_emb) = [B, T-1, embedding_dim]
            expected_emb = torch.einsum("btv,vd->btd", shift_probs, self.cell_embeddings)

            # 4) Compute distance (e.g. MSE)
            # shape of distance_loss = [B, T-1]
            # we then mean() over B*(T-1)
            dist_loss = F.mse_loss(expected_emb, ground_truth_emb, reduction="mean")

            self.dist_loss_value = dist_loss
            self.ce_loss_value = ce_loss
            # Combine the two
            loss = ce_loss + self.alpha * dist_loss


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


# class GPT2AutoencoderLeastActionModel(GenerationMixin, GPT2Model):
#     def __init__(self, config):
#         super().__init__(config)
#         self.config = config
#         self.encoder = nn.Linear(config.n_gene, config.hidden_size)  # Trainable encoder
#         self.transformer = GPT2Model(config)
#         self.decoder = nn.Linear(config.hidden_size, config.n_gene)  # Trainable decoder
#         self.init_weights()
#
#     def forward(
#             self,
#             inputs_embeds: Optional[torch.FloatTensor] = None,
#             labels_embeds: Optional[torch.FloatTensor] = None,
#             past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#             attention_mask: Optional[torch.Tensor] = None,
#             token_type_ids: Optional[torch.Tensor] = None,
#             position_ids: Optional[torch.Tensor] = None,
#             head_mask: Optional[torch.Tensor] = None,
#             encoder_hidden_states: Optional[torch.FloatTensor] = None,
#             encoder_attention_mask: Optional[torch.Tensor] = None,
#             use_cache: Optional[bool] = None,
#             output_attentions: Optional[bool] = None,
#             output_hidden_states: Optional[bool] = None,
#             return_dict: Optional[bool] = None,
#     ) -> CausalLMOutputWithCrossAttentions:
#         # import numpy as np
#         # import matplotlib.pyplot as plt
#         # xx = inputs_embeds.cpu().numpy()
#         # for day_idx in range(38):
#         #     distances = [[np.linalg.norm(xx[i, day_idx, :] - xx[j, day_idx, :]) for i in range(50) if i>j] for j in
#         #      range(50)]
#         #
#         #     # flatten the list of lists
#         #     distances = [item for sublist in distances for item in sublist]
#         #
#         #     plt.figure()
#         #     plt.hist(distances, bins=100)
#         #     plt.title(f"Distance Histogram for Day")
#         #     plt.xlabel(f" Distance")
#         #     plt.ylabel("Frequency")
#         #     plt.grid(True)
#         #     plt.show()
#
#         # Pass inputs_embeds through the trainable encoder
#         encoded_embeds = self.encoder(inputs_embeds)
#
#         # Pass the encoded embeddings through the transformer
#         transformer_outputs = self.transformer(
#             inputs_embeds=encoded_embeds,
#             past_key_values=past_key_values,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_attention_mask,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#
#         # Pass the transformer output through the trainable decoder
#         decoded_outputs = self.decoder(transformer_outputs.last_hidden_state)
#
#         # Calculate the loss if labels_embeds are provided
#         loss = None
#         # if labels_embeds is not None:
#         #     loss_fct = torch.nn.MSELoss()
#         #     distance_values = []
#         #     for i in range(decoded_outputs.shape[1]):
#         #         loss_per_day = torch.mean((decoded_outputs[:, i, :] - inputs_embeds[:, i, :]) ** 2).item()
#         #         distance_values.append(loss_per_day)
#         #     day_number = list(range(len(distance_values)))
#         #     data = [[x, y] for (x, y) in zip(day_number, distance_values)]
#         #     table = wandb.Table(data=data, columns=["recall_micro", "precision_micro"])
#         #     wandb.log({"my_lineplot_id": wandb.plot.line(table, "recall_micro",
#         #                                                  "precision_micro", stroke=None, title="Average Precision")})
#         #
#         #     loss = loss_fct(decoded_outputs, inputs_embeds)  # Compare with the input embeddings
#
#         # cross entropy
#         if labels_embeds is not None:
#             loss_fct = torch.nn.CrossEntropyLoss()
#             loss = loss_fct(decoded_outputs, labels_embeds)
#
#         return CausalLMOutputWithCrossAttentions(
#             loss=loss,
#             past_key_values=transformer_outputs.past_key_values,
#             hidden_states=decoded_outputs,
#             attentions=transformer_outputs.attentions,
#             cross_attentions=transformer_outputs.cross_attentions,
#         )

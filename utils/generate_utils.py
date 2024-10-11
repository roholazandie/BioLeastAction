import torch
from typing import Dict, Tuple


class GenerationMixin:

    def prepare_inputs_for_generation(self,
                                      inputs_embeds,
                                      position_ids,
                                      features_embeds,
                                      past_key_values,
                                      **kwargs) -> Dict[str, torch.Tensor]:
        token_type_ids = kwargs.get("token_type_ids", None)
        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if inputs_embeds.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = inputs_embeds.shape[1] - 1

            inputs_embeds = inputs_embeds[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -inputs_embeds.shape[1]:]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -inputs_embeds.shape[1]:]
        else:
            position_ids = None

        model_inputs = {
            "inputs_embeds": inputs_embeds,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        return model_inputs

    def _use_cache(self, outputs, use_cache):
        """During generation, decide whether to pass the `past` variable to the next forward pass."""
        if len(outputs) <= 1 or use_cache is False:
            return False
        if hasattr(self.config, "mem_len") and self.config.mem_len == 0:
            return False
        return True

    @torch.no_grad()
    def generate(
            self,
            inputs_embeds: torch.Tensor,
            position_ids: torch.Tensor = None,
            features_embeds: torch.Tensor = None,
            max_length: int = None,
            attention_mask: torch.LongTensor = None,
            use_cache: bool = False,
            **model_specific_kwargs
    ) -> Tuple[torch.Tensor]:
        """Generated a predicted sequence of features

        Args:
            inputs_embeds (Tensor): [batch, seq, n_embed] Input feature tensor
            position_ids (Tensor, optional): [seq, n_embed] Position tensor. Defaults to None.
            prop_embeds (Tensor, optional): [batch, seq, n_embed] Property tensor. Defaults to None.
            max_length (int, optional): Length of time series to predict. Defaults to None.
            attention_mask (LongTensor, optional): Manual attention mask. Defaults to None.
            use_cache (bool, optional): Cache past transformer states for faster generation. Defaults to False.

        Returns:
            Tuple[Tensor]: [batch, max_length, n_embed] Predicted feature tensor, additional optional transformer outputs.
        """
        max_length = max_length if max_length is not None else self.config.max_length
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(use_cache, bool), "`use_cache` should be a boolean."

        output = self._generate_time_series(
            inputs_embeds,
            position_ids,
            features_embeds,
            max_length=max_length,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **model_specific_kwargs,
        )

        return output

    def _generate_time_series(
            self,
            inputs_embeds: torch.Tensor,
            position_ids: torch.Tensor,
            features_embeds: torch.Tensor,
            max_length: int,
            use_cache: bool = True,
            **model_specific_kwargs
    ) -> Tuple[torch.Tensor]:
        """Function that calls model forward to predict

        Args:
            inputs_embeds (Tensor): [batch, seq, n_embed] Input feature tensor
            position_ids (Tensor, optional): [seq, n_embed] Position tensor. Defaults to None.
            prop_embeds (Tensor, optional): [batch, seq, n_embed] Property tensor. Defaults to None.
            max_length (int, optional): Length of time series to predict. Defaults to None.
            attention_mask (LongTensor, optional): Manual attention mask. Defaults to None.
            use_cache (bool, optional): [description]. Defaults to None.

        Returns:
            Tuple[Tensor]: [batch, max_length, n_embed] Predicted feature tensor, additional optional transformer outputs.
        """
        past_key_values = None

        cur_len = inputs_embeds.shape[1]
        assert (
                cur_len < max_length
        ), f"The input context is {cur_len}, but `max_length` is only {max_length}. Please make sure that `max_length` larger than the input"

        while cur_len < max_length:
            # Prepare inputs for transformer
            model_inputs = self.prepare_inputs_for_generation(
                inputs_embeds,
                position_ids,
                features_embeds,
                use_cache=use_cache,
                past_key_values=past_key_values,
                **model_specific_kwargs,
            )

            outputs = self.forward(**model_inputs,
                                   return_dict=True,
                                   output_attentions=False,
                                   output_hidden_states=False)

            next_output = outputs.hidden_states[:, -1:]

            if self._use_cache(outputs, use_cache):
                past_key_values = outputs.past_key_values
            else:
                past_key_values = None

            # add past output embedding and increase length by one
            inputs_embeds = torch.cat([inputs_embeds, next_output], dim=1)
            cur_len = cur_len + 1

        return inputs_embeds

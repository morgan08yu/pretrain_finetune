import math
from tqdm import tqdm
from datetime import datetime
from torch.nn import CrossEntropyLoss
import tempfile
from torch.distributed import rpc
import torch.functional as F
import re
from torch.distributed.pipeline.sync import Pipe
from collections import OrderedDict
from transformers.utils import logging
from torch import nn
from typing import Any, List
from dataclasses import dataclass
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.llama.configuration_llama import LlamaConfig
from typing import Optional, Tuple
from transformers.models.llama import LlamaModel, LlamaConfig, LlamaTokenizer, LlamaForCausalLM
import torch
import transformers
from torch.cuda.amp import autocast, GradScaler

logger = logging.get_logger(__name__)

path = "/home/devsrc/LLMs/meta-llama3-8b"

# def _make_causal_mask(
#         input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
# ):
#     """
#     Make causal mask used for bi-directional self-attention.
#     """
#     bsz, tgt_len = input_ids_shape
#     mask = torch.full((tgt_len, tgt_len),
#                       torch.finfo(dtype).min, device=device)
#     mask_cond = torch.arange(mask.size(-1), device=device)
#     mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
#     mask = mask.to(dtype)

#     if past_key_values_length > 0:
#         mask = torch.cat([torch.zeros(
#             tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
#     return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# # Copied from transformers.models.bart.modeling_bart._expand_mask
# def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
#     """
#     Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
#     """
#     bsz, src_len = mask.size()
#     tgt_len = tgt_len if tgt_len is not None else src_len

#     expanded_mask = mask[:, None, None, :].expand(
#         bsz, 1, tgt_len, src_len).to(dtype)

#     inverted_mask = 1.0 - expanded_mask

#     return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)



class LlamawithPipeEmbedding(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.config = config
        self.gradient_checkpointing = False

    def forward(self, data: torch.Tensor):  # -> PipeDecoderLayerInputOutput:
        input_ids = data
        attention_mask  = data.attention_mask
        position_ids  = data.position_ids
        past_key_values = data.past_key_values
        inputs_embeds = data.inputs_embeds
        use_cache = data.use_cache
        output_attentions = data.output_attentions
        output_hidden_states  = data.output_hidden_states
        return_dict  = data.return_dict

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size,
                             seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        res = (hidden_states, attention_mask, position_ids)
        return res



class LlamaDecoderLayerwithPipe(nn.Module):
    def __init__(self, config: LlamaConfig, layer_index: int) -> None:
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.decoder_layer = LlamaDecoderLayer(config=config, layer_idx=layer_index)

    def forward(self, *args, **kwargs):  # -> PipeDecoderLayerInputOutput:
        if len(args) == 1:
            args = args[0]
        hidden_states, attention_mask, position_ids = args

        cur_device = next(self.decoder_layer.parameters()).device

        layer_outputs = self.decoder_layer(
            hidden_states=hidden_states.to(cur_device),
            attention_mask=attention_mask.to(cur_device),
            position_ids=position_ids.to(cur_device),
            past_key_value=None,  # past_key_value,
            output_attentions=None,
            use_cache=False,
        )
        hidden_states = layer_outputs[0]
        # res = PipeDecoderLayerInputOutput(
        #     hidden_states=hidden_states,
        #     attention_mask=data.attention_mask,
        #     past_key_values=data.past_key_values,
        #     position_ids=data.position_ids,
        #     output_attentions=all_self_attns,
        #     use_cache=data.use_cache,
        #     next_decoder_cache=next_decoder_cache
        # )
        # return res
        res = (hidden_states, attention_mask, position_ids)
        return res


class LlamaCausalLMwithPipe(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

    def forward(self, *args, **kwargs) -> torch.tensor:
        if len(args) == 1:
            args = args[0]
        # LM
        hidden_states, attention_mask, position_ids = args
        # hidden_states = data.hidden_states
        hidden_states = self.norm(hidden_states)

        # causal LM part
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i])
                      for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)

        return logits
    

class LlamaModelCasualLM4pipe:
    def __init__(self,
                 config: LlamaConfig,
                 transformersmodel: transformers.PreTrainedModel,
                 ngpus: int = None) -> None:
        self.base_config = config
        self.basemodel = transformersmodel
        self.ngpus = ngpus

    def create_pipe_model(self, on_cpu: bool = True) -> nn.Sequential:
        first_gpu = 0
        last_gpu = self.ngpus - 1

        # ngpus = self.ngpus
        # small_cell = math.ceil(self.base_config.num_hidden_layers / ngpus)

        pipemodel = nn.Sequential()

        embedding_ = LlamawithPipeEmbedding(
            config=self.base_config).to(f'cuda:{last_gpu}' if not on_cpu else 'cpu')
        pipemodel.add_module(name="emebdding", module=embedding_)

        for index in range(self.base_config.num_hidden_layers):
            target_gpu = last_gpu if index < 1 else first_gpu
            pipemodel.add_module(name=f"layer{index}",
                                 module=LlamaDecoderLayerwithPipe(
                                     config=self.base_config,
                                     layer_index=index).to(f'cuda:{target_gpu}' if not on_cpu else 'cpu')
                                 )

        causallm = LlamaCausalLMwithPipe(config=self.base_config).to(
            f'cuda:{first_gpu}' if not on_cpu else 'cpu')
        if self.base_config.tie_word_embeddings:
            causallm.lm_head.weight = embedding_.embed_tokens.weight

        pipemodel.add_module(name="causallm", module=causallm)

        # add raw model params to pipemodel
        base_model_param = self.get_basemodel_params()
        pipemodel.load_state_dict(base_model_param, strict=False)
        return pipemodel

    def get_basemodel_params(self):
        base_model_param = OrderedDict(
            {self.transname(n): v for n, v in self.basemodel.named_parameters()})
        return base_model_param

    def transname(self, name: str) -> str:
        if name.find("model.embed_tokens") != -1:
            return "emebdding.embed_tokens.weight"

        if name.find("model.layers") != -1:
            layer_index = re.findall(".([0-9][0-9]*).", name)[0]
            sub_name = re.findall(".[0-9][0-9]*.(.*)", name)[0]
            new_name = f"layer{layer_index}.decoder_layer.{sub_name}"
            return new_name

        if name.find("model.norm.weight") != -1:
            return "causallm.norm.weight"

        if name.find("lm_head.weight") != -1:
            return "causallm.lm_head.weight"

        return name

    def check_model(self,
                    pipemodel: nn.Sequential = None,
                    n_times: int = 2) -> bool:
        result = []
        for index in tqdm(range(n_times)):
            input_ids = torch.randint(
                low=0, high=self.base_config.vocab_size, size=(2, 1024))

            rawresult = self.basemodel(input_ids).logits
            piperesult = pipemodel(input_ids)
            res = torch.allclose(rawresult, piperesult)
            result.extend([res])

        return all(result)

class PipeTrain:
    def __init__(self, model: nn.Sequential, config: LlamaConfig, ) -> None:
        lr = 0.05
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1.0, gamma=0.95)
        self.scaler = GradScaler()
        self.get_total_params()

        self.total_loss = 0

    # def train_mini_batchs(self, input_ids: torch.Tensor, labels: torch.Tensor):
    #     logits = self.model(input_ids)

    #     # logits = self.model(PipeEmbeddingInput(input_ids=input_ids))
    #     logits = logits.local_value().float()
    #     shift_logits = logits[..., :-1, :].contiguous()
    #     shift_labels = labels[..., 1:].contiguous()
    #     # Flatten the tokens
    #     loss_fct = CrossEntropyLoss()
    #     shift_logits = shift_logits.view(-1, self.config.vocab_size)
    #     shift_labels = shift_labels.view(-1)
    #     # Enable model parallelism
    #     shift_labels = shift_labels.to(shift_logits.device)
    #     loss = loss_fct(shift_logits, shift_labels)

    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     self.show_loss(loss=loss)
    def train_mini_batchs(self, input_ids: torch.Tensor, labels: torch.Tensor):
        first_partition_device = next(self.model.parameters()).device
        input_ids = input_ids.to(first_partition_device)
        labels = labels.to(first_partition_device)

        with autocast(enabled=True):
            logits = self.model(input_ids)
            logits = logits.local_value().float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.show_loss(loss)

        del shift_logits, shift_labels

    def train(self, total_epoch: int = 10000, dataset =  None):
        self.model.train()

        input_ids, labels = dataset.generate_data()

        for index in tqdm(range(total_epoch)):
            self.train_mini_batchs(input_ids=input_ids.cuda(), labels=labels.cuda())

    def show_loss(self, loss: torch.Tensor) -> None:
        loss = loss.item()
        self.total_loss += loss
        print(f"datetime:{datetime.now()}, loss:{loss:.3f}")

    def get_total_params(self):
        total_params = 0
        for param in self.model.parameters():
            total_params += param.numel()

        print('Total parameters in model: {:,}'.format(total_params))
        # return total_params


# rl_modules/models.py
import sys
import torch
sys.path.append("../")
from torch.cuda.amp import autocast


class LLMModel():
    def __init__(self,
                 network,
                 config) -> None:
        self.network = network
        self.config = config

    def forward(self,
                token_ids,
                mask_ids,
                target_token_ids,
                is_fp16):     
        labels = torch.clone(token_ids)
        labels[labels[:, :] == self.config["pad_token_id"]] = self.config["ignore_index"]
        if is_fp16:
            with autocast():
                outputs = self.network(input_ids=token_ids,
                                       attention_mask=mask_ids,
                                       labels=labels,
                                       use_cache=self.config["use_cache"],
                                       return_dict=self.config["return_dict"],
                                       output_attentions=self.config["output_attentions"],
                                       output_hidden_states=self.config["output_hidden_states"],)
        else:
            outputs = self.network(input_ids=token_ids,
                                    attention_mask=mask_ids,
                                    labels=labels,
                                    use_cache=self.config["use_cache"],
                                    return_dict=self.config["return_dict"],
                                    output_attentions=self.config["output_attentions"],
                                    output_hidden_states=self.config["output_hidden_states"],)
        return outputs
    
class LLMFineTuneModel(torch.nn.Module):
    def __init__(self,
                 llm_network,
                 llm_config,
                 fine_tune_config,
                 is_fp16) -> None:
        self.fine_tune_config = fine_tune_config
        self.llm_config = llm_config
        self.is_fp16 = is_fp16
        super(LLMFineTuneModel, self).__init__()
        self.llm_network = llm_network

        if fine_tune_config["use_attention"]:
            self.transformer_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=fine_tune_config["feature_size"],
                                                                              nhead=fine_tune_config["transformer"]["n_head"],
                                                                              dim_feedforward=fine_tune_config["transformer"]["dim_feedforward"],
                                                                              dropout=fine_tune_config["transformer"]["dropout_rate"],
                                                                              batch_first=True,
                                                                              device=llm_network.device)
            self.transformer_encoder = torch.nn.TransformerEncoder(self.transformer_encoder_layer,
                                                                   num_layers=fine_tune_config["transformer"]["num_layers"])                                                            

        reached_first_index = False
        if fine_tune_config["freeze_base_model"]:
            for param in self.llm_network.parameters():
                param.requires_grad = False
        
        elif fine_tune_config["freeze_until_layer"] > 0:
            for i, (name, param) in enumerate(self.llm_network.named_parameters()):
                try:
                    layer_no = int(name.split(".")[2])
                    if layer_no < fine_tune_config["freeze_until_layer"]:
                        param.requires_grad = False
                    reached_first_index = True
                except:
                    if not reached_first_index:
                        param.requires_grad = False
                    continue
        
        for name, param in self.llm_network.named_parameters():
            if not param.requires_grad:
                # print in red color
                print(f"\033[91m name: {name}, requires_grad: {param.requires_grad}, shape: {param.shape} \033[00m")
            else:
                print(f"name: {name}, requires_grad: {param.requires_grad}, shape: {param.shape}")
                
        self.regression_layers= torch.nn.ModuleList()
        if self.fine_tune_config["activation_function"] == "relu":
            self.activation_function = torch.nn.ReLU()
        elif self.fine_tune_config["activation_function"] == "sigmoid":
            self.activation_function = torch.nn.Sigmoid()
        elif self.fine_tune_config["activation_function"] == "silu":
            self.activation_function = torch.nn.SiLU()
        
        for i in range(len(self.fine_tune_config["hidden_sizes"])):
            if i == 0:
                self.regression_layers.append(torch.nn.Linear(self.fine_tune_config["feature_size"],
                                                              self.fine_tune_config["hidden_sizes"][i]))
            else:
                self.regression_layers.append(torch.nn.Linear(self.fine_tune_config["hidden_sizes"][i-1],
                                                              self.fine_tune_config["hidden_sizes"][i]))
            if i != len(self.fine_tune_config["hidden_sizes"]) - 1:
                self.regression_layers.append(self.activation_function)
                self.regression_layers.append(torch.nn.Dropout(self.fine_tune_config["dropout_rate"]))
            

    def forward(self,
                token_ids,
                mask_ids):
        if self.is_fp16:
            with autocast():
                outputs = self.llm_network(input_ids=token_ids,
                                           attention_mask=mask_ids,
                                           use_cache=self.llm_config["use_cache"],
                                           return_dict=self.llm_config["return_dict"],
                                           output_attentions=self.llm_config["output_attentions"],
                                           output_hidden_states=self.llm_config["output_hidden_states"],)
                # end_idxs = torch.where(token_ids == self.llm_config["eos_token_id"])[1]
                end_idxs = torch.zeros((0,), device=token_ids.device)
                for b_no in range(len(token_ids)):
                    end_idx = torch.where(token_ids[b_no, :] == self.llm_config["eos_token_id"])[0]
                    if len(end_idx) > 0:
                        end_idxs = torch.cat((end_idxs, end_idx[-1].unsqueeze(0)))
                    else:
                        end_idxs = torch.cat((end_idxs, 
                                              torch.tensor([token_ids.shape[1]-1], device=token_ids.device)))
                end_idxs = end_idxs.long()
                hidden_states = outputs.hidden_states[-1] # (num_layers, batch_size, seq_len, hidden_size)
                if self.fine_tune_config["use_attention"]:
                    hidden_states = self.transformer_encoder(hidden_states,
                                                                src_key_padding_mask=mask_ids)
                filtered_hidden_states = torch.zeros((hidden_states.shape[0], 
                                                        hidden_states.shape[-1]),
                                                        device=token_ids.device)

                for i in range(hidden_states.shape[0]):
                    filtered_hidden_states[i, :] = hidden_states[i, :end_idxs[i]+1, :].mean(dim=0)
                    # print(hidden_states[i, :end_idxs[i]+1, :].

                for layer in self.regression_layers:
                    filtered_hidden_states = layer(filtered_hidden_states)                  
                final_outputs = filtered_hidden_states


        else:
            # Add this else branch to handle the non-FP16 case
            outputs = self.llm_network(input_ids=token_ids,
                                    attention_mask=mask_ids,
                                    use_cache=self.llm_config["use_cache"],
                                    return_dict=self.llm_config["return_dict"],
                                    output_attentions=self.llm_config["output_attentions"],
                                    output_hidden_states=self.llm_config["output_hidden_states"],)
            
            # Copy the same processing logic from the FP16 case
            end_idxs = torch.zeros((0,), device=token_ids.device)
            for b_no in range(len(token_ids)):
                end_idx = torch.where(token_ids[b_no, :] == self.llm_config["eos_token_id"])[0]
                if len(end_idx) > 0:
                    end_idxs = torch.cat((end_idxs, end_idx[-1].unsqueeze(0)))
                else:
                    end_idxs = torch.cat((end_idxs, 
                                        torch.tensor([token_ids.shape[1]-1], device=token_ids.device)))
            end_idxs = end_idxs.long()
            hidden_states = outputs.hidden_states[-1]
            if self.fine_tune_config["use_attention"]:
                hidden_states = self.transformer_encoder(hidden_states,
                                                        src_key_padding_mask=mask_ids)
            filtered_hidden_states = torch.zeros((hidden_states.shape[0], 
                                                hidden_states.shape[-1]),
                                                device=token_ids.device)

            for i in range(hidden_states.shape[0]):
                filtered_hidden_states[i, :] = hidden_states[i, :end_idxs[i]+1, :].mean(dim=0)

            for layer in self.regression_layers:
                filtered_hidden_states = layer(filtered_hidden_states)
            final_outputs = filtered_hidden_states
        return final_outputs

if __name__ == "__main__":
    pass
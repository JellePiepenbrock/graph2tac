import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import random
# os.environ["CUDA_VISIBLE_DEVICES"] = f"{random.randint(0,7)}" # pick a random gpu
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import socket
import capnp
# import pytact.common
# import pytact.graph_visualize as gv
import torch
from transformers import GPT2LMHeadModel
from transformers import GPT2TokenizerFast
from transformers import GPT2Config
import pkg_resources
import time
import argparse
import pickle
import GPUtil
from typing import Optional, Tuple, Union
import torch
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from torch.nn import CrossEntropyLoss


parser = argparse.ArgumentParser(description='Transformer Server')

parser.add_argument('--model', type=str, help="Location of the transformer .bin file")
parser.add_argument('--tokenizer', type=str, help="Location of the tokenizer files (folder)")
parser.add_argument('--beam_width', type=int, help="How many alternative answers to keep in beam")
parser.add_argument('--dev', type=str, help="cuda | cpu", default="cuda")
parser.add_argument('--pt_threads', type=int, help="maximum threads for PyTorch", default=1)
parser.add_argument('--temp', type=float, help="temperature for softmax", default=1.0)
parser.add_argument('--sample', type=int, help="use top-k sampling | off by default", default=0)
parser.add_argument('--topk', type=int, help="number of tokens to take into account in top_k sampling - needs to be set if --sample is on.", default=-1)
parser.add_argument('--truncate_side', type=str, help= "left or right truncation of proof states that are too long", default="right")

args = parser.parse_args()
model_location = args.model
tokenizer_location = args.tokenizer
beam_w = args.beam_width
device = args.dev
n_threads = args.pt_threads
temperature = args.temp
sampling_on = bool(args.sample)
topk = args.topk
truncate_side = args.truncate_side
logspace = False
log_base = -1

if topk == -1 and sampling_on:
    raise ValueError("you turned on top k sampling, supply a number to --topk; for example 20")

# if device == "cuda":
#     DEVICE_ID_LIST  = GPUtil.getFirstAvailable(order='memory', maxLoad=0.8, maxMemory=0.8, attempts=1, interval=900,
#                                         verbose=False)
#     DEVICE_ID = DEVICE_ID_LIST[0]
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

torch.set_num_threads(n_threads)
torch.set_num_interop_threads(n_threads)

#beam_w = 10
# device = "cuda"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

capnp.remove_import_hook()

# graph_api_capnp = pytact.common.graph_api_capnp()
# graph_api_capnp = capnp.load(graph_api_capnp)

graph_api_filename = pkg_resources.resource_filename('graph2tac.loader','clib/graph_api_v15.capnp')
graph_api_capnp = capnp.load(graph_api_filename)

def log_normalize(x):

    temp = torch.logsumexp(x, dim=0)

    return x - temp

# Special forward to control temperature directly.
class CustomGPT2LMHeadModel(GPT2LMHeadModel):

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
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
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

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states) / temperature

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


def load_eval_setup(toksave, model_location):
    """"
    Load a tokenizer and a model from the specified locations.


    """

    tokenizer = GPT2TokenizerFast.from_pretrained(toksave, max_len=1024)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'eos_token': '<END>'})
    tokenizer.add_special_tokens({'bos_token': '<END>'})

    config = GPT2Config(vocab_size = 3003,
                        n_positions = 1024,
                        n_embd = 768,
                        n_head = 12,
                        n_layer = 12,
                        eos_token_id=tokenizer(["<END>"]).input_ids[0][0],
                        bos_token_id=tokenizer(["<END>"]).input_ids[0][0],
                        pad_token_id=tokenizer(["[PAD]"]).input_ids[0][0]
                        )

    model = CustomGPT2LMHeadModel(config=config)

    model.load_state_dict(torch.load(model_location, map_location=torch.device('cpu')))
    model.eval()

    model = model.to(device)

    if truncate_side == "left":
        tokenizer.truncation_side = 'left'

    return tokenizer, model

def generate(input_proof_state, tokenizer, model):
    """"
    Beam search with width beam_w, returns best beam_w sequences.

    Input:

    - string version of proofstate
    - tokenizer instance
    - gpt2 model instance

    Output:

    - [String]
    """

    sample = input_proof_state + " OUTPUT"
    print("SAMPLE")

    #print("----ä")
    
    print(sample)

    input_ids = tokenizer([sample], truncation=True, max_length=970, return_tensors="pt", padding=False).input_ids.to(device)
    
    #if input_ids.shape[1] > 1024:
    #    input_ids = input_ids[:, :1023]
        # raise ValueError("Input is too long")
    if not sampling_on:

        beam_output = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 50,
            num_beams=beam_w,
            early_stopping=True,
            num_return_sequences=beam_w,
            eos_token_id=tokenizer(["<END>"]).input_ids[0][0],
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            length_penalty=0,
        )
    else:
        beam_output = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 50,
            num_beams=beam_w,
            early_stopping=True,
            num_return_sequences=beam_w,
            eos_token_id=tokenizer(["<END>"]).input_ids[0][0],
            do_sample=True,
            output_scores=True,
            top_k=topk,
            return_dict_in_generate=True,
            length_penalty=0,
            temperature=temperature,
        )

    return_list = []
    for e, i in enumerate(beam_output['sequences']):
        #print(input_ids.shape[1])
        #print(tokenizer.decode(i[input_ids.shape[1]:], skip_special_tokens=True))
        #print(tokenizer.decode(i, skip_special_tokens=True))
        model_suggestion = tokenizer.decode(i[input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        model_suggestion = model_suggestion.rstrip()
        model_suggestion = model_suggestion.lstrip()
        return_list.append(model_suggestion)
    
    #for sugg in return_list:
    #    print(sugg)

    # sequence_probs = torch.exp(beam_output['sequences_scores']).tolist()
    if not logspace:
        seq_probs = torch.exp(beam_output['sequences_scores'])

        normalized_sequence_probs = seq_probs / torch.sum(seq_probs)
    else:
        if log_base == -1:
            # ln base
            normalized_sequence_probs = log_normalize(beam_output['sequences_scores'])
        else:
            # change of base: log_2 (x) = log_e(x) / log_e(2)
            normalized_sequence_probs = log_normalize(beam_output['sequences_scores']) / torch.log(torch.tensor(log_base))
    #sequence_probs_list = [k.item() for k in sequence_probs]

    return return_list, normalized_sequence_probs.tolist()

def prediction_loop_text(r, s, tokenizer, model):
        
    no_answers = 0
    total_reqs = 0
    for g in r:
        #print("g")
        #print(g)
        msg_type = g.which()
        #print("innerloop: message_type")
        #print(msg_type)
        if msg_type == "predict":
            #print(g.predict.state.text)
            #print(generate(g.predict.state.text, tokenizer, model))
            
            #print("STATE")
            #print(g.predict.state.text)
    

            st = time.time()

            tactics, probs = generate(g.predict.state.text.lstrip(), tokenizer, model)

            et = time.time() - st
            #print(f"Prediction takes {et} seconds")
            preds = [
                {'tacticText': t,
                 'confidence': p} for (t, p) in zip(tactics,probs) ]
            #print(preds)
            response = graph_api_capnp.PredictionProtocol.Response.new_message(textPrediction=preds)
            #print("RESPONSE") 
            #print(response)
            response.write_packed(s)
            
            #time.sleep(1)
        elif msg_type == "synchronize":
            #print("predlooptext", g)
            response = graph_api_capnp.PredictionProtocol.Response.new_message(synchronized=g.synchronize)
            #print(response)
            response.write_packed(s)
        elif msg_type == "initialize":
            return g
        else:
            print("Capnp protocol error")
            raise Exception

def initialize_loop(r, s, textmode, tokenizer, model):

    for g in r:
    #g = next(r)
        msg_type = g.which()
        #print("outerloop: Message type: ")
        #print(msg_type)
        if msg_type == "initialize":
            while g:
                #print('---------------- New prediction context -----------------')
                if not textmode:
                    gv.visualize_defs(g.initialize.graph, g.initialize.definitions)
                   # print(g.initialize.tactics)
                response = graph_api_capnp.PredictionProtocol.Response.new_message(initialized=None)
                response.write_packed(s)
                #print(response)
    
                #time.sleep(1)
                if textmode:
                    g = prediction_loop_text(r, s, tokenizer, model)
                else:
                    tacs = list(g.initialize.tactics)
                    g = prediction_loop(r, s, tacs, g.initialize.graph, g.initialize.definitions)
        elif msg_type == "synchronize":
            #print("outerloop", g)
            response = graph_api_capnp.PredictionProtocol.Response.new_message(synchronized=g.synchronize)
            #print(response)
            response.write_packed(s)
            initialize_loop(r, s, textmode, tokenizer, model)
        else:
            print("Capnp protocol error")
            raise Exception

def main():
        
    tokenizer, model = load_eval_setup(tokenizer_location, model_location)
    # print("Model Loaded")
    # print(sys.stdin.fileno())


    host = '127.0.0.1'
    port = 33333
    tcp = False
    textmode = True
    if not tcp:
        s = socket.socket(fileno=sys.stdin.fileno())
        reader = graph_api_capnp.PredictionProtocol.Request.read_multiple_packed(s, traversal_limit_in_words=2**64-1)

        initialize_loop(reader, s, textmode, tokenizer, model)
    else:
        print("TCP MODE")
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((host, port))

        try: 
            server_sock.listen(1)
            session_idx = 0

            while True:
                sock, remove_addr = server_sock.accept()
                reader = graph_api_capnp.PredictionProtocol.Request.read_multiple_packed(sock, traversal_limit_in_words=2**64-1)
                initialize_loop(reader, sock, textmode, tokenizer, model)

                session_idx += 1
        
        finally:
            server_sock.close()

        

    #if sys.argv[1] == 'text':
    #    print('Python server running in text mode')
    #    textmode = True
    #elif sys.argv[1] == 'graph':
    #    print('Python server running in graph mode')
    #    textmode = False
    #r = graph_api_capnp.PredictionProtocol.Request.read_multiple_packed(s, traversal_limit_in_words=2**64-1)
    #initialize_loop(r, s, textmode, tokenizer, model)

if __name__ == '__main__':
    main()

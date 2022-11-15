import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
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
import random
parser = argparse.ArgumentParser(description='Transformer Server')

parser.add_argument('--model', type=str, help="Location of the transformer .bin file")
parser.add_argument('--tokenizer', type=str, help="Location of the tokenizer files (folder)")
parser.add_argument('--beam_width', type=int, help="How many alternative answers to keep in beam")
args = parser.parse_args()
model_location = args.model
tokenizer_location= args.tokenizer
beam_w = args.beam_width

model_location = "/home/piepejel/projects/coq-gpt-train/1110/checkpoint-3189000/pytorch_model.bin"
tokenizer_location = "/home/piepejel/projects/coq-gpt-train/1110/"
beam_w = 10
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda:4"
capnp.remove_import_hook()

cheat = False
if cheat:
    with open('/home/piepejel/projects/coq-gpt-train/data/answers.pickle', 'rb') as handle:
            answer_dict = pickle.load(handle)
# graph_api_capnp = pytact.common.graph_api_capnp()
# graph_api_capnp = capnp.load(graph_api_capnp)

graph_api_filename = pkg_resources.resource_filename('graph2tac.loader','clib/graph_api_v11.capnp')
graph_api_capnp = capnp.load(graph_api_filename)

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
                        n_embd = 192,
                        n_head = 12,
                        n_layer = 8,
                        eos_token_id=tokenizer(["<END>"]).input_ids[0][0],
                        bos_token_id=tokenizer(["<END>"]).input_ids[0][0],
                        pad_token_id=tokenizer(["[PAD]"]).input_ids[0][0]
                        )

    model = GPT2LMHeadModel(config=config)
    device = "cuda"
    model.load_state_dict(torch.load(model_location, map_location=torch.device('cpu')))
    model.eval()


    model = model.to(device)

    return tokenizer, model



def generate(input_proof_state, tokenizer, model):
    """"
    Beam search with width 10, returns best 10 sequences.

    Input:

    - string version of proofstate
    - tokenizer instance
    - gpt2 model instance

    Output:

    - [String]
    """

    sample = input_proof_state + " OUTPUT"
    #print("SAMPLE")
    print("----ä")
    
    print(sample)
    device = "cuda"
    input_ids = tokenizer([sample], truncation=True, max_length=900, return_tensors="pt", padding=False).input_ids.to(device)
    
    #if input_ids.shape[1] > 1024:
    #    input_ids = input_ids[:, :1023]
        # raise ValueError("Input is too long")

    beam_output = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + 50,
        num_beams=10,
        early_stopping=True,
        num_return_sequences=10,
        eos_token_id=tokenizer(["<END>"]).input_ids[0][0],
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
        length_penalty=0,
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
    
    for sugg in return_list:
        print(sugg)
    sequence_probs = torch.exp(beam_output['sequences_scores']).tolist()
    #sequence_probs_list = [k.item() for k in sequence_probs]
    return return_list, sequence_probs

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
            
            print("STATE")
            print(g.predict.state.text)
    

            st = time.time()
            if not cheat:
                tactics, probs = generate(g.predict.state.text.lstrip(), tokenizer, model)
            elif cheat:
                current_proofstate = g.predict.state.text.lstrip()
                total_reqs += 1
                if current_proofstate in answer_dict:
                    num_answers = len(answer_dict[current_proofstate])
                    sample_size = min(10, num_answers)
                    print(current_proofstate)
                    
                    tactics = random.sample(answer_dict[current_proofstate], k = sample_size)
                    print(tactics)
                    probs = [1.0/len(tactics) for k in range(len(tactics))]
                else:
                    no_answers += 1
                    print(f"No matching proofstates for {no_answers} / {total_reqs}")
                    print(current_proofstate)
                    print("------------")
                    tactics = []
                    probs = [1.0]
                #tactics = [answer_dict[g.predict.state.text]]
            et = time.time() - st
            print(f"Prediction takes {et} seconds")
            preds = [
                {'tacticText': t,
                 'confidence': p} for (t, p) in zip(tactics,probs) ]
            print(preds)
            response = graph_api_capnp.PredictionProtocol.Response.new_message(textPrediction=preds)
            #print("RESPONSE") 
            print(response)
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
    print("Model Loaded")
    print(sys.stdin.fileno())
    

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

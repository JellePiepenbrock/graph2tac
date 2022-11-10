import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import socket
import capnp
# import pytact.common
# import pytact.graph_visualize as gv
import pkg_resources
import time
import argparse
import pickle
import random
parser = argparse.ArgumentParser(description='Text Server that can be loaded with a dictionary that contains proofstate -> [tactics] ')

parser.add_argument('--model', type=str, help="Location of the pickled dictionary")
parser.add_argument('--beam_width', type=int, help="How many alternative answers to send")
args = parser.parse_args()

model_location = args.model
beam_w = args.beam_width

capnp.remove_import_hook()


# graph_api_capnp = pytact.common.graph_api_capnp()
# graph_api_capnp = capnp.load(graph_api_capnp)

graph_api_filename = pkg_resources.resource_filename('graph2tac.loader','clib/graph_api_v11.capnp')
graph_api_capnp = capnp.load(graph_api_filename)

def prediction_loop_text(r, s, tokenizer, model):
        
    no_answers = 0
    total_reqs = 0
    for g in r:
        
        msg_type = g.which()

        if msg_type == "predict":
            
            print("STATE")
            print(g.predict.state.text)
    

            st = time.time()
            
        
            current_proofstate = g.predict.state.text
            total_reqs += 1
            if current_proofstate in model:
                num_answers = len(model[current_proofstate])
                sample_size = min(beam_w, num_answers)
                tactics = random.sample(model[current_proofstate], k = sample_size)
                probs = [1.0/len(tactics) for k in range(len(tactics))]
            else:
                no_answers += 1
                print(f"No matching proofstates for {no_answers} / {total_reqs}")
                print(current_proofstate)
                print("------------")
                tactics = []
                probs = []
                #tactics = [answer_dict[g.predict.state.text]]
            et = time.time() - st
            print(f"Prediction takes {et} seconds")
            preds = [
                {'tacticText': t,
                 'confidence': p} for (t, p) in zip(tactics,probs) ]
            print(preds)
            response = graph_api_capnp.PredictionProtocol.Response.new_message(textPrediction=preds)
    
            print(response)
            response.write_packed(s)
            
            
        elif msg_type == "synchronize":
            
            response = graph_api_capnp.PredictionProtocol.Response.new_message(synchronized=g.synchronize)
            
            response.write_packed(s)
        elif msg_type == "initialize":
            return g
        else:
            print("Capnp protocol error")
            raise Exception

def initialize_loop(r, s, textmode, model):

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
                   # print(g.initialize.:tactics)
                response = graph_api_capnp.PredictionProtocol.Response.new_message(initialized=None)
                response.write_packed(s)
                #print(response)
    
                #time.sleep(1)
                if textmode:
                    g = prediction_loop_text(r, s, model)
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
        
        
    with open('/home/piepejel/projects/coq-gpt-train/data/answers.pickle', 'rb') as handle:
            model = pickle.load(handle)
    print("Model Loaded")
    print(sys.stdin.fileno())
    

    host = '127.0.0.1'
    port = 33333
    tcp = False
    textmode = True
    if not tcp:
        s = socket.socket(fileno=sys.stdin.fileno())
        reader = graph_api_capnp.PredictionProtocol.Request.read_multiple_packed(s, traversal_limit_in_words=2**64-1)

        initialize_loop(reader, s, textmode, model)
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
                initialize_loop(reader, sock, textmode, model)

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

import json
import unittest
from unittest.mock import patch
import openai

import pytest

##################
####
#### set for openai 

#MODEL_NAME = "gpt-4o-mini"
#client =openai. OpenAI(api_key="sk-ac47gsEsGuxxxxxxxxx")

####
##################


##################
####
#### set for local server, vLLM
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
#client =openai. OpenAI(base_url="http://127.0.0.1:8002/v1/",api_key="123")


### install the server  from https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# vllm serve meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port  8002  --tensor-parallel-size 4 

####
##################



##################
####
#### set for nvidia NIM

#MODEL_NAME = "meta/llama-3.2-3b-instruct"
#client =openai. OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key="nvapi-xxxxxxxxx")

####
##################

##################
####
#### set for bitdeer
#MODEL_NAME = "meta/llama-3.1-8b-instruct"
#client =openai. OpenAI(base_url="https://www.bitdeer.ai/api/inference/v1/",api_key="xxxxxxxx")

####
##################


print ("please paste the setting or uncomment MODEL_NAME and client in then above, then delete the exit")
exit(1 )

#global_messages=[{"role":"system","content":"you are a helpful coding assistant"},{"role": "user", "content": "you are tasking to output the code to implementation the fibonacci sequence "}]
global_messages=[{"role":"system","content":"you are a helpful coding assistant"},{"role": "user", "content": "you are tasking to output the code to implementation the average score of a list "}]


def test_temperature_zero_same_seed():
    """
    When temperature=0 and the same seed is used, the output should be deterministic.
    """
    seed = 12345  # Fix seed for deterministic response
    seed2 = 3456  # Fix seed for deterministic response
    temperature = 0
    response1 = client.chat.completions.create(
        model=MODEL_NAME,
        messages=global_messages , 
        temperature=temperature,
        top_p=0.95,
        seed=seed,
    )

    response2 = client.chat.completions.create(
        model=MODEL_NAME,
        messages=global_messages , 
        temperature=temperature,
        top_p=0.95,
        seed=seed,
    )
    #with open("test_temperature.jsonl","a") as fw :
    #    info = {"test_name":"test_temperature_zero_same_seed","expect1":response1.choices[0].message.content , "expect2":response2.choices[0].message.content,"judge":"eq" } 
    #    fw.write( json.dumps(info)+"\n" )
    assert response1.choices[0].message.content[:128] == \
            response2.choices[0].message.content[:128]

@pytest.mark.skipif( "openai.com/" in str(client.base_url), reason="openai: This feature is in Beta. If specified, our system will make a best effort to sample deterministically  ")
def test_temperature_zero_diff_seed():
    """
    When temperature=0 and the same seed is used, the output should be deterministic.
    """
    seed = 12345  # Fix seed for deterministic response
    seed2 = 3456  # Fix seed for deterministic response
    temperature = 0
    response1 = client.chat.completions.create(
        model=MODEL_NAME,
        messages=global_messages , 
        temperature=temperature,
        seed=seed,
    )
    response3 = client.chat.completions.create(
        model=MODEL_NAME,
        messages=global_messages , 
        temperature=temperature,
        seed=seed2,
    )
    #with open("test_temperature.jsonl","a") as fw :
    #    info = {"test_name":"test_temperature_zero_same_seed2","expect1":response1.choices[0].message.content , "expect2":response3.choices[0].message.content,"judge":"eq" } 
    #    fw.write( json.dumps(info)+"\n" )

    assert response1.choices[0].message.content == \
                     response3.choices[0].message.content


def test_temperature_nonzero_same_seed():
    """
    When temperature>0 and different seeds are used, outputs should be stochastic.
    """
    # Configure the mock to return different results for different calls

    seed1 = 12345
    seed2 = 54321
    temperature = 0.8
    response1 = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "you are tasking to output the code to implementation the fibonacci sequence "}],
        temperature=temperature,
        seed=seed1,
    )

    response1_same_seed = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "you are tasking to output the code to implementation the fibonacci sequence "}],
        temperature=temperature,
        seed=seed1,
    )
    assert response1.choices[0].message.content == \
                        response1_same_seed.choices[0].message.content


def test_temperature_nonzero_different_seed():
    """
    When temperature>0 and different seeds are used, outputs should be stochastic.
    """
    # Configure the mock to return different results for different calls

    seed1 = 12345
    seed2 = 54321
    temperature = 0.8
    response1 = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "you are tasking to output the code to implementation the fibonacci sequence "}],
        temperature=temperature,
        seed=seed1,
    )
    response2 = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "you are tasking to output the code to implementation the fibonacci sequence "}],
        temperature=temperature,
        seed=seed2,
    )


    assert response1.choices[0].message.content !=  \
                        response2.choices[0].message.content






@pytest.mark.skipif( "openai.com/" in str(client.base_url), reason="openai: This feature is in Beta for openai ")
def test_seed_deterministically_for_programming_code():
    """
    This may be the same as the previous test_temperature_zero_same_seed ... , this is for code generation's deterministic; We may argue that code generation may ignore such code determinism when given greedy and fixed seed. Let's try it
    """
    seed1 = 12345
    seed2 = 54321
    temperature_0 = 0
    temperature_8 = 0.8
    response_8_1_retries = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "you are tasking to output the code to implementation the fibonacci sequence "}],
        temperature=temperature_8,
        seed=seed1,
        n=1,
    )
    response_8_1 = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "you are tasking to output the code to implementation the fibonacci sequence "}],
        temperature=temperature_8,
        seed=seed1,
        n=1,
    )
    assert  (response_8_1.choices[0].message.content ==                         response_8_1_retries.choices[-1].message.content)

    response_8_1_sed2 = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "you are tasking to output the code to implementation the fibonacci sequence "}],
        temperature=temperature_8,
        seed=seed2,
#        top_p=0.90,
        n=1,
    )
    assert  (response_8_1.choices[0].message.content !=         response_8_1_sed2.choices[-1].message.content)

def test_n_samples():
    """
    When n:int is set, the output should be list[response]
    """
    seed1 = 12345
    seed2 = 54321
    temperature_0 = 0
    temperature_8 = 0.8
    response_0_1 = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "you are tasking to output the code to implementation the fibonacci sequence "}],
        temperature=temperature_0,
        #seed=seed1,
        n=1
    )
    assert len( response_0_1.choices) == 1 

@pytest.mark.skipif( "nvidia.com/" in str(client.base_url), reason="nim not support the n_samples, https://platform.openai.com/docs/api-reference/chat/create#chat-create-n  ")
def test_n_samples_greater_than_1():
    """
    https://platform.openai.com/docs/api-reference/chat/create#chat-create-n 
    Please take a look at this API, it is important to us. 
    Unfortunately, bitdeer.ai does not support it, instead, its output is a stringified result of the list[n_response].
    You should investigate why to output a single string, instead of a  list.
    """
    seed1 = 12345
    seed2 = 54321
    temperature_0 = 0
    temperature_8 = 0.8
    response_8_10 = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "you are tasking to output the code to implementation the fibonacci sequence "}],
        temperature=temperature_8,
        #seed=seed1,
        n=10,
    )

    assert  len( response_8_10.choices) == 10 


    msg_list = [x.message.content for x in response_8_10.choices ] 
    assert len(msg_list) ==10 , ( len(msg_list) , ) 
    #with open("a1.jsonl","w") as fw :
    #    fw.write( "\n".join( [json.dumps(x) for x in msg_list ]))
    assert len(msg_list) >1,( msg_list , response_8_10.choices ) 
    assert len(set(msg_list)) >1,( msg_list , response_8_10.choices ) 



def test_stop_words():
    """
    https://platform.openai.com/docs/api-reference/chat/create#chat-create-stop
    indicate the stop words:List[str] to stop generating further tokens
    """
    seed1 = 12345
    seed2 = 54321
    temperature_0 = 0
    temperature_8 = 0.8
    

    response_8_10 = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "echo 1 to 10 digits and ensure to end your response with [/STOP]."}],
        temperature=temperature_8,
        seed=seed1,
        n=1,
        max_tokens=128,
    )
    msg ="\n".join( [x.message.content for x in response_8_10.choices ]  )
    assert "STOP"  in msg , msg 

    response_8_10 = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "echo 1 to 10 digits and ensure to end your response with [/STOP]."}],
        temperature=temperature_8,
        seed=seed1,
        stop=["[/STOP]","[STOP]","STOP"],
        n=1,
        max_tokens=128,
    )
    msg ="\n".join( [x.message.content for x in response_8_10.choices ]  )
    assert "STOP" not in msg , msg 

    response_8_10 = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "echo 1 to 10 digits and ensure to end your response with [/STOP] [/OVER]."}],
        temperature=temperature_8,
        seed=seed1,
        stop=["[/OVER]","[OVER]","OVER"],
        n=1,
        max_tokens=128,
    )
    msg ="\n".join( [x.message.content for x in response_8_10.choices ]  )
    assert "STOP"  in msg , msg 
    assert "OVER" not in msg , msg 

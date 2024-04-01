from langchain.llms import HuggingFaceTextGenInference
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# LLM inference
llm = HuggingFaceTextGenInference(
    inference_server_url="http://130.207.125.98:8080/",
    max_new_tokens=128,
    # top_k=10,
    # top_p=0.95,
    # typical_p=0.95,
    # temperature=0.9,
    # repetition_penalty=1.15
)


llama2_prompt ="""
    <s>[INST] <<SYS>>
    {system_message}
    <</SYS>>

    hello, I am test [/INST] I'm a large language model, so I don't have feelings like humans do, but I'm always happy to chat with you. Is there something specific you'd like to talk about or ask me? I'm here to help with any questions you might have. </s><s>[INST] {human_input} [/INST]
"""



output = llm(llama2_prompt.format(system_message="You are a Chatbot", human_input="Hello, do you know what time it is?"))
print(output)

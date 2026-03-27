from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm =  HuggingFacePipeline.from_model_id(
    model_id='MiniMaxAI/MiniMax-M2.5',
    task='text-generation',
    pipeline_kwargs={
        'temperature': 0.7,
        'max_new_tokens': 100
    }
)
model = ChatHuggingFace(llm = llm)

result = model.invoke("What is the capital of India?")

print(result.content)
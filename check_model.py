from rag_chatbot.pipeline import LocalRAGPipeline

pipeline = LocalRAGPipeline()
print("✅ Model in use:", pipeline.get_model_name())

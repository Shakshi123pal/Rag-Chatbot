from rag_chatbot.pipeline import LocalRAGPipeline

# 1️⃣ Pipeline initialize karo
pipeline = LocalRAGPipeline()

# 2️⃣ Dummy chatbot context (UI ke liye placeholder)
# Ye list of list of strings hai, jisme pehle se koi chat history ho sakti hai
chatbot = [["User", "Hi"], ["Bot", "Hello!"]]

# 3️⃣ User message (jo normally UI se aata hai)
message = "Explain Retrieval-Augmented Generation in simple words."

# 4️⃣ Mode select karo (default / streaming / whatever your pipeline supports)
mode = "default"

# 5️⃣ Query pipeline
response = pipeline.query(mode, message, chatbot)

# 6️⃣ Output dekho
print("\n🤖 Chatbot Response:\n", response)

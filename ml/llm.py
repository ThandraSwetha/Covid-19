import google.generativeai as genai 

genai.configure(api_key = "AIzaSyDkqHB0LjhtZaiF_uTqQZnFeKPoCmwPCNA")

model = genai.GenerativeModel(model_name = "gemini-2.0-flash")

chat = model.start_chat(history=[])

while True:
    prmt = input("Enter Prompt")
    if (prmt =="exit"):
        break

    res = chat.send_message(prmt)

    print(res.text)
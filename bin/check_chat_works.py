import ollama

client = ollama.Client()

model = "dolphin3"
prompt = "Hi there"

response = client.generate(model=model, prompt=prompt)

print("Response from ollama:")
print(response.response)

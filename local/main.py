from transformers import pipeline

model = pipeline("text-classification")
res = model("good job")
print(res)

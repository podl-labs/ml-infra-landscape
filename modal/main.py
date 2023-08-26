import modal

stub = modal.Stub("text-classification")
transformers_image = modal.Image.debian_slim().pip_install("transformers", "tensorflow")

@stub.function(image=transformers_image)
def get_sentiment(text: str):
    from transformers import pipeline

    model = pipeline("text-classification")
    return model(text)

@stub.local_entrypoint()
def main():
    res = get_sentiment.remote("good job")
    print(res)

from beam import App, Runtime, Image
from transformers import pipeline

app = App(
    name="text-classification",
    runtime=Runtime(
        image=Image(
            python_version="python3.10",
            python_packages=["transformers", "tensorflow"],
        ),
    ),
)


def get_sentiment(text: str):
    model = pipeline("text-classification")
    return model(text)


@app.run()
def main():
    res = get_sentiment("good job")
    print(res)

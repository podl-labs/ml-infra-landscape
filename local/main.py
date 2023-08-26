from transformers import pipeline

def get_sentiment(text: str):
    model = pipeline("text-classification")
    return model(text)

def main():
    res = get_sentiment("good job")
    print(res)

if __name__ == "__main__":
    main()
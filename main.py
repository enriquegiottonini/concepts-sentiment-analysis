import math
import random


def load_data(filepath: str):
    with open(filepath, "r", encoding="latin-1") as file:
        for _ in range(100):
            yield next(file).strip()


stopwords = ("and", "or", "a", "is", "are", "to", "in", "that", "the", "as", "at")


def tokenize(x: str) -> list[str]:
    tokens = []
    for token in x.split(" "):
        if "'" in token:
            token = token.split("'", 1)[0]
        if token.startswith(("#", "@")):
            continue
        if token in stopwords:
            continue
        if token.endswith(("s", "?", "!", "%")):
            token = token[:-1]
        if token.endswith(("ed")):
            token = token[:-2]
        if token.isnumeric():
            continue

        tokens.append(token.lower())
    return tokens


# def feature_extractor(tweet: str, idxlookup: dict) -> list[int]:
#     x = [0] * len(idxlookup)
#     for token in tokenize(tweet):
#         index_in_vocab = idxlookup[token]
#         x[index_in_vocab] = 1
#     return x
#
def feature_extractor(tweet: str, freqs: dict):
    x = [1, 0, 0]
    for token in set(tokenize(tweet)):
        x[1] += freqs.get((token, 1), 0)
        x[2] += freqs.get((token, 0), 0)
    return x


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(x))


def predictor(input, weights) -> float:
    z = sum(x * y for x, y in zip(input, weights))
    return sigmoid(z)


def train_validate_test_split(inputs, idxs: list[int]):
    N = len(inputs)
    train_size = math.floor(0.7 * N)
    val_size = math.ceil(0.15 * N)
    test_size = math.ceil(0.15 * N)

    train = [inputs[i] for i in idxs[:train_size]]
    validate = [inputs[i] for i in idxs[train_size : train_size + val_size]]
    test = [inputs[i] for i in idxs[-test_size:]]

    return train, validate, test


def evaluate(preds: list[int], y: list[int]) -> float:
    return sum(preds[i] == y[i] for i in range(len(preds))) / len(preds)


def logreg(preds: list[float], target: list[int]) -> float:
    return -sum(
        target[i] * math.log(preds[i] + 1e-7)
        + (1 - target[i]) * math.log(1 - preds[i] + 1e-7)
        for i in range(len(preds))
    ) / len(preds)


def gradient_descent(X, y, weights, phi, learning_rate: float, iters: int):
    for t in range(iters):
        preds: list[float] = [predictor(phi(X[i]), weights) for i in range(len(X))]
        errors: list[float] = [preds[i] - y[i] for i in range(len(preds))]

        dweights: list[float] = [
            sum(error * xs for error, xs in zip(errors, phi(X[i])))
            for i in range(len(X))
        ]
        weights = [w - learning_rate * dw for w, dw in zip(weights, dweights)]
        # print(f"{weights=}")
        # print(f"{dweights=}")

        if t % 10 == 0:
            loss = logreg(preds, y)
            pred_to_class = [1 if pred < 0.5 else 0 for pred in preds]
            acc = evaluate(pred_to_class, y)
            print(f"{t=} {loss=:.5f} {acc=:.5f}")

    return weights


def main():
    # data = load_data("./dataset.txt")
    # sdata = [line.rsplit(",", 1) for line in data]
    # X: list[str] = [line[0] for line in sdata]
    # y: list[int] = [int(line[1]) for line in sdata]
    data = load_data("./Ukraine_10K_tweets_sentiment_analysis.csv")
    sdata = (line.rsplit(",", 3) for line in data)
    X: list[str] = []
    y: list[int] = []
    for line in sdata:
        label = line[-1].strip()
        tweet = line[0].strip()
        if label == "Negative":
            X.append(tweet)
            y.append(0)
        elif label == "Positive":
            X.append(tweet)
            y.append(1)

    print(len(X), "tweets")
    # vocab = {}
    # for tweet in X:
    #     for word in tokenize(tweet):
    #         vocab[word] = 1
    # token2index = { token:i for i, token in enumerate(vocab)}
    N = len(X)
    freqs = {}
    for i in range(N):
        tweet = X[i]
        sentiment = y[i]
        for token in tokenize(tweet):
            freqs[(token, sentiment)] = freqs.get((token, sentiment), 0) + 1

    idxs = list(range(N))
    random.seed(42)
    random.shuffle(idxs)

    Xtrain, Xval, Xtest = train_validate_test_split(X, idxs)
    ytrain, yval, ytest = train_validate_test_split(y, idxs)

    weights = [0.01 * random.uniform(-1, 1) for _ in range(3)]
    phi = lambda x: feature_extractor(x, freqs)

    optimal_weights = gradient_descent(Xtrain, ytrain, weights, phi, 1e-9, 100)

    vpreds = [predictor(phi(Xval[i]), optimal_weights) for i in range(len(Xval))]
    vpreds_to_class = [1 if pred < 0.5 else 0 for pred in vpreds]
    loss = logreg(vpreds, yval)
    acc = evaluate(vpreds_to_class, yval)
    print(f"validation {loss=:.5f} {acc=:.2f}")

    xs = Xtest[9]
    ys = ytest[9]
    print(f"tweet = {xs}")
    print(f"features= {phi(xs)}")
    print(f"label = {ys}")
    print(f"predicted = {1 if predictor(phi(xs),optimal_weights) < 0.5 else 0}")


if __name__ == "__main__":
    main()

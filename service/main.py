from fastapi import FastAPI
import joblib

nlp = joblib.load('models/toxic_nlp.joblib')

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/NLP/")
def read_item(q: str = None):
    pred, proba = nlp_predict([q,])
    print(pred, proba)
    # return { "q": str(pred[0])}

    return {
        "predicted":str(pred[0]),
        "proba":str(proba)
    }


def nlp_predict(text):
    """
    :param text: text to classifiy toxic
    :return: toxic or not & probability
    """
    return  nlp.predict(text), nlp.predict_proba(text)
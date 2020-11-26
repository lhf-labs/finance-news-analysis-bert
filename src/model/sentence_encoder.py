from sentence_transformers import SentenceTransformer


def sentence_encoder(model_name):
    model = SentenceTransformer(model_name)
    return model

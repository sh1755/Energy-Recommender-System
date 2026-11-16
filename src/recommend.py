"""Load a trusted local model artifact and rank a requested context."""
import joblib,numpy as np

def recommend(model_path,household_id,daytype=0,k=5):
    artifact=joblib.load(model_path); items=np.arange(daytype*48,(daytype+1)*48)
    scores=artifact['model'].predict(household_id,items); order=np.argsort(-scores,kind='stable')[:k]
    return [{'item':int(items[i]),'slot':int(items[i]%48),'score':float(scores[i])} for i in order]

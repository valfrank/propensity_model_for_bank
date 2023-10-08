from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from backend.database import SessionLocal
from backend.models import Clients
from backend.train_model_func import train_model, predict_on_input, preprocess_data
import pandas as pd
from backend.schema import Client, ClientPrediction, ClientList, Classifiers

app = FastAPI()


def get_session():
    """
    Create a new session
    """
    with SessionLocal() as session:
        return session


@app.get("/sample", response_model=ClientList)
def get_sample_clients(limit: int, db: Session = Depends(get_session)):
    """
    Function that returns a sample from the database
    :param limit: number of rows
    :param db: database connection
    :return: ClientList
    """
    clients = db.query(Clients).limit(limit).all()
    return clients


@app.get("/id", response_model=ClientList)
def get_client_by_id(id: int, db: Session = Depends(get_session)):
    """
    Function that return a row filtered by client id
    :param id: unique client id
    :param db: database connection
    :return: Client
    """
    result = db.query(Clients).filter(Clients.id == id).all()
    if len(result) == 0:
        return JSONResponse(status_code=400, content={"message": f"That id {id} does not exist"})
    else:
        return result


@app.get("/train_model")
def get_train_model(classifier: Classifiers, db: Session = Depends(get_session)):
    """
    Function that train and save model
    :param classifier: name of model
    :param db: database connection
    """
    clients = db.query(Clients).limit(12000).all()

    data = pd.DataFrame([(r.id, r.age, r.gender, r.education, r.marital_status, r.child_total,
                        r.dependants, r.socstatus_work_fl, r.socstatus_pens_fl, r.own_auto,
                        r.fl_presence_fl, r.family_income, r.personal_income, r.credit, r.loan_num_total,
                        r.loan_num_closed, r.target) for r in clients],
                        columns=['id', 'age', 'gender', 'education', 'marital_status', 'child_total',
                               'dependants', 'socstatus_work_fl', 'socstatus_pens_fl', 'own_auto',
                               'fl_presence_fl', 'family_income', 'personal_income', 'credit',
                               'loan_num_total', 'loan_num_closed', 'target'])

    train_model(data, classifier)

    return {"Response": "Training completed."}


@app.post("/predict_model")
def get_predict_model(client: ClientPrediction):
    """
    Function that return prediction
    :param client: dictionary ClientPrediction
    :return: float
    """
    data = pd.DataFrame([[client.age, client.gender, client.education, client.marital_status, client.child_total,
                        client.dependants, client.socstatus_work_fl, client.socstatus_pens_fl, client.own_auto,
                        client.fl_presence_fl, client.family_income, client.personal_income, client.credit,
                        client.loan_num_total, client.loan_num_closed]],
                        columns=['age', 'gender', 'education', 'marital_status', 'child_total',
                                 'dependants', 'socstatus_work_fl', 'socstatus_pens_fl', 'own_auto',
                                 'fl_presence_fl', 'family_income', 'personal_income', 'credit',
                                 'loan_num_total', 'loan_num_closed'])

    data = preprocess_data(data)
    pred = predict_on_input(data)
    return {"Probability of client to deny bank offer": round(pred[0, 0], 2)}



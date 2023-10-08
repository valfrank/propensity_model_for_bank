from pydantic import BaseModel, RootModel, ConfigDict
from enum import Enum
from typing import List


class Client(BaseModel):
    id: int
    age: int
    gender: int
    education: str
    marital_status: str
    child_total: str
    dependants: str
    socstatus_work_fl: int
    socstatus_pens_fl: int
    own_auto: int
    fl_presence_fl: int
    family_income: str
    personal_income: int
    credit: int
    loan_num_total: int
    loan_num_closed: int
    target: int


class ClientPrediction(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    age: int
    gender: int
    education: str
    marital_status: str
    child_total: str
    dependants: str
    socstatus_work_fl: int
    socstatus_pens_fl: int
    own_auto: int
    fl_presence_fl: int
    family_income: str
    personal_income: int
    credit: int
    loan_num_total: int
    loan_num_closed: int


class ClientList(RootModel):
    root: List[Client]


class Classifiers(str, Enum):
    logistic = 'LogisticRegression'
    catboost = 'CatBoost'
    svm = 'SVM'
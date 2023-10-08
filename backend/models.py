from sqlalchemy import Column, Integer, String
from backend.database import Base


class Clients(Base):
    __tablename__ = "clients"

    id = Column(Integer, primary_key=True, name='agreement_rk')
    age = Column(Integer)
    gender = Column(Integer)
    education = Column(String)
    marital_status = Column(String)
    child_total = Column(String)
    dependants = Column(String)
    socstatus_work_fl = Column(Integer)
    socstatus_pens_fl = Column(Integer)
    own_auto = Column(Integer)
    fl_presence_fl = Column(Integer)
    family_income = Column(String)
    personal_income = Column(Integer)
    credit = Column(Integer)
    loan_num_total = Column(Integer)
    loan_num_closed = Column(Integer)
    target = Column(Integer)

import pandas as pd
import dill
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

pickle_path = 'api/pickles/model_dict.pkl'

app = FastAPI()
main_pipline = object

with open(pickle_path, 'rb') as in_strm:
    model_dict = dill.load(in_strm)

metadata = model_dict['metadata']
main_pipline = model_dict['pipline']



class Form(BaseModel):
    Age: int
    BusinessTravel: str
    DailyRate: int
    Department: str
    DistanceFromHome: int
    Education: int
    EducationField: str
    EmployeeCount: int
    EmployeeNumber: int
    EnvironmentSatisfaction: int
    Gender: str
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    Over18: str
    OverTime: str
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StandardHours: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int


class Prediction(BaseModel):
    id: int
    Result: str

@app.get('/status')
def status():
    return "Dismissal project is running!"


@app.get('/version')
def version():
    return metadata


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = main_pipline.predict(df)
    res = 'Yes' if y[0] == 1 else 'No'


    return {
        'id': form.EmployeeNumber,
        'Result': res
    }


# if __name__ == '__main__':
#      uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
#

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

preprocessing_dict = joblib.load('../api/pickles/prep_tools.pickle')

encoder = preprocessing_dict['encoder']
scaler = preprocessing_dict['scaler']

def columnsDropping(X, y=None):
    cols_for_del = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    return X.drop(
          cols_for_del, axis=1)

def additionalColumnsGeneration(X, y=None):
    X['Age_category'] = X.Age.apply(lambda x: 'Old' if x > X.Age.quantile(0.75) else ('Middle' if x <= X.Age.quantile(0.75) and x > X.Age.quantile(0.25) else 'Young') )
    X['DailyRate_category'] = X.DailyRate.apply(lambda x: 'High' if x > X.DailyRate.quantile(0.75) else ('Middle' if x <= X.DailyRate.quantile(0.75) and x > X.DailyRate.quantile(0.25) else 'Small') )
    X['DistanceFromHome_category'] = X.DistanceFromHome.apply(lambda x: 'Far' if x > X.DistanceFromHome.quantile(0.75) else ('Middle' if x <= X.DistanceFromHome.quantile(0.75) and x > X.DistanceFromHome.quantile(0.25) else 'Close') )
    X['Education_category'] = X.Education.apply(lambda x: 'High' if x > X.Education.quantile(0.75) else ('Middle' if x <= X.Education.quantile(0.75) and x > X.Education.quantile(0.25) else 'Small') )
    X['EnvironmentSatisfaction_category'] = X.EnvironmentSatisfaction.apply(lambda x: 'Satisfied' if x > X.EnvironmentSatisfaction.quantile(0.75) else ('Middle' if x <= X.EnvironmentSatisfaction.quantile(0.75) and x > X.EnvironmentSatisfaction.quantile(0.25) else 'Doesnot_Satisfied'))
    X['HourlyRate_category'] = X.HourlyRate.apply(lambda x: 'High' if x > X.HourlyRate.quantile(0.75) else ('Middle' if x <= X.HourlyRate.quantile(0.75) and x > X.HourlyRate.quantile(0.25) else 'Small') )
    X['JobInvolvement_category'] = X.JobInvolvement.apply(lambda x: 'High' if x > X.JobInvolvement.quantile(0.75) else ('Middle' if x <= X.JobInvolvement.quantile(0.75) and x > X.JobInvolvement.quantile(0.25) else 'Small') )
    X['JobLevel_category'] = X.JobLevel.apply(lambda x: 'High' if x > X.JobLevel.quantile(0.75) else ('Middle' if x <= X.JobLevel.quantile(0.75) and x > X.JobLevel.quantile(0.25) else 'Small') )
    X['JobSatisfaction_category'] = X.JobSatisfaction.apply(lambda x: 'Satisfied' if x > X.JobSatisfaction.quantile(0.75) else ('Middle' if x <= X.JobSatisfaction.quantile(0.75) and x > X.JobSatisfaction.quantile(0.25) else 'Doesnot_Satisfied') )
    X['MonthlyIncome_category'] = X.MonthlyIncome.apply(lambda x: 'High' if x > X.MonthlyIncome.quantile(0.75) else ('Middle' if x <= X.MonthlyIncome.quantile(0.75) and x > X.MonthlyIncome.quantile(0.25) else 'Small') )
    X['MonthlyRate_category'] = X.MonthlyRate.apply(lambda x: 'High' if x > X.MonthlyRate.quantile(0.75) else ('Middle' if x <= X.MonthlyRate.quantile(0.75) and x > X.MonthlyRate.quantile(0.25) else 'Small') )
    X['NumCompaniesWorked_category'] = X.NumCompaniesWorked.apply(lambda x: 'Many' if x > X.NumCompaniesWorked.quantile(0.75) else ('Middle' if x <= X.NumCompaniesWorked.quantile(0.75) and x > X.NumCompaniesWorked.quantile(0.25) else 'Few') )
    X['PercentSalaryHike_category'] = X.PercentSalaryHike.apply(lambda x: 'High' if x > X.PercentSalaryHike.quantile(0.75) else ('Middle' if x <= X.PercentSalaryHike.quantile(0.75) and x > X.PercentSalaryHike.quantile(0.25) else 'Small') )
    X['PerformanceRating_category'] = X.PerformanceRating.apply(lambda x: 'High' if x > X.PerformanceRating.quantile(0.75) else ('Middle' if x <= X.PerformanceRating.quantile(0.75) and x > X.PerformanceRating.quantile(0.25) else 'Small') )
    X['RelationshipSatisfaction_category'] = X.RelationshipSatisfaction.apply(lambda x: 'Satisfied' if x > X.RelationshipSatisfaction.quantile(0.75) else ('Middle' if x <= X.RelationshipSatisfaction.quantile(0.75) and x > X.RelationshipSatisfaction.quantile(0.25) else 'Doesnot_Satisfied') )
    X['StockOptionLevel_category'] = X.StockOptionLevel.apply(lambda x: 'High' if x > X.StockOptionLevel.quantile(0.75) else ('Middle' if x <= X.StockOptionLevel.quantile(0.75) and x > X.StockOptionLevel.quantile(0.25) else 'Small') )
    X['TotalWorkingYears_category'] = X.TotalWorkingYears.apply(lambda x: 'Many' if x > X.TotalWorkingYears.quantile(0.75) else ('Middle' if x <= X.TotalWorkingYears.quantile(0.75) and x > X.TotalWorkingYears.quantile(0.25) else 'Few') )
    X['TrainingTimesLastYear_category'] = X.TrainingTimesLastYear.apply(lambda x: 'Many' if x > X.TrainingTimesLastYear.quantile(0.75) else ('Middle' if x <= X.TrainingTimesLastYear.quantile(0.75) and x > X.TrainingTimesLastYear.quantile(0.25) else 'Few') )
    X['WorkLifeBalance_category'] = X.WorkLifeBalance.apply(lambda x: 'High' if x > X.WorkLifeBalance.quantile(0.75) else ('Middle' if x <= X.WorkLifeBalance.quantile(0.75) and x > X.WorkLifeBalance.quantile(0.25) else 'Small') )
    X['YearsAtCompany_category'] = X.YearsAtCompany.apply(lambda x: 'Many' if x > X.YearsAtCompany.quantile(0.75) else ('Middle' if x <= X.YearsAtCompany.quantile(0.75) and x > X.YearsAtCompany.quantile(0.25) else 'Few') )
    X['YearsInCurrentRole_category'] = X.YearsInCurrentRole.apply(lambda x: 'Many' if x > X.YearsInCurrentRole.quantile(0.75) else ('Middle' if x <= X.YearsInCurrentRole.quantile(0.75) and x > X.YearsInCurrentRole.quantile(0.25) else 'Few') )
    X['YearsSinceLastPromotion_category'] = X.YearsSinceLastPromotion.apply(lambda x: 'Many' if x > X.YearsSinceLastPromotion.quantile(0.75) else ('Middle' if x <= X.YearsSinceLastPromotion.quantile(0.75) and x > X.YearsSinceLastPromotion.quantile(0.25) else 'Few') )
    X['YearsWithCurrManager_category'] = X.YearsWithCurrManager.apply(lambda x: 'Many' if x > X.YearsWithCurrManager.quantile(0.75) else ('Middle' if x <= X.YearsWithCurrManager.quantile(0.75) and x > X.YearsWithCurrManager.quantile(0.25) else 'Few') )

    return X


def categoricalEncoding(X, y=None):
    categorical_features = X.select_dtypes('object').columns

    ft = encoder.transform(X[categorical_features])
    df_ft = pd.DataFrame(ft, columns=encoder.get_feature_names_out())
    X.reset_index(drop=True, inplace=True)
    df_ft.reset_index(drop=True, inplace=True)
    df_ft = df_ft.astype('int64')
    X = pd.concat([X, df_ft], axis=1)
    X.drop(categorical_features, inplace=True, axis=1)

    return X


def numericalScaling(X, y=None):
    numerical_features = X.select_dtypes('number').columns

    std_scaled = scaler.transform(X[numerical_features])

    X[numerical_features] = std_scaled

    return X



preprocessor_pipe = Pipeline(
    steps=
    [
        ('columnDropper', FunctionTransformer(func=columnsDropping)),
        ('additionalColumnsGenerator', FunctionTransformer(func=additionalColumnsGeneration)),
        ('numericalTransformer', FunctionTransformer(func=numericalScaling)),
        ('categoricalTransformer', FunctionTransformer(func=categoricalEncoding)),

    ]
)

model = joblib.load('../api/pickles/best_model.pickle')

main_pipline_pipe = Pipeline(
    steps = [
        ('preprocessing', preprocessor_pipe),
        ('model', model)
    ]
)
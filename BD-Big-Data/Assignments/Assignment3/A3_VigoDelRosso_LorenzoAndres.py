import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2022, 6, 4),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'schedule_interval': '0 20 * * MON',
}

airflow_variables = Variable.get("variables", deserialize_json=True)
local_path = airflow_variables["local_path"]
data_url = airflow_variables["data_url"]
predict_url = airflow_variables["predict_url"]


def task1(): 
    i = 1
    for file in requests.get(data_url).text.split("\n"):
        pd.read_csv(file).to_csv(local_path+"train" + str(i) + ".csv")
        i += 1

def task2():
    train1 = pd.read_csv(local_path+"train1.csv")
    train2 = pd.read_csv(local_path+"train2.csv")
    train = pd.concat([train1,train2])

    train, test = train_test_split(train, test_size=0.2)
    train_targets, train = train["Species"], train.drop(["Species"], axis=1)

    clf = LogisticRegression()
    clf.fit(train, train_targets)

    with open('iris_trained_model.pkl', 'wb') as f:
        pickle.dump(clf, f)

def task3():
    pd.read_csv(predict_url).to_csv(local_path+"predict.csv")

def task4():
    with open('iris_trained_model.pkl', 'rb') as f:
        clf_loaded = pickle.load(f)

    predictions = clf_loaded.predict(pd.read_csv(local_path+"predict.csv"))
    prediction_df = pd.DataFrame({"Prediction": predictions})
    prediction_df.to_csv(local_path+"predictions.csv")

    print(prediction_df)


dag = DAG("a3_dag", catchup=False, default_args=default_args)
with dag:
    task1 = PythonOperator(task_id="task1",python_callable=task1)
    task2 = PythonOperator(task_id="task2",python_callable=task2)
    task3 = PythonOperator(task_id="task3",python_callable=task3)
    task4 = PythonOperator(task_id="task4",python_callable=task4)

    task1 >> task2 >> task4
    task3 >> task4

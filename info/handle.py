import math
from datetime import datetime

import mariadb
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import VarianceThreshold
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, recall_score, f1_score, \
    roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, \
    PowerTransformer, Normalizer
import time
import numpy as np
from sys import platform
import multiprocessing as mp
from sklearn.utils.validation import column_or_1d
# from pai4sk import RandomForestClassifier
# from pai4sk import DecisionTreeClassifier
# from pai4sk import SupportVectorMachine
# from pai4sk import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def getScalers():
    scalers = {'StandardScaler': StandardScaler(),
               'MinMaxScaler': MinMaxScaler(),
               'MaxAbsScaler': MaxAbsScaler(),
               'RobustScaler': RobustScaler(),
               'QuantileTransformer-Normal': QuantileTransformer(output_distribution='normal'),
               'QuantileTransformer-Uniform': QuantileTransformer(output_distribution='uniform'),
               # 'PowerTransformer-Yeo-Johnson': PowerTransformer(method='yeo-johnson'),
               'Normalizer': Normalizer(),
               'NoScaler': None
               }
    return scalers


def getmodels():
    models = {
        # 'Logistic Regression GPU': LogisticRegression(use_gpu=True, device_ids=[0, 1]),
        # 'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
        # 'K-Nearest Neighbor': KNeighborsClassifier(n_jobs=-1),
        # 'DecisionTree GPU': DecisionTreeClassifier(use_gpu=True, use_histograms=True),
        'GaussianNB': GaussianNB(),
        # 'SupportVectorMachine GPU': SupportVectorMachine(use_gpu=True, device_ids=[0, 1]),
        # 'Random Forest GPU': RandomForestClassifier(use_gpu=True, gpu_ids=[0, 1], use_histograms=True),
        'Random Forest': RandomForestClassifier(n_jobs=-1),
        # 'MLP': MLPClassifier(),
        'Light GBM': LGBMClassifier(n_jobs=-1),
        'XGBoost': XGBClassifier(tree_method='gpu_hist', gpu_id=0)
    }
    return models


def getRandomSamplers():
    samplers = {
        'UnderSampler': RandomUnderSampler(sampling_strategy=0.5),
        'OverSampler': RandomOverSampler(sampling_strategy=0.5),
        'NoSampler': None
    }
    return samplers


def getUnderAndOverSamplers():
    samplers = {
        'SMOTEENN': SMOTEENN(sampling_strategy=0.5, n_jobs=-1),
        # 'SMOTEENN': SMOTEENN(sampling_strategy=0.5, n_jobs=-1)
        'SMOTETomek': SMOTETomek(sampling_strategy=0.5, n_jobs=-1)
    }
    return samplers


def getAllRecordsFromDatabase(databaseName):
    start = time.time()
    connection = mariadb.connect(
        pool_name="read_pull",
        pool_size=4,
        host="store.usr.user.hu",
        user="mki",
        password="pwd",
        database=databaseName
    )
    # connection = mysql.connector.connect(
    #     pool_size=16,
    #     host="localhost",
    #     user="root",
    #     password="TOmi_1970",
    #     database=databaseName)
    cursor = connection.cursor()
    sql_use_Query = "USE " + databaseName
    cursor.execute(sql_use_Query)
    sql_select_Query = "select * from transaction order by timestamp"
    cursor.execute(sql_select_Query)
    result = cursor.fetchall()
    connection.close()
    numpy_array = np.array(result)
    end = time.time()
    elapsedTime = end - start
    print(f'{databaseName} beolvasva, betöltési idő: {elapsedTime}, rekordszám: {numpy_array.shape}')
    return numpy_array[:, :]


def getDatabaseNames():
    # databaseNames = ["card_10000_5_i", "card_100000_1_i", "card_250000_02_i", "ordered_field_creditcard_transaction_i"]
    # databaseNames = ["ordered_field_creditcard_transaction_i"]
    # databaseNames = ["card_10000_5_i"]
    databaseNames = ["card_10000_5_i", "card_100000_1_i", "card_250000_02_i"]
    # databaseNames = ["card_10000_5_i", "card_100000_1_i"]
    # databaseNames = ["card_100000_1_i", "card_250000_02_i"]
    # databaseNames = ["card_250000_02_i"]
    return databaseNames


def calculateF(beta, precision, recall):
    temp = beta * beta * precision + recall
    if temp != 0:
        f_beta = (1 + beta) * (1 + beta) * precision * recall / temp
    else:
        f_beta = 0
    return f_beta


def saveMetrics(statisticalDatabaseName, databaseName, currentScalerName, currentSamplerName, currentModelName,
                scaleRunningTime, PCArunningTime, PCAcomponentNumber, sampleRunningTime, modelRunningTime,
                predictedLabels,
                testLabels):
    try:
        connection = mariadb.connect(
            pool_name="create_pool",
            pool_size=32,
            host="store.usr.user.hu",
            user="mki",
            password="pwd",
            database=statisticalDatabaseName
        )
        cursor = connection.cursor()
        sqlUSEQuery = "USE " + statisticalDatabaseName
        cursor.execute(sqlUSEQuery)

        confusionMatrix = confusion_matrix(testLabels, predictedLabels)
        print(f"Confusion Matrix: {confusionMatrix}")
        TN = int(confusionMatrix[0][0])
        FP = int(confusionMatrix[0][1])
        FN = int(confusionMatrix[1][0])
        TP = int(confusionMatrix[1][1])
        temp = TP + FN
        sensitivity = 0
        if temp != 0:
            sensitivity = TP / (TP + FN)
        temp = TN + FP
        specificity = 0
        if temp != 0:
            specificity = TN / (TN + FP)
        accuracy = accuracy_score(testLabels, predictedLabels)
        balanced_accuracy = balanced_accuracy_score(testLabels, predictedLabels)
        precision = 0
        temp = TP + FP
        if temp != 0:
            precision = TP / (TP + FP)
        recall = recall_score(testLabels, predictedLabels)
        temp = TP + FN
        PPV = 0
        if temp != 0:
            PPV = TP / (TP + FN)
        temp = TN + FN
        NPV = 0
        if temp != 0:
            NPV = TN / (TN + FN)
        temp = FN + TP
        FNR = 0
        if temp != 0:
            FNR = FN / (FN + TP)
        temp = FP + TN
        FPR = 0
        if temp != 0:
            FPR = FP / (FP + TN)
        FDR = 0
        temp = FP + TP
        if temp != 0:
            FDR = FP / (FP + TP)
        temp = FN + TN
        FOR=0
        if temp != 0:
            FOR = FN / (FN + TN)
        f1 = f1_score(testLabels, predictedLabels)
        f_05 = calculateF(0.5, precision, recall)
        f2 = calculateF(2, precision, recall)
        temp = math.sqrt(TP + FP) * math.sqrt(TP + FN) * math.sqrt(TN + FP) * math.sqrt(TN + FN)
        MCC = 0
        if temp != 0:
            MCC = (TP * TN - FP * FN) / temp
        ROCAUC = roc_auc_score(testLabels, predictedLabels)
        Youdens_statistic = sensitivity + specificity - 1

        sql_insert_Query = "INSERt INTO metrics (database_name,scaler_name,sampler_name,model_name,scaler_running_time,PCA_running_time,PCA_component_number, sampler_running_time,model_running_time," \
                           "TP,FP,TN,FN,sensitivity,specificity,accuracy,balanced_accuracy,prec,recall,PPV,NPV,FNR,FPR,FDR,F_OR,f1,f_05,f2,MCC,ROCAUC,Youdens_statistic) VALUES" \
                           "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        values = (
            databaseName, currentScalerName, currentSamplerName, currentModelName, scaleRunningTime, PCArunningTime,
            PCAcomponentNumber,
            sampleRunningTime, modelRunningTime, TP, FP, TN, FN, sensitivity, specificity, accuracy, balanced_accuracy,
            precision, recall, PPV, NPV, FNR, FPR, FDR, FOR, f1, f_05, f2, MCC, ROCAUC, Youdens_statistic)
        cursor.execute(sql_insert_Query, values)
        connection.commit()
    except Exception as e:
        print(type(e, e))
        print(
            f'Hiba a mutatószámok képzésekor vagy adatbázisba íráskor, adatbásis neve {databaseName}; skálázó: {currentScalerName}; sampler: {currentSamplerName}; model: {currentModelName}')
    finally:
        cursor.close()
        connection.close()


def createStatisticalDatabase(statisticalDatabaseName):
    connection = mariadb.connect(
        pool_name="create_pool",
        pool_size=4,
        host="store.usr.user.hu",
        user="mki",
        password="pwd")
    sqlCreateSchemaScript = "CREATE DATABASE IF NOT EXISTS " + statisticalDatabaseName
    cursor = connection.cursor()
    cursor.execute(sqlCreateSchemaScript)
    connection.commit()
    sqlUseScript = "USE " + statisticalDatabaseName
    cursor.execute(sqlUseScript)
    file = open("SQL create table metrics.txt", "r")
    sqlCreataTableScript = file.read()
    cursor.execute(sqlCreataTableScript)
    connection.commit()
    cursor.close()
    connection.close()


if __name__ == '__main__':
    print(f'Platform: {platform}')
    cpuCount = mp.cpu_count()
    print(f'Cpu count: {cpuCount}')
    now = datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")
    hour = now.strftime("%H")
    min = now.strftime("%M")
    sec = now.strftime("%S")
    statisticalDatabaseName = "statistic_" + year + "_" + month + "_" + day + "_" + hour + "_" + min + "_" + sec + "_desktop"
    # statisticalDatabaseName = "statistic_" + year + "_" + month + "_" + day + "_" + hour + "_" + min + "_" + sec + "_ac922"
    createStatisticalDatabase(statisticalDatabaseName)
    databaseNames = getDatabaseNames()
    availableScalers = getScalers()
    availableRandomSamplers = getRandomSamplers()
    availableComplexSamplers = getUnderAndOverSamplers()
    availableUnderAndOverSamplers = getUnderAndOverSamplers()
    availableRandomModels = getmodels()
    sampledAndScaledDatabaseNames = list()
    scaledDataContainer = dict()
    PCAnComponents = [8, 42]
    for databaseName in databaseNames:
        imputedDatas = getAllRecordsFromDatabase(databaseName)
        features = imputedDatas[:, 1:-1]
        binaries = imputedDatas[:, -1:]
        binaries = binaries.astype(int)
        transform = VarianceThreshold()
        dataWithoutNullVariancies = transform.fit_transform(features)
        for scalerName in availableScalers.keys():
            scaledDatabaseName = databaseName + scalerName
            if scalerName != 'NoScaler':
                currentScaler = availableScalers.get(scalerName)
                print(f'Scaler: {currentScaler}')
                startOfScale = time.time()
                scaledData = currentScaler.fit_transform(dataWithoutNullVariancies)
                endOfScale = time.time()
                processTimeOfScale = endOfScale - startOfScale
                print(f'Scale process time: {processTimeOfScale}')
            else:
                scaledData = dataWithoutNullVariancies
                processTimeOfScale = 0
            for pcaComponentNumber in PCAnComponents:
                pca = PCA(n_components=pcaComponentNumber)
                startOfPCA = time.time()
                pcaTransformedData = pca.fit_transform(scaledData)
                endOfPCA = time.time()
                processTimeOfPCA = endOfPCA - startOfPCA
                print(f'PCA process time: {processTimeOfPCA}')
                trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(scaledData, binaries,
                                                                                        test_size=0.20, random_state=0)
                for samplerName in availableRandomSamplers.keys():
                    if samplerName != 'NoSampler':
                        currentSampler = availableRandomSamplers.get(samplerName)
                        print(f'Sampler: {currentSampler}')
                        startOfSample = time.time()
                        sampledFeatures, sampledLabels = currentSampler.fit_resample(trainFeatures, trainLabels)
                        endOfSample = time.time()
                        processTimeOfSample = endOfSample - startOfSample
                        print(f'Sample process time: {processTimeOfSample}')
                    else:
                        sampledFeatures = trainFeatures
                        sampledLabels = trainLabels
                        processTimeOfSample = 0
                    for modelName in availableRandomModels.keys():
                        currentModel = availableRandomModels.get(modelName)
                        print(f'model: {currentModel}')
                        startOfModelFit = time.time()
                        modifiedTrainLabels = column_or_1d(trainLabels)
                        currentModel.fit(trainFeatures, modifiedTrainLabels)
                        endOfModelFit = time.time()
                        processTimeOfModelFit = endOfModelFit - startOfModelFit
                        print(f'Model fit process time: {processTimeOfModelFit}')
                        predictedLabels = currentModel.predict(testFeatures)
                        saveMetrics(statisticalDatabaseName, databaseName, scalerName, samplerName, modelName,
                                    processTimeOfScale,
                                    processTimeOfPCA, pcaComponentNumber, processTimeOfModelFit, processTimeOfSample,
                                    predictedLabels,
                                    testLabels)

                # for samplerName in availableUnderAndOverSamplers:
                #     currentSampler = availableUnderAndOverSamplers.get(samplerName)
                #     print(f'Sampler: {currentSampler}')
                #     startOfSample = time.time()
                #     sampledFeatures, sampledLabels = currentSampler.fit_resample(trainFeatures, trainLabels)
                #     endOfSample = time.time()
                #     processTimeOfSample = endOfSample - startOfSample
                #     print(f'Sample process time: {processTimeOfSample}')
                #     for modelName in availableRandomModels.keys():
                #         currentModel = availableRandomModels.get(modelName)
                #         print(f'model: {currentModel}')
                #         startOfModelFit = time.time()
                #         modifiedTrainLabels = column_or_1d(trainLabels)
                #         currentModel.fit(trainFeatures, modifiedTrainLabels)
                #         endOfModelFit = time.time()
                #         processTimeOfModelFit = endOfModelFit - startOfModelFit
                #         print(f'Model fit process time: {processTimeOfModelFit}')
                #         predictedLabels = currentModel.predict(testFeatures)
                #         saveMetrics(statisticalDatabaseName, databaseName, scalerName, samplerName, modelName,
                #                     processTimeOfScale,
                #                     processTimeOfPCA, pcaComponentNumber, processTimeOfModelFit, processTimeOfSample,
                #                     predictedLabels,
                #                     testLabels)

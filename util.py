import smtplib
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import xlsxwriter

def textMe(string):
    # needs to abilitate less secure app: https://www.google.com/settings/security/lesssecureapps
    # NB: avoid ":" in the string!
    content = string
    mail = smtplib.SMTP('smtp.gmail.com', 587)
    mail.ehlo()
    mail.starttls()
    address = 'rudy.rizzo.tv@gmail.com'
    mail.login('amsmdeepmrs@gmail.com', 'amsmdeepmrs20')
    mail.sendmail('amsmdeepmrs@gmail.com', address, content)
    mail.close()
    print(">>> sent E-mail @" + address)


def scores_index(gt, pred, index):
    regr = linear_model.LinearRegression()
    x = gt[:, index].reshape(-1, 1)
    y = pred[:, index]
    regr.fit(x, y)
    lin = regr.predict(np.arange(0, np.max(gt[:, index]), 0.01).reshape(-1, 1))
    mse = mean_squared_error(x, y)
    r_sq = regr.score(x, y)
    return regr.coef_[0], regr.intercept_, r_sq, mse

def save_scores_tab(filename, filepath, gt, pred):
    excelname = '/' + filename + '.xlsx'
    workbook = xlsxwriter.Workbook(filepath + excelname)
    worksheet = workbook.add_worksheet()
    for i in range(16):
        a, q, r2, mse = scores_index(gt, pred, i)
        s = 'A' + str(i * 4 + 1)
        worksheet.write(s, a)
        s = 'A' + str(i * 4 + 2)
        worksheet.write(s, q)
        s = 'A' + str(i * 4 + 3)
        worksheet.write(s, r2)
        s = 'A' + str(i * 4 + 4)
        worksheet.write(s, mse)

    workbook.close()
    print('xlsx SAVED')


def model_similarity(model1, model2, per_layer):
    """
    NB: to return TRUE as similarity, layers must have the same name as well

    :param model1: keras model #1 to compare
    :param model2: keras model #2 to compare
    :param per_layer: boolean ==1 if comparison layer by layer
    :return: boolean -> true or false in comparison
    """

    print("Models are identical: " + model1.get_config() == model2.get_config())
    if per_layer == 1:
        for l1, l2 in zip(model1.layers, model2.layers):
            print("Layer " + str(l1) + "is identical: " + l1.get_config() == l2.get_config())



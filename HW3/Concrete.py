import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

data = pd.read_csv("Concrete_Data.csv")
#data normalization
data = (data - data.mean())/(data.max()-data.min())
data1v = data[['Cement (component 1)(kg in a m^3 mixture)','Concrete compressive strength(MPa, megapascals) ']]

#Renaming columns
data1v.columns = ['feat', 'output']


def GD(x, y,n_iteration=1000 ,criterion="MSE", method = "Adam"):
    n_instances = x.shape[0]
    ones = np.ones(n_instances).reshape(-1,1)
    x = np.append(ones, x.values, axis=1)
    y = y.values 
    n_feats = x.shape[1]

    if criterion == "MSE":        
        if method == "Adam":
            #ADAM parameters
            alpha = 0.05
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 10 ** (-8)
            momentum = np.zeros((n_feats,1))
            RMSprop_m = np.zeros((n_feats,1))
            #Linear Regression parameters
            theta = np.random.random((n_feats,1))
            for i in range(n_iteration):
                #gradient of MSE
                grad = 2/n_instances * x.T.dot(x.dot(theta)-y)
                #Adam Algorithm
                momentum = beta1 * momentum + (1-beta1) * grad
                RMSprop_m = beta2 * RMSprop_m + (1-beta2) * grad * grad
                m_hat = momentum / (1 - beta1 ** (i+1))
                v_hat = RMSprop_m / (1 - beta2 ** (i+1))                
                theta = theta - alpha * m_hat/(np.sqrt(v_hat + epsilon))            
            print("Train MSE: " ,mean_squared_error(y, x.dot(theta)))
            print("Train R^2: " ,r2_score(y, x.dot(theta)))
            return theta

        elif method == "SGD":
            #learning rate
            lr = 0.05
            t0, t1 = 5, 50
            #Linear Regression parameters
            theta = np.random.random((n_feats,1))
            for itr in range(n_iteration):
                for i in range(n_instances):
                    rnd_idx = np.random.randint(n_instances)
                    xi = x[rnd_idx : rnd_idx + 1]
                    yi = y[rnd_idx : rnd_idx + 1]
                    grad = 2 * xi.T.dot(xi.dot(theta) - yi)
                    lr = t0 / (itr *  n_instances + i + t1)
                    theta = theta - lr * grad
            print("Train MSE: " ,mean_squared_error(y, x.dot(theta)))
            print("Train R^2: " ,r2_score(y, x.dot(theta)))
            return theta

        elif method == "Naive":
            #learning rate
            lr = 0.05
            #Linear Regression parameters
            theta = np.random.random((n_feats,1))
            for itr in range(n_iteration):
                rnd_grad = np.random.randint(n_feats)
                parDiff = np.zeros((1,1))
                for i in range(n_instances):
                    xi = x[i : i+1]
                    yi = y[i : i+1]
                    parDiff += (xi.dot(theta)-yi) *  xi[:,rnd_grad:rnd_grad+1]
                parDiff = 2/n_instances * parDiff
                grad = np.zeros((n_feats,1))
                grad[rnd_grad : rnd_grad+1] = parDiff
                theta = theta - lr * grad
            return theta

def training_sigVar(x_train, x_test, y_train, y_test, model="both", n_iteration=1000, criterion = "MSE", method = "Adam"):
    if model == "skLinReg" or "both":
        print("--------------------IN SKLEARN LINEAR REGRESSION--------------------")
        lin_reg = LinearRegression()
        lin_reg.fit(x_train,y_train)
        print("Coefficient of Regression line: ", lin_reg.coef_)
        print("Incterception of Regression line: ", lin_reg.intercept_)
        y_pred_train = lin_reg.predict(x_train)
        y_pred = lin_reg.predict(x_test)
        print("train_MSE: ",mean_squared_error(y_train, y_pred_train))
        print("test_R^2: ", r2_score(y_test, y_pred))
        print("test_MSE: ", mean_squared_error(y_test, y_pred))
    if model == "self_GD" or "both":
        print("\n------------------IN SELF LINEAR REGRESSION--------------------------")
        theta = GD(x_train, y_train, n_iteration, criterion, method)
        print("Coefficient of Regression line: ", theta[0])
        print("Incterception of Regression line: ", theta[1])
       
        x_test = np.append(np.ones(x_test.shape[0]).reshape(-1,1), x_test.values, axis=1)
        y_pred = x_test.dot(theta)
        R2_score = r2_score(y_test, y_pred)
        MSE = mean_squared_error(y_test, y_pred)
        print("test_MSE: ", MSE)

def training_mulVar(x_train, x_test, y_train, y_test, poly = False, n_iteration=1000, criterion = "MSE", method = "Adam"):     
    print("Gradient Descent Method :", method)
    theta = GD(x_train, y_train, n_iteration, criterion, method)
    if not poly:
        print("Coefficient of Regression hyperplane: ", theta[:-1].reshape(-1))
        print("Incterception of Regression hyperplane: ", theta[-1])
       
    x_test = np.append(np.ones(x_test.shape[0]).reshape(-1,1), x_test.values, axis=1)
    y_pred = x_test.dot(theta)

    #Evaluation of prediction
    R2_score = r2_score(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    print("test_R^2: ", R2_score)
    print("test_MSE: ", MSE)

if __name__ == "__main__":    
    X_sigVar = data1v[['feat']]
    y = data1v[['output']]
    X = data.drop(columns=['Concrete compressive strength(MPa, megapascals) '])

    poly_features = PolynomialFeatures(degree=4, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    X_poly = pd.DataFrame(X_poly)

    # data for sigle variable
    x_sigVar_train, x_sigVar_test, y_sigVar_train, y_sigVar_test = train_test_split(X_sigVar, y , test_size = 0.2)
    # data for multivariable
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
    # data for polynomial regression
    x_poly_train, x_poly_test, y_poly_train, y_poly_test = train_test_split(X_poly, y, test_size = 0.2)

    model = "both"
    training_sigVar(x_sigVar_train, x_sigVar_test, y_sigVar_train, y_sigVar_test, model)
    print("\n------------------IN MULTIVARIABLE LINEAR REGRESSION--------------------------")
    training_mulVar(x_train, x_test, y_train, y_test)
    print("\n------------------IN MULTIVARIABLE LINEAR REGRESSION--------------------------")
    training_mulVar(x_train, x_test, y_train, y_test, method="Naive")
    print("\n------------------IN POLYNOMIAL LINEAR REGRESSION--------------------------")
    training_mulVar(x_poly_train, x_poly_test, y_poly_train, y_poly_test, poly=True)

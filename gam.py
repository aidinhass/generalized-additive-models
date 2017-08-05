# Additive models

import random
from copy import copy
import numpy as np
from numpy.linalg import eigh
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plot


class AdditiveModel():
    
    def __init__(self):
        self.knots = None
        self.X = None
        self.Y = None
        self.lambdas = None
        self.S = None
        self.B = None
        self.X_aug = None
        self.Y_aug = None
        self.lr = None
        self.X_names = None
        self.Y_name = None
        self.coefficients = None
        self.original_coefficients = None
        self.means = None
        self.overall_mean = None
        self.Y_predictions = None
        self.Y_overall_prediction = None
        
    def set_X(self,X):
        self.X = X
        
    def set_Y(self,Y):
        self.Y = Y
        
    def set_knots(self,knots):
        self.knots = knots
        
    def set_lambdas(self,lambdas):
        self.lambdas = lambdas
        
    def set_variable_names(self,X_names=None,Y_name=None):
        self.X_names = X_names
        self.Y_name = Y_name
        
    def run(self):
        print 'BUILDING S...'
        self._build_S()
        
        print 'BUILDING B...'
        self._build_B()
        
        print 'BUILDING X...'
        self.build_X()
        
        print 'BUILDING Y...'
        self._build_Y()
        
        print 'ESTIMATING PARAMETERS...'
        self.lr = LinearRegression()
        self.lr.fit(self.X_aug,self.Y_aug)
        
        print 'GENERATE COEFFICIENTS...'
        self._generate_coefficients()
        
        print 'GENERATE PREDICTIONS...'
        self._generate_predictions()
        
    def plot(self):
        number_of_rows,number_of_columns = self.X.shape
        for i in xrange(number_of_columns):
            X = self.X[:,i]
            X_name = self.X_names[i]
            predicted_Y = self.predicted_Y[i]
            
            plot.xlabel(X_name)
            plot.ylabel(self.Y_name)
            plot.title('Variable '+str(i+1))
            plot.scatter(X,self.Y,color='blue',s=0.5)
            plot.scatter(X,predicted_Y[0:number_of_rows,0],color='orange',s=10.0)
            plot.show()
            
            plot.xlabel(X_name)
            plot.ylabel('Residules')
            plot.title('Variable '+str(i+1))
            residules = predicted_Y[0:number_of_rows,:] - self.Y
            plot.scatter(X,residules,color='blue',s=1.0)
            plot.show()
            
        all_predicted_Y = copy(self.predicted_Y[0])
        for i in xrange(1,number_of_columns):
            predicted_Y = self.predicted_Y[i]
            all_predicted_Y = all_predicted_Y + predicted_Y - self.overall_mean
            
        X = range(number_of_rows)
        plot.xlabel('Data Item')
        plot.ylabel('Residules')
        plot.scatter(X,self.Y - all_predicted_Y[0:number_of_rows],color='blue',s=1.0)
        plot.show()
        
    def predict(self,X,Y):
        X_aug = self.build_X(X)
        raw_predictions = []
        predicted_Ys = []
        number_of_rows = X_aug.shape[0] - self.B.shape[0]
        number_of_columns = X.shape[1]
        for i in xrange(number_of_columns):
            coefficients = self.coefficients[i]
            self.lr.coef_ = coefficients
            raw_prediction = self.lr.predict(X_aug)
            raw_predictions.append(raw_prediction)
            
        self.overall_mean = float(sum(self.Y[:,0])/len(self.Y[:,0]))
            
        l = len(raw_predictions)
        for i in xrange(l):
            mean = self.means[i]
            predicted_Y = raw_predictions[i] - mean + self.overall_mean
            predicted_Ys.append(predicted_Y)
            
        all_predicted_Y = copy(predicted_Ys[0])
        for i in xrange(1,len(predicted_Ys)):
            predicted_Y = predicted_Ys[i]
            all_predicted_Y = all_predicted_Y + predicted_Y - self.overall_mean
            
        errors = []
        absolute_error = 0.0
        number_of_rows = Y.shape[0] # NOTE: DO NOT USE all_predicted_Y!
        for i in xrange(number_of_rows):
            y_est = all_predicted_Y[i][0]
            y_act = Y[i][0]
            diff = y_act - y_est
            errors.append(diff)
            absolute_error = absolute_error + abs(diff)
        
        prediction = (predicted_Ys,all_predicted_Y,errors,absolute_error)
        
        return prediction
        
    def _generate_predictions(self):
        raw_predictions = []
        self.predicted_Y = []
        self.means = []
        number_of_rows = self.X.shape[0]
        for coefficients in self.coefficients:
            self.lr.coef_ = coefficients
            raw_prediction = self.lr.predict(self.X_aug)
            raw_predictions.append(raw_prediction)
            mean = float(sum(raw_prediction[0:number_of_rows,0])/len(raw_prediction[0:number_of_rows,0]))
            self.means.append(mean)
            
        self.overall_mean = float(sum(self.Y[:,0])/len(self.Y[:,0]))
            
        l = len(raw_predictions)
        for i in xrange(l):
            mean = self.means[i]
            predicted_Y = raw_predictions[i] - mean + self.overall_mean
            self.predicted_Y.append(predicted_Y)
            
        self.lr.coef_ = self.original_coefficients
        self.Y_overall_prediction = self.lr.predict(self.X_aug)
            
    def _generate_coefficients(self):
        self.original_coefficients = copy(self.lr.coef_)
        self.coefficients = []
        
        offset = 2
        offsets = []
        for knots in self.knots:
            offsets.append(offset)
            offset = 1
            
        indices = [0]
        j_star = 0
        for j in xrange(len(self.knots)):
            knots = self.knots[j]
            offset = offsets[j]
            l = len(knots)
            j_star = j_star + offset + l
            indices.append(j_star)
            
        for i in xrange(1,len(indices)):
            start_i = indices[i-1]
            end_i = indices[i]
            number_of_columns = self.lr.coef_.shape[1]
            new_coefficients = [[0.0]*number_of_columns]
            for j in xrange(start_i,end_i):
                new_coefficients[0][j] = self.lr.coef_[0][j]
                
            new_coefficients = np.array(new_coefficients)
            new_coefficients.resize((1,number_of_columns))
            self.coefficients.append(new_coefficients)
        
    def _build_S(self):
        dim = 0
        offset = 2
        offsets = []
        for knots in self.knots:
            l = len(knots)
            dim = dim + l + offset
            offsets.append(offset)
            offset = 1
            
        self.S = np.array([0.0]*dim**2)
        self.S.resize((dim,dim))
        i_star = 0
        j_star = 0
        for inx in xrange(len(self.knots)):
            knots = self.knots[inx]
            offset = offsets[inx]
            lamb = self.lambdas[inx]
            i_star = i_star + offset
            j_star = j_star + offset
            S = np.array([0.0]*dim**2)
            S.resize((dim,dim))
            l = len(knots)
            for i in xrange(0,l):
                x_star_i = knots[i]
                for j in xrange(0,l):
                    x_star_j = knots[j]
                    s = self._basis_function(x_star_i,x_star_j)
                    S[i_star+i][j_star+j] = s
            i_star = i_star + i + 1
            j_star = j_star + j + 1
                    
            self.S = self.S + lamb*S
                
    def _build_B(self):
        eigenvalues,eigenvectors = eigh(self.S)
        
        l = eigenvalues.shape[0]
        D = np.array([0.0]*l*l)
        D.resize((l,l))
        for i in xrange(l):
            eigenvalue = eigenvalues[i]
            D[i][i] = eigenvalue**0.5
            
        self.B = np.dot(np.dot(eigenvectors,D),eigenvectors.T)
        
    def build_X(self,X_est=None):
        if X_est == None:
            X = self.X
        else:
            X = X_est
            
        offset = 2
        offsets = []
        for knots in self.knots:
            offsets.append(offset)
            offset = 1
            
        number_of_rows,number_of_columns = X.shape
        number_of_rows_B,number_of_columns_B = self.B.shape
        number_of_rows_X = number_of_rows + number_of_rows_B
        number_of_columns_X = number_of_columns_B
        X_aug = np.array([1.0]*number_of_rows_X*number_of_columns_X)
        X_aug.resize((number_of_rows_X,number_of_columns_X))
        
        # scale the explanatory variables so they have the same domain as the basis functions.
        for j in xrange(number_of_columns):
            x_min = min(X[:,j])
            x_max = max(X[:,j])
            
            for i in xrange(number_of_rows):
                x = X[i][j]
                X[i][j] = (x - x_min)/x_max
        
        for i in xrange(number_of_rows):
            j_star = 0
            for j in xrange(number_of_columns):
                x = X[i][j]
                knots = self.knots[j]
                offset = offsets[j]
                j_star = j_star + offset
                l = len(knots)
                for inx in xrange(0,l):
                    knot = knots[inx]
                    s = self._basis_function(x,knot)
                    X_aug[i][j_star+inx] = s
                    
                X_aug[i][j_star-1] = x
                j_star = j_star + inx + 1
                
        for i in xrange(number_of_rows_B):
            for j in xrange(number_of_columns_B):
                b = self.B[i][j]
                X_aug[i+number_of_rows][j] = b
                
        if X_est == None:
            self.X_aug = X_aug
        else:
            return X_aug
        
    def _build_Y(self):
        number_of_rows = self.Y.shape[0]
        number_of_rows_B = self.B.shape[0]
        number_of_rows_Y = number_of_rows + number_of_rows_B
        self.Y_aug = np.array([0.0]*number_of_rows_Y)
        self.Y_aug.resize((number_of_rows_Y,1))
        
        for i in xrange(number_of_rows):
            y = self.Y[i]
            self.Y_aug[i] = y
                
    def _basis_function(self,x_i,x_j):
        result = (((x_j - 0.5)**2 - 1.0/12)*((x_i - 0.5)**2 - 1.0/12))/4.0 - ((abs(x_i - x_j) - 0.5)**4 - 0.5*(abs(x_i - x_j) - 0.5)**2 + 7.0/240)/24.0
        return result
                
    def generate_example_1(self,sample_size=1000):
        X = []
        Y = []
        slope = 0.5
        for i in xrange(sample_size):
            x = i*3.0*np.pi/sample_size
            y = 2.0*np.sin(x) + random.gauss(0,0.1)
            X.append(x)
            Y.append(y)
            
        X = np.array(X)
        X.resize((sample_size,1))
        Y = np.array(Y)
        Y.resize((sample_size,1))
        
        self.set_X(X)
        self.set_Y(Y)
        self.set_knots([[0.2,0.4,0.6,0.8]])
        self.set_lambdas([0.01])
        self.run()
        
        number_of_rows = self.X.shape[0] 
        plot.scatter(self.X,self.Y,color='blue',s=0.5)

        predicted_Y = self.lr.predict(self.X_aug)
        plot.scatter(self.X,predicted_Y[0:number_of_rows,0],color='orange',s=1.0)
        plot.show()
        
        residules = predicted_Y[0:number_of_rows,:] - self.Y
        plot.scatter(self.X,residules,s=1.0,color='blue')
        plot.show()
        
    def generate_example_2(self,sample_size=1000):
        X = []
        Y = []
        slope = 2.75
        for i in xrange(sample_size):
            x_1 = random.uniform(0,1)
            x_2 = random.uniform(0,1)
            x_3 = random.uniform(0,1)
            y = 1.5*np.sin(2.0*np.pi*x_1) + 3.0*abs(np.cos(np.pi*(x_2-0.75))) + slope*x_3**2 + random.gauss(0,1.0)
            X.append((x_1,x_2,x_3))
            Y.append(y)
        
        X = np.array(X)
        X.resize((sample_size,3))
        Y = np.array(Y)
        Y.resize((sample_size,1))
        
        self.set_X(X)
        self.set_Y(Y)
        self.set_knots([[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]])
        self.set_lambdas([0.01,0.01,0.1])
        self.set_variable_names(['X1','X2','X3'],'Response')
        self.run()
        self.plot()
        
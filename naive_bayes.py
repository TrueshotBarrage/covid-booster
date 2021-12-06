import numpy as np


def naivebayesPY(y):
    """
    function [pos,neg] = naivebayesPY(y);

    Computation of P(Y)
    Input:
        y : n labels (-1 or +1) (nx1)

    Output:
    pos: probability p(y=1)
    neg: probability p(y=-1)
    """
    n = y.shape[0]
    pos = np.count_nonzero(y == 1) / n
    neg = np.count_nonzero(y == -1) / n

    return pos, neg


def naivebayesPXY_mle(x, y):
    """
    function [posprob,negprob] = naivebayesPXY(x,y);
    
    Computation of P(X|Y) -- Maximum Likelihood Estimate
    Input:
        x : n input vectors of d dimensions (nxd)
        y : n labels (-1 or +1) (nx1)
    
    Output:
    posprob: probability vector of p(x|y=1) (1xd)
    negprob: probability vector of p(x|y=-1) (1xd)
    """
    neg_ind = y == -1
    pos_ind = y == 1

    german_words = x[neg_ind, :]
    english_words = x[pos_ind, :]

    ger_sum = np.sum(german_words, axis=0)
    eng_sum = np.sum(english_words, axis=0)

    negprob = ger_sum / np.sum(ger_sum)
    posprob = eng_sum / np.sum(eng_sum)

    return posprob, negprob


def naivebayesPXY_smoothing(x, y):
    """
    function [posprob,negprob] = naivebayesPXY(x,y);
    
    Computation of P(X|Y) -- Smoothing with Laplace estimate
    Input:
        x : n input vectors of d dimensions (nxd)
        y : n labels (-1 or +1) (nx1)
    
    Output:
    posprob: probability vector of p(x|y=1) (1xd)
    negprob: probability vector of p(x|y=-1) (1xd)
    """
    d = x.shape[1]

    neg_ind = y == -1
    pos_ind = y == 1

    german_words = x[neg_ind, :]
    english_words = x[pos_ind, :]

    ger_sum = np.sum(german_words, axis=0)
    eng_sum = np.sum(english_words, axis=0)

    negprob = (ger_sum + 1) / (np.sum(ger_sum) + d)
    posprob = (eng_sum + 1) / (np.sum(eng_sum) + d)

    return posprob, negprob


def naivebayes(x, y, xtest, naivebayesPXY):
    """
    function logratio = naivebayes(x,y);
    
    Computation of log P(Y|X=x1) using Bayes Rule
    Input:
    x : n input vectors of d dimensions (nxd)
    y : n labels (-1 or +1)
    xtest: input vector of d dimensions (1xd)
    naivebayesPXY: input function for getting conditional probabilities (naivebayesPXY_smoothing OR naivebayesPXY_mle)
    
    Output:
    logratio: log (P(Y = 1|X=xtest)/P(Y=-1|X=xtest))
    """
    n = y.shape[0]

    theta_eng, theta_ger = naivebayesPXY(x, y)
    coeff = np.math.factorial(np.sum(xtest)) / np.prod(
        np.asarray(list(map(np.math.factorial, xtest))))

    p_x_given_eng = coeff * np.prod(
        np.asarray(list(map(lambda x, y: x**y, theta_eng, xtest))))
    p_x_given_ger = coeff * np.prod(
        np.asarray(list(map(lambda x, y: x**y, theta_ger, xtest))))

    p_eng, p_ger = naivebayesPY(y)

    logratio = np.log(p_x_given_eng) + np.log(p_eng) - np.log(
        p_x_given_ger) - np.log(p_ger)

    return logratio


def naivebayesCL(x, y, naivebayesPXY):
    """
    function [w,b]=naivebayesCL(x,y);
    Implementation of a Naive Bayes classifier
    Input:
    x : n input vectors of d dimensions (nxd)
    y : n labels (-1 or +1)
    naivebayesPXY: input function for getting conditional probabilities (naivebayesPXY_smoothing OR naivebayesPXY_mle)

    Output:
    w : weight vector of d dimensions
    b : bias (scalar)
    """
    n, d = x.shape

    theta_eng, theta_ger = naivebayesPXY(x, y)
    print(theta_eng.shape)
    p_eng, p_ger = naivebayesPY(y)
    w = np.log(theta_eng) - np.log(theta_ger)
    b = np.log(p_eng) - np.log(p_ger)

    return w, b


def classifyLinear(x, w, b=0):
    """
    function preds=classifyLinear(x,w,b);
    
    Make predictions with a linear classifier
    Input:
    x : n input vectors of d dimensions (nxd)
    w : weight vector of d dimensions
    b : bias (optional)
    
    Output:
    preds: predictions
    """
    w = w.flatten()
    preds = np.sign(x @ w + b)

    return preds


# print('Training error (Smoothing with Laplace estimate): %.2f%%' % (100 *(classifyLinear(X, w_smoothing, b_smoothing) != Y).mean()))
# print('Training error (Maximum Likelihood Estimate): %.2f%%' % (100 *(classifyLinear(X, w_mle, b_mle) != Y).mean()))
# print('Test error (Smoothing with Laplace estimate): %.2f%%' % (100 *(classifyLinear(xTe, w_smoothing, b_smoothing) != yTe).mean()))
# print('Test error (Maximum Likelihood Estimate): %.2f%%' % (100 *(classifyLinear(xTe, w_mle, b_mle) != yTe).mean()))

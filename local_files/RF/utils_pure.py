import numpy as np

def predict_tree(example, tree):
    """Return tree's prediction for given examples
    
    Parameters
    ----------
    example : ndarray
        Array of target examples to make inference of
    tree: dict
        Decision Tree Dict. A dict of essential tree parameters.

    Returns
    -------
    """
    def recurse(example, node):
        if tree['children_left'][node] >=0 or tree['children_right'][node] >= 0:
            if example[tree['feature'][node]] <= tree['threshold'][node]:
                return recurse(example, tree['children_left'][node])
            else:
                return recurse(example, tree['children_right'][node])
        else:
            # value == Number of examples that fell in each class during the training phase.            
            proba = tree['value'][node]
            proba /= np.sum(proba)
            return proba
        
    return recurse(example, 0)


def RF_predict(example, RF):
    """Make prediction of a RF in the Scikit-learn's way.

    Parameters
    ----------
    example : ndarray
        Array of target examples to make inference of

    RF : List 
        Random Forest as a list of Decision Tree Dict

    Returns
    -------
        Most probable class (index) prediction as the weighted sum of tree's predictions
    """
    ans = []
    for dt in RF:
        ans.append(predict_tree(example, dt))
    # argmax of weighted sum of all tree's predictions
    
    return np.sum(np.vstack(ans), axis=0).argmax()
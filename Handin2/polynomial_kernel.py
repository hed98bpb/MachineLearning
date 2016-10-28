from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from Handin2.util import get_data

images, labels = get_data('Data/auTrain.npz')
images_test, labels_test = get_data('Data/auTest.npz')



param_grid = [{'C': [1000, 2000], 'degree': [2, 4], 'kernel': ['poly']},]

scores = ['precision', 'recall']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)

    clf = GridSearchCV(svm.SVC(), param_grid, cv=5, scoring='%s_macro' % score)
    clf.fit(images, labels)

    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = labels_test, clf.predict(images_test)
    print(classification_report(y_true, y_pred))
    print()

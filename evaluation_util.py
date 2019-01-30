from sklearn.metrics import accuracy_score


def eval_model_on_data(model, sklearn_model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Own model has accuracy of {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

    sklearn_model.fit(X_train, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test)
    print("Sklean model has accuracy of {:.2f}%".format(accuracy_score(y_test, y_pred_sklearn) *
                                                       100))
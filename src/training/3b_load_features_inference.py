import numpy as np
import sklearn
import sklearn.linear_model
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import os
import argparse
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description="Linear probe evaluation script")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory containing feature files"
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        required=True,
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--model_choice",
        type=str,
        default="densenet_600k_crop",
        help="Model architecture choice",
    )
    parser.add_argument(
        "--epoch", type=int, default=-1, help="Epoch number for model checkpoint"
    )
    parser.add_argument(
        "--month_date", type=str, default="", help="Month/date string for model naming"
    )
    parser.add_argument(
        "--tune_linearprobe",
        action="store_true",
        help="Whether to tune hyperparameters",
    )
    parser.add_argument(
        "--feature_prefix", type=str, default="", help="Prefix for feature files"
    )

    return parser.parse_args()


def run_analysis(
    title: str,
    y_train,
    y_train_proba,
    y_test=None,
    y_test_proba=None,
    label_col="finetune_label",
):
    if y_test is None or y_test_proba is None:
        print(f"---- {title} {label_col} ----")
        print("Test:")
        auroc = print_metrics(y_train, y_train_proba)
        return auroc
    print(f"---- {title} {label_col} ----")
    print("Train:")
    auroc_train = print_metrics(y_train, y_train_proba)
    print("Test:")
    auroc = print_metrics(y_test, y_test_proba)
    return auroc_train, auroc


def print_metrics(y_true, y_proba):
    y_pred = y_proba > 0.5
    auroc = sklearn.metrics.roc_auc_score(y_true, y_proba)
    aps = sklearn.metrics.average_precision_score(y_true, y_proba)
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    f1 = sklearn.metrics.f1_score(y_true, y_pred)
    auprc = sklearn.metrics.precision_recall_curve(y_true, y_proba)
    print("\tAUROC:", auroc)
    print("\tAPS:", aps)
    print("\tAccuracy:", accuracy)
    print("\tF1 Score:", f1)
    return auroc


def logistic_regression(
    column_ls,
    X_train,
    y_train,
    X_val,
    y_val,
    model_save_path,
    model_choice,
    epoch,
    month_date,
    metric_values,
    run,
    writer,
    tune_linearprobe,
):
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    auroc_train_dict = {}
    auroc_val_dict = {}

    # y_train is a list
    if type(y_train) == list:
        linear_model = sklearn.linear_model.LogisticRegressionCV(
            penalty="l2", solver="liblinear"
        ).fit(X_train, y_train)

        y_train_proba = linear_model.predict_proba(X_train)[::, 1]
        y_val_proba = linear_model.predict_proba(X_val)[::, 1]
        auroc_train, auroc_val = run_analysis(
            "Logistic Regression", y_train, y_train_proba, y_val, y_val_proba
        )
        # run.track(auroc_train, name='auroc_train', step=epoch, context={'subset': 'train'})
        auroc_train_dict["auroc_train_finetune_label"] = auroc_train
        auroc_val_dict["auroc_val_finetune_label"] = auroc_val
        # writer.add_scalar("val_accuracy", auroc_val, epoch)
        # run.track(auroc_val, name='auroc_val', step=epoch, context={'subset': 'valid'})

        linear_model_save_path = os.path.join(
            model_save_path,
            f"linear_model_{epoch}epoch_{model_choice}_{month_date}.pkl",
        )
        with open(linear_model_save_path, "wb") as f:
            pickle.dump(linear_model, f)

        metric_values.append(auroc_val)

    # y_train is an array
    else:
        auroc_val_list = []

        for idx, label_col in enumerate(column_ls):
            y_train_list = y_train[:, idx]
            idx_uncensored = np.where(
                (y_train_list != "Censored") & (y_train_list != -1)
            )[0]

            X_train_list = X_train[idx_uncensored]
            y_train_list = y_train_list[idx_uncensored]

            y_train_list = y_train_list.astype(int)
            print(
                f"num of censored for {label_col} in train",
                len(y_train.tolist()) - len(y_train_list),
                "out of",
                len(y_train.tolist()),
            )

            if tune_linearprobe:
                linear_model = sklearn.linear_model.LogisticRegressionCV(
                    penalty="l2", solver="liblinear", cv=2, n_jobs=-1
                ).fit(X_train_list, y_train_list)
                # linear_model = tune_hyperparameter_LR(linear_model, X_train_list, y_train_list)

                # linear_model = sklearn.linear_model.LogisticRegressionCV(penalty="l2", solver="liblinear").fit(X_train_list, y_train_list)

                # linear_model = sklearn.linear_model.LogisticRegression(penalty="l2", solver="liblinear").fit(X_train_list, y_train_list)

                tree_model = xgb.XGBClassifier()
                tree_model = tune_hyperparameter_tree(
                    tree_model, X_train_list, y_train_list
                )
            else:
                linear_model = sklearn.linear_model.LogisticRegressionCV(
                    penalty="l2", solver="liblinear", cv=2, n_jobs=-1
                ).fit(X_train_list, y_train_list)
                tree_model = xgb.XGBClassifier().fit(X_train_list, y_train_list)

            y_val_task = y_val[:, idx]
            idx_uncensored = np.where((y_val_task != "Censored") & (y_val_task != -1))[
                0
            ]
            X_val_list = X_val[idx_uncensored]
            y_val_task = y_val_task[idx_uncensored]
            y_val_task = y_val_task.astype(int)
            print(
                f"num of censored for {label_col} in val",
                len(y_val.tolist()) - len(y_val_task),
                "out of",
                len(y_val.tolist()),
            )

            y_train_proba = linear_model.predict_proba(X_train_list)[::, 1]
            y_val_proba = linear_model.predict_proba(X_val_list)[::, 1]
            y_train_proba_tree = tree_model.predict_proba(X_train_list)[::, 1]
            y_val_proba_tree = tree_model.predict_proba(X_val_list)[::, 1]
            auroc_train, auroc_val = run_analysis(
                "Logistic Regression",
                y_train_list,
                y_train_proba,
                y_val_task,
                y_val_proba,
                label_col,
            )
            auroc_train_tree, auroc_val_tree = run_analysis(
                "XGBoost",
                y_train_list,
                y_train_proba_tree,
                y_val_task,
                y_val_proba_tree,
                label_col,
            )

            auroc_train_dict[f"auroc_train_{label_col}"] = auroc_train
            auroc_val_dict[f"auroc_val_{label_col}"] = auroc_val
            auroc_train_dict[f"auroc_train_tree_{label_col}"] = auroc_train_tree
            auroc_val_dict[f"auroc_val_tree_{label_col}"] = auroc_val_tree
            if model_save_path:
                linear_model_save_path = os.path.join(
                    model_save_path,
                    f"linear_model_{epoch}epoch_{model_choice}_{label_col}_{month_date}.pkl",
                )
                with open(linear_model_save_path, "wb") as f:
                    pickle.dump(linear_model, f)
                tree_model_save_path = os.path.join(
                    model_save_path,
                    f"tree_model_{epoch}epoch_{model_choice}_{label_col}_{month_date}.pkl",
                )
                with open(tree_model_save_path, "wb") as f:
                    pickle.dump(tree_model, f)

            auroc_val_list.append(auroc_val)
            auroc_val = np.mean(auroc_val_list)
        metric_values.append(np.mean(auroc_val_list))

    return metric_values, run, writer, auroc_val, auroc_train_dict, auroc_val_dict


def tune_hyperparameter_tree(tree_model, X_val, y_val):
    param_grid = {
        "max_depth": [3, 6],  # , 10
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [100, 500],  # , 1000
        # 'colsample_bytree': [0.3, 0.7]
    }
    grid_search = GridSearchCV(
        estimator=tree_model,
        param_grid=param_grid,
        scoring="roc_auc",
        n_jobs=-1,
        cv=3,
        verbose=1,
    )

    grid_search.fit(X_val, y_val)
    best_model = grid_search.best_estimator_
    return best_model


def tune_hyperparameter_LR(linear_model, X_val, y_val):
    param_grid = {
        "penalty": ["l2", "elasticnet", "none"],  # 'l1'
        #'C': [0.01, 0.1, 1, 10, 100],
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    }

    # Setup GridSearchCV
    grid_search = GridSearchCV(
        estimator=linear_model,
        param_grid=param_grid,
        scoring="roc_auc",
        n_jobs=-1,
        cv=5,
        verbose=1,
    )

    # Fit the model
    grid_search.fit(X_val, y_val)

    best_model = grid_search.best_estimator_  # or random_search.best_estimator_

    return best_model


def main():
    args = parse_args()

    # Define feature paths
    feature_prefix = (
        f"{args.feature_prefix}_{args.epoch}epoch_{args.model_choice}"
        if args.feature_prefix
        else f"{args.epoch}epoch_{args.model_choice}"
    )
    X_train = pickle.load(
        open(os.path.join(args.data_dir, f"X_train_{feature_prefix}.pkl"), "rb")
    )
    y_train = pickle.load(
        open(os.path.join(args.data_dir, f"y_train_{feature_prefix}.pkl"), "rb")
    )
    X_val = pickle.load(
        open(os.path.join(args.data_dir, f"X_val_{feature_prefix}.pkl"), "rb")
    )
    y_val = pickle.load(
        open(os.path.join(args.data_dir, f"y_val_{feature_prefix}.pkl"), "rb")
    )

    column_ls = [
        "12_month_PH",
        "pe_positive_nlp",
        "1_month_mortality",
        "6_month_mortality",
        "12_month_mortality",
        "1_month_readmission",
        "6_month_readmission",
        "12_month_readmission",
    ]

    metric_values = []
    run, writer = None, None

    start_time = datetime.now()
    print(f"Start time: {start_time}")
    metric_values, run, writer, auroc_val, auroc_train_dict, auroc_val_dict = (
        logistic_regression(
            column_ls,
            X_train,
            y_train,
            X_val,
            y_val,
            args.model_save_path,
            args.model_choice,
            args.epoch,
            args.month_date,
            metric_values,
            run,
            writer,
            args.tune_linearprobe,
        )
    )
    print(f"Time taken: {datetime.now() - start_time}")


if __name__ == "__main__":
    main()

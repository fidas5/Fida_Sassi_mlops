import argparse

from model_pipeline import (
    load_data,
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main():

    # Creation du parser
    parser = argparse.ArgumentParser(description="Customer Churn ML Pipeline")

    # Definition des arguments
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--load_data", action="store_true", help="Load the dataset")

    group.add_argument("--prepare", action="store_true", help="Prepare the data")

    group.add_argument("--train", action="store_true", help="Train the model")

    group.add_argument("--evaluate", action="store_true", help="Evaluate the model")

    group.add_argument("--save", action="store_true", help="Train and save the model")

    group.add_argument("--load", action="store_true", help="Load the saved model")

    group.add_argument(
        "--all", action="store_true", help="Execute the complete lifecycle"
    )

    args = parser.parse_args()

    # Chargement des donnees
    if args.load_data:
        data = load_data()
        print(data.head())

    # Preparation des donnees
    elif args.prepare:
        x_train, x_test, y_train, y_test = prepare_data()
        print("Data prepared successfully.")

    # Entrainement du modele
    elif args.train:
        x_train, x_test, y_train, y_test = prepare_data()
        model = train_model(x_train, y_train)
        save_model(model)
        print("Model trained successfully.")

    # Evaluation du modele
    elif args.evaluate:
        x_train, x_test, y_train, y_test = prepare_data()
        model = load_model()
        evaluate_model(model, x_test, y_test)

    # Sauvegarde du modele
    elif args.save:
        x_train, x_test, y_train, y_test = prepare_data()
        model = train_model(x_train, y_train)
        save_model(model)

    # Chargement du modele
    elif args.load:
        model = load_model()

    # Cycle de vie complet
    # Cycle de vie complet
    elif args.all:

        print("\n===== 1. DATA LOADING =====")
        data = load_data()

        print("\n===== 2. DATA PREPARATION =====")
        x_train, x_test, y_train, y_test = prepare_data(data)

        print("\n===== 3. MODEL TRAINING =====")
        model = train_model(x_train, y_train)

        print("\n===== 4. MODEL SAVING =====")
        save_model(model)

        print("\n===== 5. MODEL LOADING =====")
        model = load_model()

        print("\n===== 6. MODEL EVALUATION =====")
        accuracy = evaluate_model(model, x_test, y_test)

        print("\n===== COMPLETE LIFECYCLE FINISHED =====")
        print(f"Final Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()

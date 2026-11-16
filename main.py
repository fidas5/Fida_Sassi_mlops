import argparse

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model
)


def main():

    # Création du parser
    parser = argparse.ArgumentParser(
        description="Customer Churn ML Pipeline"
    )

    # Définition des commandes
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare the data"
    )

    group.add_argument(
        "--train",
        action="store_true",
        help="Train the model"
    )

    group.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the model"
    )

    group.add_argument(
        "--save",
        action="store_true",
        help="Train and save the model"
    )

    group.add_argument(
        "--load",
        action="store_true",
        help="Load the saved model"
    )

    # Récupération des arguments
    args = parser.parse_args()

    # Préparation des données
    if args.prepare:

        x_train, x_test, y_train, y_test = prepare_data()

        print("Data prepared successfully.")

    # Entraînement du modèle
    elif args.train:

        x_train, x_test, y_train, y_test = prepare_data()

        model = train_model(x_train, y_train)

        save_model(model)

        print("Model trained successfully.")

    # Évaluation du modèle
    elif args.evaluate:

        x_train, x_test, y_train, y_test = prepare_data()

        model = load_model()

        evaluate_model(model, x_test, y_test)

    # Sauvegarde du modèle
    elif args.save:

        x_train, x_test, y_train, y_test = prepare_data()

        model = train_model(x_train, y_train)

        save_model(model)

        print("Model trained and saved.")

    # Chargement du modèle
    elif args.load:

        model = load_model()


if __name__ == "__main__":
    main()

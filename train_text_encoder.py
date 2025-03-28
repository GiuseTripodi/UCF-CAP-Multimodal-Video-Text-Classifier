import argparse
import os
from datetime import date

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from transformers import DistilBertTokenizer, TFDistilBertModel
from tensorflow.keras import layers, Model
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from parse_config import ConfigParser


def load_and_preprocess_data(config: ConfigParser, max_seq_len, bert_name):
    """Carica il dataset, esegue la pulizia, prepara i dati e codifica le etichette."""
    # Caricamento del dataset
    train_csv_path, val_csv_path = config.train_path
    test__csv_path = config.test_path

    train_df = pd.read_csv(train_csv_path,  encoding='utf-8').sample(frac=1).reset_index(drop=True)[:20]
    test_df = pd.read_csv(test__csv_path,  encoding='utf-8').sample(frac=1).reset_index(drop=True)[:20]
    val_df = pd.read_csv(val_csv_path,  encoding='utf-8').sample(frac=1).reset_index(drop=True)[:20]


    train_sentences = train_df["caption"].values
    val_sentences = val_df["caption"].values
    test_sentences = test_df["caption"].values

    # Codifica delle etichette
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["label"])
    y_val = label_encoder.transform(val_df["label"])
    y_test = label_encoder.transform(test_df["label"])

    num_classes = len(label_encoder.classes_)

    # Preparazione input BERT
    tokenizer = DistilBertTokenizer.from_pretrained(bert_name)

    def prepare_bert_input(sentences):
        encodings = tokenizer(
            list(sentences),
            truncation=True,
            padding='max_length',
            max_length=max_seq_len,
            return_tensors="np"
        )
        return [encodings["input_ids"], encodings["attention_mask"]]

    X_train = prepare_bert_input(train_sentences)
    X_val = prepare_bert_input(val_sentences)
    X_test = prepare_bert_input(test_sentences)

    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes, label_encoder

def create_model(max_seq_len, num_classes, bert_name):
    """Crea e restituisce un modello BERT compilato per la classificazione multiclasse."""
    input_ids = layers.Input(shape=(max_seq_len,), dtype=tf.int32, name='input_ids')
    input_mask = layers.Input(shape=(max_seq_len,), dtype=tf.int32, name='attention_mask')
    inputs = [input_ids, input_mask]

    bert = TFDistilBertModel.from_pretrained(bert_name)
    bert_outputs = bert(inputs)
    last_hidden_states = bert_outputs.last_hidden_state

    avg = layers.GlobalAveragePooling1D()(last_hidden_states)
    output = layers.Dense(num_classes, activation="softmax")(avg)

    model = Model(inputs=inputs, outputs=output)

    opt = tfa.optimizers.RectifiedAdam(learning_rate=3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    model.compile(
        loss=loss,
        optimizer=opt,
        metrics=[accuracy]
    )

    return model


def train_model(model, X_train, y_train, X_val, y_val, output_weights_file, initial_epoch=0, total_epochs=8, batch_size=8):
    """Addestra il modello e salva i pesi migliori."""
    m_ckpt = ModelCheckpoint(
        output_weights_file,
        monitor='val_sparse_categorical_accuracy',
        mode='max',
        verbose=2,
        save_weights_only=True,
        save_best_only=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=total_epochs,
        initial_epoch=initial_epoch,
        batch_size=batch_size,
        callbacks=[m_ckpt],
        verbose=2
    )


def evaluate_model(model, X_test, y_test, label_encoder):
    """Valuta il modello e stampa le metriche."""
    test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=32, verbose=2)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    # Predizioni
    y_pred_probs = model.predict(X_test, batch_size=32, verbose=2)
    y_pred = tf.argmax(y_pred_probs, axis=1).numpy()

    # Metriche
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def continue_training(model, X_train, y_train, X_val, y_val, weights_file, current_epochs, additional_epochs, batch_size=8):
    """Continua l'addestramento di un modello esistente per ulteriori epoche."""
    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"I pesi non sono stati trovati: {weights_file}")

    print(f"Caricamento pesi da {weights_file}")
    model.load_weights(weights_file)

    total_epochs = current_epochs + additional_epochs

    m_ckpt = ModelCheckpoint(
        weights_file,
        monitor='val_sparse_categorical_accuracy',
        mode='max',
        verbose=2,
        save_weights_only=True,
        save_best_only=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=total_epochs,
        initial_epoch=current_epochs,
        batch_size=batch_size,
        callbacks=[m_ckpt],
        verbose=2
    )


def print_embeddings(model, X_data, output_layer_name="tf_distil_bert_model"):
    """
    Stampa gli embedding intermedi del modello (senza la classificazione finale).

    Args:
        model: Modello Keras.
        X_data: Input per il modello (ad esempio, [input_ids, attention_mask]).
        output_layer_name: Nome del livello di output da cui prendere gli embedding.
    """
    # Crea un modello intermedio che restituisce l'output del livello BERT
    intermediate_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer(output_layer_name).output
    )

    # Ottieni l'output intermedio
    outputs = intermediate_model.predict(X_data, batch_size=32, verbose=2)

    # Gli embedding si trovano in last_hidden_state
    embeddings = outputs.last_hidden_state

    print("Embedding Shape:", embeddings.shape)
    print("Embeddings:")
    print(embeddings)


def main_train_text_embedding(config: ConfigParser):
    # ================= Configurazione ====================
    SAVE_DIR = config.save_dir

    WEIGHTS_FILE = f"{SAVE_DIR}\\weights_multiclass_{config.exper_name}.h5"
    MAX_SEQ_LEN = config.max_seq_len
    BERT_NAME = "distilbert-base-uncased"
    TOTAL_EPOCHS = config.num_epochs
    ADDITIONAL_EPOCHS = 5

    # Caricamento e preprocessing dei dati
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes, label_encoder = load_and_preprocess_data(
        config, MAX_SEQ_LEN, BERT_NAME
    )

    # Creazione del modello
    model = create_model(MAX_SEQ_LEN, num_classes, BERT_NAME)

    # Selezione modalità
    mode = input("Scegli la modalità (train/eval/continue/embeddings): ").strip().lower()

    if mode == "train":
        print("Avvio del training...")
        if os.path.exists(WEIGHTS_FILE):
            print(f"Caricamento pesi da {WEIGHTS_FILE}")
            model.load_weights(WEIGHTS_FILE)
            train_model(
                model, X_train, y_train, X_val, y_val, WEIGHTS_FILE,
                initial_epoch=TOTAL_EPOCHS, total_epochs=TOTAL_EPOCHS + ADDITIONAL_EPOCHS
            )
        else:
            train_model(model, X_train, y_train, X_val, y_val, WEIGHTS_FILE, total_epochs=TOTAL_EPOCHS)

    elif mode == "eval":
        if os.path.exists(WEIGHTS_FILE):
            print(f"Caricamento pesi da {WEIGHTS_FILE}")
            model.load_weights(WEIGHTS_FILE)
            evaluate_model(model, X_test, y_test, label_encoder)
        else:
            print("Nessun modello salvato trovato per l'evaluation. Esegui prima il training.")

    elif mode == "continue":
        print("Continuazione del training...")
        if os.path.exists(WEIGHTS_FILE):
            current_epochs = TOTAL_EPOCHS
            continue_training(
                model, X_train, y_train, X_val, y_val, WEIGHTS_FILE,
                current_epochs=current_epochs, additional_epochs=ADDITIONAL_EPOCHS
            )
        else:
            print("Nessun modello salvato trovato per continuare l'addestramento. Esegui prima il training.")

    elif mode == "embeddings":
        print("Estrazione degli embeddings...")
        if os.path.exists(WEIGHTS_FILE):
            print(f"Caricamento pesi da {WEIGHTS_FILE}")
            model.load_weights(WEIGHTS_FILE)
            print_embeddings(model, X_test)
        else:
            print("Nessun modello salvato trovato per estrarre gli embeddings. Esegui prima il training.")

    else:
        print("Modalità non valida. Usa 'train', 'eval', 'continue' o 'embeddings'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to train model")
    parser.add_argument('--config', default=None, help='Path to configuration file')
    parser.add_argument('--model_name',default=None, help='Path to CSV file with dataset information to eval the model')
    parser.add_argument('--name', default=None, help='Name of the experiment (used for saving the model)')
    parser.add_argument('--save_dir', default=None, help='Path to where get the saves file')
    args = parser.parse_args()


    config = ConfigParser(args)
    if args.model_name is None:
        exper_name = config.exper_name
        model_name = f'space_time_{exper_name}_{date.today().strftime("%d-%m-%y")}'
    else:
        model_name = args.model_name

    main_train_text_embedding(config)

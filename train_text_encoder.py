import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from transformers import DistilBertTokenizer, TFDistilBertModel
from tensorflow.keras import layers, Model
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint



def load_and_preprocess_data(file_path, max_seq_len, bert_name):
    """Carica il dataset, esegue la pulizia, prepara i dati e codifica le etichette."""
    # Caricamento del dataset
    data = pd.read_csv(file_path, sep=';', encoding='utf-8')
    data.dropna(subset=["caption", "class", "split"], inplace=True)

    # Divisione del dataset
    train_data = data[data["split"] == "train"]
    val_data = data[data["split"] == "val"]
    test_data = data[data["split"] == "test"]

    def shuffle_data(df):
        return df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_data = shuffle_data(train_data)
    val_data = shuffle_data(val_data)
    test_data = shuffle_data(test_data)

    train_sentences = train_data["caption"].values
    val_sentences = val_data["caption"].values
    test_sentences = test_data["caption"].values

    # Codifica delle etichette
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_data["class"])
    y_val = label_encoder.transform(val_data["class"])
    y_test = label_encoder.transform(test_data["class"])

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


def main():
    # ================= Configurazione ====================
    BASE_DIR = "C:\\Users\\CristianCos\\Desktop\\Python\\Multimodal"
    DATASET_PATH = f"{BASE_DIR}\\texts_dataset.csv"
    SAVE_DIR = f"{BASE_DIR}\\DistilBert"
    os.makedirs(SAVE_DIR, exist_ok=True)

    WEIGHTS_FILE = f"{SAVE_DIR}\\weights_multiclass.h5"
    MAX_SEQ_LEN = 128
    BERT_NAME = "distilbert-base-uncased"
    TOTAL_EPOCHS = 8
    ADDITIONAL_EPOCHS = 5

    # Caricamento e preprocessing dei dati
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes, label_encoder = load_and_preprocess_data(
        DATASET_PATH, MAX_SEQ_LEN, BERT_NAME
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
    main()

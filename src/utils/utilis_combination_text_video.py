import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import date
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from torchvision import transforms
from transformers import DistilBertTokenizer, TFDistilBertModel, DistilBertModel, TimesformerModel, AutoTokenizer, \
    TFAutoModel
import tensorflow as tf
import tensorflow_addons as tfa
import h5py
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from transformers import DistilBertTokenizer, TFDistilBertModel
from tensorflow.keras import layers, Model
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint


from model.MLP_classifier import MLPClassifier
from model.video_transformer import SpaceTimeTransformer
from transformers import TimesformerForVideoClassification
from data_loader.ucf_cap_dataset_embeddings import UCF101Dataset
from parse_config import ConfigParser


# ===========================
#   CARICAMENTO MODELLI NLP
# ===========================
def load_pretrained_text_model_with_embeddings(weights_path, bert_name='distilbert-base-uncased'):
    """Carica il modello DistilBERT con pesi pre-addestrati e consente l'estrazione degli embeddings."""

    max_seq_len = 245
    num_classes = 7

    # Input layers
    input_ids = layers.Input(shape=(max_seq_len,), dtype=tf.int32, name='input_ids')
    input_mask = layers.Input(shape=(max_seq_len,), dtype=tf.int32, name='attention_mask')
    inputs = [input_ids, input_mask]

    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(bert_name)

    # DistilBERT model
    bert = TFDistilBertModel.from_pretrained(bert_name)
    bert_outputs = bert(inputs)

    # Extract embeddings (hidden states)
    last_hidden_states = bert_outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

    # Pooling and classification
    avg = layers.GlobalAveragePooling1D()(last_hidden_states)
    output = layers.Dense(num_classes, activation="softmax")(avg)

    # Create the model that outputs both embeddings and predictions
    model = Model(inputs=inputs, outputs=[last_hidden_states, output])
    print(model.summary())

    # Load pre-trained weights
    if os.path.exists(weights_path):
        print(f"[INFO] Caricamento pesi pre-addestrati da {weights_path}")
        model.load_weights(weights_path)
    else:
        raise FileNotFoundError(f"I pesi {weights_path} non sono stati trovati.")

    return tokenizer, model


def load_pretrained_text_model(weights_path, bert_name='distilbert-base-uncased'):
    """Carica il modello DistilBERT con pesi pre-addestrati."""
    max_seq_len = 245
    num_classes = 13
    input_ids = layers.Input(shape=(max_seq_len,), dtype=tf.int32, name='input_ids')
    input_mask = layers.Input(shape=(max_seq_len,), dtype=tf.int32, name='attention_mask')
    inputs = [input_ids, input_mask]
    tokenizer = DistilBertTokenizer.from_pretrained(bert_name)

    bert = TFDistilBertModel.from_pretrained(bert_name)
    bert_outputs = bert(inputs)
    last_hidden_states = bert_outputs.last_hidden_state

    avg = layers.GlobalAveragePooling1D()(last_hidden_states)
    output = layers.Dense(num_classes, activation="softmax")(avg)

    model = Model(inputs=inputs, outputs=output)


    # Carichiamo i pesi pre-addestrati
    if os.path.exists(weights_path):
        print('Pesi del modello, vedere se sono quali a sopra')
        weights_path = '/Users/user/PycharmProjects/frozen-in-time/data/models/weights_multiclass.h5'
        with h5py.File(weights_path, 'r') as f:
            print("Layers in weights file:", len(f.keys()))

        print(f"[INFO] Caricamento pesi pre-addestrati da {weights_path}")
        model.load_weights(weights_path)
    else:
        raise FileNotFoundError(f"I pesi {weights_path} non sono stati trovati.")

    return tokenizer, model


def load_text_encoder(bert_name='distilbert-base-uncased'):
    """Carica il tokenizer e il modello DistilBERT."""
    tokenizer = DistilBertTokenizer.from_pretrained(bert_name)
    bert_model = DistilBertModel.from_pretrained(bert_name)
    return tokenizer, bert_model

# ===========================
#   ESTRAZIONE EMBEDDING NLP
# ===========================

import torch

def extract_text_embeddings(bert_model, tokenizer, sentences, max_seq_len):
    """Extracts embeddings from the DistilBERT model."""
    input_ids, attention_mask = sentences["input_ids"], sentences["attention_mask"]

    # Compute embeddings
    outputs = bert_model(input_ids, attention_mask)
    last_hidden_states = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

    # Compute mean across tokens (dimension 1)
    avg_embeddings = last_hidden_states.mean(dim=1)  # ✅ PyTorch operation

    return avg_embeddings.detach()  # ✅ Detach from computation graph


def extract_text_embeddings_weight(model, tokenizer, sentences, max_seq_len):
    """Estrai embedding da un modello DistilBERT pre-addestrato."""
    # Tokenize the sentences
    '''
    encodings = tokenizer(
        sentences,
        truncation=True,
        padding='max_length',
        max_length=245,
        return_tensors="tf"
    )
    '''

    # Convert PyTorch tensors to NumPy and then to TensorFlow tensors
    input_ids_tf = tf.convert_to_tensor(sentences['input_ids'].numpy(), dtype=tf.int32)
    attention_mask_tf = tf.convert_to_tensor(sentences['attention_mask'].numpy(), dtype=tf.int32)

    # Extract embeddings and predictions
    embeddings, predictions = model.predict([input_ids_tf, attention_mask_tf])
    return embeddings

# ===========================
#   CARICAMENTO MODELLO VIDEO ENCOER
# ===========================

def load_model_embeddings(config: ConfigParser, model_name, logger):
    if config.modality == 0:
        model_path = os.path.join(config.save_dir, f'{model_name}')
        model = SpaceTimeTransformer(
            img_size=config.img_size,
            num_frames=config.num_frames,
            in_chans=config.in_chans,
            num_classes=config.num_classes,
            depth=config.depth,
            num_heads=config.num_heads,
            embed_dim=768,
            attention_style='frozen-in-time'
        )
        model.load_state_dict(torch.load(model_path))
        processor = None

    elif config.modality == 1:
        # Use model pre_trained and the fine_tuned
        model_path = os.path.join(config.save_dir, f'{model_name}')
        logger.info(f"[INFO] Loaded fine_tuned model from: {model_path}")
        model = TimesformerModel.from_pretrained(model_path)
        # processor = AutoImageProcessor.from_pretrained(model_path)
        processor = None
        print(model.config)

    return model, processor
# ===========================
#   ESTRAZIONE EMBEDDING VIDEO
# ===========================

def extract_videos_embedding(model, inputs):
    outputs = model(inputs)
    return outputs.last_hidden_state

def project_text_video(text_encoder, video_encoder, text_embedding, video_embedding, projection_dim=256):
    # code from: https://github.com/m-bain/frozen-in-time/blob/main/model/model.py
    txt_proj = nn.Sequential(
        nn.ReLU(),
        # Use distilBert hidden dimension
        nn.Linear(768, projection_dim),
    )

    vid_proj = nn.Sequential(
        nn.Linear(video_encoder.config.hidden_size, projection_dim)
    )


    text_embedding = txt_proj(torch.from_numpy(text_embedding))
    video_embedding = vid_proj(video_embedding)
    return text_embedding, video_embedding


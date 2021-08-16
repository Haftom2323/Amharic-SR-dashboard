# sys.path.insert(0, './scripts')
from logging import log

import numpy as np
from json import load
from python_speech_features import mfcc, logfbank
# from model_trainer import CTCLossLayer
import tensorflow as tf

try:
    from logger_creator import CreateLogger
except:
    from scripts.logger_creator import CreateLogger

# Initializing Logger
logger = CreateLogger('ModelInference', handlers=1)
logger = logger.get_default_logger()

class ModelInference:
    def __init__(self,audio,alphabets_path = 'data/alphabets_data.json',model_path = 'models/stacked-lstm_predict.h5'):
        try:
            self.alphabets_path = alphabets_path
            self.model_path = model_path
            self.FEAT_MASK_VALUE = 1e+10
            self.infer(audio)
            logger.info('Successfully Initialized ModelInference.')
        except Exception as e:
            logger.exception('Failed to Initialize ModelInference.')

    def infer(self,audio):
        self.load_files()
        self.prepare_feature(audio)
        self.decode()

    def load_files(self):
        try:
            with open(self.alphabets_path, 'r', encoding='UTF-8') as alphabets_file:
                self.alphabets = load(alphabets_file)
            logger.info('Successfully Loaded alphabets_data.')
        except Exception as e:
            logger.exception('Failed to Load alphabets data.')
        
        try:
            self.model_pred = tf.keras.models.load_model(self.model_path)
            logger.info('Successfully Loaded Model for inference.')
        except Exception as e:
            logger.exception('Failed to Load Model for inference.')

    def prepare_feature(self,audio):
        try:
            input_val = logfbank(
                audio, 16000, nfilt=26)
            input_val = (input_val - np.mean(input_val)) / np.std(input_val)
            # transform in 3d array
            train_input = tf.ragged.constant([input_val], dtype=np.float32)
            # train_input = tf.expand_dims(train_input, axis=0)
            self.train_seq_len = tf.cast(train_input.row_lengths(), tf.int32)
            self.train_input = train_input.to_tensor(
                    default_value=self.FEAT_MASK_VALUE)
            logger.info('Successfully Extracted features form inputs.')

        except Exception as e:
            logger.exception('Failed to Extract features from inputs')

    def decode(self):
        try:
            decoded, _ = tf.nn.ctc_greedy_decoder(tf.transpose(self.model_pred(self.train_input), (1, 0, 2)), self.train_seq_len)
            d = tf.sparse.to_dense(decoded[0], default_value=-1).numpy()
            str_decoded = [''.join([self.alphabets['num_to_char'][str(x)]
                                    for x in np.asarray(row) if x != -1]) for row in d]
            
            check_augmentation = False
            pred_statement = ""
            for index, prediction in enumerate(str_decoded):
                if(check_augmentation):
                    prediction = prediction.replace(self.alphabets['num_to_char']['0'], ' ')
                    pred_statement += prediction
                else:
                    if(index % 7 == 0):
                        # Replacing space label to space
                        prediction = prediction.replace(self.alphabets['num_to_char']['0'], ' ')
                        pred_statement += prediction
            self.pred_statement = pred_statement
            logger.info(
                    'Successfully Produced Predicted Transcription.')

        except Exception as e:
            logger.exception(
                'Failed to Get Transcription for the inputs.')

    def get_prediction(self):
        try:
            return self.pred_statement
        except Exception as e:
            logger.exception(
                'Failed to Return Predictions.')
            

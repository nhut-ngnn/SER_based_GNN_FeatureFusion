class Config:

    DATA_PATH = 'Datasets/CASIA/6'
    CLASS_LABELS = ("angry", "fear", "happy", "neutral", "sad", "surprise")
    # CLASS_LABELS = ("positive", "negative", "neutral")
    # CLASS_LABELS = ("angry", "boredom", "disgust", "fear", "happy", "neutral", "sad")

    # Opensmile 
    CONFIG = 'IS10_paraling'
    # Opensmile 
    OPENSMILE_PATH = '/Users/zou/opensmile-2.3.0'
    FEATURE_NUM = {
        'IS09_emotion': 384,
        'IS10_paraling': 1582,
        'IS11_speaker_state': 4368,
        'IS12_speaker_trait': 6125,
        'IS13_ComParE': 6373,
        'ComParE_2016': 6373
    }

    FEATURE_PATH = 'Features/6-category/'
    TRAIN_FEATURE_PATH_OPENSMILE = FEATURE_PATH + 'train_opensmile_casia.csv'
    PREDICT_FEATURE_PATH_OPENSMILE = FEATURE_PATH + 'test_opensmile_casia.csv'
    TRAIN_FEATURE_PATH_LIBROSA = FEATURE_PATH + 'train_librosa_casia.p'
    PREDICT_FEATURE_PATH_LIBROSA = FEATURE_PATH + 'test_librosa_casia.p'

    MODEL_PATH = 'Models/'
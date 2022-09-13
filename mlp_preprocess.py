import os
import json
import librosa as lbs
import math

SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

DATASET_PATH = "musicdata/genres_original"
JSON_PATH = "data.json"

# store data in json for quick access when training

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, n_segments=5): 
    # number of segments chops up each track into segments to increase number of input data
    
    # build dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }
    
    n_samples_per_segment = int(SAMPLES_PER_TRACK / n_segments)
    expected_mfcc_vectors_per_segment = math.ceil(n_samples_per_segment / hop_length) # ceil rounds up number

    
    # fill data dictionary
    # loop through all genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        # ensure not at root of dataset
        if dirpath is not dataset_path:
            # save semantic label
            dirpath_components = dirpath.split("/")
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))
            
            # process files for a genre
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                signal, sr = lbs.load(file_path, sr=SAMPLE_RATE)
                
                
                # divide signal into segments and extract mfcc
                for s in range(n_segments):
                    start_sample = n_samples_per_segment * s
                    finish_sample = start_sample + n_samples_per_segment
                    
                        
                    mfcc = lbs.feature.mfcc(signal[start_sample:finish_sample],
                                            sr=sr,
                                            n_fft=n_fft,
                                            n_mfcc=n_mfcc,
                                            hop_length=hop_length)
                    
                    mfcc = mfcc.T # transpose
                    
                    # store mfcc for segment if it has the expected length
                    if len(mfcc) == expected_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        #print("{}, segment: {}".format(file_path, s))
                        
                        
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
        
if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, n_segments=5)
    print("finished!!!")
"""" Demo script for extracting acoustic features from interjection audio clips.
Currently set up to process **Surprise/Shock** clips as an example.
The workflow can be extended to other labels and languages.""""

import os
import librosa
import numpy as np
import pandas as pd
import parselmouth

folder = input("Enter folder containing audio clips: ")
output_csv = input("Enter path to save CSV: ")

all_results = []

for label in os.listdir(folder):
    label_path = os.path.join(folder, label)
    
    if os.path.isdir(label_path):
        print(f"Processing Label: {label}...")
        
        for file_name in os.listdir(label_path):
            if file_name.lower().endswith(".wav"):
                file_path = os.path.join(label_path, file_name)
                
                # --- LOAD AUDIO ---
                y, sr = librosa.load(file_path)
                snd = parselmouth.Sound(file_path)
                
                # --- BASIC ACOUSTIC ---
                duration = librosa.get_duration(y=y, sr=sr)
                
                # --- PITCH & MOVEMENT ---
                pitch = snd.to_pitch()
                mean_f0 = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")
                std_f0 = parselmouth.praat.call(pitch, "Get standard deviation", 0, 0, "Hertz")
                min_f0 = parselmouth.praat.call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
                max_f0 = parselmouth.praat.call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
                pitch_range = max_f0 - min_f0
                
                # Slope Calculation (Rise/Fall)
                pitch_values = pitch.selected_array['frequency']
                pitch_values = pitch_values[pitch_values > 0] # Ignore unvoiced
                slope = np.polyfit(range(len(pitch_values)), pitch_values, 1)[0] if len(pitch_values) > 1 else 0

                # --- SPECTRAL (Librosa) ---
                centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                # Spectral Flux (Onset Strength)
                flux = np.mean(librosa.onset.onset_strength(y=y, sr=sr))
                
                # --- PHONOLOGICAL (Formants) ---
                formants = snd.to_formant_burg()
                f1 = parselmouth.praat.call(formants, "Get mean", 1, 0, 0, "Hertz")
                f2 = parselmouth.praat.call(formants, "Get mean", 2, 0, 0, "Hertz")

                all_results.append({
                    "Label": label,
                    "Duration": duration,
                    "Pitch_Mean": mean_f0,
                    "Pitch_SD": std_f0,
                    "Pitch_Range": pitch_range,
                    "Pitch_Slope": slope,
                    "Spectral_Centroid": centroid,
                    "Spectral_Flux": flux,
                    "F1_Vowel_Height": f1,
                    "F2_Vowel_Backness": f2
                })

# 2. GENERATE MEAN TABLE
df = pd.DataFrame(all_results)
mean_table = df.groupby("Label").mean().reset_index()
mean_table.to_csv(output_csv, index=False)

print(f"\nSuccess! Table created: {output_csv}")
print(mean_table)

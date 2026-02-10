import parselmouth
import os
import numpy as np
import pandas as pd

# ===== FOLDER SETTINGS =====
folder = r"C:\Users\user\Desktop\TESTBATCH\SS"  # folder containing your WAV clips
output_csv = r"C:\Users\user\Desktop\TESTBATCH\SS\Suprise_results.csv"

# ===== DATAFRAME SETUP =====
columns = ["File Name", "Duration(s)", "Mean Pitch (Hz)", "SD Pitch (Hz)",
           "Mean Intensity (dB)", "HNR", "Vowel Type", "Consonant Type",
           "Number of Repeats", "Rhythm", "Label", "Context"]

df = pd.DataFrame(columns=columns)

# ===== LOOP THROUGH CLIPS =====
for file in os.listdir(folder):
    if file.lower().endswith(".wav"):
        path = os.path.join(folder, file)
        sound = parselmouth.Sound(path)

        # Duration
        duration = sound.get_total_duration()

        # Pitch (mean & SD)
        pitch = sound.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=300)
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values != 0]  # remove unvoiced
        if len(pitch_values) > 0:
            mean_pitch = np.mean(pitch_values)
            sd_pitch = np.std(pitch_values)
        else:
            mean_pitch = np.nan
            sd_pitch = np.nan

        # Intensity
        intensity = sound.to_intensity(time_step=0.01)
        intensity_values = intensity.values.T.flatten()
        mean_intensity = np.mean(intensity_values)

        # HNR (cleaned)
        harmonicity = sound.to_harmonicity_cc(time_step=0.01, minimum_pitch=75)
        hnr_values = harmonicity.values.T.flatten()
        # Remove NaNs and non-positive values
        hnr_values = hnr_values[~np.isnan(hnr_values)]
        hnr_values = hnr_values[hnr_values > 0]
        if len(hnr_values) > 0:
            hnr_mean = np.mean(hnr_values)
        else:
            hnr_mean = np.nan  # flag as invalid

        # PLACEHOLDERS for manual input
        vowel_type = ""
        consonant_type = ""
        number_of_repeats = ""
        rhythm = ""
        label = "Suprise/Shock"
        context = ""

        # Append row to dataframe
        df = pd.concat([df, pd.DataFrame([[file, duration, mean_pitch, sd_pitch,
                                           mean_intensity, hnr_mean, vowel_type, consonant_type,
                                           number_of_repeats, rhythm, label, context]],
                                         columns=columns)], ignore_index=True)

# ===== SAVE CSV =====
df.to_csv(output_csv, index=False)
print("CSV saved to:", output_csv)

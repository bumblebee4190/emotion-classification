# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 12:05:28 2025

@author: gourmetcai
"""
if __name__ == '__main__':
    
    
    import os
    import time
    import json
    import re
    import torch
    import pandas as pd
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from accelerate import disk_offload
    from sklearn.metrics import classification_report
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from dotenv import load_dotenv
    
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, f1_score
    from sklearn.calibration import calibration_curve
    from scipy.stats import chi2_contingency
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import joblib
    
    def va_map_to_emotion(valence, arousal):					
    # Tiny neutral core					
        if 0.47 <= valence <= 0.53 and 0.47 <= arousal <= 0.53:					
            return "unknown"					
     					
        # Positive					
        if valence >= 0.76:					
            if arousal >= 0.70: return "joy"					
            if arousal >= 0.50: return "love"					
            return "calm"           # anything else under 0.50 arousal					
     					
        # Negative					
        if valence <= 0.24:					
            if arousal >= 0.72: return "anger"					
            if arousal >= 0.48: return "fear"					
            if arousal <= 0.28: return "sadness"					
     					
        # Surprise band (narrow, very high arousal)					
        if 0.45 <= valence <= 0.55 and arousal >= 0.85:					
            return "surprise"					
     					
        # Mid‑valence fallback					
        return "love" if arousal >= 0.55 else "calm"					


    # === Parse & Confidence Model ===
    def parse_output(entry):
        text = entry["llm_output"]
        try:
            emotion = re.search(r'Emotion:\s*(\w+)', text).group(1)
            confidence = int(re.search(r'Confidence:\s*(\d+)', text).group(1))
            reason = re.search(r'Reason:\s*(.*)', text, re.DOTALL).group(1)
            artist = entry["artist"]
            title = entry["title"]
            return emotion, confidence, reason, artist, title
        except:
            return None, None, None, None, None
    
    def loadjson(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                return data
        except:
            None
    
    def mergeData(df,data):
        
        # base_path = 'F:\\music4all'
    
        # df_info = pd.read_csv(os.path.join(base_path, 'id_information.csv'), sep='\t')
        # df1 = pd.merge(df,df_info,left_on=['id'],right_on=['id'])
        
        
        records = []
        for d in data:
            emotion, confidence, reason, artist, title = parse_output(d)
            if emotion and confidence and reason:
                records.append({
                    "lyrics": d["lyrics"],
                    "true_emotion": d["true_emotion"],
                    "gpt_emotion": emotion,
                    "confidence": confidence,
                    "artist" : artist,
                    "song" : title
                    # "reason": reason
                })
                
        df_result = pd.DataFrame(records)
        
        merged_df_result = pd.merge(df_result,df, left_on=['artist','song'],right_on=['artist','song'])
        merged_df_result = merged_df_result[merged_df_result['gpt_emotion']!='confusion']
        merged_df_result['new_emotion']=merged_df_result.apply(lambda x: va_map_to_emotion(x.valence,x.arousal),axis=1)
        return merged_df_result
    
    def compute_ece(y_true, y_proba, n_bins=10):
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        confidences = np.max(y_proba, axis=1)
        predictions = np.argmax(y_proba, axis=1)
        accuracies = (predictions == y_true)
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            mask = (confidences > bin_lower) & (confidences <= bin_upper)
            if np.any(mask):
                bin_accuracy = np.mean(accuracies[mask])
                bin_confidence = np.mean(confidences[mask])
                ece += np.abs(bin_confidence - bin_accuracy) * np.sum(mask) / len(y_true)
        return ece   
    
    def fitModel(df):
        if 'id' in df.columns:
            df = df.drop(columns=["id"])
            
        label_encoder = LabelEncoder()
        df["emotion_encoded"] = label_encoder.fit_transform(df["new_emotion"])
        # tfidf = TfidfVectorizer(max_features=500)
        # X = tfidf.fit_transform(df["lyrics_x"])
        X = df[["arousal","valence","tempo","danceability"]]
        # X = df[["arousal","valence"]]
        y = df["emotion_encoded"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        print("=== Classification Report ===\n")
        print(report)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        print(f"Macro F1 Score: {macro_f1:.4f}")
        kappa = cohen_kappa_score(y_test, y_pred)
        print(f"Cohen's Kappa: {kappa:.4f}")
        contingency = pd.crosstab(y_test, y_pred)
        chi2, p, dof, expected = chi2_contingency(contingency)
        print(f"Chi-squared Test: chi2={chi2:.2f}, p={p:.4f}, dof={dof}")
        
        ece = compute_ece(y_test.to_numpy(), y_proba)
        print(f"Expected Calibration Error (ECE): {ece:.4f}")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("F:\\musicresult\\lyrics_confusion_matrix.png")
        plt.show()
        # # joblib.dump(clf, "output/audio_rf_model.pkl")
        # # joblib.dump(label_encoder, "output/audio_label_encoder.pkl")
        # # joblib.dump(scaler, "output/audio_scaler.pkl")
        
        plt.figure(figsize=(6, 6))
        for i in range(len(label_encoder.classes_)):
            prob_true, prob_pred = calibration_curve((y_test == i).astype(int), y_proba[:, i], n_bins=10)
            plt.plot(prob_pred, prob_true, label=f"{label_encoder.classes_[i]}")
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            plt.xlabel("Predicted confidence")
            plt.ylabel("True probability")
            plt.title("Calibration Curves per Class")
            plt.legend()
            plt.grid(True)
     
        plt.savefig("F:\\musicresult\\lyrics_cali.png")
        plt.show()

        
    def build_confidence_model1(df):
    
    
        # df["correct"] = (df["gpt_emotion"].str.lower() == df["true_emotion"].str.lower()).astype(int)
    
        tfidf = TfidfVectorizer(max_features=500)
        X = tfidf.fit_transform(df["lyrics_x"])
        label_encoder = LabelEncoder()
        df["emotion_encoded"] = label_encoder.fit_transform(df["new_emotion"])

        y = df["emotion_encoded"]
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # model = GradientBoostingRegressor()
        # model.fit(X_train, y_train)
        y_pred1 = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred1)
        print(f"Confidence regression MSE: {mse}")
        
        # y_true = df["emotion_encoded"]
        # y_pred = model.predict(X)
        
        y_true = y_test
        y_pred = y_pred1

        
        print(classification_report(y_true, y_pred, labels=label_encoder.classes_.tolist()))
        # print(classification_report(y_true, y_pred))        
        
        #macro f1
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        print(f"Macro F1 Score: {macro_f1:.4f}")
        
        #cohen's kappa
        kappa = cohen_kappa_score(y_true, y_pred)
        print(f"Cohen's Kappa: {kappa:.4f}")

        #x^2
        contingency = pd.crosstab(y_true, y_pred)
        chi2, p, dof, expected = chi2_contingency(contingency)
        print(f"Chi-squared Test: chi2={chi2:.2f}, p={p:.4f}, dof={dof}")

        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {accuracy:.5f}")
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap="Blues")
        # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")        
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("F:\\musicresult\\newlyrics_confusion_matrix.png")
        plt.show()
        
        y_proba = model.predict_proba(X_test)
        plt.figure(figsize=(6, 6))
        for i in range(len(label_encoder.classes_)):
            prob_true, prob_pred = calibration_curve((y_test == i).astype(int), y_proba[:, i], n_bins=10)
            plt.plot(prob_pred, prob_true, label=f"{label_encoder.classes_[i]}")
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            plt.xlabel("Predicted confidence")
            plt.ylabel("True probability")
            plt.title("Calibration Curves per Class")
            plt.legend()
            plt.grid(True)
     
        plt.savefig("F:\\musicresult\\lyrics_cali.png")
        plt.show()

    
    df = pd.read_csv('F:\\musicresult\\dflyricsoutput.csv')
    data = loadjson("F:\\musicresult\\openchat_results6")  
    df1 = mergeData(df, data)
    # aaa = df[df['gpt_emotion']!='confusion']
    # df =  pd.read_csv("F:\\musicresult\\mergedresult1.csv")
    
    
    # df = pd.read_csv("F:\\musicresult\\dfaudiofeatures.csv")    
    # data = loadjson("F:\\musicresult\\openchat_results6")      
    # mergeData(df, data)
    # fitModel(df1)
    build_confidence_model1(df1)
    
    
    

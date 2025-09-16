import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class AdvancedTrafficAnalysis:
    def __init__(self):
        self.models = {}
        self.evaluation_results = {}
        
    def load_data(self):
        print("Učitavanje podataka")
        try:
            self.df1 = pd.read_csv('Traffic.csv')
            self.df2 = pd.read_csv('TrafficTwoMonth.csv')
            
            # Kombinuj podatke
            self.df = pd.concat([self.df1, self.df2], ignore_index=True)
            print(f"Podaci učitani: {self.df.shape[0]} redova, {self.df.shape[1]} kolona")
            return True
        except Exception as e:
            print(f"Greška pri učitavanju: {e}")
            return False
    
    def preprocess_data(self):
        print("\nPriprema podataka")
        
        # Izvuci sat iz Time kolone
        if 'Time' in self.df.columns:
            self.df['Hour'] = pd.to_datetime(self.df['Time'], format='%I:%M:%S %p').dt.hour
        
        # Enkoduj dan u nedelji
        if 'Day of the week' in self.df.columns:
            day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 
                          'Friday': 5, 'Saturday': 6, 'Sunday': 7}
            self.df['DayNum'] = self.df['Day of the week'].map(day_mapping)
            
            
            self.df['IsWeekend'] = self.df['DayNum'].apply(lambda x: 1 if x in [6, 7] else 0)
        
        # Enkoduj situaciju u saobraćaju
        if 'Traffic Situation' in self.df.columns:
            situation_mapping = {'low': 1, 'normal': 2, 'high': 3, 'heavy': 4}
            self.df['TrafficLevel'] = self.df['Traffic Situation'].map(situation_mapping)
        
        # Kreiraj dodatne features
        self.df['TotalVehicles'] = (self.df['CarCount'] + self.df['BikeCount'] + 
                                   self.df['BusCount'] + self.df['TruckCount'])
        self.df['CarRatio'] = self.df['CarCount'] / (self.df['TotalVehicles'] + 1)
        self.df['CommercialRatio'] = (self.df['BusCount'] + self.df['TruckCount']) / (self.df['TotalVehicles'] + 1)
        
        print("Podaci pripremljeni!")
    
    def train_and_evaluate_models(self):
        print("\n" + "="*60)
        print("KREIRANJE I EVALUACIJA MODELA REGRESIJE")
        print("="*60)
        
        # Pripremi podatke za modelovanje
        features = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Hour', 'DayNum']
        X = self.df[features].fillna(0)
        y = self.df['Total']
        
        # Podeli podatke na trening i test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Trening set: {X_train.shape[0]} uzoraka")
        print(f"Test set: {X_test.shape[0]} uzoraka")
        
        # MODEL 1: LINEAR REGRESSION
        print("\n--- MODEL 1: LINEAR REGRESSION ---")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        
        r2_lr = r2_score(y_test, y_pred_lr)
        rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
        mae_lr = mean_absolute_error(y_test, y_pred_lr)
        sse_lr = np.sum((y_test - y_pred_lr) ** 2)
        
        self.models['Linear Regression'] = lr
        self.evaluation_results['Linear Regression'] = {
            'R²': r2_lr, 'RMSE': rmse_lr, 'MAE': mae_lr, 'SSE': sse_lr
        }
        
        print(f"R² (koeficijent determinacije): {r2_lr:.4f}")
        print(f"RMSE: {rmse_lr:.4f}")
        print(f"MAE: {mae_lr:.4f}")
        print(f"SSE: {sse_lr:.4f}")
        
        # MODEL 2: RIDGE REGRESSION
        print("\n--- MODEL 2: RIDGE REGRESSION ---")
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        y_pred_ridge = ridge.predict(X_test)
        
        r2_ridge = r2_score(y_test, y_pred_ridge)
        rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
        mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
        sse_ridge = np.sum((y_test - y_pred_ridge) ** 2)
        
        self.models['Ridge Regression'] = ridge
        self.evaluation_results['Ridge Regression'] = {
            'R²': r2_ridge, 'RMSE': rmse_ridge, 'MAE': mae_ridge, 'SSE': sse_ridge
        }
        
        print(f"R² (koeficijent determinacije): {r2_ridge:.4f}")
        print(f"RMSE: {rmse_ridge:.4f}")
        print(f"MAE: {mae_ridge:.4f}")
        print(f"SSE: {sse_ridge:.4f}")
        
        # MODEL 3: RANDOM FOREST REGRESSION
        print("\n--- MODEL 3: RANDOM FOREST REGRESSION ---")
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        
        r2_rf = r2_score(y_test, y_pred_rf)
        rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
        mae_rf = mean_absolute_error(y_test, y_pred_rf)
        sse_rf = np.sum((y_test - y_pred_rf) ** 2)
        
        self.models['Random Forest'] = rf
        self.evaluation_results['Random Forest'] = {
            'R²': r2_rf, 'RMSE': rmse_rf, 'MAE': mae_rf, 'SSE': sse_rf
        }
        
        print(f"R² (koeficijent determinacije): {r2_rf:.4f}")
        print(f"RMSE: {rmse_rf:.4f}")
        print(f"MAE: {mae_rf:.4f}")
        print(f"SSE: {sse_rf:.4f}")
        
        # POREĐENJE MODELA
        print("\n" + "="*50)
        print("POREĐENJE MODELA")
        print("="*50)
        
        best_r2 = max(self.evaluation_results.keys(), 
                     key=lambda x: self.evaluation_results[x]['R²'])
        best_rmse = min(self.evaluation_results.keys(), 
                       key=lambda x: self.evaluation_results[x]['RMSE'])
        
        print(f"Najbolji R² score: {best_r2} ({self.evaluation_results[best_r2]['R²']:.4f})")
        print(f"Najmanji RMSE: {best_rmse} ({self.evaluation_results[best_rmse]['RMSE']:.4f})")
        
        return X_test, y_test
        
    def answer_research_questions(self):
        print("\n" + "="*80)
        print("ODGOVORI NA 4 KLJUČNA PITANJA REGRESIJSKE ANALIZE")
        print("="*80)
        
        best_model_name = max(self.evaluation_results.keys(), 
                             key=lambda x: self.evaluation_results[x]['R²'])
        best_model = self.models[best_model_name]
        
        print("\n1. KOJI FAKTOR NAJVIŠE UTIČE NA PREDVIĐANJE SAOBRAĆAJA?")
        print("-" * 80)
        
        rf = self.models['Random Forest']
        feature_names = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Hour', 'DayNum']
        importance_dict = dict(zip(feature_names, rf.feature_importances_))
        
        print("Važnost faktora (Random Forest):")
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_importance):
            print(f"   {i+1}. {feature}: {importance:.4f}")
        
        most_important = sorted_importance[0][0]
        print(f"\n ODGOVOR: {most_important} NAJVIŠE utiče na predviđanje saobraćaja")
        print(f"   Važnost: {sorted_importance[0][1]:.4f}")
        
        print("\n2. U KOJE DOBA DANA JE SAOBRAĆAJ NAJINTENZIVNIJI?")
        print("-" * 80)
        
        hourly_traffic = self.df.groupby('Hour')['Total'].agg(['mean', 'max', 'std']).round(2)
        top_hours = hourly_traffic.sort_values('mean', ascending=False).head(3)
        
        print("Top 3 sata sa najvećim prosečnim saobraćajem:")
        for hour, data in top_hours.iterrows():
            print(f"   {hour:02d}:00 - Prosek: {data['mean']:.1f}, Maksimum: {data['max']:.0f}, Std: {data['std']:.1f}")
        
        peak_hour = top_hours.index[0]
        print(f"\n ODGOVOR: Najintenzivniji saobraćaj je u {peak_hour:02d}:00")
        
        print("\n3. KAKVA JE GUSTINA SAOBRAĆAJA PO DANIMA U NEDELJI?")
        print("-" * 80)
        
        # Analiziraj distribuciju Traffic Situation po danima
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        traffic_situations = ['low', 'normal', 'high', 'heavy']
        
        print("Distribucija gustine saobraćaja po danima:")
        daily_traffic_density = {}
        
        for day in day_names:
            day_data = self.df[self.df['Day of the week'] == day]['Traffic Situation']
            if len(day_data) > 0:
                situation_counts = day_data.value_counts()
                total_count = len(day_data)
                
                daily_traffic_density[day] = {}
                print(f"\n{day}:")
                
                for situation in traffic_situations:
                    count = situation_counts.get(situation, 0)
                    percentage = (count / total_count) * 100 if total_count > 0 else 0
                    daily_traffic_density[day][situation] = percentage
                    print(f"   {situation}: {count} ({percentage:.1f}%)")
                
                # Najčešća situacija za taj dan
                most_common = situation_counts.index[0] if len(situation_counts) > 0 else 'N/A'
                print(f"   Najčešća situacija: {most_common}")

        print("\n DETALJNE METRIKE PO DANIMA:")
        print("-" * 50)
        
        weights = {'low': 1, 'normal': 2, 'high': 3, 'heavy': 4}
        daily_scores = {}
        problem_scores = {}
        
        for day in day_names:
            if day in daily_traffic_density:
                data = daily_traffic_density[day]
                
                # Weighted score (1-4 skala) - VEĆI = GORI
                weighted_score = sum(data.get(situation, 0) * weight 
                                   for situation, weight in weights.items())
                daily_scores[day] = weighted_score
                
                # Problem score (high + heavy)
                problem_score = data.get('high', 0) + data.get('heavy', 0)
                problem_scores[day] = problem_score
                
                print(f"{day}:")
                for situation in traffic_situations:
                    print(f"   {situation}: {data.get(situation, 0):.1f}%")
                print(f"Weighted Score: {weighted_score:.1f} (veći = gori)")
                print(f"Problem Score: {problem_score:.1f}% (high+heavy)")
                print()
        
        # Različiti načini određivanja najgoreg dana
        worst_day_weighted = max(daily_scores, key=daily_scores.get) if daily_scores else 'N/A'
        best_day_weighted = min(daily_scores, key=daily_scores.get) if daily_scores else 'N/A'
        
        worst_day_problems = max(problem_scores, key=problem_scores.get) if problem_scores else 'N/A'
        
        heavy_percentages = {day: data.get('heavy', 0) for day, data in daily_traffic_density.items()}
        worst_day_heavy = max(heavy_percentages, key=heavy_percentages.get) if heavy_percentages else 'N/A'
        
        low_percentages = {day: data.get('low', 0) for day, data in daily_traffic_density.items()}
        best_day_low = max(low_percentages, key=low_percentages.get) if low_percentages else 'N/A'
        
        print("RAZLIČITI NAČINI ANALIZE:")
        print("-" * 50)
        print(f"Najgori dan (Weighted Score): {worst_day_weighted} ({daily_scores.get(worst_day_weighted, 0):.1f})")
        print(f"Najbolji dan (Weighted Score): {best_day_weighted} ({daily_scores.get(best_day_weighted, 0):.1f})")
        print(f"Najgori dan (High+Heavy): {worst_day_problems} ({problem_scores.get(worst_day_problems, 0):.1f}%)")
        print(f"Najgori dan (Samo Heavy): {worst_day_heavy} ({heavy_percentages.get(worst_day_heavy, 0):.1f}%)")
        print(f"Najbolji dan (Najviše Low): {best_day_low} ({low_percentages.get(best_day_low, 0):.1f}%)")
        
        # Koristi weighted score kao glavnu metriku
        worst_day = worst_day_weighted
        best_day = best_day_weighted
        
        print(f"\n FINALNI ODGOVOR (na osnovu Weighted Score):")
        print(f"Najgori dan: {worst_day} (Score: {daily_scores.get(worst_day, 0):.1f})")
        print(f"Najbolji dan: {best_day} (Score: {daily_scores.get(best_day, 0):.1f})")
        print(f"\nOBJAŠNJENJE: Weighted Score = Low×1 + Normal×2 + High×3 + Heavy×4")
        print(f"   Uzima u obzir SVE situacije, ne samo jednu kategoriju!")
        
       
    def run_analysis(self):
        if not self.load_data():
            return
        
        self.preprocess_data()
        X_test, y_test = self.train_and_evaluate_models()
        self.answer_research_questions()
        #self.create_individual_visualizations(X_test, y_test)
        
        print("\n" + "="*60)
        print("KOMPLETNA ANALIZA ZAVRŠENA!")
        print("="*60)

# Pokreni analizu
if __name__ == "__main__":
    analyzer = AdvancedTrafficAnalysis()
    analyzer.run_analysis()
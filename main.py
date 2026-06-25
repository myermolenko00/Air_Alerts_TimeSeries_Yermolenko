import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Set elegant typography defaults globally
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# ==========================================================================
# 1. PIPELINE ENGINE OBJECT
# ==========================================================================
class ConflictForecastingPipeline:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.regional_data = None
        self.regions = ['West', 'East', 'Central', 'South']

    def load_and_clean(self):
        print("--- [1/4] Ingesting & Aggregating Dataset ---")
        self.df = pd.read_csv(self.file_path)
        self.df['started_at'] = pd.to_datetime(self.df['started_at'])

        region_mapping = {
            'Lvivska oblast': 'West', 'Ivano-Frankivska oblast': 'West', 'Volynska oblast': 'West',
            'Rivnenska oblast': 'West', 'Ternopilska oblast': 'West', 'Khmelnytska oblast': 'West',
            'Chernivetska oblast': 'West', 'Zakarpatska oblast': 'West',
            'Vinnytska oblast': 'Central', 'Zhytomyrska oblast': 'Central', 'Cherkaska oblast': 'Central',
            'Kirovohradska oblast': 'Central', 'Kyivska oblast': 'Central', 'Kyiv City': 'Central',
            'Chernihivska oblast': 'Central', 'Poltavska oblast': 'Central', 'Sumska oblast': 'Central',
            'Donetska oblast': 'East', 'Dnipropetrovska oblast': 'East', 'Kharkivska oblast': 'East',
            'Luhanska oblast': 'East',
            'Mykolaivska oblast': 'South', 'Odeska oblast': 'South', 'Zaporizka oblast': 'South',
            'Khersonska oblast': 'South'
        }

        self.df['region'] = self.df['oblast'].map(region_mapping)
        self.df['date'] = self.df['started_at'].dt.date

        # Build daily matrix
        self.regional_data = self.df.groupby(['date', 'region']).size().unstack(fill_value=0)
        self.regional_data.index = pd.to_datetime(self.regional_data.index)
        self.regional_data = self.regional_data.asfreq('D', fill_value=0)

    def build_exogenous_features(self, index_range):
        """Constructs a DataFrame of symbolic/strategic holiday flags (Exogenous Matrix)"""
        exog = pd.DataFrame(index=index_range)
        exog['is_weekend'] = exog.index.dayofweek.isin([5, 6]).astype(int)
        exog['symbolic_date'] = 0
        exog.loc[exog.index.month == 1, 'symbolic_date'] = 1
        exog.loc[exog.index.month == 12, 'symbolic_date'] = 1
        exog.loc[(exog.index.month == 8) & (exog.index.day.isin([23, 24, 25])), 'symbolic_date'] = 1
        return exog

    def execute_cross_validation(self, region, train_end, val_days=45):
        """Performs rigorous Time Series Backtesting to measure absolute performance error"""
        total_data = self.regional_data[region]
        cv_train_end = pd.to_datetime(train_end) - pd.Timedelta(days=val_days)

        y_train = total_data.loc[:cv_train_end]
        y_val = total_data.loc[cv_train_end + pd.Timedelta(days=1):pd.to_datetime(train_end)]

        exog_total = self.build_exogenous_features(total_data.index)
        exog_train = exog_total.loc[y_train.index]
        exog_val = exog_total.loc[y_val.index]

        cv_model = SARIMAX(y_train, exog=exog_train, order=(2, 1, 1), seasonal_order=(1, 0, 0, 7)).fit(maxiter=200, method='nm', disp=False)
        cv_preds = cv_model.forecast(steps=len(y_val), exog=exog_val)
        cv_preds = np.clip(cv_preds, 0, None)

        mae = mean_absolute_error(y_val, cv_preds)
        rmse = root_mean_squared_error(y_val, cv_preds)
        return mae, rmse

    def generate_forecast(self, cutoff_date, end_date):
        print("--- [2/4] Training Production SARIMAX Models & Running Validation ---")
        history = self.regional_data.loc[:cutoff_date].copy()
        future_dates = pd.date_range(start=pd.to_datetime(cutoff_date) + pd.Timedelta(days=1), end=end_date, freq='D')

        exog_total = self.build_exogenous_features(self.regional_data.index.union(future_dates))
        exog_hist = exog_total.loc[history.index]
        exog_future = exog_total.loc[future_dates]

        forecast_results = pd.DataFrame(index=future_dates)
        performance_metrics = {}

        np.random.seed(42)

        for region in self.regions:
            mae, rmse = self.execute_cross_validation(region, cutoff_date)
            performance_metrics[region] = {'MAE': mae, 'RMSE': rmse}
            print(f" > {region} Region Backtest Error -> MAE: {mae:.2f} alerts, RMSE: {rmse:.2f}")

            model = SARIMAX(history[region], exog=exog_hist, order=(2, 1, 1), seasonal_order=(1, 0, 0, 7))
            fitted_model = model.fit(maxiter=200, method='nm', disp=False)

            trend_pred = fitted_model.forecast(steps=len(future_dates), exog=exog_future)
            forecast_results[f'{region}_Trend'] = np.clip(trend_pred.values, 0, None)

            res_std = fitted_model.resid.std()
            simulated_peaks = []
            recent_shocks = list(fitted_model.resid.iloc[-3:].values)

            for idx in range(len(future_dates)):
                base_val = forecast_results[f'{region}_Trend'].iloc[idx]
                if np.random.rand() > 0.88:
                    spike = max(0, np.random.standard_t(df=3) * 2.3 * res_std)
                else:
                    spike = np.random.normal(0, res_std * 0.4)

                propagated = spike + (0.35 * recent_shocks[-1])
                recent_shocks.append(propagated)
                simulated_peaks.append(max(0, base_val + propagated))

            forecast_results[f'{region}_Peaks'] = simulated_peaks
            forecast_results[f'{region}_Upper95'] = forecast_results[f'{region}_Trend'] + (1.96 * res_std * (1 + 0.04 * np.sqrt(idx)))
            forecast_results[f'{region}_Lower95'] = np.clip(forecast_results[f'{region}_Trend'] - (1.28 * res_std), 0, None)

        return history, forecast_results, performance_metrics

# ==========================================================================
# 2. EXECUTIVE VISUALIZATION INTERFACE
# ==========================================================================
def plot_premium_dashboard(history, forecast, metrics):
    print("--- [3/4] Architecting Production Dashboard Layout ---")
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 11))
    axes_list = axes.flatten()

    colors = ['#1e4620', '#8b0000', '#0f4c81', '#d97706']

    fig.suptitle("M A C R O - R E G I O N A L   A I R   R A I D   A L E R T   F O R E C A S T I N G", fontsize=15, fontweight='bold', color='#111827', y=0.96)
    fig.text(0.5, 0.92,
             "Data Specification: 1 alert in 1 oblast = 1 unique count. Simultaneous regional activations accumulate concurrently.",
             ha='center', fontsize=11, color='#7f1d1d', style='italic', fontweight='medium')

    for i, region in enumerate(metrics.keys()):
        ax = axes_list[i]
        color = colors[i]

        hist_slice = history.loc['2026-01-01':, region]

        ax.plot(hist_slice.index, hist_slice, label='Observed History', color='#9ca3af', linewidth=1.1, alpha=0.5)
        ax.fill_between(forecast.index, forecast[f'{region}_Lower95'], forecast[f'{region}_Upper95'], color=color, alpha=0.03, label='95% Predictive Range')
        ax.plot(forecast.index, forecast[f'{region}_Peaks'], label='Forecast with Kinetic Peaks', color=color, linestyle='-', linewidth=1.3, alpha=0.9)
        ax.plot(forecast.index, forecast[f'{region}_Trend'], label='Exogenous Baseline Mean', color='#111827', linestyle='--', linewidth=1.4, alpha=0.75)

        ax.axvline(pd.to_datetime('2026-06-15'), color='#4b5563', linestyle=':', linewidth=1.2)

        mae_score = metrics[region]['MAE']
        rmse_score = metrics[region]['RMSE']
        metric_box_text = f"Backtest Validation:\nMAE: {mae_score:.2f}\nRMSE: {rmse_score:.2f}"
        ax.text(0.03, 0.06, metric_box_text, transform=ax.transAxes, fontsize=8.5, fontweight='semibold',
                color='#1f2937', bbox=dict(facecolor='#f3f4f6', edgecolor='none', boxstyle='round,pad=0.5', alpha=0.85))

        ax.set_title(f"{region.upper()} UKRAINE REGIONAL COMPLEX", fontsize=11.5, fontweight='bold', color='#1f2937', loc='left', pad=8)
        ax.set_ylabel('Daily Alerts Count', fontsize=9.5, fontweight='medium', color='#4b5563')
        ax.grid(axis='y', linestyle='-', linewidth=0.5, color='#e5e7eb')
        ax.grid(axis='x', visible=False)

        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(False)

        ax.legend(loc='upper right', frameon=True, facecolor='#f9fafb', edgecolor='none', fontsize=8.5)
        ax.tick_params(axis='both', which='major', labelsize=9, labelcolor='#4b5563')

        if i >= 2:
            ax.set_xlabel('Timeline Operational Horizon', fontsize=9.5, fontweight='medium', color='#4b5563', labelpad=8)

    plt.tight_layout()
    plt.subplots_adjust(top=0.86, hspace=0.38, wspace=0.22)

    # CRITICAL FIX: Save MUST occur before running plt.show()
    output_filename = "macro_regional_forecast.png"
    print(f"Saving high-resolution dashboard asset to: {os.path.abspath(output_filename)}")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')

    print("--- [4/4] Rendering Output Engine Canvas ---")
    plt.show()

# ==========================================================================
# 3. EXECUTION ENTRANCE ROUTINE
# ==========================================================================
if __name__ == "__main__":
    pipeline = ConflictForecastingPipeline(file_path="official_data_en.csv")
    pipeline.load_and_clean()

    hist_df, fore_df, error_metrics = pipeline.generate_forecast(cutoff_date='2026-06-15', end_date='2026-12-31')
    plot_premium_dashboard(hist_df, fore_df, error_metrics)
    print("Process Finished Successfully.")
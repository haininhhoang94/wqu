dw = durbin_watson(pd.to_numeric(nyse_hist_10_years.Close).pct_change().dropna().values)
print(f"DW-statistic of {dw}")

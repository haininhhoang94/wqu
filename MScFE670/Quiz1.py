# %%
import pandas as pd
import numpy as np

# %%
country_data = pd.read_excel("./CountryData.xlsx")
country_data.set_index("CountryName", inplace=True)
np.mean(country_data.GDP)

# %%
country_data.LifeExpectancy["Japan"]
country_data.LifeExpectancy["Hong Kong SAR, China"]
country_data.LifeExpectancy["Italy"]
country_data.LifeExpectancy["Iceland"]
country_data.columns

# %%
columns_ = np.array([])
null_counts = np.array([])

for col in country_data.columns:
    columns_ = np.append(columns_, col)
    null_count = country_data[col].isna().sum()
    null_counts = np.append(null_counts, null_count)

print(null_counts)
print(columns_)
# null_count_pd = pd.DataFrame(null_counts, index=columns_)
np.where(null_counts == np.max(null_counts))
columns_[5]


# %%
np.median(country_data.GDP.dropna())
# %%


